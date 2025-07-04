#[macro_use]
extern crate log;

use crate::radixtrie_kv_router::{DstProcess, RadixTrieKvRouter};
use anyhow::{anyhow, ensure, Context as anyhow_ctx, Result};
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::request::Parts;
use axum::http::HeaderValue;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use chat_prompts::chat::{BuildChatPrompt, ChatPrompt};
use chat_prompts::PromptTemplateType;
use chrono::Utc;
use clap::Parser;
use endpoints::chat::{ChatCompletionRequest, ChatCompletionRequestMessage, Tool};
use http_body_util::BodyExt;
use hyper_util::rt::TokioIo;
use log::LevelFilter;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use parking_lot::{ RwLock};
use reqwest::{Client, Url};
use serde::Deserialize;
use std::net::{IpAddr, SocketAddr};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::Arc;
use stream_guard::GuardStreamExt;
use tokenizers::{ Tokenizer};
use tokio::net::TcpStream;

mod radixtrie_kv_router;

type TokenId = u32;

/// SGLang Load Balancer Server
#[derive(Parser)]
#[command(version)]
struct Args {
    /// URLs for prefill servers
    #[arg(short, long)]
    prefill: Vec<Url>,

    /// URLs for decode servers
    #[arg(short, long)]
    decode: Vec<Url>,

    /// Bootstrap ports for prefill servers
    #[arg(long)]
    prefill_bootstrap_ports: Vec<u16>,

    /// Host to bind the server (default: 0.0.0.0)
    #[arg(short, long, default_value = "0.0.0.0")]
    host: IpAddr,

    /// Port to bind the server (default: 8000)
    #[arg(short, long, default_value_t = 8000)]
    port: u16,

    #[arg(long)]
    prompt_template_type: PromptTemplateType,

    #[command(flatten)]
    score_weight: ScoreWeight,

    #[arg(long)]
    model_repo: String,

    #[arg(long, default_value_t = 1)]
    prefill_dp_size: usize,

    #[arg(long, default_value_t = 1)]
    decode_dp_size: usize,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct PrefillConfig {
    url: Url,
    bootstrap_port: Option<u16>
}

#[derive(Copy, Clone, clap::Args)]
struct ScoreWeight {
    task_queue: f64,
    num_of_cache_token: f64
}

struct Context {
    // prefill servers: (config, dp node guard)
    prefill_servers: RwLock<Vec<(PrefillConfig, Vec<Arc<()>>)>>,
    decode_servers: RwLock<Vec<(Url, Vec<Arc<()>>)>>,
    client: Client,
    room_id: AtomicU32,
    radix_trie_kv_router: RwLock<RadixTrieKvRouter>,
    tokenizer: Tokenizer,
    chat_prompt: ChatPrompt,
    score_weight: ScoreWeight,
    prefill_dp_size: usize,
    decode_dp_size: usize,
}

struct ServerPair {
    prefill_server: Url,
    prefill_rank: usize,
    prefill_server_guard: Arc<()>,
    bootstrap_port: Option<u16>,

    decode_server: Url,
    decode_rank: usize,
    decode_server_guard: Arc<()>,
}

impl Context {
    fn select_pair(&self, prompts: (&mut Vec<ChatCompletionRequestMessage>, Option<&[Tool]>)) -> Result<ServerPair> {
        let (prefill_server, bootstrap_port, prefill_rank, prefill_server_guard) = {
            let raw_prompts= self.chat_prompt.build_with_tools(prompts.0, prompts.1)?;
            let encoding = self.tokenizer.encode_fast(raw_prompts, false)
                .map_err(|e| anyhow::anyhow!("failed to encode tokens: {}", e))?;

            let cache_prefill = self.radix_trie_kv_router.read().get(encoding.get_ids());

            let (dst, c) = {
                let guard = self.prefill_servers.read();
                // score, prefill config, prefill server guard, select node timestamp
                let mut max_score_node: Option<(f64, DstProcess, &Arc<()>, Option<&Arc<AtomicI64>>)> = None;

                for (config, dp_count) in guard.iter() {
                    for rank in 0..self.prefill_dp_size {
                        let count = &dp_count[rank];
                        // w / (1 + task_queue)
                        let load_score = self.score_weight.task_queue / (1f64 + Arc::strong_count(count) as f64);

                        let dst = DstProcess {
                            node: config.clone(),
                            dp_rank: rank,
                        };
                        let select_node  = cache_prefill.get(&dst)
                            .map(|(cache_num, timestamp)| (*cache_num as f64, timestamp));

                        let cache_token = select_node.as_ref().map(|(v, _)| *v).unwrap_or(0f64);
                        // w * num of cache token / 100
                        let cache_score = self.score_weight.num_of_cache_token * cache_token / 100f64;

                        if let Some((max_score, _, _, _)) = max_score_node {
                            if load_score + cache_score > max_score {
                                max_score_node = Some((load_score + cache_score, dst, count, select_node.map(|(_, t)| t)));
                            }
                        } else {
                            max_score_node = Some((load_score + cache_score, dst, count, select_node.map(|(_, t)| t)));
                        }
                    }
                }

                let select_node= max_score_node.unwrap();

                if let Some(v) = select_node.3 {
                    v.store(Utc::now().timestamp(), Ordering::Relaxed);
                }
                (select_node.1.clone(), select_node.2.clone())
            };

            self.radix_trie_kv_router.write().update(encoding.get_ids(), dst.clone());
            (dst.node.url, dst.node.bootstrap_port, dst.dp_rank, c)
        };

        let (decode_server, decode_rank, decode_server_guard) = {
            let guard = self.decode_servers.read();
            let mut nodes = Vec::with_capacity(guard.len() * self.decode_dp_size);

            for (decode_server, dp_count) in guard.iter() {
                for (rank, c) in dp_count.iter().enumerate() {
                    nodes.push((decode_server.clone(), rank, c));
                }
            }

            let (decode_server, rank, c) = nodes.into_iter()
                .min_by_key(|(_, _, c)| Arc::strong_count(c))
                .ok_or_else(|| anyhow!("failed to find node"))?;

            (decode_server.clone(), rank, c.clone())
        };

        Ok(ServerPair {
            prefill_server,
            prefill_server_guard,
            prefill_rank,
            bootstrap_port,
            decode_server,
            decode_rank,
            decode_server_guard
        })
    }

    // prefill, decode
    fn all_serves(&self) -> (Vec<Url>, Vec<Url>) {
        let prefill_servers = {
            let guard = self.prefill_servers.read();

            guard.iter()
                .map(|(c, _)| c.url.clone())
                .collect::<Vec<_>>()
        };

        let decode_servers = {
            let guard = self.decode_servers.read();

            guard.iter()
                .map(|(c, _)| c.clone())
                .collect::<Vec<_>>()
        };

        (prefill_servers, decode_servers)
    }
}

#[derive(Copy, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Prefill,
    Decode
}

#[derive(Deserialize)]
struct PDRegistryRequest {
    mode: Mode,
    registry_url: Url,
    bootstrap_port: Option<u16>
}

async fn health() -> Response {
    Response::default()
}

async fn health_generate(
    State(ctx): State<Arc<Context>>,
) -> Response {
    let (prefill_servers, _decode_servers) = ctx.all_serves();

    let fut = async {
        for dst in prefill_servers.iter() {
            let dst = dst.join("health_generate")?;
            ctx.client.get(dst.clone()).send().await
                .with_context(|| format!("failed send health_generate to: {}", dst))?;
        }
        Result::<_, anyhow::Error>::Ok(())
    };

    match fut.await {
        Ok(_) => Response::default(),
        Err(e) => {
            error!("(/health_generate) Error: {:?}", e);

            Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("(/health_generate) Error: {:?}", e)))
                .unwrap()
        }
    }
}

async fn flush_cache(
    State(ctx): State<Arc<Context>>,
) -> Response {
    let (prefill_servers, decode_servers) = ctx.all_serves();

    let fut = async {
        for dst in prefill_servers.iter().chain(&decode_servers) {
            let dst = dst.join("flush_cache")?;
            ctx.client.get(dst.clone()).send().await
                .with_context(|| format!("failed send flush_cache to: {}", dst))?;
        }
        Result::<_, anyhow::Error>::Ok(())
    };

    match fut.await {
        Ok(_) => Response::default(),
        Err(e) => {
            error!("(/flush_cache) Error: {:?}", e);

            Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("(/flush_cache) Error: {:?}", e)))
                .unwrap()
        }
    }
}

async fn get_server_info(
    State(ctx): State<Arc<Context>>,
) -> Response {
    let (prefill_servers, decode_servers) = ctx.all_serves();

    let fut = async {
        let mut prefill_infos = Vec::with_capacity(prefill_servers.len());
        let mut decode_infos = Vec::with_capacity(decode_servers.len());

        for x in prefill_servers {
            let dst = x.join("get_server_info")?;
            let resp = ctx.client.get(dst.clone()).send().await
                .with_context(|| format!("failed send get_server_info to: {}", dst))?;

            let out: serde_json::Value = resp.json().await?;
            prefill_infos.push(out);
        }

        for x in decode_servers {
            let dst = x.join("get_server_info")?;
            let resp = ctx.client.get(dst.clone()).send().await
                .with_context(|| format!("failed send get_server_info to: {}", dst))?;

            let out: serde_json::Value = resp.json().await?;
            decode_infos.push(out);
        }

        let out = serde_json::json!({
            "prefill": prefill_infos,
            "decode": decode_infos
        });

        Result::<_, anyhow::Error>::Ok(out)
    };

    match fut.await {
        Ok(out) => {
            Response::new(axum::body::Body::from(out.to_string()))
        },
        Err(e) => {
            error!("(/get_server_info) Error: {:?}", e);

            Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("(/get_server_info) Error: {:?}", e)))
                .unwrap()
        }
    }
}

async fn get_model_info(
    State(ctx): State<Arc<Context>>,
) -> Response {
    let fut = async {
        let dst = (*ctx.prefill_servers.read())[0].0.url.join("get_model_info")?;
        let resp = ctx.client.get(dst).send().await?;
        let out = resp.text().await?;
        Result::<_, anyhow::Error>::Ok(out)
    };

    match fut.await {
        Ok(out) => {
            Response::new(axum::body::Body::from(out))
        }
        Err(e) => {
            error!("(/get_model_info) Error: {:?}", e);

            Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("(/get_model_info) Error: {:?}", e)))
                .unwrap()
        }
    }
}

fn get_request_batch_size(request: &serde_json::Map<String, serde_json::Value>) -> Option<usize> {
    if let Some(text_value) = request.get("text") {
        match text_value {
            serde_json::Value::String(_) => return None,
            serde_json::Value::Array(arr) => return Some(arr.len()),
            _ => return None,
        }
    }

    if let Some(input_ids_value) = request.get("input_ids") {
        if let serde_json::Value::Array(arr) = input_ids_value {
            if arr.first().map(|v| v.is_number()).unwrap_or(false) {
                return None;
            } else if arr.first().map(|v| matches!(v, serde_json::Value::Array(_))).unwrap_or(false) {
                return Some(arr.len());
            }
        }
    }

    None
}

async fn gen(
    mut parts: Parts,
    body: Bytes,
    dst: &str,
    stream_guard: Arc<()>,
) -> Result<Response> {
    let headers = &mut parts.headers;
    let authority = dst;

    let stream = TcpStream::connect(authority).await?;
    let stream = TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(stream).await?;
    tokio::spawn(conn);

    headers.insert("host", authority.parse()?);
    headers.insert("content-length", HeaderValue::from(body.len()));

    let body = axum::body::Body::from(body);
    let req = Request::from_parts(parts, body);
    let resp = sender.send_request(req).await?;

    let (resp_parts, resp_body) = resp.into_parts();
    let resp_stream = resp_body.into_data_stream();

    let resp_stream = resp_stream.guard(move || {
        let _guard = stream_guard;
    });

    let resp_body = axum::body::Body::from_stream(resp_stream);
    Ok(Response::from_parts(resp_parts, resp_body))
}

// todo
// async fn generate<Model: BuildChatPrompt>(
//     State(ctx): State<Arc<Context<Model>>>,
//     (parts, Json(mut body)): (Parts, Json<serde_json::Map<String, serde_json::Value>>)
// ) -> Response {
//     let fut = async {
//         let pair = ctx.select_pair()?;
//         let batch_size = get_request_batch_size(&body);
//
//         let bootstrap_host = pair.prefill_server.host_str().ok_or_else(|| anyhow!("prefill server host is empty"))?;
//         let bootstrap_host = serde_json::Value::from(bootstrap_host);
//
//         let bootstrap_port = pair.bootstrap_port.map(|port| serde_json::Value::from(port));
//
//         match batch_size {
//             Some(batch_size) => {
//                 let mut bootstrap_host_list = Vec::with_capacity(batch_size);
//                 let mut bootstrap_port_list = Vec::with_capacity(batch_size);
//                 let mut bootstrap_room_list = Vec::with_capacity(batch_size);
//
//                 for _ in 0..batch_size {
//                     bootstrap_host_list.push(bootstrap_host.clone());
//
//                     if let Some(bootstrap_port) = &bootstrap_port {
//                         bootstrap_port_list.push(bootstrap_port.clone());
//                     }
//
//                     bootstrap_room_list.push(ctx.room_id.fetch_add(1, Ordering::Relaxed));
//                 }
//
//                 body.insert(String::from("bootstrap_host"), serde_json::Value::from(bootstrap_host_list));
//
//                 if !bootstrap_port_list.is_empty() {
//                     body.insert(String::from("bootstrap_port"), serde_json::Value::from(bootstrap_port_list));
//                 }
//
//                 body.insert(String::from("bootstrap_room"), serde_json::Value::from(bootstrap_room_list));
//             }
//             None => {
//                 body.insert(String::from("bootstrap_host"), bootstrap_host);
//
//                 if let Some(bootstrap_port) = bootstrap_port {
//                     body.insert(String::from("bootstrap_port"), bootstrap_port);
//                 }
//
//                 body.insert(String::from("bootstrap_room"), serde_json::Value::from(ctx.room_id.fetch_add(1, Ordering::Relaxed)));
//             }
//         }
//
//         let body = serde_json::to_vec(&body)?;
//         let body = Bytes::from(body);
//
//         let prefill_fut = gen(parts.clone(), body.clone(), pair.prefill_server.authority(), pair.prefill_server_guard);
//         let decode_fut = gen(parts, body, pair.decode_server.authority(), pair.decode_server_guard);
//
//         let (prefill_resp, decode_resp) = tokio::try_join!(prefill_fut, decode_fut)?;
//         ensure!(prefill_resp.status().is_success());
//         let _body_bytes = prefill_resp.collect().await?;
//
//         Result::<_, anyhow::Error>::Ok(decode_resp)
//     };
//
//     match fut.await {
//         Ok(resp) => resp,
//         Err(e) => {
//             error!("(/generate) Error: {:?}", e);
//
//             Response::builder()
//                 .status(500)
//                 .body(axum::body::Body::from(format!("(/generate) Error: {:?}", e)))
//                 .unwrap()
//         }
//     }
// }

async fn chat_completions(
    State(ctx): State<Arc<Context>>,
    (parts, Json(mut body)): (Parts, Json<ChatCompletionRequest>)
) -> Response {
    let fut = async {
        let pair = ctx.select_pair((&mut body.messages, body.tools.as_ref().map(|v| v.as_slice())))?;
        let bootstrap_host = pair.prefill_server.host_str().ok_or_else(|| anyhow!("prefill server host is empty"))?;
        let bootstrap_port = pair.bootstrap_port.map(|port| serde_json::Value::from(port));

        let bootstrap_room = {
            let curr_room: u32 = ctx.room_id.fetch_add(ctx.prefill_dp_size as u32, Ordering::Relaxed);
            curr_room + pair.prefill_rank as u32
        };

        let mut body = serde_json::to_value(body)?;
        let body = body.as_object_mut().unwrap();

        body.insert(String::from("bootstrap_host"), serde_json::Value::from(bootstrap_host));
        if let Some(bootstrap_port) = bootstrap_port {
            body.insert(String::from("bootstrap_port"), bootstrap_port);
        }
        body.insert(String::from("bootstrap_room"), serde_json::Value::from(bootstrap_room));
        body.insert(String::from("decode_rank"), serde_json::Value::from(pair.decode_rank));

        let body = serde_json::to_vec(&body)?;
        let body = Bytes::from(body);

        let prefill_fut = gen(parts.clone(), body.clone(), pair.prefill_server.authority(), pair.prefill_server_guard);
        let decode_fut = gen(parts, body, pair.decode_server.authority(), pair.decode_server_guard);

        let (prefill_resp, decode_resp) = tokio::try_join!(prefill_fut, decode_fut)?;
        ensure!(prefill_resp.status().is_success());
        let _body_bytes = prefill_resp.collect().await?;

        Result::<_, anyhow::Error>::Ok(decode_resp)
    };

    match fut.await {
        Ok(resp) => resp,
        Err(e) => {
            error!("(/v1/chat/completions) Error: {:?}", e);

            Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("(/v1/chat/completions) Error: {:?}", e)))
                .unwrap()
        }
    }
}

async fn models(
    State(ctx): State<Arc<Context>>,
) -> Response {
    let fut = async {
        let prefill_server = ctx.prefill_servers.read()[0].0.url.clone();
        let dst = prefill_server.join("v1/models")?;
        let body = ctx.client.get(dst).send().await?.text().await?;

        Result::<_, anyhow::Error>::Ok(Response::new(Body::from(body)))
    };

    match fut.await {
        Ok(resp) => resp,
        Err(e) => {
            error!("(/v1/models) Error: {:?}", e);

            Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("(/v1/models) Error: {:?}", e)))
                .unwrap()
        }
    }
}

async fn register(
    State(ctx): State<Arc<Context>>,
    Json(req): Json<PDRegistryRequest>
) -> Response {
    match req.mode {
        Mode::Prefill => {
            let c = PrefillConfig {
                url: req.registry_url,
                bootstrap_port: req.bootstrap_port
            };

            let mut guard = ctx.prefill_servers.write();
            let mut dp_count = Vec::with_capacity(ctx.prefill_dp_size);

            for _ in 0..ctx.prefill_dp_size {
                dp_count.push(Arc::new(()));
            }
            guard.push((c, dp_count));
        }
        Mode::Decode => {
            let mut guard = ctx.decode_servers.write();

            let mut dp_count = Vec::with_capacity(ctx.decode_dp_size);

            for _ in 0..ctx.decode_dp_size {
                dp_count.push(Arc::new(()));
            }

            guard.push((req.registry_url, dp_count));
        }
    }
    Response::default()
}

fn logger_init() -> Result<()> {
    let pattern = if cfg!(debug_assertions) {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {f}:{L} - {m}{n}"
    } else {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {t} - {m}{n}"
    };

    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();

    let config = log4rs::Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(
            Root::builder()
                .appender("stdout")
                .build(LevelFilter::from_str(
                    &std::env::var("SGLANG_LOAD_BALANCER_LOG").unwrap_or_else(|_| String::from("INFO")),
                )?),
        )?;

    log4rs::init_config(config)?;
    Ok(())
}

fn launch(mut args: Args) -> Result<()> {
    logger_init()?;
    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        let prefill_dp_size = args.prefill_dp_size;
        let decode_dp_size = args.decode_dp_size;

        while args.prefill_bootstrap_ports.len() < args.prefill.len() {
            args.prefill_bootstrap_ports.push(0);
        }

        let prefill_servers = args.prefill.into_iter().zip(args.prefill_bootstrap_ports)
            .map(|(url, port)| {
                let config = PrefillConfig {
                    url,
                    bootstrap_port: {
                        if port == 0 {
                            None
                        } else {
                            Some(port)
                        }
                    },
                };
                let mut dp_count = Vec::with_capacity(prefill_dp_size);

                for _ in 0..prefill_dp_size {
                    dp_count.push(Arc::new(()));
                }

                (config, dp_count)
            })
            .collect::<Vec<_>>();

        let decode_servers = args.decode.into_iter()
            .map(|url| {
                let mut dp_count = Vec::with_capacity(decode_dp_size);

                for _ in 0..decode_dp_size {
                    dp_count.push(Arc::new(()));
                }
                (url, dp_count)
            })
            .collect::<Vec<_>>();

        let tokenizer = Tokenizer::from_pretrained(&args.model_repo, None)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

        let ctx = Context {
            prefill_servers: RwLock::new(prefill_servers),
            decode_servers: RwLock::new(decode_servers),
            client: Client::new(),
            room_id: AtomicU32::new(0),
            chat_prompt: ChatPrompt::from(args.prompt_template_type),
            score_weight: args.score_weight,
            tokenizer,
            radix_trie_kv_router: RwLock::new(RadixTrieKvRouter::new(10000)),
            decode_dp_size,
            prefill_dp_size
        };

        let ctx = Arc::new(ctx);

        let app = Router::new()
            .route("/health", get(health))
            .route("/health_generate", get(health_generate))
            .route("/flush_cache", post(flush_cache))
            .route("/get_server_info", get(get_server_info))
            .route("/get_model_info", get(get_model_info))
            // .route("/generate", post(generate))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/models", get(models))
            .route("/register", post(register))
            .with_state(ctx);

        let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port)).await?;
        let listen_addr = listener.local_addr()?;

        info!("Listening on http://{}", listen_addr);
        axum::serve(listener, app).await?;

        Ok(())
    })
}

// cargo run --release -- --prefill http://127.0.0.1 --decode http://127.0.0.1 --prefill-bootstrap-ports 10000 --prompt-template-type qwen3-no-think --task_queue 4 --num-of-cache-token 256 --model-repo Qwen/Qwen3-0.6B --dp-size 8
fn main() -> ExitCode {
    let args = Args::parse();

    match launch(args) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{:?}", e);
            ExitCode::FAILURE
        }
    }
}
