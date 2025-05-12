use crate::{PrefillConfig, TokenId};
use chrono::Utc;
use radix_trie::{Trie, TrieCommon};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct DstProcess {
    pub node: PrefillConfig,
    pub dp_rank: usize,
}

pub struct RadixTrieKvRouter {
    // seq -> (prefill config, utc timestamp)
    trie: Trie<Vec<TokenId>, (DstProcess, Arc<AtomicI64>)>,
    max_seq_len: usize,
}

impl RadixTrieKvRouter {
    pub fn new(max_seq_len: usize) -> RadixTrieKvRouter {
        RadixTrieKvRouter {
            trie: Trie::new(),
            max_seq_len,
        }
    }

    pub fn update(&mut self, seq: &[TokenId], select_node: DstProcess) {
        if self.trie.len() >= self.max_seq_len {
            let mut seqs = self.trie.iter()
                .collect::<Vec<_>>();

            seqs.sort_unstable_by_key(|(_, (_, attach_time))| attach_time.load(Ordering::Relaxed));

            let keys = seqs[0..std::cmp::min(10, seqs.len())].iter()
                .map(|(key, _)| key.to_vec())
                .collect::<Vec<_>>();

            for key in &keys {
                self.trie.remove(key);
            }
        }

        self.trie.insert(seq.to_vec(), (select_node, Arc::new(AtomicI64::new(Utc::now().timestamp()))));
    }

    pub fn get(&self, seq: &[TokenId]) -> HashMap<DstProcess, (usize, Arc<AtomicI64>)> {
        // node -> (num of cached tokens, timestamp)
        let mut map = HashMap::new();

        for cache_num in 1..=seq.len() {
            let sub_trie = self.trie
                .get_raw_descendant(&seq[0..cache_num].to_vec());

            match sub_trie {
                None => break,
                Some(sub_trie) => {
                    for (_, (dst, timestamp)) in sub_trie.iter() {
                        map.insert(dst.clone(), (cache_num, timestamp.clone()));
                    }
                }
            }
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use url::Url;
    use super::*;

    #[test]
    fn test() {
        let mut router = RadixTrieKvRouter::new(10000);
        router.update(&[1, 2, 3, 4, 5, 6, 7, 8], DstProcess{node: PrefillConfig{url: "https://example.com".parse::<Url>().unwrap(), bootstrap_port: None}, dp_rank: 1});
        router.update(&[1, 2, ], DstProcess{node: PrefillConfig{url: "https://example.com".parse::<Url>().unwrap(), bootstrap_port: None}, dp_rank: 0});

        let r = router.get(&[1, 2, 3, 4, 5]);

        for (dst, (cache, b)) in r {
            println!("cache tokens {}, rank {}", cache, dst.dp_rank)
        }
    }
}