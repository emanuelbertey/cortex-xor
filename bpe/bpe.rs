use std::collections::HashMap;

use crate::common::{
    update_counts,
    calculate_counts,
    merge,
    bytes_to_u32,
    bytes_to_byte_string_literal,
};

fn tokenize_with_special(text: &str, special_tokens: &Vec<String>, start_id: u32) -> Vec<u32> {
    let mut ids = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    // Create a map for quick lookup of special tokens
    let mut special_map: HashMap<&[u8], u32> = HashMap::new();
    for (idx, token) in special_tokens.iter().enumerate() {
        special_map.insert(token.as_bytes(), start_id + idx as u32);
    }

    while i < bytes.len() {
        let mut matched = false;
        // Check if any special token matches at current position
        // Sort keys by length descending to match longest token first
        // (For efficiency in production one would use Aho-Corasick or Trie)
        for (token_bytes, id) in &special_map {
            if bytes[i..].starts_with(token_bytes) {
                ids.push(*id);
                i += token_bytes.len();
                matched = true;
                break;
            }
        }

        if !matched {
            ids.push(bytes[i] as u32);
            i += 1;
        }
    }
    ids
}

pub fn train(
    text: &str,
    merges: &mut HashMap<(u32, u32), u32>,
    vocab: &mut HashMap<u32, String>,
    counts: &mut HashMap<(u32, u32), u32>,
    vocab_size: usize,
    special_tokens: &Vec<String>,
) -> Vec<u32> {
    // 1. Initialize base vocab (0-255)
    for i in 0..=255 {
        vocab.insert(i as u32, format!("{}", i as u8 as char));
    }
    
    // 2. Register special tokens
    let mut current_id = 256;
    for token in special_tokens {
        vocab.insert(current_id, token.clone());
        current_id += 1;
    }

    if vocab_size < current_id as usize {
        panic!("Vocab size {} is too small to hold 256 bytes + {} special tokens", vocab_size, special_tokens.len());
    }

    let num_merges = vocab_size - current_id as usize;
    
    // 3. Convert text to IDs, respecting special tokens
    let mut u32_ids = tokenize_with_special(text, special_tokens, 256);

    // 4. Run BPE training
    for i in 0..num_merges {
        *counts = calculate_counts(&u32_ids);
        // We must filter out pairs that involve special tokens to prevent merging them?
        // Actually, calculate_counts will count pairs like (SPECIAL_ID, some_byte).
        // If we want to prevent merging special tokens, we should filter counts.
        // But for simplicity, and to follow standard BPE behavior (which might learn to merge special tokens if they appear in context),
        // we will leave it. However, typically special tokens are boundaries.
        // Let's assume the user wants them fixed. 
        // If we don't want them merged, we should ignore pairs involving IDs >= 256 and < 256 + special_tokens.len().
        
        let special_start = 256;
        let special_end = 256 + special_tokens.len() as u32;

        let max_pair_opt = counts
            .iter()
            .filter(|((p1, p2), _)| {
                // Prevent merging if either part is a special token
                !(*p1 >= special_start && *p1 < special_end) && !(*p2 >= special_start && *p2 < special_end)
            })
            .max_by_key(|entry| entry.1);

        if max_pair_opt.is_none() {
            println!("No more valid pairs to merge.");
            break;
        }
        
        let max_pair = max_pair_opt.unwrap();
        
        let idx = current_id + i as u32; // Start IDs after special tokens
        u32_ids = merge(u32_ids, *max_pair.0, idx);
        merges.insert(*max_pair.0, idx);
        
        let merged_bytes =
            vocab.get(&max_pair.0.0).cloned().unwrap_or_default() +
            &vocab.get(&max_pair.0.1).cloned().unwrap_or_default();
            
        vocab.insert(idx, merged_bytes);
        println!(
            "Epoch {}/{}: {} {} -> {} ({:?}) had {:?} occurrences",
            i + 1,
            num_merges,
            max_pair.0.0,
            max_pair.0.1,
            idx,
            vocab.get(&idx).unwrap(),
            counts.get(max_pair.0).unwrap()
        );
    }
    u32_ids
}

pub fn encode(text: &str, merges: &HashMap<(u32, u32), u32>, special_tokens: &Vec<String>) -> Vec<u32> {
    // 1. Convert text to IDs, respecting special tokens
    let mut ids = tokenize_with_special(text, special_tokens, 256);

    // 2. Apply BPE merges
    while ids.len() >= 2 {
        let mut stats = HashMap::new();
        update_counts(&ids, &mut stats);
        let min_pair = stats
            .iter()
            .min_by_key(|&(p, _)| merges.get(p).cloned().unwrap_or(std::u32::MAX))
            .unwrap().0;

        if !merges.contains_key(&min_pair) {
            break;
        }

        let idx = merges.get(&min_pair).unwrap();
        ids = merge(ids, *min_pair, *idx);
    }

    ids
}

pub fn decode(ids: &[u32], vocab: &HashMap<u32, String>) -> String {
    let mut text = String::new();
    for &id in ids {
        if let Some(s) = vocab.get(&id) {
            text.push_str(s);
        }
    }
    text
}
