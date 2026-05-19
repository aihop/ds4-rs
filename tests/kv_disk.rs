use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ds4_rust::{KvCache, KvEntry, KvToolReplayEntry, SessionSnapshot, Tokens};

const SESSION_PAYLOAD_MAGIC: u32 = 0x3456_5344;

fn unique_temp_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("ds4-rs-kv-test-{}-{stamp}", std::process::id()))
}

#[test]
fn kv_cache_round_trips_entries_through_disk() {
    let root = unique_temp_dir();
    let key = "visible:resp_test:42";
    let entry = KvEntry {
        reason: "responses".to_string(),
        rendered_text: "User: hello\nAssistant: hi".to_string(),
        tokens: Tokens(vec![11, 22, 33]),
        snapshot: SessionSnapshot {
            bytes: dsv4_payload(4096, &[11, 22, 33], &[0.1, 0.2]),
        },
        tool_replay: vec![KvToolReplayEntry {
            tool_call_id: "call_resp_42".to_string(),
            sampled_block: "AssistantToolCall[call_resp_42] function read_file({\"path\":\"README.md\"})"
                .to_string(),
        }],
        ctx_size: 4096,
        hit_count: 7,
        created_at_unix: 1234,
        last_used_at_unix: 5678,
    };

    let mut cache = KvCache::with_disk_dir(&root).expect("disk cache should initialize");
    cache.store(key, entry.clone()).expect("store should succeed");
    assert!(cache.has_candidate(key));
    assert!(cache.has_rendered_candidate(&entry.rendered_text));

    let mut restarted = KvCache::with_disk_dir(&root).expect("disk cache should reopen");
    assert!(restarted.has_rendered_candidate(&entry.rendered_text));
    let loaded = restarted
        .load_rendered_text(&entry.rendered_text)
        .expect("load should succeed")
        .clone()
        .expect("entry should exist");
    assert_eq!(loaded, entry);

    fs::remove_dir_all(root).ok();
}

#[test]
fn kv_disk_file_uses_kvc_header_and_embeds_rendered_text() {
    let root = unique_temp_dir();
    let key = "visible:resp_test:99";
    let rendered_text = "Visible prefix bytes";
    let entry = KvEntry {
        reason: "continued".to_string(),
        rendered_text: rendered_text.to_string(),
        tokens: Tokens(vec![]),
        snapshot: SessionSnapshot {
            bytes: dsv4_payload(2048, &[], &[]),
        },
        tool_replay: vec![KvToolReplayEntry {
            tool_call_id: "call_resp_99".to_string(),
            sampled_block: "AssistantToolCall[call_resp_99] function bash({\"command\":\"pwd\"})"
                .to_string(),
        }],
        ctx_size: 2048,
        hit_count: 3,
        created_at_unix: 11,
        last_used_at_unix: 22,
    };
    let mut cache = KvCache::with_disk_dir(&root).expect("disk cache should initialize");
    cache.store(key, entry).expect("store should succeed");

    let file = root.join(format!("{}.kv", sha1_hex(rendered_text.as_bytes())));
    let bytes = fs::read(file).expect("kv file should exist");
    assert_eq!(&bytes[0..3], b"KVC");
    assert_eq!(bytes[3], 1);
    let rendered_len = u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]) as usize;
    let rendered = std::str::from_utf8(&bytes[52..52 + rendered_len]).expect("rendered text should be utf-8");
    assert_eq!(rendered, rendered_text);
    let payload_start = 52 + rendered_len;
    let payload_len = u64::from_le_bytes([
        bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
    ]) as usize;
    let tool_map_offset = payload_start + payload_len;
    assert_eq!(&bytes[tool_map_offset..tool_map_offset + 3], b"KTM");

    fs::remove_dir_all(root).ok();
}

#[test]
fn kv_disk_filename_uses_sha1_of_rendered_text() {
    let root = unique_temp_dir();
    let rendered_text = "abc";
    let entry = KvEntry {
        reason: "responses".to_string(),
        rendered_text: rendered_text.to_string(),
        tokens: Tokens(vec![]),
        snapshot: SessionSnapshot {
            bytes: dsv4_payload(512, &[], &[]),
        },
        tool_replay: Vec::new(),
        ctx_size: 512,
        hit_count: 0,
        created_at_unix: 0,
        last_used_at_unix: 0,
    };
    let mut cache = KvCache::with_disk_dir(&root).expect("disk cache should initialize");
    cache.store("visible:anything", entry).expect("store should succeed");

    let file = root.join("a9993e364706816aba3e25717850c26c9cd0d89d.kv");
    assert!(file.is_file(), "expected sha1-based kv filename");

    fs::remove_dir_all(root).ok();
}

#[test]
fn kv_cache_can_lookup_tool_replay_after_restart() {
    let root = unique_temp_dir();
    let key = "visible:resp_test:replay";
    let replay = KvToolReplayEntry {
        tool_call_id: "call_replay_1".to_string(),
        sampled_block:
            "AssistantToolCall[call_replay_1] function grep({\"pattern\":\"todo\"})".to_string(),
    };
    let entry = KvEntry {
        reason: "responses".to_string(),
        rendered_text: "tool replay source".to_string(),
        tokens: Tokens(vec![1, 2]),
        snapshot: SessionSnapshot {
            bytes: dsv4_payload(1024, &[1, 2], &[0.4]),
        },
        tool_replay: vec![replay.clone()],
        ctx_size: 1024,
        hit_count: 1,
        created_at_unix: 1,
        last_used_at_unix: 2,
    };
    let mut cache = KvCache::with_disk_dir(&root).expect("disk cache should initialize");
    cache.store(key, entry).expect("store should succeed");

    let mut restarted = KvCache::with_disk_dir(&root).expect("disk cache should reopen");
    let loaded = restarted
        .find_tool_replay(&replay.tool_call_id)
        .expect("tool replay lookup should succeed");
    assert_eq!(loaded.as_deref(), Some(replay.sampled_block.as_str()));

    fs::remove_dir_all(root).ok();
}

fn dsv4_payload(ctx_size: u32, tokens: &[i32], logits: &[f32]) -> Vec<u8> {
    let header = [
        SESSION_PAYLOAD_MAGIC,
        1,
        ctx_size,
        0,
        0,
        0,
        0,
        tokens.len() as u32,
        0,
        0,
        0,
        logits.len() as u32,
        0,
    ];
    let mut bytes = Vec::new();
    for field in header {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    for logit in logits {
        bytes.extend_from_slice(&logit.to_le_bytes());
    }
    bytes
}

fn sha1_hex(bytes: &[u8]) -> String {
    let digest = sha1_digest(bytes);
    let mut out = String::with_capacity(40);
    for byte in digest {
        out.push(hex_digit(byte >> 4));
        out.push(hex_digit(byte & 0x0f));
    }
    out
}

fn sha1_digest(bytes: &[u8]) -> [u8; 20] {
    let mut h0 = 0x6745_2301u32;
    let mut h1 = 0xefcd_ab89u32;
    let mut h2 = 0x98ba_dcfeu32;
    let mut h3 = 0x1032_5476u32;
    let mut h4 = 0xc3d2_e1f0u32;

    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(((bytes.len() + 9).div_ceil(64)) * 64);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0u32; 80];
    for chunk in padded.chunks_exact(64) {
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let start = i * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'a' + (nibble - 10)) as char,
    }
}
