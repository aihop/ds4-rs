use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Ds4Error, Result};
use crate::types::{SessionSnapshot, Tokens};

const KV_DISK_MAGIC: &[u8; 3] = b"KVC";
const KV_DISK_VERSION: u8 = 1;
const KV_TOOL_MAP_FLAG: u8 = 1;
const KV_TOOL_MAP_MAGIC: &[u8; 3] = b"KTM";
const KV_TOOL_MAP_VERSION: u8 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvToolReplayEntry {
    pub tool_call_id: String,
    pub sampled_block: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvEntry {
    pub reason: String,
    pub rendered_text: String,
    pub tokens: Tokens,
    pub snapshot: SessionSnapshot,
    pub tool_replay: Vec<KvToolReplayEntry>,
    pub ctx_size: usize,
    pub hit_count: u32,
    pub created_at_unix: u64,
    pub last_used_at_unix: u64,
}

#[derive(Default)]
pub struct KvCache {
    by_key: HashMap<String, KvEntry>,
    disk: Option<KvDiskStore>,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_disk_dir(path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            by_key: HashMap::new(),
            disk: Some(KvDiskStore::create(path.into())?),
        })
    }

    pub fn has_candidate(&self, key: &str) -> bool {
        self.by_key.contains_key(key)
            || self
                .disk
                .as_ref()
                .is_some_and(|disk| disk.has_rendered_candidate(key))
    }

    pub fn store(&mut self, key: impl Into<String>, entry: KvEntry) -> Result<()> {
        let key = key.into();
        if let Some(disk) = &self.disk {
            disk.store(&key, &entry)?;
        }
        self.by_key.insert(key, entry);
        Ok(())
    }

    /// Loads the cache entry associated with `key`.
    pub fn load(&mut self, key: &str) -> Result<Option<KvEntry>> {
        if !self.by_key.contains_key(key) {
            if let Some(disk) = &self.disk {
                if let Some(entry) = disk.load_rendered_text(key)? {
                    self.by_key.insert(key.to_string(), entry);
                }
            }
        }

        Ok(self.by_key.get(key).cloned())
    }

    pub fn has_rendered_candidate(&self, rendered_text: &str) -> bool {
        self.disk
            .as_ref()
            .is_some_and(|disk| disk.has_rendered_candidate(rendered_text))
    }

    pub fn load_rendered_text(&mut self, rendered_text: &str) -> Result<Option<KvEntry>> {
        if !self.by_key.contains_key(rendered_text) {
            if let Some(disk) = &self.disk {
                if let Some(entry) = disk.load_rendered_text(rendered_text)? {
                    self.by_key.insert(rendered_text.to_string(), entry);
                }
            }
        }
        Ok(self.by_key.get(rendered_text).cloned())
    }

    pub fn find_tool_replay(&mut self, tool_call_id: &str) -> Result<Option<String>> {
        for entry in self.by_key.values() {
            if let Some(found) = entry
                .tool_replay
                .iter()
                .find(|item| item.tool_call_id == tool_call_id)
            {
                return Ok(Some(found.sampled_block.clone()));
            }
        }
        if let Some(disk) = &self.disk {
            return disk.find_tool_replay(tool_call_id);
        }
        Ok(None)
    }
}

#[derive(Clone, Debug)]
struct KvDiskStore {
    root: PathBuf,
}

impl KvDiskStore {
    fn create(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn has_rendered_candidate(&self, rendered_text: &str) -> bool {
        self.path_for_rendered_text(rendered_text).is_file()
    }

    fn store(&self, key: &str, entry: &KvEntry) -> Result<()> {
        let rendered_text_bytes = entry.rendered_text.as_bytes();
        let reason_bytes = entry.reason.as_bytes();
        let token_count = u32::try_from(entry.tokens.len())
            .map_err(|_| Ds4Error::Protocol("too many tokens to persist kv entry".to_string()))?;
        let rendered_text_len = u32::try_from(rendered_text_bytes.len())
            .map_err(|_| Ds4Error::Protocol("rendered text too large".to_string()))?;
        let reason_len = u32::try_from(reason_bytes.len())
            .map_err(|_| Ds4Error::Protocol("kv reason too large".to_string()))?;
        let payload_len = u64::try_from(entry.snapshot.bytes.len())
            .map_err(|_| Ds4Error::Protocol("kv snapshot too large".to_string()))?;
        let ctx_size = u32::try_from(entry.ctx_size)
            .map_err(|_| Ds4Error::Protocol("kv context size too large".to_string()))?;
        let quant_bits = infer_routed_quant_bits(&entry.snapshot.bytes);
        let save_reason = encode_save_reason(&entry.reason);
        let tool_map_bytes = encode_tool_map(&entry.tool_replay)?;
        let extension_flags = if tool_map_bytes.is_empty() {
            0
        } else {
            KV_TOOL_MAP_FLAG
        };

        let mut bytes = Vec::with_capacity(
            52 + rendered_text_bytes.len() + entry.snapshot.bytes.len() + tool_map_bytes.len() + reason_bytes.len(),
        );
        bytes.extend_from_slice(KV_DISK_MAGIC);
        bytes.push(KV_DISK_VERSION);
        bytes.push(quant_bits);
        bytes.push(save_reason);
        bytes.push(extension_flags);
        bytes.push(0);
        bytes.extend_from_slice(&token_count.to_le_bytes());
        bytes.extend_from_slice(&entry.hit_count.to_le_bytes());
        bytes.extend_from_slice(&ctx_size.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        bytes.extend_from_slice(&entry.created_at_unix.to_le_bytes());
        bytes.extend_from_slice(&entry.last_used_at_unix.to_le_bytes());
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(&rendered_text_len.to_le_bytes());
        bytes.extend_from_slice(rendered_text_bytes);
        bytes.extend_from_slice(&entry.snapshot.bytes);
        bytes.extend_from_slice(&tool_map_bytes);
        bytes.extend_from_slice(&reason_len.to_le_bytes());
        bytes.extend_from_slice(reason_bytes);
        let _ = key;
        fs::write(self.path_for_rendered_text(&entry.rendered_text), bytes)?;
        Ok(())
    }

    fn load_rendered_text(&self, rendered_text: &str) -> Result<Option<KvEntry>> {
        let path = self.path_for_rendered_text(rendered_text);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(path)?;
        if bytes.len() >= 52 && &bytes[..3] == KV_DISK_MAGIC && bytes[3] == KV_DISK_VERSION {
            return self.load_kvc_entry(&bytes).map(Some);
        }
        if bytes.len() < 24 {
            return Err(Ds4Error::Protocol("invalid kv disk header".to_string()));
        }
        self.load_legacy_entry(rendered_text, &bytes).map(Some)
    }

    fn load_kvc_entry(&self, bytes: &[u8]) -> Result<KvEntry> {
        let extension_flags = bytes[6];
        let token_count = read_u32(bytes, 8)? as usize;
        let hit_count = read_u32(bytes, 12)?;
        let ctx_size = read_u32(bytes, 16)? as usize;
        let created_at_unix = read_u64(bytes, 24)?;
        let last_used_at_unix = read_u64(bytes, 32)?;
        let payload_len = read_u64(bytes, 40)? as usize;
        let rendered_text_len = read_u32(bytes, 48)? as usize;
        let rendered_text_end = 52usize
            .checked_add(rendered_text_len)
            .ok_or_else(|| Ds4Error::Protocol("kv rendered text length overflow".to_string()))?;
        let payload_end = rendered_text_end
            .checked_add(payload_len)
            .ok_or_else(|| Ds4Error::Protocol("kv payload length overflow".to_string()))?;
        if payload_end > bytes.len() {
            return Err(Ds4Error::Protocol("truncated kv payload".to_string()));
        }

        let rendered_text = std::str::from_utf8(&bytes[52..rendered_text_end])
            .map_err(|_| Ds4Error::Protocol("rendered text is not valid utf-8".to_string()))?
            .to_string();
        let snapshot = SessionSnapshot {
            bytes: bytes[rendered_text_end..payload_end].to_vec(),
        };
        let (tokens, derived_ctx_size) = snapshot_payload_meta(&snapshot.bytes)?;
        if token_count != 0 && token_count != tokens.len() {
            return Err(Ds4Error::Protocol("kv cached token count mismatch".to_string()));
        }
        let (tool_replay, trailer_offset) = if extension_flags & KV_TOOL_MAP_FLAG != 0 {
            parse_tool_map(bytes, payload_end)?
        } else {
            (Vec::new(), payload_end)
        };
        let reason = if trailer_offset + 4 <= bytes.len() {
            let reason_len = read_u32(bytes, trailer_offset)? as usize;
            let reason_start = trailer_offset + 4;
            let reason_end = reason_start
                .checked_add(reason_len)
                .ok_or_else(|| Ds4Error::Protocol("kv reason length overflow".to_string()))?;
            if reason_end != bytes.len() {
                return Err(Ds4Error::Protocol("kv reason length mismatch".to_string()));
            }
            std::str::from_utf8(&bytes[reason_start..reason_end])
                .map_err(|_| Ds4Error::Protocol("kv reason is not valid utf-8".to_string()))?
                .to_string()
        } else {
            decode_save_reason(bytes[5]).to_string()
        };

        Ok(KvEntry {
            reason,
            rendered_text,
            tokens,
            snapshot,
            tool_replay,
            ctx_size: if ctx_size == 0 { derived_ctx_size } else { ctx_size },
            hit_count,
            created_at_unix,
            last_used_at_unix,
        })
    }

    fn load_legacy_entry(&self, key: &str, bytes: &[u8]) -> Result<KvEntry> {
        if &bytes[..4] != b"DS4K" {
            return Err(Ds4Error::Protocol("invalid kv disk header".to_string()));
        }
        let version = read_u32(&bytes, 4)?;
        if version != u32::from(KV_DISK_VERSION) {
            return Err(Ds4Error::Protocol("unsupported kv disk version".to_string()));
        }
        let key_len = read_u32(&bytes, 8)? as usize;
        let reason_len = read_u32(&bytes, 12)? as usize;
        let token_count = read_u32(&bytes, 16)? as usize;
        let snapshot_len = read_u32(&bytes, 20)? as usize;
        let token_bytes = token_count
            .checked_mul(4)
            .ok_or_else(|| Ds4Error::Protocol("kv token byte count overflow".to_string()))?;
        let expected_len = 24usize
            .checked_add(key_len)
            .and_then(|v| v.checked_add(reason_len))
            .and_then(|v| v.checked_add(token_bytes))
            .and_then(|v| v.checked_add(snapshot_len))
            .ok_or_else(|| Ds4Error::Protocol("kv entry length overflow".to_string()))?;
        if bytes.len() != expected_len {
            return Err(Ds4Error::Protocol("kv entry length mismatch".to_string()));
        }

        let mut offset = 24usize;
        let stored_key = std::str::from_utf8(&bytes[offset..offset + key_len])
            .map_err(|_| Ds4Error::Protocol("kv key is not valid utf-8".to_string()))?;
        offset += key_len;
        if stored_key != key {
            return Err(Ds4Error::Protocol("kv key hash collision detected".to_string()));
        }
        let reason = std::str::from_utf8(&bytes[offset..offset + reason_len])
            .map_err(|_| Ds4Error::Protocol("kv reason is not valid utf-8".to_string()))?
            .to_string();
        offset += reason_len;
        let mut tokens = Vec::with_capacity(token_count);
        for chunk in bytes[offset..offset + token_bytes].chunks_exact(4) {
            tokens.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        offset += token_bytes;
        let snapshot = SessionSnapshot {
            bytes: bytes[offset..offset + snapshot_len].to_vec(),
        };
        Ok(KvEntry {
            reason,
            rendered_text: stored_key.to_string(),
            tokens: Tokens(tokens),
            snapshot,
            tool_replay: Vec::new(),
            ctx_size: 0,
            hit_count: 0,
            created_at_unix: 0,
            last_used_at_unix: 0,
        })
    }

    fn path_for_rendered_text(&self, rendered_text: &str) -> PathBuf {
        self.root
            .join(format!("{}.kv", sha1_hex(rendered_text.as_bytes())))
    }

    fn find_tool_replay(&self, tool_call_id: &str) -> Result<Option<String>> {
        for entry in fs::read_dir(&self.root)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("kv") {
                continue;
            }
            let bytes = fs::read(&path)?;
            let maybe_entry = if bytes.len() >= 52 && &bytes[..3] == KV_DISK_MAGIC && bytes[3] == KV_DISK_VERSION {
                Some(self.load_kvc_entry(&bytes)?)
            } else if bytes.len() >= 24 && &bytes[..4] == b"DS4K" {
                None
            } else {
                None
            };
            if let Some(entry) = maybe_entry {
                if let Some(found) = entry
                    .tool_replay
                    .iter()
                    .find(|item| item.tool_call_id == tool_call_id)
                {
                    return Ok(Some(found.sampled_block.clone()));
                }
            }
        }
        Ok(None)
    }
}

fn encode_tool_map(entries: &[KvToolReplayEntry]) -> Result<Vec<u8>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let count = u32::try_from(entries.len())
        .map_err(|_| Ds4Error::Protocol("too many tool replay entries".to_string()))?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(KV_TOOL_MAP_MAGIC);
    bytes.push(KV_TOOL_MAP_VERSION);
    bytes.extend_from_slice(&count.to_le_bytes());
    for entry in entries {
        let id_bytes = entry.tool_call_id.as_bytes();
        let block_bytes = entry.sampled_block.as_bytes();
        let id_len = u32::try_from(id_bytes.len())
            .map_err(|_| Ds4Error::Protocol("tool replay id too large".to_string()))?;
        let block_len = u32::try_from(block_bytes.len())
            .map_err(|_| Ds4Error::Protocol("tool replay block too large".to_string()))?;
        bytes.extend_from_slice(&id_len.to_le_bytes());
        bytes.extend_from_slice(&block_len.to_le_bytes());
        bytes.extend_from_slice(id_bytes);
        bytes.extend_from_slice(block_bytes);
    }
    Ok(bytes)
}

fn parse_tool_map(bytes: &[u8], offset: usize) -> Result<(Vec<KvToolReplayEntry>, usize)> {
    if bytes.len() < offset + 8 {
        return Err(Ds4Error::Protocol("truncated tool replay section".to_string()));
    }
    if &bytes[offset..offset + 3] != KV_TOOL_MAP_MAGIC || bytes[offset + 3] != KV_TOOL_MAP_VERSION {
        return Err(Ds4Error::Protocol("invalid tool replay section header".to_string()));
    }
    let count = read_u32(bytes, offset + 4)? as usize;
    let mut cursor = offset + 8;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let id_len = read_u32(bytes, cursor)? as usize;
        let block_len = read_u32(bytes, cursor + 4)? as usize;
        cursor += 8;
        let id_end = cursor
            .checked_add(id_len)
            .ok_or_else(|| Ds4Error::Protocol("tool replay id length overflow".to_string()))?;
        let block_end = id_end
            .checked_add(block_len)
            .ok_or_else(|| Ds4Error::Protocol("tool replay block length overflow".to_string()))?;
        if block_end > bytes.len() {
            return Err(Ds4Error::Protocol("truncated tool replay entry".to_string()));
        }
        let tool_call_id = std::str::from_utf8(&bytes[cursor..id_end])
            .map_err(|_| Ds4Error::Protocol("tool replay id is not valid utf-8".to_string()))?
            .to_string();
        let sampled_block = std::str::from_utf8(&bytes[id_end..block_end])
            .map_err(|_| Ds4Error::Protocol("tool replay block is not valid utf-8".to_string()))?
            .to_string();
        out.push(KvToolReplayEntry {
            tool_call_id,
            sampled_block,
        });
        cursor = block_end;
    }
    Ok((out, cursor))
}

fn infer_routed_quant_bits(payload: &[u8]) -> u8 {
    let _ = payload;
    0
}

pub fn unix_time_secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or_default()
}

fn encode_save_reason(reason: &str) -> u8 {
    match reason {
        "cold" => 1,
        "continued" => 2,
        "evict" => 3,
        "shutdown" => 4,
        _ => 0,
    }
}

fn decode_save_reason(reason: u8) -> &'static str {
    match reason {
        1 => "cold",
        2 => "continued",
        3 => "evict",
        4 => "shutdown",
        _ => "unknown",
    }
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

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Ds4Error::Protocol("truncated kv disk header".to_string()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Ds4Error::Protocol("truncated kv disk header".to_string()))?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn snapshot_payload_meta(payload: &[u8]) -> Result<(Tokens, usize)> {
    if payload.len() < 52 {
        return Ok((Tokens::default(), 0));
    }
    let magic = read_u32(payload, 0)?;
    let version = read_u32(payload, 4)?;
    if magic != 0x3456_5344 || version != 1 {
        return Ok((Tokens::default(), 0));
    }
    let ctx_size = read_u32(payload, 8)? as usize;
    let token_count = read_u32(payload, 28)? as usize;
    let token_bytes = token_count
        .checked_mul(4)
        .ok_or_else(|| Ds4Error::Protocol("snapshot token byte count overflow".to_string()))?;
    let start = 52usize;
    let end = start
        .checked_add(token_bytes)
        .ok_or_else(|| Ds4Error::Protocol("snapshot token section overflow".to_string()))?;
    if end > payload.len() {
        return Err(Ds4Error::Protocol("snapshot token section truncated".to_string()));
    }
    let mut tokens = Vec::with_capacity(token_count);
    for chunk in payload[start..end].chunks_exact(4) {
        tokens.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((Tokens(tokens), ctx_size))
}

#[derive(Clone, Debug, Default)]
pub struct TransformerKvLayer {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub seq_len: usize,
    pub metal_tensor: Option<*mut crate::ffi::ds4_gpu_tensor>,
}

unsafe impl Send for TransformerKvLayer {}
unsafe impl Sync for TransformerKvLayer {}

impl TransformerKvLayer {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
            metal_tensor: None,
        }
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }

    pub fn push(&mut self, key: &[f32], value: &[f32]) {
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.seq_len += 1;
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransformerKvCache {
    layers: Vec<TransformerKvLayer>,
}

impl TransformerKvCache {
    pub fn with_layers(layer_count: usize) -> Self {
        Self {
            layers: vec![TransformerKvLayer::default(); layer_count],
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn ensure_layers(&mut self, layer_count: usize) {
        if self.layers.len() < layer_count {
            self.layers
                .resize_with(layer_count, TransformerKvLayer::default);
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn layer(&self, index: usize) -> Option<&TransformerKvLayer> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut TransformerKvLayer> {
        self.layers.get_mut(index)
    }
}
