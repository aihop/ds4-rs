use super::Engine;
use crate::kv::TransformerKvCache;
use crate::types::Tokens;
use crate::weights::{checksum_prefix, BoundWeights};

impl Engine {
    /// Infers logits for the next token given the sequence of current `tokens`.
    /// Attempts to use hardware acceleration if available; otherwise falls back to a stub.
    pub fn infer_logits(&self, tokens: &Tokens) -> Vec<f32> {
        if let Some(token) = tokens.0.last() {
            let pos = tokens.len().saturating_sub(1);
            let mut dummy_kv = TransformerKvCache::with_layers(61);
            let mut dummy_scratch = crate::kernels::decode_scratch::DecodeScratch::new();
            if let Some(logits) = self.try_reference_decode_next(&mut dummy_kv, &mut dummy_scratch, *token, pos) {
                return logits;
            }
        }
        self.stub_infer_logits(tokens)
    }

    pub(crate) fn try_reference_prefill(
        &self,
        kv_cache: &mut TransformerKvCache,
        scratch: &mut crate::kernels::decode_scratch::DecodeScratch,
        tokens: &[i32],
        start_pos: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    fn try_infer_logits_from_weights(&self, tokens: &Tokens) -> Option<Vec<f32>> {
        None
    }

    pub(crate) fn try_reference_decode_next(
        &self,
        _kv_cache: &mut TransformerKvCache,
        _scratch: &mut crate::kernels::decode_scratch::DecodeScratch,
        token: i32,
        pos: usize,
    ) -> Option<Vec<f32>> {
        let model = self.model.as_ref()?;
        let weights = self.weights.as_ref()?;
        
        if self.options.backend == crate::engine::Backend::Metal {
            let mut graph_guard = self.metal_graph.lock().unwrap();
            if graph_guard.is_none() {
                // Initialize the metal graph
                *graph_guard = Some(crate::kernels::metal_graph::MetalGraph::new(1, 7168, 4));
            }
            if let Some(graph) = graph_guard.as_mut() {
                let success = graph.execute_decode_step(model, weights, token, pos);
                if success {
                    // For now, return stub logits to satisfy the Rust type signature
                    let mut logits = vec![0.0f32; 131072];
                    // SAFETY: `logits` is allocated with `131072` f32 elements (131072 * 4 bytes).
                    // `graph.logits` points to a valid GPU tensor.
                    // `ds4_gpu_tensor_read` copies up to `131072 * 4` bytes from the GPU tensor into `logits`.
                    unsafe {
                        crate::ffi::ds4_gpu_tensor_read(
                            graph.logits,
                            0,
                            logits.as_mut_ptr() as *mut std::ffi::c_void,
                            (131072 * 4) as u64,
                        );
                    }
                    return Some(logits);
                }
            }
        }
        
        None
    }

    pub(super) fn should_use_reference_logits(&self, weights: &BoundWeights) -> bool {
        let _ = weights;
        true
    }

    fn stub_infer_logits(&self, tokens: &Tokens) -> Vec<f32> {
        let vocab = self.vocab_size().clamp(16, 8192);
        let mut logits = vec![0.0; vocab];
        let seed = self.inference_seed(tokens);
        let repeat_bias = tokens.0.last().copied().unwrap_or_default().unsigned_abs() as usize % vocab;

        for (idx, logit) in logits.iter_mut().enumerate() {
            let mixed = mix64(seed ^ idx as u64);
            let frac = ((mixed >> 11) as f64 / ((1u64 << 53) as f64)) as f32;
            let mut value = frac * 2.0 - 1.0;
            if idx == repeat_bias {
                value += 0.75;
            }
            if idx == (tokens.len() % vocab) {
                value += 0.25;
            }
            *logit = value;
        }

        logits
    }

    fn inference_seed(&self, tokens: &Tokens) -> u64 {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        seed ^= self.tensor_count() as u64;
        seed ^= (self.vocab_size() as u64) << 17;
        if let (Some(model), Some(weights)) = (&self.model, &self.weights) {
            seed ^= weights.token_embd.abs_offset.rotate_left(7);
            seed ^= weights.output.abs_offset.rotate_left(19);
            seed ^= checksum_prefix(model, &weights.token_embd, 64);
            seed ^= checksum_prefix(model, &weights.output, 64).rotate_left(9);
        }
        for (idx, token) in tokens.0.iter().enumerate() {
            seed ^= mix64((*token as i64 as u64).wrapping_add(idx as u64 * 0x1000_0001));
        }
        seed
    }
}

fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn debug_engine_event(hypothesis_id: &str, location: &str, msg: &str, data_json: String) {
    let event = format!(
        "{{\"sessionId\":\"slow-prefill-startup\",\"runId\":\"pre-fix\",\"hypothesisId\":\"{}\",\"location\":{},\"msg\":{},\"data\":{},\"ts\":{}}}",
        hypothesis_id,
        debug_json_string(location),
        debug_json_string(msg),
        data_json,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default()
    );

    let mut host = "127.0.0.1".to_string();
    let mut port = 7777;
    let mut path = "/event".to_string();

    if let Ok(content) = std::fs::read_to_string(".dbg/slow-prefill-startup.env") {
        for line in content.lines() {
            if let Some(url) = line.strip_prefix("DEBUG_SERVER_URL=") {
                let url = url.trim();
                if let Some(url) = url.strip_prefix("http://") {
                    let hp_path = url;
                    let (hp, p) = if let Some(idx) = hp_path.find('/') {
                        (&hp_path[..idx], &hp_path[idx..])
                    } else {
                        (hp_path, "/")
                    };
                    path = p.to_string();
                    if let Some((h, pt)) = hp.split_once(':') {
                        host = h.to_string();
                        port = pt.parse().unwrap_or(80);
                    } else {
                        host = hp.to_string();
                        port = 80;
                    }
                }
                break;
            }
        }
    }

    use std::net::ToSocketAddrs;
    if let Ok(mut addrs) = (host.as_str(), port).to_socket_addrs() {
        if let Some(addr) = addrs.next() {
            if let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(1)) {
                let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(1)));
                use std::io::Write;
                let request = format!(
                    "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    path, host, port, event.len(), event
                );
                let _ = stream.write_all(request.as_bytes());
            }
        }
    }
}

fn debug_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}
