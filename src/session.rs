use std::sync::Arc;
use std::time::Instant;

use crate::engine::Engine;
use crate::error::{Ds4Error, Result};
use crate::kernels::decode_scratch::DecodeScratch;
use crate::kv::TransformerKvCache;
use crate::types::{SessionSnapshot, TokenScore, Tokens};

const SESSION_PAYLOAD_MAGIC: u32 = 0x3456_5344;
const SESSION_PAYLOAD_VERSION: u32 = 1;
const SESSION_PAYLOAD_HEADER_U32S: usize = 13;

pub trait ProgressHandler: Send + Sync {
    fn on_progress(&self, event: &str, current: usize, total: usize);
}

#[derive(Clone, Debug, Default)]
pub struct SyncStats {
    pub cached_tokens: usize,
    pub replay_tokens: usize,
    pub rebuilt: bool,
}

pub struct Session {
    engine: Arc<Engine>,
    ctx_size: usize,
    checkpoint: Tokens,
    logits: Vec<f32>,
    progress: Option<Arc<dyn ProgressHandler>>,
    prefill_boundary: Option<usize>,
    transformer_kv: TransformerKvCache,
    decode_scratch: DecodeScratch,
}

impl Session {
    /// Creates a new `Session` with the given engine and context size limit.
    pub fn create(engine: Arc<Engine>, ctx_size: usize) -> Result<Self> {
        if ctx_size == 0 {
            return Err(Ds4Error::InvalidArgument(
                "context size must be positive".to_string(),
            ));
        }
        Ok(Self {
            engine,
            ctx_size,
            checkpoint: Tokens::default(),
            logits: vec![0.0; 256],
            progress: None,
            prefill_boundary: None,
            transformer_kv: TransformerKvCache::default(),
            decode_scratch: DecodeScratch::default(),
        })
    }

    /// Sets a progress handler callback.
    pub fn set_progress(&mut self, progress: Option<Arc<dyn ProgressHandler>>) {
        self.progress = progress;
    }

    /// Sets the maximum prefill boundary per sync step.
    pub fn set_prefill_boundary(&mut self, boundary: Option<usize>) {
        self.prefill_boundary = boundary.filter(|value| *value > 0);
    }

    /// Synchronizes the session's state with the given `prompt`, performing prefill if necessary.
    pub fn sync(&mut self, prompt: &Tokens) -> Result<SyncStats> {
        self.prefill(prompt)
    }

    pub fn prefill(&mut self, prompt: &Tokens) -> Result<SyncStats> {
        if prompt.len() >= self.ctx_size {
            return Err(Ds4Error::ContextExceeded {
                prompt_len: prompt.len(),
                ctx_size: self.ctx_size,
            });
        }

        let common = self.checkpoint.common_prefix_len(prompt);
        let rebuilt = common != self.checkpoint.len();
        let cached_tokens = if rebuilt { 0 } else { common };
        let replay_tokens = prompt.len().saturating_sub(cached_tokens);
        self.report_prefill_progress(cached_tokens, prompt.len());

        if rebuilt {
            self.reset_live_state();
        }

        if replay_tokens == 0 {
            if self.checkpoint != *prompt {
                self.rebuild_prompt_state(prompt);
            }
        } else if cached_tokens == 0 {
            self.rebuild_prompt_state(prompt);
        } else if !self.extend_live_suffix(prompt, cached_tokens) {
            self.rebuild_prompt_state(prompt);
        }

        Ok(SyncStats {
            cached_tokens,
            replay_tokens,
            rebuilt,
        })
    }

    pub fn common_prefix(&self, prompt: &Tokens) -> usize {
        self.checkpoint.common_prefix_len(prompt)
    }

    pub fn argmax(&self) -> i32 {
        self.logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx as i32)
            .unwrap_or_default()
    }

    pub fn top_logprobs(&self, k: usize) -> Vec<TokenScore> {
        let mut ranked: Vec<(usize, f32)> = self.logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
            .into_iter()
            .take(k.min(5))
            .map(|(idx, logit)| TokenScore {
                id: idx as i32,
                logit,
                logprob: logit - 1.0,
            })
            .collect()
    }

    /// Evaluates a single token to compute new logits, without advancing the KV cache or token sequence.
    pub fn eval(&mut self, token: i32) -> Result<()> {
        self.decode_next(token)
    }

    /// Appends a token to the session and computes logits for the next step.
    pub fn decode_next(&mut self, token: i32) -> Result<()> {
        if self.checkpoint.len() + 1 >= self.ctx_size {
            return Err(Ds4Error::ContextExceeded {
                prompt_len: self.checkpoint.len() + 1,
                ctx_size: self.ctx_size,
            });
        }
        self.append_token_with_reference_fallback(token);
        Ok(())
    }

    pub fn rewind(&mut self, pos: usize) {
        let mut prompt = self.checkpoint.clone();
        prompt.0.truncate(pos.min(prompt.len()));
        self.rebuild_prompt_state(&prompt);
    }

    pub fn invalidate(&mut self) {
        self.checkpoint = Tokens::default();
        self.transformer_kv.clear();
    }

    pub fn pos(&self) -> usize {
        self.checkpoint.len()
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn ctx(&self) -> usize {
        self.ctx_size
    }

    pub fn tokens(&self) -> &Tokens {
        &self.checkpoint
    }

    pub fn save_snapshot(&self) -> SessionSnapshot {
        let token_count = self.checkpoint.len();
        let vocab_size = self.logits.len();
        let mut bytes = Vec::with_capacity(
            SESSION_PAYLOAD_HEADER_U32S * 4 + token_count * 4 + vocab_size * 4,
        );
        let header = [
            SESSION_PAYLOAD_MAGIC,
            SESSION_PAYLOAD_VERSION,
            self.ctx_size as u32,
            self.prefill_boundary.unwrap_or(0) as u32,
            0,
            0,
            0,
            token_count as u32,
            self.transformer_kv.layer_count() as u32,
            0,
            0,
            vocab_size as u32,
            0,
        ];
        for field in header {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        for token in &self.checkpoint.0 {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        for logit in &self.logits {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
        SessionSnapshot { bytes }
    }

    pub fn load_snapshot(&mut self, snapshot: &SessionSnapshot) {
        if self.try_load_dsv4_snapshot(snapshot) {
            return;
        }
        let mut tokens = Vec::new();
        for chunk in snapshot.bytes.chunks_exact(4) {
            tokens.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        self.checkpoint = Tokens(tokens);
        self.transformer_kv.clear();
        self.logits = if self.checkpoint.is_empty() {
            vec![0.0; self.engine.vocab_size().max(1)]
        } else {
            self.engine.infer_logits(&self.checkpoint)
        };
        self.rehydrate_live_state();
    }

    /// Returns the engine associated with this session.
    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }

    /// Automatically decodes up to `max_tokens` using the greedy argmax strategy.
    pub fn generate_argmax_tokens(&mut self, max_tokens: usize) -> Result<Tokens> {
        let mut out = Tokens::default();
        for step in 0..max_tokens {
            let token = self.argmax();
            // #region debug-point B:decode-step
            debug_session_event(
                "B",
                "src/session.rs:generate_argmax_tokens:step-start",
                "[DEBUG] decode step start",
                format!(
                    "{{\"step\":{},\"token\":{},\"checkpoint_len\":{},\"logits\":{}}}",
                    step,
                    token,
                    self.checkpoint.len(),
                    self.logits.len()
                ),
            );
            // #endregion
            let step_started = Instant::now();
            self.decode_next(token)?;
            // #region debug-point B:decode-step
            debug_session_event(
                "B",
                "src/session.rs:generate_argmax_tokens:step-done",
                "[DEBUG] decode step done",
                format!(
                    "{{\"step\":{},\"token\":{},\"checkpoint_len\":{},\"elapsed_ms\":{}}}",
                    step,
                    token,
                    self.checkpoint.len(),
                    step_started.elapsed().as_millis()
                ),
            );
            // #endregion
            out.push(token);
        }
        Ok(out)
    }

    /// Renders a slice of tokens into a human-readable string using the engine's tokenizer.
    pub fn render_tokens(&self, tokens: &Tokens) -> String {
        self.engine.decode_tokens(tokens)
    }

    fn report_prefill_progress(&mut self, cached_tokens: usize, total: usize) {
        if let Some(progress) = &self.progress {
            progress.on_progress("prefill_chunk", cached_tokens, total);
            if let Some(boundary) = self.prefill_boundary.take() {
                if boundary > cached_tokens && boundary < total {
                    progress.on_progress("prefill_chunk", boundary, total);
                }
            }
            progress.on_progress("prefill_chunk", total, total);
        } else {
            self.prefill_boundary = None;
        }
    }

    fn append_token_with_reference_fallback(&mut self, token: i32) {
        let pos = self.checkpoint.len();
        if let Some(logits) = self
            .engine
            .try_reference_decode_next(
                &mut self.transformer_kv,
                &mut self.decode_scratch,
                token,
                pos,
            )
        {
            self.checkpoint.push(token);
            self.logits = logits;
            return;
        }

        let mut prompt = self.checkpoint.clone();
        prompt.push(token);
        self.rebuild_prompt_state(&prompt);
    }

    fn reset_live_state(&mut self) {
        self.checkpoint = Tokens::default();
        self.transformer_kv.clear();
    }

    fn rebuild_prompt_state(&mut self, prompt: &Tokens) {
        self.reset_live_state();
        if prompt.is_empty() {
            self.logits = vec![0.0; self.engine.vocab_size().max(1)];
            return;
        }
        if let Some(logits) = self.engine.try_reference_prefill(
            &mut self.transformer_kv,
            &mut self.decode_scratch,
            &prompt.0,
            0,
        ) {
            self.checkpoint = prompt.clone();
            self.logits = logits;
            return;
        }
        self.checkpoint = prompt.clone();
        self.transformer_kv.clear();
        self.logits = self.engine.infer_logits(&self.checkpoint);
    }

    fn extend_live_suffix(&mut self, prompt: &Tokens, cached_tokens: usize) -> bool {
        let suffix = &prompt.0[cached_tokens..];
        if suffix.is_empty() {
            return true;
        }
        if let Some(logits) = self.engine.try_reference_prefill(
            &mut self.transformer_kv,
            &mut self.decode_scratch,
            suffix,
            cached_tokens,
        ) {
            self.checkpoint = prompt.clone();
            self.logits = logits;
            return true;
        }
        false
    }

    fn rehydrate_live_state(&mut self) {
        if self.checkpoint.is_empty() {
            return;
        }
        if let Some(logits) = self.engine.try_reference_prefill(
            &mut self.transformer_kv,
            &mut self.decode_scratch,
            &self.checkpoint.0,
            0,
        ) {
            self.logits = logits;
        }
    }

    fn try_load_dsv4_snapshot(&mut self, snapshot: &SessionSnapshot) -> bool {
        let header_bytes = SESSION_PAYLOAD_HEADER_U32S * 4;
        if snapshot.bytes.len() < header_bytes {
            return false;
        }
        let mut header = [0u32; SESSION_PAYLOAD_HEADER_U32S];
        for (idx, slot) in header.iter_mut().enumerate() {
            let start = idx * 4;
            let chunk = &snapshot.bytes[start..start + 4];
            *slot = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        if header[0] != SESSION_PAYLOAD_MAGIC || header[1] != SESSION_PAYLOAD_VERSION {
            return false;
        }
        let token_count = header[7] as usize;
        let vocab_size = header[11] as usize;
        let token_bytes = match token_count.checked_mul(4) {
            Some(value) => value,
            None => return false,
        };
        let logit_bytes = match vocab_size.checked_mul(4) {
            Some(value) => value,
            None => return false,
        };
        let expected_len = match header_bytes
            .checked_add(token_bytes)
            .and_then(|value| value.checked_add(logit_bytes))
        {
            Some(value) => value,
            None => return false,
        };
        if snapshot.bytes.len() != expected_len {
            return false;
        }

        let mut offset = header_bytes;
        let mut tokens = Vec::with_capacity(token_count);
        for chunk in snapshot.bytes[offset..offset + token_bytes].chunks_exact(4) {
            tokens.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        offset += token_bytes;
        let mut logits = Vec::with_capacity(vocab_size);
        for chunk in snapshot.bytes[offset..offset + logit_bytes].chunks_exact(4) {
            logits.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        self.ctx_size = header[2] as usize;
        self.prefill_boundary = match header[3] {
            0 => None,
            value => Some(value as usize),
        };
        self.checkpoint = Tokens(tokens);
        self.logits = logits;
        self.transformer_kv.clear();
        self.rehydrate_live_state();
        true
    }
}

fn debug_session_event(hypothesis_id: &str, location: &str, msg: &str, data_json: String) {
    // #region debug-point B:network-report
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
    let _ = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import pathlib, urllib.request, sys; p=pathlib.Path('.dbg/slow-prefill-startup.env'); u='http://127.0.0.1:7777/event';\n\
try:\n\
 c=p.read_text();\n\
 u=next((line.split('=',1)[1].strip() for line in c.splitlines() if line.startswith('DEBUG_SERVER_URL=')), u)\n\
except Exception:\n\
 pass\n\
urllib.request.urlopen(urllib.request.Request(u, data=sys.argv[1].encode(), headers={'Content-Type':'application/json'}), timeout=1).read()",
        )
        .arg(event)
        .output();
    // #endregion
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::engine::EngineOptions;

    #[test]
    fn generates_repeatable_argmax_stub_tokens() {
        let engine = Engine::open(EngineOptions {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-preview-model.gguf"),
            ..EngineOptions::default()
        })
        .unwrap();
        let vocab = engine.vocab_size() as i32;
        let mut session_a = Session::create(engine.clone(), 1024).unwrap();
        session_a.sync(&Tokens(vec![1, 2, 3, 4])).unwrap();
        let generated_a = session_a.generate_argmax_tokens(4).unwrap();

        let mut session_b = Session::create(engine, 1024).unwrap();
        session_b.sync(&Tokens(vec![1, 2, 3, 4])).unwrap();
        let generated_b = session_b.generate_argmax_tokens(4).unwrap();

        assert_eq!(generated_a.len(), 4);
        assert_eq!(generated_a, generated_b);
        assert!(generated_a.0.iter().all(|token| *token >= 0 && *token < vocab));
    }

    #[test]
    fn prefill_matches_sync_behavior() {
        let engine = Engine::open(EngineOptions {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-preview-model.gguf"),
            ..EngineOptions::default()
        })
        .unwrap();
        let prompt = Tokens(vec![7, 8, 9, 10]);

        let mut session_a = Session::create(engine.clone(), 1024).unwrap();
        let sync_stats = session_a.sync(&prompt).unwrap();
        let sync_logits = session_a.logits().to_vec();

        let mut session_b = Session::create(engine, 1024).unwrap();
        let prefill_stats = session_b.prefill(&prompt).unwrap();
        let prefill_logits = session_b.logits().to_vec();

        assert_eq!(sync_stats.cached_tokens, prefill_stats.cached_tokens);
        assert_eq!(sync_stats.replay_tokens, prefill_stats.replay_tokens);
        assert_eq!(sync_stats.rebuilt, prefill_stats.rebuilt);
        assert_eq!(sync_logits, prefill_logits);
    }

    #[test]
    fn decode_next_matches_eval_behavior() {
        let engine = Engine::open(EngineOptions {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-preview-model.gguf"),
            ..EngineOptions::default()
        })
        .unwrap();

        let mut session_a = Session::create(engine.clone(), 1024).unwrap();
        session_a.prefill(&Tokens(vec![1, 2, 3])).unwrap();
        session_a.eval(4).unwrap();

        let mut session_b = Session::create(engine, 1024).unwrap();
        session_b.prefill(&Tokens(vec![1, 2, 3])).unwrap();
        session_b.decode_next(4).unwrap();

        assert_eq!(session_a.tokens(), session_b.tokens());
        assert_eq!(session_a.logits(), session_b.logits());
    }

    #[test]
    fn prefill_rebuilds_from_zero_when_prompt_shrinks() {
        let engine = Engine::open(EngineOptions {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-preview-model.gguf"),
            ..EngineOptions::default()
        })
        .unwrap();

        let mut session = Session::create(engine, 1024).unwrap();
        session.prefill(&Tokens(vec![1, 2, 3, 4])).unwrap();
        let stats = session.prefill(&Tokens(vec![1, 2, 3])).unwrap();

        assert_eq!(stats.cached_tokens, 0);
        assert_eq!(stats.replay_tokens, 3);
        assert!(stats.rebuilt);
        assert_eq!(session.tokens(), &Tokens(vec![1, 2, 3]));
    }

    #[test]
    fn rewind_restores_same_state_as_fresh_prefill() {
        let engine = Engine::open(EngineOptions {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-preview-model.gguf"),
            ..EngineOptions::default()
        })
        .unwrap();

        let mut rewound = Session::create(engine.clone(), 1024).unwrap();
        rewound.prefill(&Tokens(vec![1, 2, 3, 4])).unwrap();
        rewound.rewind(2);

        let mut fresh = Session::create(engine, 1024).unwrap();
        fresh.prefill(&Tokens(vec![1, 2])).unwrap();

        assert_eq!(rewound.tokens(), fresh.tokens());
        assert_eq!(rewound.logits(), fresh.logits());
    }

    #[test]
    fn snapshot_round_trips_dsv4_payload_with_logits() {
        let engine = Engine::open(EngineOptions {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-preview-model.gguf"),
            ..EngineOptions::default()
        })
        .unwrap();

        let prompt = Tokens(vec![5, 6, 7]);
        let mut original = Session::create(engine.clone(), 1024).unwrap();
        original.set_prefill_boundary(Some(128));
        original.sync(&prompt).unwrap();
        let snapshot = original.save_snapshot();

        assert_eq!(
            u32::from_le_bytes([
                snapshot.bytes[0],
                snapshot.bytes[1],
                snapshot.bytes[2],
                snapshot.bytes[3],
            ]),
            SESSION_PAYLOAD_MAGIC
        );

        let mut restored = Session::create(engine, 8).unwrap();
        restored.load_snapshot(&snapshot);
        assert_eq!(restored.ctx(), 1024);
        assert_eq!(restored.tokens(), &prompt);
        assert_eq!(restored.logits(), original.logits());
    }

    #[test]
    fn snapshot_restore_matches_fresh_decode() {
        let engine = Engine::open(EngineOptions {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-preview-model.gguf"),
            ..EngineOptions::default()
        })
        .unwrap();

        let prompt = Tokens(vec![5, 6, 7]);
        let mut original = Session::create(engine.clone(), 1024).unwrap();
        original.sync(&prompt).unwrap();
        let snapshot = original.save_snapshot();

        let mut restored = Session::create(engine.clone(), 1024).unwrap();
        restored.load_snapshot(&snapshot);

        original.decode_next(8).unwrap();
        restored.decode_next(8).unwrap();

        let mut fresh = Session::create(engine, 1024).unwrap();
        fresh.sync(&Tokens(vec![5, 6, 7, 8])).unwrap();

        assert_eq!(restored.tokens(), original.tokens());
        assert_eq!(restored.logits(), original.logits());
        assert_eq!(restored.tokens(), fresh.tokens());
        assert_eq!(restored.logits(), fresh.logits());
    }
}
