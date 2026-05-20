use std::path::PathBuf;
mod inference;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::error::{Ds4Error, Result};
use crate::gguf::{load_model, GgufModel, GgufTensor};
use crate::tokenizer::Tokenizer;
use crate::types::Tokens;
use crate::weights::{bind_weights, BoundWeights};

/// Specifies the compute backend for inference operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Metal,
    Cuda,
    Cpu,
}

impl Backend {
    /// Returns the short string name of the backend.
    pub fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

/// Specifies the "think mode" quality or detail level during inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkMode {
    None,
    #[default]
    High,
    Max,
}

/// Configuration options for initializing an `Engine`.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub model_path: PathBuf,
    pub mtp_path: Option<PathBuf>,
    pub backend: Backend,
    pub n_threads: usize,
    pub mtp_draft_tokens: usize,
    pub mtp_margin: f32,
    pub warm_weights: bool,
    pub quality: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            model_path: resolve_default_model_path(),
            mtp_path: None,
            backend: Backend::Metal,
            n_threads: 8,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            warm_weights: false,
            quality: false,
        }
    }
}

fn resolve_default_model_path() -> PathBuf {
    let manifest_default = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ds4flash.gguf");
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("ds4flash.gguf"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("ds4flash.gguf"));
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("ds4flash.gguf"));
            }
        }
    }
    candidates.push(manifest_default.clone());
    candidates
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or(manifest_default)
}

/// Estimates of memory consumption for maintaining inference context.
#[derive(Clone, Debug, Default)]
pub struct ContextMemory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

/// The core inference engine orchestrating weights, backend, and computation.
// #[derive(Debug)]
pub struct Engine {
    pub(crate) options: EngineOptions,
    pub(crate) model: Option<GgufModel>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) weights: Option<BoundWeights>,
    pub(crate) metal_graph: std::sync::Mutex<Option<crate::kernels::metal::MetalGraph>>,
}

impl Engine {
    /// Opens and initializes the engine with the provided options.
    /// Returns an `Arc<Self>` upon successful setup.
    pub fn open(options: EngineOptions) -> Result<Arc<Self>> {
        if options.model_path.as_os_str().is_empty() {
            return Err(Ds4Error::InvalidArgument(
                "model path must not be empty".to_string(),
            ));
        }

        let (model, tokenizer, weights) = if options.model_path.exists() {
            let model = load_model(&options.model_path)?;
            let tokenizer = Tokenizer::from_gguf(&model)?;
            let weights = bind_weights(&model)?;
            (Some(model), tokenizer, Some(weights))
        } else {
            (None, Tokenizer::preview(), None)
        };

        if options.backend == Backend::Metal {
            if let Some(model) = &model {
                unsafe {
                    let ok = crate::ffi::ds4_gpu_init();
                    if ok == 0 {
                        eprintln!("ds4-rs: ds4_gpu_init failed");
                    }
                    let map_ok = crate::ffi::ds4_gpu_set_model_map_range(
                        model.model_map_ptr(),
                        model.file_size,
                        model.tensor_data_pos,
                        model.file_size - model.tensor_data_pos,
                    );
                    if map_ok == 0 {
                        eprintln!("ds4-rs: ds4_gpu_set_model_map_range failed");
                    }
                }
            }
        }

        Ok(Arc::new(Self {
            options,
            model,
            tokenizer,
            weights,
            metal_graph: std::sync::Mutex::new(None),
        }))
    }

    /// Returns the engine's configuration options.
    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    /// Returns a human-readable summary of the engine's current state.
    pub fn summary(&self) -> String {
        if let Some(model) = &self.model {
            format!(
                "ds4-rust engine model={} backend={} threads={} gguf=v{} arch={} tensors={} kv={} tensor_data_pos={} vocab={} tokenizer=real",
                self.options.model_path.display(),
                self.options.backend.name(),
                self.options.n_threads,
                model.version,
                model.architecture.as_deref().unwrap_or("unknown"),
                model.n_tensors,
                model.n_kv,
                model.tensor_data_pos,
                model.vocab_size
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| self.tokenizer.vocab_size().unwrap_or_default().to_string()),
            )
        } else {
            format!(
                "ds4-rust preview engine model={} backend={} threads={} tokenizer=preview",
                self.options.model_path.display(),
                self.options.backend.name(),
                self.options.n_threads
            )
        }
    }

    /// Estimates context memory requirements for a given context size.
    pub fn context_memory_estimate(&self, ctx_size: usize) -> ContextMemory {
        let token_bytes = ctx_size as u64 * 4096;
        ContextMemory {
            total_bytes: token_bytes * 3,
            raw_bytes: token_bytes,
            compressed_bytes: token_bytes,
            scratch_bytes: token_bytes,
            prefill_cap: 2048,
            raw_cap: ctx_size.min(128) as u32,
            comp_cap: ctx_size as u32,
        }
    }

    /// Tokenizes a raw text string into a sequence of tokens.
    pub fn tokenize_text(&self, text: &str) -> Tokens {
        self.tokenizer.tokenize_text(text)
    }

    /// Retrieves the string representation for a single token ID.
    pub fn token_text(&self, token: i32) -> String {
        self.tokenizer.token_text(token)
    }

    /// Decodes a sequence of tokens back into a string.
    pub fn decode_tokens(&self, tokens: &Tokens) -> String {
        self.tokenizer.decode_tokens(tokens)
    }

    /// Prepares a chat prompt with the specified system and user text.
    pub fn render_chat_prompt(&self, system: &str, prompt: &str, think_mode: ThinkMode) -> Tokens {
        self.tokenizer.render_chat_prompt(system, prompt, think_mode)
    }

    /// Checks if a valid GGUF model is loaded.
    pub fn has_real_model(&self) -> bool {
        self.model.is_some()
    }

    /// Checks if weights have been successfully bound to memory.
    pub fn has_bound_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Indicates whether the engine can run trustworthy generations.
    pub fn supports_trustworthy_generation(&self) -> bool {
        self.weights.as_ref().is_some_and(|weights| {
            self.should_use_reference_logits(weights)
                && !weights.blocks.is_empty()
                && weights.blocks.iter().all(|block| block.ffn.is_some())
        })
    }

    /// Provides a reference to the loaded `GgufModel`, if any.
    pub fn gguf_model(&self) -> Option<&GgufModel> {
        self.model.as_ref()
    }

    /// Retrieves a specific `GgufTensor` by its name.
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.model.as_ref().and_then(|model| model.tensor(name))
    }

    /// Returns the number of tensors currently loaded.
    pub fn tensor_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.tensors.len())
            .unwrap_or(0)
    }

    /// Returns the active vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.model
            .as_ref()
            .and_then(|m| m.vocab_size.map(|v| v as usize))
            .or_else(|| self.tokenizer.vocab_size())
            .unwrap_or(256)
    }

}
