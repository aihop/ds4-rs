pub const DS4_N_LAYER: usize = 43;
pub const DS4_N_EMBD: usize = 4096;
pub const DS4_N_VOCAB: usize = 129280;
pub const DS4_N_HEAD: usize = 64;
pub const DS4_N_HEAD_KV: usize = 1;
pub const DS4_N_HEAD_DIM: usize = 512;
pub const DS4_N_VALUE_DIM: usize = 512;
pub const DS4_N_ROT: usize = 64;
pub const DS4_N_OUT_GROUP: usize = 8;
pub const DS4_N_LORA_Q: usize = 1024;
pub const DS4_N_LORA_O: usize = 1024;
pub const DS4_N_EXPERT: usize = 256;
pub const DS4_N_EXPERT_USED: usize = 6;
pub const DS4_N_EXPERT_SHARED: usize = 1;
pub const DS4_N_FF_EXP: usize = 2048;
pub const DS4_N_HASH_LAYER: usize = 3;
pub const DS4_N_SWA: usize = 128;
pub const DS4_N_INDEXER_HEAD: usize = 64;
pub const DS4_N_INDEXER_HEAD_DIM: usize = 128;
pub const DS4_N_INDEXER_TOP_K: usize = 512;
pub const DS4_N_HC: usize = 4;
pub const DS4_N_HC_SINKHORN_ITER: u32 = 20;

pub const DS4_RMS_EPS: f32 = 1.0e-6;
pub const DS4_HC_EPS: f32 = 1.0e-6;
pub const DS4_ROPE_SCALE_FACTOR: f32 = 16.0;
pub const DS4_ROPE_ORIG_CTX: u32 = 65536;
pub const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
pub const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;pub const DS4_SWIGLU_CLAMP_EXP: f32 = 0.0;
pub fn layer_rope_freq_base(_il: usize) -> f32 { 10000.0 }
pub fn layer_rope_freq_scale(_il: usize) -> f32 { 1.0 }
