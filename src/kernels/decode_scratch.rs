use crate::kernels::quant::BlockQ8K;

#[derive(Debug, Default)]
pub(crate) struct DecodeScratch {
    pub attn_norm: Vec<f32>,
    pub attn_qr: Vec<f32>,
    pub attn_qr_norm: Vec<f32>,
    pub attn_kv_raw: Vec<f32>,
    pub attn_low: Vec<f32>,

    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_mid: Vec<f32>,
    pub shared_out: Vec<f32>,

    pub routed_xq: Vec<BlockQ8K>,
    pub routed_mid_all: Vec<f32>,
    pub routed_midq: Vec<BlockQ8K>,
    pub routed_pair_out: Vec<f32>,
    pub routed_out: Vec<f32>,
}

impl DecodeScratch {
    pub fn new() -> Self {
        Self::default()
    }
}
