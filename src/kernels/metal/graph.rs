use crate::ffi::*;
use crate::gguf::GgufModel;
use crate::weights::BoundWeights;
use std::ffi::c_void;

/// A wrapper representing a Metal computation graph for inference.
pub struct MetalGraph {
    pub batch_cur_hc: *mut ds4_gpu_tensor,
    pub batch_next_hc: *mut ds4_gpu_tensor,
    pub batch_flat_hc: *mut ds4_gpu_tensor,
    pub batch_hc_mix: *mut ds4_gpu_tensor,
    pub batch_hc_split: *mut ds4_gpu_tensor,
    pub batch_attn_cur: *mut ds4_gpu_tensor,
    pub batch_attn_norm: *mut ds4_gpu_tensor,
    pub batch_qr: *mut ds4_gpu_tensor,
    pub batch_qr_norm: *mut ds4_gpu_tensor,
    pub batch_q: *mut ds4_gpu_tensor,
    pub batch_kv_raw: *mut ds4_gpu_tensor,
    pub batch_kv: *mut ds4_gpu_tensor,
    pub batch_heads: *mut ds4_gpu_tensor,
    pub batch_attn_low: *mut ds4_gpu_tensor,
    pub batch_attn_out: *mut ds4_gpu_tensor,
    pub batch_group_tmp: *mut ds4_gpu_tensor,
    pub batch_low_tmp: *mut ds4_gpu_tensor,
    pub batch_after_attn_hc: *mut ds4_gpu_tensor,
    pub batch_ffn_cur: *mut ds4_gpu_tensor,
    pub batch_ffn_norm: *mut ds4_gpu_tensor,
    pub batch_shared_gate: *mut ds4_gpu_tensor,
    pub batch_shared_up: *mut ds4_gpu_tensor,
    pub batch_shared_mid: *mut ds4_gpu_tensor,
    pub batch_shared_out: *mut ds4_gpu_tensor,
    pub batch_router_logits: *mut ds4_gpu_tensor,
    pub batch_router_probs: *mut ds4_gpu_tensor,
    pub batch_router_selected: *mut ds4_gpu_tensor,
    pub batch_router_weights: *mut ds4_gpu_tensor,
    pub batch_routed_gate: *mut ds4_gpu_tensor,
    pub batch_routed_up: *mut ds4_gpu_tensor,
    pub batch_routed_mid: *mut ds4_gpu_tensor,
    pub batch_routed_down: *mut ds4_gpu_tensor,
    pub batch_routed_out: *mut ds4_gpu_tensor,
    pub batch_ffn_out: *mut ds4_gpu_tensor,
    pub logits: *mut ds4_gpu_tensor,
    pub tokens: *mut ds4_gpu_tensor,
    pub batch_layer_n_index_comp: [u32; 61],
    pub batch_layer_n_comp: [u32; 61],
    pub batch_layer_comp_cap: [u32; 61],
    pub batch_layer_attn_comp_cache: [*mut ds4_gpu_tensor; 61],
    pub batch_comp_kv_cur: *mut ds4_gpu_tensor,
    pub batch_comp_sc_cur: *mut ds4_gpu_tensor,
    pub batch_layer_index_state_kv: [*mut ds4_gpu_tensor; 61],
    pub batch_layer_index_state_score: [*mut ds4_gpu_tensor; 61],
    pub batch_indexer_q: *mut ds4_gpu_tensor,
    pub batch_indexer_scores: *mut ds4_gpu_tensor,
    pub batch_indexer_weights: *mut ds4_gpu_tensor,
    pub batch_layer_index_comp_cache: [*mut ds4_gpu_tensor; 61],
    pub batch_comp_selected: *mut ds4_gpu_tensor,
    pub batch_raw_window: u32,
    pub router_selected: *mut ds4_gpu_tensor,
    pub router_weights: *mut ds4_gpu_tensor,
    pub router_probs: *mut ds4_gpu_tensor,
    pub router_logits: *mut ds4_gpu_tensor,
    pub routed_out: *mut ds4_gpu_tensor,
    pub routed_gate: *mut ds4_gpu_tensor,
    pub routed_up: *mut ds4_gpu_tensor,
    pub routed_mid: *mut ds4_gpu_tensor,
    pub routed_down: *mut ds4_gpu_tensor,
    pub ffn_norm: *mut ds4_gpu_tensor,
    pub quality: bool,
    pub shared_gate: *mut ds4_gpu_tensor,
    pub shared_up: *mut ds4_gpu_tensor,
    pub shared_mid: *mut ds4_gpu_tensor,
    pub shared_out: *mut ds4_gpu_tensor,
    pub ffn_out: *mut ds4_gpu_tensor,
    pub after_ffn_hc: *mut ds4_gpu_tensor,
    pub batch_hc_post: *mut ds4_gpu_tensor,
    pub batch_hc_comb: *mut ds4_gpu_tensor,    pub qr: *mut ds4_gpu_tensor,    pub kv: *mut ds4_gpu_tensor,    pub kv_raw: *mut ds4_gpu_tensor,    pub qr_norm: *mut ds4_gpu_tensor,    pub layer_attn_state_kv: [*mut ds4_gpu_tensor; 61],    pub layer_attn_state_score: [*mut ds4_gpu_tensor; 61],
    pub cur_hc: *mut crate::ffi::ds4_gpu_tensor,
    pub flat_hc: *mut crate::ffi::ds4_gpu_tensor,
    pub output_pre: *mut crate::ffi::ds4_gpu_tensor,
    pub output_weights: *mut crate::ffi::ds4_gpu_tensor,
    pub output_embd: *mut crate::ffi::ds4_gpu_tensor,
    pub output_norm: *mut crate::ffi::ds4_gpu_tensor,
    pub layer_raw_cache: [*mut crate::ffi::ds4_gpu_tensor; 61],
    pub raw_cap: u32,
}

impl MetalGraph {
    /// Creates a new `MetalGraph` and allocates necessary GPU memory buffers.
    pub fn new(prefill_cap: usize, n_embd: usize, n_hc: usize) -> Self {
        // SAFETY: `ds4_gpu_tensor_alloc` is called with correctly calculated byte sizes.
        // It returns valid pointers to GPU memory, which are safely encapsulated within `MetalGraph`.
        unsafe {
            let mut res: Self = std::mem::zeroed();
            res.batch_cur_hc = ds4_gpu_tensor_alloc((prefill_cap * n_hc * n_embd * 4) as u64);
            res.batch_next_hc = ds4_gpu_tensor_alloc((prefill_cap * n_hc * n_embd * 4) as u64);
            res.batch_flat_hc = ds4_gpu_tensor_alloc((prefill_cap * n_hc * n_embd * 4) as u64);
            res.batch_hc_mix = ds4_gpu_tensor_alloc((prefill_cap * (n_hc * 2 + n_hc * n_hc) * 4) as u64);
            res.batch_hc_split = ds4_gpu_tensor_alloc((prefill_cap * (n_hc * 2 + n_hc * n_hc) * 4) as u64);
            res.batch_attn_cur = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_attn_norm = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_qr = ds4_gpu_tensor_alloc((prefill_cap * 1536 * 4) as u64);
            res.batch_qr_norm = ds4_gpu_tensor_alloc((prefill_cap * 1536 * 4) as u64);
            res.batch_q = ds4_gpu_tensor_alloc((prefill_cap * 16384 * 4) as u64);
            res.batch_kv_raw = ds4_gpu_tensor_alloc((prefill_cap * 512 * 4) as u64);
            res.batch_kv = ds4_gpu_tensor_alloc((prefill_cap * 512 * 4) as u64);
            res.batch_heads = ds4_gpu_tensor_alloc((prefill_cap * 16384 * 4) as u64);
            res.batch_attn_low = ds4_gpu_tensor_alloc((prefill_cap * 1536 * 4) as u64);
            res.batch_attn_out = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_group_tmp = ds4_gpu_tensor_alloc((prefill_cap * 1536 * 4) as u64);
            res.batch_low_tmp = ds4_gpu_tensor_alloc((prefill_cap * 1536 * 4) as u64);
            res.batch_after_attn_hc = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_ffn_cur = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_ffn_norm = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_shared_gate = ds4_gpu_tensor_alloc((prefill_cap * 2048 * 4) as u64);
            res.batch_shared_up = ds4_gpu_tensor_alloc((prefill_cap * 2048 * 4) as u64);
            res.batch_shared_mid = ds4_gpu_tensor_alloc((prefill_cap * 2048 * 4) as u64);
            res.batch_shared_out = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_router_logits = ds4_gpu_tensor_alloc((prefill_cap * 256 * 4) as u64);
            res.batch_router_probs = ds4_gpu_tensor_alloc((prefill_cap * 256 * 4) as u64);
            res.batch_router_selected = ds4_gpu_tensor_alloc((prefill_cap * 8 * 4) as u64);
            res.batch_router_weights = ds4_gpu_tensor_alloc((prefill_cap * 8 * 4) as u64);
            res.batch_routed_gate = ds4_gpu_tensor_alloc((prefill_cap * 8 * 2048 * 4) as u64);
            res.batch_routed_up = ds4_gpu_tensor_alloc((prefill_cap * 8 * 2048 * 4) as u64);
            res.batch_routed_mid = ds4_gpu_tensor_alloc((prefill_cap * 8 * 2048 * 4) as u64);
            res.batch_routed_down = ds4_gpu_tensor_alloc((prefill_cap * 8 * n_embd * 4) as u64);
            res.batch_routed_out = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.batch_ffn_out = ds4_gpu_tensor_alloc((prefill_cap * n_embd * 4) as u64);
            res.logits = ds4_gpu_tensor_alloc((1 * 131072 * 4) as u64);
            res.tokens = ds4_gpu_tensor_alloc((prefill_cap * 4) as u64);
            
            res.cur_hc = res.batch_cur_hc; // For now, share with batch
            res.flat_hc = res.batch_flat_hc;
            res.after_ffn_hc = res.batch_next_hc;
            
            let hc_dim = (crate::kernels::ds4_constants::DS4_N_HC as u64) * (crate::kernels::ds4_constants::DS4_N_EMBD as u64);
            res.output_pre = ds4_gpu_tensor_alloc(hc_dim * 4);
            res.output_weights = ds4_gpu_tensor_alloc((crate::kernels::ds4_constants::DS4_N_HC as u64) * (crate::kernels::ds4_constants::DS4_N_EMBD as u64) * 4);
            res.output_embd = ds4_gpu_tensor_alloc((crate::kernels::ds4_constants::DS4_N_EMBD as u64) * 4);
            res.output_norm = ds4_gpu_tensor_alloc((crate::kernels::ds4_constants::DS4_N_EMBD as u64) * 4);
            
            res.raw_cap = 4096;
            res.batch_raw_window = 4096;
            for i in 0..61 {
                res.layer_raw_cache[i] = ds4_gpu_tensor_alloc((res.raw_cap as u64) * (crate::kernels::ds4_constants::DS4_N_HEAD_KV as u64) * (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u64) * 4);
            }
            res
        }
    }

}

impl Drop for MetalGraph {
    fn drop(&mut self) {
        // SAFETY: All `ds4_gpu_tensor_free` calls receive valid pointers previously allocated via `ds4_gpu_tensor_alloc`.
        // This is called exactly once when `MetalGraph` is dropped, preventing double-free.
        unsafe {
            ds4_gpu_tensor_free(self.batch_cur_hc);
            ds4_gpu_tensor_free(self.batch_next_hc);
            ds4_gpu_tensor_free(self.batch_flat_hc);
            ds4_gpu_tensor_free(self.batch_hc_mix);
            ds4_gpu_tensor_free(self.batch_hc_split);
            ds4_gpu_tensor_free(self.batch_attn_cur);
            ds4_gpu_tensor_free(self.batch_attn_norm);
            ds4_gpu_tensor_free(self.batch_qr);
            ds4_gpu_tensor_free(self.batch_qr_norm);
            ds4_gpu_tensor_free(self.batch_q);
            ds4_gpu_tensor_free(self.batch_kv_raw);
            ds4_gpu_tensor_free(self.batch_kv);
            ds4_gpu_tensor_free(self.batch_heads);
            ds4_gpu_tensor_free(self.batch_attn_low);
            ds4_gpu_tensor_free(self.batch_attn_out);
            ds4_gpu_tensor_free(self.batch_group_tmp);
            ds4_gpu_tensor_free(self.batch_low_tmp);
            ds4_gpu_tensor_free(self.batch_after_attn_hc);
            ds4_gpu_tensor_free(self.batch_ffn_cur);
            ds4_gpu_tensor_free(self.batch_ffn_norm);
            ds4_gpu_tensor_free(self.batch_shared_gate);
            ds4_gpu_tensor_free(self.batch_shared_up);
            ds4_gpu_tensor_free(self.batch_shared_mid);
            ds4_gpu_tensor_free(self.batch_shared_out);
            ds4_gpu_tensor_free(self.batch_router_logits);
            ds4_gpu_tensor_free(self.batch_router_probs);
            ds4_gpu_tensor_free(self.batch_router_selected);
            ds4_gpu_tensor_free(self.batch_router_weights);
            ds4_gpu_tensor_free(self.batch_routed_gate);
            ds4_gpu_tensor_free(self.batch_routed_up);
            ds4_gpu_tensor_free(self.batch_routed_mid);
            ds4_gpu_tensor_free(self.batch_routed_down);
            ds4_gpu_tensor_free(self.batch_routed_out);
            ds4_gpu_tensor_free(self.batch_ffn_out);
            ds4_gpu_tensor_free(self.logits);
            ds4_gpu_tensor_free(self.tokens);
        }
    }
}
// SAFETY: `MetalGraph` only holds pointers to GPU memory which can be safely sent across threads.
// FFI functions that interact with these tensors must be thread-safe or externally synchronized.
unsafe impl Send for MetalGraph {}
// SAFETY: Methods accessing the pointers require `&mut self` or external synchronization.
unsafe impl Sync for MetalGraph {}
