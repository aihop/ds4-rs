use crate::GgufModel;
use crate::weights::BoundWeights;
use super::graph::MetalGraph;

impl MetalGraph {
    pub unsafe fn encode_output_head(
        &mut self,
        model: &GgufModel,
        weights: &BoundWeights,
        vocab_dim: u64,
    ) -> bool {
        let hc_dim = (crate::kernels::ds4_constants::DS4_N_HC as u64) * (crate::kernels::ds4_constants::DS4_N_EMBD as u64);
        let mut ok = crate::ffi::ds4_gpu_rms_norm_plain_tensor(
            self.batch_flat_hc,
            self.batch_cur_hc,
            hc_dim as u32,
            crate::kernels::ds4_constants::DS4_RMS_EPS as f32,
        ) != 0;

        if ok {
            ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                self.output_pre,
                model.model_map_ptr(),
                model.file_size,
                weights.output_hc_fn.as_ref().unwrap().abs_offset,
                hc_dim,
                crate::kernels::ds4_constants::DS4_N_HC as u64,
                self.batch_flat_hc,
                1,
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_output_hc_weights_tensor(
                self.output_weights,
                self.output_pre,
                model.model_map_ptr(),
                model.file_size,
                weights.output_hc_scale.as_ref().unwrap().abs_offset,
                weights.output_hc_base.as_ref().unwrap().abs_offset,
                crate::kernels::ds4_constants::DS4_N_HC as u32,
                crate::kernels::ds4_constants::DS4_HC_EPS as f32,
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_hc_weighted_sum_tensor(
                self.output_embd,
                self.batch_cur_hc,
                self.output_weights,
                crate::kernels::ds4_constants::DS4_N_EMBD as u32,
                crate::kernels::ds4_constants::DS4_N_HC as u32,
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_rms_norm_weight_tensor(
                self.output_norm,
                self.output_embd,
                model.model_map_ptr(),
                model.file_size,
                weights.output_norm.as_ref().unwrap().abs_offset,
                crate::kernels::ds4_constants::DS4_N_EMBD as u32,
                crate::kernels::ds4_constants::DS4_RMS_EPS as f32,
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                self.logits,
                model.model_map_ptr(),
                model.file_size,
                weights.output.abs_offset,
                crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                vocab_dim,
                self.output_norm,
                1,
            ) != 0;
        }

        ok
    }
}
