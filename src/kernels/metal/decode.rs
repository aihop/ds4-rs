use crate::GgufModel;
use crate::weights::BoundWeights;
use super::graph::MetalGraph;

impl MetalGraph {

    pub unsafe fn metal_graph_matmul_plain_tensor(
        out: *mut crate::ffi::ds4_gpu_tensor,
        model: &crate::GgufModel,
        w: &crate::weights::BoundTensor,
        in_dim: u64,
        out_dim: u64,
        x: *mut crate::ffi::ds4_gpu_tensor,
        n_tok: u64,
    ) -> bool {
        if w.tensor_type == 1 {
            crate::ffi::ds4_gpu_matmul_f16_tensor(
                out,
                model.model_map_ptr(),
                model.file_size,
                w.abs_offset,
                in_dim,
                out_dim,
                x,
                n_tok,
            ) != 0
        } else if w.tensor_type == 0 {
            crate::ffi::ds4_gpu_matmul_f32_tensor(
                out,
                model.model_map_ptr(),
                model.file_size,
                w.abs_offset,
                in_dim,
                out_dim,
                x,
                n_tok,
            ) != 0
        } else {
            false
        }
    }

    /// Executes a single decode step on the GPU.
    /// Returns `true` if the commands were successfully dispatched.
    pub fn execute_decode_step(
        &mut self,
        model: &GgufModel,
        weights: &BoundWeights,
        token: i32,
        pos: usize,
    ) -> bool {
        unsafe {
            if crate::ffi::ds4_gpu_begin_commands() == 0 {
                return false;
            }

            // Execute the single token inference loop
            let mut ok = self.encode_token_raw_swa(model, weights, token, pos, true);

            crate::ffi::ds4_gpu_end_commands();
            crate::ffi::ds4_gpu_synchronize();
            ok
        }
    }

    pub unsafe fn encode_token_raw_swa(
        &mut self,
        model: &GgufModel,
        weights: &BoundWeights,
        token: i32,
        pos: usize,
        need_logits: bool,
    ) -> bool {
        if self.raw_cap == 0 {
            return false;
        }
        let raw_row = (pos as u32) % self.raw_cap;
        
        let window = if self.batch_raw_window != 0 { self.batch_raw_window } else { crate::kernels::ds4_constants::DS4_N_SWA as u32 };
        let mut needed = 1_u64;
        if window > 0 {
            needed += (window as u64) - 1;
        }
        let available = (pos as u64) + 1;
        if needed > available {
            needed = available;
        }
        if needed > (self.raw_cap as u64) {
            needed = self.raw_cap as u64;
        }
        let n_raw = needed as u32;

        let mut ok = crate::ffi::ds4_gpu_embed_token_hc_tensor(
                self.batch_cur_hc,
                model.model_map_ptr(),
                model.file_size,
                weights.token_embd.abs_offset,
                weights.token_embd.dims[1] as u32,
                token as u32,
                crate::kernels::ds4_constants::DS4_N_EMBD as u32,
                crate::kernels::ds4_constants::DS4_N_HC as u32,
            ) != 0;
            
            if !ok {
                eprintln!("ds4-rs: ds4_gpu_embed_token_hc_tensor failed");
            }

            for il in 0..61 {
                if !ok {
                    eprintln!("ds4-rs: loop broken at layer {}", il);
                    break;
                }
                ok = self.encode_decode_layer(
                    model,
                    weights,
                    il,
                    pos,
                    self.layer_raw_cache[il],
                    self.raw_cap,
                    raw_row,
                    n_raw,
                    token,
                );
                if !ok {
                    eprintln!("ds4-rs: encode_decode_layer failed at layer {}", il);
                }
                
                let tmp = self.batch_cur_hc;
                self.batch_cur_hc = self.after_ffn_hc;
                self.after_ffn_hc = tmp;
            }

            if ok && need_logits {
                ok = self.encode_output_head(model, weights, weights.output.dims[1] as u64);
                if !ok {
                    eprintln!("ds4-rs: encode_output_head failed");
                }
            }
        ok
    }
    pub unsafe fn encode_decode_layer(
        &mut self,
        model: &GgufModel,
        weights: &crate::weights::BoundWeights,
        il: usize,
        pos: usize,
        raw_cache: *mut crate::ffi::ds4_gpu_tensor,
        raw_cap: u32,
        raw_row: u32,
        n_raw: u32,
        token: i32,
    ) -> bool {
        let layer_attn = &weights.blocks[il].attention;
        let layer_ffn = weights.blocks[il].ffn.as_ref().unwrap();
        let hc_dim = (crate::kernels::ds4_constants::DS4_N_HC as u64)
            * (crate::kernels::ds4_constants::DS4_N_EMBD as u64);
        let mix_hc = 2 * (crate::kernels::ds4_constants::DS4_N_HC as u64)
            + (crate::kernels::ds4_constants::DS4_N_HC as u64)
                * (crate::kernels::ds4_constants::DS4_N_HC as u64);
        let q_rank = layer_attn.attn_q_a.dims[1];
        let q_dim = (crate::kernels::ds4_constants::DS4_N_HEAD as u64)
            * (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u64);
        let n_groups = (crate::kernels::ds4_constants::DS4_N_OUT_GROUP as u32);
        let group_heads = (crate::kernels::ds4_constants::DS4_N_HEAD as u32) / (n_groups as u32);
        let group_dim = (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32) * group_heads;
        let rank = (crate::kernels::ds4_constants::DS4_N_LORA_O as u32);
        let shared_dim = (layer_ffn.ffn_gate_shexp.dims[1] as u32);
        let expert_in_dim = layer_ffn.ffn_gate_exps.dims[0];
        let expert_mid_dim = layer_ffn.ffn_gate_exps.dims[1];
        let down_in_dim = layer_ffn.ffn_down_exps.dims[0];
        let routed_out_dim = layer_ffn.ffn_down_exps.dims[1];
        let compressed = false;
        let freq_base = crate::kernels::ds4_constants::layer_rope_freq_base(il);
        let freq_scale = crate::kernels::ds4_constants::layer_rope_freq_scale(il);
        let ext_factor = if compressed && (crate::kernels::ds4_constants::DS4_ROPE_SCALE_FACTOR as f64) > 1.0
        {
            1.0
        } else {
            0.0
        };
        let mut attn_factor = 1.0;
        if ext_factor != 0.0 && freq_scale > 0.0 {
            attn_factor /= 1.0 + 0.1 * (1.0_f32 / freq_scale).ln();
        }
        let qkv_rms_fused = true;

        let mut ok = true;
        if ok {
            ok = crate::ffi::ds4_gpu_rms_norm_plain_tensor(
                self.batch_flat_hc,
                self.batch_cur_hc,
                (hc_dim as u32),
                (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
            ) != 0;
        }
        if ok {
            ok = Self::metal_graph_matmul_plain_tensor(
                self.batch_hc_mix,
                model,
                layer_attn.hc_attn_fn.as_ref().unwrap(),
                hc_dim as u64,
                mix_hc as u64,
                self.batch_flat_hc,
                1,
            );
        }
        let fuse_hc_norm =
            true && true;
        if ok && fuse_hc_norm {
            ok = crate::ffi::ds4_gpu_hc_split_weighted_sum_norm_tensor(
                self.batch_attn_cur,
                self.batch_attn_norm,
                self.batch_hc_split,
                self.batch_hc_mix,
                self.batch_cur_hc,
                model.model_map_ptr(),
                model.file_size,
                layer_attn.hc_attn_scale.as_ref().unwrap().abs_offset,
                layer_attn.hc_attn_base.as_ref().unwrap().abs_offset,
                layer_attn.attn_norm.abs_offset,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_N_HC as u32),
                (crate::kernels::ds4_constants::DS4_N_HC_SINKHORN_ITER as u32),
                (crate::kernels::ds4_constants::DS4_HC_EPS as f32),
                (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
            ) != 0;
        } else if ok {
            ok = true; /* metal_graph_decode_hc_pre
                self.batch_attn_cur,
                self.batch_hc_split,
                self.batch_hc_mix,
                self.batch_cur_hc,
                model,
                layer_attn.hc_attn_scale.as_ref().unwrap().abs_offset,
                layer_attn.hc_attn_base.as_ref().unwrap().abs_offset,
            */;
        }

        if ok && !fuse_hc_norm {
            ok = crate::ffi::ds4_gpu_rms_norm_weight_tensor(
                self.batch_attn_norm,
                self.batch_attn_cur,
                model.model_map_ptr(),
                model.file_size,
                layer_attn.attn_norm.abs_offset,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                self.qr,
                model.model_map_ptr(),
                model.file_size,
                layer_attn.attn_q_a.abs_offset,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                q_rank,
                self.batch_attn_norm,
                1,
            ) != 0;
        }

        if qkv_rms_fused {
            if ok {
                ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                    self.kv_raw,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_kv.abs_offset,
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u64),
                    self.batch_attn_norm,
                    1,
                ) != 0;
            }

            if ok {
                ok = crate::ffi::ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                    self.qr_norm,
                    self.qr,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_q_a_norm.abs_offset,
                    (q_rank as u32),
                    self.kv,
                    self.kv_raw,
                    layer_attn.attn_kv_a_norm.abs_offset,
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                    1,
                    (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
                ) != 0;
            }
        } else {
            if ok {
                ok = crate::ffi::ds4_gpu_rms_norm_weight_tensor(
                    self.qr_norm,
                    self.qr,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_q_a_norm.abs_offset,
                    (q_rank as u32),
                    (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
                ) != 0;
            }
        }

        if ok {
            ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                self.batch_q,
                model.model_map_ptr(),
                model.file_size,
                layer_attn.attn_q_b.abs_offset,
                q_rank,
                q_dim,
                self.qr_norm,
                1,
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_head_rms_norm_tensor(
                self.batch_q,
                1,
                (crate::kernels::ds4_constants::DS4_N_HEAD as u32),
                (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_rope_tail_tensor(
                self.batch_q,
                1,
                (crate::kernels::ds4_constants::DS4_N_HEAD as u32),
                (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                (crate::kernels::ds4_constants::DS4_N_ROT as u32),
                ((pos as u32) as u32),
                if compressed {
                    (crate::kernels::ds4_constants::DS4_ROPE_ORIG_CTX as u32) as u32
                } else {
                    0
                },
                false,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_FAST as f32),
                (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_SLOW as f32),
            ) != 0;
        }

        if !qkv_rms_fused {
            if ok {
                ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                    self.kv_raw,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_kv.abs_offset,
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u64),
                    self.batch_attn_norm,
                    1,
                ) != 0;
            }

            if ok {
                ok = crate::ffi::ds4_gpu_rms_norm_weight_tensor(
                    self.kv,
                    self.kv_raw,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_kv_a_norm.abs_offset,
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                    (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
                ) != 0;
            }
        }
        if ok {
            ok = crate::ffi::ds4_gpu_rope_tail_tensor(
                self.kv,
                1,
                (crate::kernels::ds4_constants::DS4_N_HEAD_KV as u32),
                (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                (crate::kernels::ds4_constants::DS4_N_ROT as u32),
                ((pos as u32) as u32),
                if compressed {
                    (crate::kernels::ds4_constants::DS4_ROPE_ORIG_CTX as u32) as u32
                } else {
                    0
                },
                false,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_FAST as f32),
                (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_SLOW as f32),
            ) != 0;
        }

        /* RoPE stays as the exact standalone kernel above.  The decode fusion
         * starts after that, where FP8 KV quantization and raw-cache storage can
         * share one pass without changing the trigonometric path. */
        if ok {
            ok = true;
        }

        let mut n_comp = 0;
        let mut comp_cache: *mut crate::ffi::ds4_gpu_tensor = std::ptr::null_mut();
        let mut comp_selected: *mut crate::ffi::ds4_gpu_tensor = std::ptr::null_mut();
        let mut n_selected = 0;
        let mut decode_index_stage_t0 = 0.0;
        let decode_index_stage_profile = false;
        if ok && compressed {
            let ratio = 0;
            let coff = if ratio == 4 { 2 } else { 1 };
            let comp_width = (coff as u64) * (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u64);
            let emit = (((pos as u32) + 1_u32) % ratio) == 0_u32;
            if layer_attn.attn_compressor_kv.is_none()
                || layer_attn.attn_compressor_gate.is_none()
                || layer_attn.attn_compressor_ape.is_none()
                || layer_attn.attn_compressor_norm.is_none()
                || layer_attn.attn_compressor_kv.as_ref().unwrap().tensor_type != 1
                || layer_attn
                    .attn_compressor_gate
                    .as_ref()
                    .unwrap()
                    .tensor_type
                    != 1
                || layer_attn.attn_compressor_kv.as_ref().unwrap().dims[0]
                    != (crate::kernels::ds4_constants::DS4_N_EMBD as u64)
                || layer_attn.attn_compressor_gate.as_ref().unwrap().dims[0]
                    != (crate::kernels::ds4_constants::DS4_N_EMBD as u64)
                || layer_attn.attn_compressor_kv.as_ref().unwrap().dims[1] != comp_width
                || layer_attn.attn_compressor_gate.as_ref().unwrap().dims[1] != comp_width
            {
                ok = false;
            }
            if ok && emit && self.batch_layer_n_comp[il] >= self.batch_layer_comp_cap[il] {
                ok = false;
            }
            if (ok && !false) {
                ok = crate::ffi::ds4_gpu_matmul_f16_pair_tensor(
                    self.batch_comp_kv_cur,
                    self.batch_comp_sc_cur,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_compressor_kv.as_ref().unwrap().abs_offset,
                    layer_attn.attn_compressor_gate.as_ref().unwrap().abs_offset,
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                    comp_width,
                    self.batch_attn_norm,
                    1,
                ) != 0;
            } else {
                if ok {
                    ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                        self.batch_comp_kv_cur,
                        model.model_map_ptr(),
                        model.file_size,
                        layer_attn.attn_compressor_kv.as_ref().unwrap().abs_offset,
                        crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                        comp_width,
                        self.batch_attn_norm,
                        1,
                    ) != 0;
                }
                if ok {
                    ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                        self.batch_comp_sc_cur,
                        model.model_map_ptr(),
                        model.file_size,
                        layer_attn.attn_compressor_gate.as_ref().unwrap().abs_offset,
                        crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                        comp_width,
                        self.batch_attn_norm,
                        1,
                    ) != 0;
                }
            }
            let comp_row = self.batch_layer_n_comp[il];
            if ok {
                ok = crate::ffi::ds4_gpu_compressor_update_tensor(
                    self.batch_comp_kv_cur,
                    self.batch_comp_sc_cur,
                    self.layer_attn_state_kv[il],
                    self.layer_attn_state_score[il],
                    self.batch_layer_attn_comp_cache[il],
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_compressor_ape.as_ref().unwrap().abs_offset,
                    layer_attn.attn_compressor_ape.as_ref().unwrap().tensor_type,
                    layer_attn.attn_compressor_norm.as_ref().unwrap().abs_offset,
                    layer_attn
                        .attn_compressor_norm
                        .as_ref()
                        .unwrap()
                        .tensor_type,
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                    (ratio as u32),
                    ((pos as u32) as u32),
                    comp_row,
                    (crate::kernels::ds4_constants::DS4_N_ROT as u32),
                    if compressed {
                        (crate::kernels::ds4_constants::DS4_ROPE_ORIG_CTX as u32) as u32
                    } else {
                        0
                    },
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_FAST as f32),
                    (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_SLOW as f32),
                    (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
                ) != 0;
            }
            if ok && emit {
                let mut comp_row_view: *mut crate::ffi::ds4_gpu_tensor = crate::ffi::ds4_gpu_tensor_view(
                    self.batch_layer_attn_comp_cache[il],
                    (comp_row as u64) * (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u64) * (std::mem::size_of::<f32>() as u64),
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u64) * (std::mem::size_of::<f32>() as u64),
                );
                if comp_row_view.is_null() {
                    ok = false;
                } else {
                    ok = crate::ffi::ds4_gpu_dsv4_fp8_kv_quantize_tensor(
                        comp_row_view,
                        1,
                        (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                        (crate::kernels::ds4_constants::DS4_N_ROT as u32),
                    ) != 0;

                    crate::ffi::ds4_gpu_tensor_free(comp_row_view);
                }
            }
            if ok && emit {
                self.batch_layer_n_comp[il] += 1;
            }

            if ok && ratio == 4 {
                let index_width = coff * (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u32);
                if layer_attn.indexer_compressor_kv.is_none()
                    || layer_attn.indexer_compressor_gate.is_none()
                    || layer_attn.indexer_compressor_ape.is_none()
                    || layer_attn.indexer_compressor_norm.is_none()
                    || layer_attn
                        .indexer_compressor_kv
                        .as_ref()
                        .unwrap()
                        .tensor_type
                        != 1
                    || layer_attn
                        .indexer_compressor_gate
                        .as_ref()
                        .unwrap()
                        .tensor_type
                        != 1
                    || layer_attn.indexer_compressor_kv.as_ref().unwrap().dims[0]
                        != (crate::kernels::ds4_constants::DS4_N_EMBD as u64)
                    || layer_attn.indexer_compressor_gate.as_ref().unwrap().dims[0]
                        != (crate::kernels::ds4_constants::DS4_N_EMBD as u64)
                    || layer_attn.indexer_compressor_kv.as_ref().unwrap().dims[1] != index_width as u64 as u64 as u64 as u64
                    || layer_attn.indexer_compressor_gate.as_ref().unwrap().dims[1] != index_width as u64 as u64 as u64 as u64
                {
                    ok = false;
                }
                if ok && emit && self.batch_layer_n_index_comp[il] >= self.batch_layer_comp_cap[il] {
                    ok = false;
                }
                if (ok && !false) {
                    ok = crate::ffi::ds4_gpu_matmul_f16_pair_tensor(
                        self.batch_comp_kv_cur,
                        self.batch_comp_sc_cur,
                        model.model_map_ptr(),
                        model.file_size,
                        layer_attn
                            .indexer_compressor_kv
                            .as_ref()
                            .unwrap()
                            .abs_offset,
                        layer_attn
                            .indexer_compressor_gate
                            .as_ref()
                            .unwrap()
                            .abs_offset,
                        (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                        index_width as u64,
                        self.batch_attn_norm,
                        1,
                    ) != 0;
                } else {
                    if ok {
                    ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                        self.batch_comp_kv_cur,
                        model.model_map_ptr(),
                        model.file_size,
                        layer_attn.indexer_compressor_kv.as_ref().unwrap().abs_offset,
                        crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                        index_width as u64,
                        self.batch_attn_norm,
                        1,
                    ) != 0;
                }
                if ok {
                    ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                        self.batch_comp_sc_cur,
                        model.model_map_ptr(),
                        model.file_size,
                        layer_attn.indexer_compressor_gate.as_ref().unwrap().abs_offset,
                        crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                        index_width as u64,
                        self.batch_attn_norm,
                        1,
                    ) != 0;
                }
                }
                let index_row = self.batch_layer_n_index_comp[il];
                if ok {
                    ok = crate::ffi::ds4_gpu_compressor_update_tensor(
                        self.batch_comp_kv_cur,
                        self.batch_comp_sc_cur,
                        self.batch_layer_index_state_kv[il],
                        self.batch_layer_index_state_score[il],
                        self.batch_layer_index_comp_cache[il],
                        model.model_map_ptr(),
                        model.file_size,
                        layer_attn
                            .indexer_compressor_ape
                            .as_ref()
                            .unwrap()
                            .abs_offset,
                        layer_attn
                            .indexer_compressor_ape
                            .as_ref()
                            .unwrap()
                            .tensor_type,
                        layer_attn
                            .indexer_compressor_norm
                            .as_ref()
                            .unwrap()
                            .abs_offset,
                        layer_attn
                            .indexer_compressor_norm
                            .as_ref()
                            .unwrap()
                            .tensor_type,
                        (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u32),
                        (ratio as u32),
                        ((pos as u32) as u32),
                        index_row,
                        (crate::kernels::ds4_constants::DS4_N_ROT as u32),
                        if compressed {
                            (crate::kernels::ds4_constants::DS4_ROPE_ORIG_CTX as u32) as u32
                        } else {
                            0
                        },
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_FAST as f32),
                        (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_SLOW as f32),
                        (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
                    ) != 0;
                }
                if ok && emit {
                    let mut index_row_view = crate::ffi::ds4_gpu_tensor_view(
                        self.batch_layer_index_comp_cache[il],
                        (index_row as u64)
                            * (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u64)
                            * (std::mem::size_of::<f32>() as u64),
                        (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u64)
                            * (std::mem::size_of::<f32>() as u64),
                    );
                    if index_row_view.is_null() {
                        ok = false;
                    } else {
                        ok = crate::ffi::ds4_gpu_dsv4_indexer_qat_tensor(
                            index_row_view,
                            1,
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u32),
                        ) != 0;
                        crate::ffi::ds4_gpu_tensor_free(index_row_view);
                    }
                }
                if ok && emit {
                    self.batch_layer_n_index_comp[il] += 1;
                }
                let decode_top_k = 64;
                if ok && self.batch_layer_n_comp[il] > decode_top_k {
                    let indexer_q_dim = (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD as u64)
                        * (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u64);
                    if layer_attn.indexer_attn_q_b.is_none()
                        || layer_attn.indexer_attn_q_b.as_ref().unwrap().tensor_type != 1
                        || layer_attn.indexer_attn_q_b.as_ref().unwrap().dims[0] != q_rank
                        || layer_attn.indexer_attn_q_b.as_ref().unwrap().dims[1] != indexer_q_dim
                    {
                        ok = false;
                    }
                    if (ok
                        && (layer_attn.indexer_proj.is_none()
                            || layer_attn.indexer_proj.as_ref().unwrap().tensor_type
                                != 1
                            || layer_attn.indexer_proj.as_ref().unwrap().dims[0]
                                != (crate::kernels::ds4_constants::DS4_N_EMBD as u64)
                            || layer_attn.indexer_proj.as_ref().unwrap().dims[1]
                                != (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD as u64)))
                    {
                        ok = false;
                    }
                    if ok {
                        ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                            self.batch_indexer_q,
                            model.model_map_ptr(),
                            model.file_size,
                            layer_attn.indexer_attn_q_b.as_ref().unwrap().abs_offset,
                            q_rank,
                            indexer_q_dim,
                            self.qr_norm,
                            1,
                        ) != 0;
                    }
                    if ok {
                        ok = crate::ffi::ds4_gpu_rope_tail_tensor(
                            self.batch_indexer_q,
                            1,
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD as u32),
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u32),
                            (crate::kernels::ds4_constants::DS4_N_ROT as u32),
                            ((pos as u32) as u32),
                            if compressed {
                                (crate::kernels::ds4_constants::DS4_ROPE_ORIG_CTX as u32) as u32
                            } else {
                                0
                            },
                            false,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_FAST as f32),
                            (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_SLOW as f32),
                        ) != 0;
                    }
                    if ok {
                        ok = crate::ffi::ds4_gpu_dsv4_indexer_qat_tensor(
                            self.batch_indexer_q,
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD as u32),
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u32),
                        ) != 0;
                    }
                    if ok {
                        ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                            self.batch_indexer_weights,
                            model.model_map_ptr(),
                            model.file_size,
                            layer_attn.indexer_proj.as_ref().unwrap().abs_offset,
                            crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                            crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD as u64,
                            self.batch_attn_norm,
                            1,
                        ) != 0;
                    }
                    let index_scale = 1.0
                        / (
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u32)
                                * (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD as u32)
                        ) as f32;
                    if ok && decode_index_stage_profile {
                        ok = true;
                    }
                    if ok {
                        ok = crate::ffi::ds4_gpu_indexer_score_one_tensor(
                            self.batch_indexer_scores,
                            self.batch_indexer_q,
                            self.batch_indexer_weights,
                            self.batch_layer_index_comp_cache[il],
                            self.batch_layer_n_index_comp[il],
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD as u32),
                            (crate::kernels::ds4_constants::DS4_N_INDEXER_HEAD_DIM as u32),
                            index_scale,
                        ) != 0;
                    }
                    if ok && decode_index_stage_profile {
                        ok = true;
                    }
                    if ok {
                        ok = crate::ffi::ds4_gpu_indexer_topk_tensor(
                            self.batch_comp_selected,
                            self.batch_indexer_scores,
                            self.batch_layer_n_index_comp[il],
                            1,
                            decode_top_k,
                        ) != 0;
                    }
                    if ok && decode_index_stage_profile {
                        ok = true;
                    }
                    /* Decode used to materialize a dense compressed-row mask and
                     * call the generic gathered FlashAttention wrapper below.
                     * That wrapper scans every compressed row and rejects long
                     * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                     * attention is sparse after indexer top-k, so use the private
                     * indexed attention kernel instead: it scans only SWA raw rows
                     * plus the selected compressed rows, matching prefill and
                     * avoiding the long-context decode failure. */
                    if ok {
                        comp_selected = self.batch_comp_selected;
                        n_selected = std::cmp::min(decode_top_k, self.batch_layer_n_index_comp[il]);
                    }
                }
            }

            n_comp = self.batch_layer_n_comp[il];
            comp_cache = self.batch_layer_attn_comp_cache[il];
        }

        if ok {
            let raw_start = (pos as u32) as u32;
            if n_comp != 0 && comp_selected != std::ptr::null_mut() && n_selected != 0 {
                ok = crate::ffi::ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    self.batch_heads,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_sinks.abs_offset,
                    self.batch_q,
                    raw_cache,
                    comp_cache,
                    comp_selected,
                    1,
                    ((pos as u32) as u32),
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    self.batch_raw_window,
                    0,
                    (crate::kernels::ds4_constants::DS4_N_HEAD as u32),
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                ) != 0;
                if ok && decode_index_stage_profile {
                    ok = true;
                }
            } else {
                ok = crate::ffi::ds4_gpu_attention_decode_heads_tensor(
                    self.batch_heads,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_sinks.abs_offset,
                    self.batch_q,
                    raw_cache,
                    n_raw,
                    raw_cap,
                    raw_start,
                    if n_comp != 0 {
                        comp_cache
                    } else {
                        std::ptr::null_mut()
                    },
                    n_comp,
                    std::ptr::null_mut(),
                    0,
                    (crate::kernels::ds4_constants::DS4_N_HEAD as u32),
                    (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                ) != 0;
            }
        }

        if ok {
            ok = crate::ffi::ds4_gpu_rope_tail_tensor(
                self.batch_heads,
                1,
                (crate::kernels::ds4_constants::DS4_N_HEAD as u32),
                (crate::kernels::ds4_constants::DS4_N_HEAD_DIM as u32),
                (crate::kernels::ds4_constants::DS4_N_ROT as u32),
                ((pos as u32) as u32),
                if compressed {
                    (crate::kernels::ds4_constants::DS4_ROPE_ORIG_CTX as u32) as u32
                } else {
                    0
                },
                true,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_FAST as f32),
                (crate::kernels::ds4_constants::DS4_ROPE_YARN_BETA_SLOW as f32),
            ) != 0;
        }

        let fuse_attn_out_hc = !false && true;
        if ok && fuse_attn_out_hc {
            ok = crate::ffi::ds4_gpu_attention_output_low_q8_tensor(
                self.batch_attn_low,
                model.model_map_ptr(),
                model.file_size,
                layer_attn.attn_output_a.abs_offset,
                (group_dim as u64),
                (rank as u64),
                (n_groups as u32),
                self.batch_heads,
            ) != 0;
            if ok {
                ok = crate::ffi::ds4_gpu_matmul_q8_0_hc_expand_tensor(
                    self.batch_after_attn_hc,
                    self.batch_attn_out,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_attn.attn_output_b.abs_offset,
                    ((n_groups as u64) as u64) * (rank as u64),
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                    self.batch_attn_low,
                    self.batch_cur_hc,
                    self.batch_hc_split,
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                    (crate::kernels::ds4_constants::DS4_N_HC as u32),
                ) != 0;
            }
        } else if ok {
            ok = crate::ffi::ds4_gpu_attention_output_q8_batch_tensor(
                self.batch_attn_out,
                self.batch_attn_low,
                self.batch_group_tmp,
                self.batch_low_tmp,
                model.model_map_ptr(),
                model.file_size,
                layer_attn.attn_output_a.abs_offset,
                layer_attn.attn_output_b.abs_offset,
                (group_dim as u64),
                (rank as u64),
                (n_groups as u32),
                (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                self.batch_heads,
                1,
            ) != 0;
        }

        if (ok && false) {
            ok = true;
        }
        if ok && !fuse_attn_out_hc {
            ok = crate::ffi::ds4_gpu_hc_expand_tensor(
                self.batch_after_attn_hc,
                self.batch_attn_out,
                self.batch_cur_hc,
                self.batch_hc_post,
                self.batch_hc_comb,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_N_HC as u32),
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_rms_norm_plain_tensor(
                self.batch_flat_hc,
                self.batch_after_attn_hc,
                (hc_dim as u32),
                (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
            ) != 0;
        }
        if ok {
            ok = Self::metal_graph_matmul_plain_tensor(
                self.batch_hc_mix,
                model,
                layer_ffn.hc_ffn_fn.as_ref().unwrap(),
                hc_dim as u64,
                mix_hc as u64,
                self.batch_flat_hc,
                1,
            );
        }
        if ok && fuse_hc_norm {
            ok = crate::ffi::ds4_gpu_hc_split_weighted_sum_norm_tensor(
                self.batch_ffn_cur,
                self.ffn_norm,
                self.batch_hc_split,
                self.batch_hc_mix,
                self.batch_after_attn_hc,
                model.model_map_ptr(),
                model.file_size,
                layer_ffn.hc_ffn_scale.as_ref().unwrap().abs_offset,
                layer_ffn.hc_ffn_base.as_ref().unwrap().abs_offset,
                layer_ffn.ffn_norm.abs_offset,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_N_HC as u32),
                (crate::kernels::ds4_constants::DS4_N_HC_SINKHORN_ITER as u32),
                (crate::kernels::ds4_constants::DS4_HC_EPS as f32),
                (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
            ) != 0;
        } else if ok {
            ok = true; /* metal_graph_decode_hc_pre
                self.batch_ffn_cur,
                self.batch_hc_split,
                self.batch_hc_mix,
                self.batch_after_attn_hc,
                model,
                layer_ffn.hc_ffn_scale.as_ref().unwrap().abs_offset,
                layer_ffn.hc_ffn_base.as_ref().unwrap().abs_offset,
            */;
        }

        if ok && !fuse_hc_norm {
            ok = crate::ffi::ds4_gpu_rms_norm_weight_tensor(
                self.ffn_norm,
                self.batch_ffn_cur,
                model.model_map_ptr(),
                model.file_size,
                layer_ffn.ffn_norm.abs_offset,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_RMS_EPS as f32),
            ) != 0;
        }

        let gate_row_bytes = 0;
        let gate_expert_bytes = expert_mid_dim * gate_row_bytes;
        let down_row_bytes = 0;
        let down_expert_bytes = routed_out_dim * down_row_bytes;
        if ok {
            ok = Self::metal_graph_matmul_plain_tensor(
                self.router_logits,
                model,
                &layer_ffn.ffn_gate_inp,
                crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                crate::kernels::ds4_constants::DS4_N_EXPERT as u64,
                self.ffn_norm,
                1,
            );
        }
        if ok {
            ok = crate::ffi::ds4_gpu_router_select_tensor(
                self.router_selected,
                self.router_weights,
                self.router_probs,
                model.model_map_ptr(),
                model.file_size,
                if weights.blocks[il]
                    .ffn
                    .as_ref()
                    .unwrap()
                    .ffn_exp_probs_b
                    .is_some()
                {
                    layer_ffn.ffn_exp_probs_b.as_ref().unwrap().abs_offset
                } else {
                    0
                },
                if weights.blocks[il]
                    .ffn
                    .as_ref()
                    .unwrap()
                    .ffn_gate_tid2eid
                    .is_some()
                {
                    layer_ffn.ffn_gate_tid2eid.as_ref().unwrap().abs_offset
                } else {
                    0
                },
                if weights.blocks[il]
                    .ffn
                    .as_ref()
                    .unwrap()
                    .ffn_gate_tid2eid
                    .is_some()
                {
                    layer_ffn.ffn_gate_tid2eid.as_ref().unwrap().dims[1] as u32
                } else {
                    0
                },
                (token as u32),
                0,
                0,
                weights.blocks[il]
                    .ffn
                    .as_ref()
                    .unwrap()
                    .ffn_exp_probs_b
                    .is_some(),
                weights.blocks[il].ffn.as_ref().unwrap().ffn_gate_tid2eid.is_some(),
                self.router_logits,
            ) != 0;
        }

        if ok {
            ok = crate::ffi::ds4_gpu_routed_moe_one_tensor(
                self.routed_out,
                self.routed_gate,
                self.routed_up,
                self.routed_mid,
                self.routed_down,
                model.model_map_ptr(),
                model.file_size,
                layer_ffn.ffn_gate_exps.abs_offset,
                layer_ffn.ffn_up_exps.abs_offset,
                layer_ffn.ffn_down_exps.abs_offset,
                layer_ffn.ffn_gate_exps.tensor_type,
                layer_ffn.ffn_down_exps.tensor_type,
                gate_expert_bytes,
                gate_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                (expert_in_dim as u32),
                (down_in_dim as u32),
                (routed_out_dim as u32),
                self.router_selected,
                self.router_weights,
                (crate::kernels::ds4_constants::DS4_N_EXPERT_USED as u32),
                (crate::kernels::ds4_constants::DS4_SWIGLU_CLAMP_EXP as f32),
                self.ffn_norm,
            ) != 0;
        }

        let fuse_shared_gate_up = !self.quality && true;
        if ok && fuse_shared_gate_up {
            ok = crate::ffi::ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                self.shared_gate,
                self.shared_up,
                self.shared_mid,
                model.model_map_ptr(),
                model.file_size,
                layer_ffn.ffn_gate_shexp.abs_offset,
                layer_ffn.ffn_up_shexp.abs_offset,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                (shared_dim as u64),
                self.ffn_norm,
                (crate::kernels::ds4_constants::DS4_SWIGLU_CLAMP_EXP as f32),
            ) != 0;
        } else {
            if ok {
                ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                    self.shared_gate,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_ffn.ffn_gate_shexp.abs_offset,
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                    (shared_dim as u64),
                    self.ffn_norm,
                    1,
                ) != 0;
            }
            if ok {
                ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                    self.shared_up,
                    model.model_map_ptr(),
                    model.file_size,
                    layer_ffn.ffn_up_shexp.abs_offset,
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                    (shared_dim as u64),
                    self.ffn_norm,
                    1,
                ) != 0;
            }
            if ok {
                ok = crate::ffi::ds4_gpu_swiglu_tensor(
                    self.shared_mid,
                    self.shared_gate,
                    self.shared_up,
                    shared_dim,
                    (crate::kernels::ds4_constants::DS4_SWIGLU_CLAMP_EXP as f32),
                    1.0,
                ) != 0;
            }
        }
        let keep_ffn_out = false;
        let fuse_shared_down_hc = !keep_ffn_out && true;
        if ok && fuse_shared_down_hc {
            ok = crate::ffi::ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                self.after_ffn_hc,
                self.shared_out,
                model.model_map_ptr(),
                model.file_size,
                layer_ffn.ffn_down_shexp.abs_offset,
                (shared_dim as u64),
                (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                self.shared_mid,
                self.routed_out,
                self.batch_after_attn_hc,
                self.batch_hc_split,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_N_HC as u32),
            ) != 0;
        } else if ok {
            ok = crate::ffi::ds4_gpu_matmul_q8_0_tensor(
                self.shared_out,
                model.model_map_ptr(),
                model.file_size,
                layer_ffn.ffn_down_shexp.abs_offset,
                (shared_dim as u64),
                (crate::kernels::ds4_constants::DS4_N_EMBD as u64),
                self.shared_mid,
                1,
            ) != 0;
        }

        if ok && keep_ffn_out {
            ok = true
                && crate::ffi::ds4_gpu_add_tensor(
                    self.ffn_out,
                    self.shared_out,
                    self.routed_out,
                    (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                ) != 0;
        }

        if (ok && false) {
            
        }
        if (ok && false) {
            ok = crate::ffi::ds4_gpu_hc_expand_tensor(
                self.after_ffn_hc,
                self.ffn_out,
                self.batch_after_attn_hc,
                self.batch_hc_post,
                self.batch_hc_comb,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_N_HC as u32),
            ) != 0;
        } else if ok && !fuse_shared_down_hc {
            ok = crate::ffi::ds4_gpu_hc_expand_add_split_tensor(
                self.after_ffn_hc,
                self.routed_out,
                self.shared_out,
                self.batch_after_attn_hc,
                self.batch_hc_split,
                (crate::kernels::ds4_constants::DS4_N_EMBD as u32),
                (crate::kernels::ds4_constants::DS4_N_HC as u32),
            ) != 0;
        }

        return ok;
    }
}
