import re

with open("src/kernels/metal/decode.rs", "r") as f:
    content = f.read()

# Add metal_graph_matmul_plain_tensor helper function
helper_func = """
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
"""

if "pub unsafe fn metal_graph_matmul_plain_tensor" not in content:
    content = content.replace("impl MetalGraph {", "impl MetalGraph {\n" + helper_func)

# Replace FIXMEs

content = re.sub(
    r'if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}\s*let fuse_hc_norm =',
    r'''if ok {
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
        let fuse_hc_norm =''',
    content
)

content = re.sub(
    r'if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}\s*if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}',
    r'''if ok {
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
                }''',
    content,
    count=1 # First occurrence for attn_compressor
)

content = re.sub(
    r'if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}\s*if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}',
    r'''if ok {
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
                }''',
    content,
    count=1 # Second occurrence for indexer_compressor
)

content = re.sub(
    r'if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}\s*if ok \{\s*ok = crate::ffi::ds4_gpu_rope_tail_tensor\(',
    r'''if ok {
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
                        ok = crate::ffi::ds4_gpu_rope_tail_tensor(''',
    content
)

content = re.sub(
    r'if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}\s*let index_scale =',
    r'''if ok {
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
                    let index_scale =''',
    content
)

content = re.sub(
    r'if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}\s*if ok && fuse_hc_norm \{',
    r'''if ok {
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
        if ok && fuse_hc_norm {''',
    content
)

content = re.sub(
    r'if ok \{\s*ok = true; // FIXME: ds4_gpu_matmul_f16_tensor manually patched\s*\}\s*if ok \{\s*ok = crate::ffi::ds4_gpu_router_select_tensor\(',
    r'''if ok {
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
            ok = crate::ffi::ds4_gpu_router_select_tensor(''',
    content
)

with open("src/kernels/metal/decode.rs", "w") as f:
    f.write(content)
