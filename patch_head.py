with open("src/kernels/metal/head.rs", "r") as f:
    content = f.read()

content = content.replace(
"""        if ok {
            ok = crate::ffi::ds4_gpu_matmul_f16_tensor(
                self.logits,
                model.model_map_ptr(),
                model.file_size,
                weights.output.abs_offset,
                vocab_dim,
                crate::kernels::ds4_constants::DS4_N_EMBD as u64,
                self.output_norm,
                1,
            ) != 0;
        }""",
"""        if ok {
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
        }"""
)

with open("src/kernels/metal/head.rs", "w") as f:
    f.write(content)
