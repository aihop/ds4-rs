#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ds4ModelShape {
    pub n_head: usize,
    pub n_head_dim: usize,
    pub n_rot: usize,
    pub n_out_group: usize,
    pub n_lora_q: usize,
    pub n_lora_o: usize,
}

impl Ds4ModelShape {
    pub const fn head_width(self) -> usize {
        self.n_head * self.n_head_dim
    }

    pub const fn rope_tail_width(self) -> usize {
        self.n_head_dim - self.n_rot
    }
}

pub const DS4_V4_FLASH_SHAPE: Ds4ModelShape = Ds4ModelShape {
    n_head: 64,
    n_head_dim: 512,
    n_rot: 64,
    n_out_group: 8,
    n_lora_q: 1024,
    n_lora_o: 1024,
};

pub const DS4_ROPE_FREQ_BASE: f32 = 10_000.0;
