use crate::gguf::GgufModel;
use crate::kernels::matmul::f16_to_f32;
use crate::weights::BoundTensor;

pub(crate) const QK_K: usize = 256;
const BLOCK_Q2_K_SIZE: usize = 84;
const BLOCK_IQ2_XXS_SIZE: usize = 66;

#[derive(Clone, Copy, Debug)]
pub(crate) struct QuantizedTensorAccessor<'a> {
    data: &'a [u8],
    row_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

impl Default for BlockQ8K {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

#[allow(dead_code)]
pub(crate) fn quantize_row_q8_k(x: &[f32]) -> Option<Vec<BlockQ8K>> {
    if x.is_empty() || x.len() % QK_K != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(x.len() / QK_K);
    for chunk in x.chunks_exact(QK_K) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &value in chunk {
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                max = value;
            }
        }
        if amax == 0.0 {
            out.push(BlockQ8K {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; QK_K / 16],
            });
            continue;
        }

        let iscale = -127.0 / max;
        let mut block = BlockQ8K {
            d: 1.0 / iscale,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        for (idx, &value) in chunk.iter().enumerate() {
            let quant = (iscale * value).round().clamp(-128.0, 127.0) as i32;
            block.qs[idx] = quant as i8;
        }
        for (group_idx, group) in block.qs.chunks_exact(16).enumerate() {
            let sum: i32 = group.iter().map(|&v| i32::from(v)).sum();
            block.bsums[group_idx] = sum as i16;
        }
        out.push(block);
    }
    Some(out)
}

pub(crate) fn quantize_row_q8_k_into(dst: &mut [BlockQ8K], x: &[f32]) -> Option<()> {
    if x.is_empty() || x.len() % QK_K != 0 || dst.len() != x.len() / QK_K {
        return None;
    }
    for (block, chunk) in dst.iter_mut().zip(x.chunks_exact(QK_K)) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &value in chunk {
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                max = value;
            }
        }
        if amax == 0.0 {
            *block = BlockQ8K::default();
            continue;
        }

        let iscale = -127.0 / max;
        let mut out = BlockQ8K {
            d: 1.0 / iscale,
            ..BlockQ8K::default()
        };
        for (idx, &value) in chunk.iter().enumerate() {
            let quant = (iscale * value).round().clamp(-128.0, 127.0) as i32;
            out.qs[idx] = quant as i8;
        }
        for (group_idx, group) in out.qs.chunks_exact(16).enumerate() {
            let sum: i32 = group.iter().map(|&v| i32::from(v)).sum();
            out.bsums[group_idx] = sum as i16;
        }
        *block = out;
    }
    Some(())
}

#[allow(dead_code)]
pub(crate) fn dot_q2_k_row_prequant(
    model: &GgufModel,
    tensor: &BoundTensor,
    row_idx: usize,
    input: &[BlockQ8K],
    in_dim: usize,
) -> Option<f32> {
    let accessor = quantized_tensor_accessor(model, tensor, BLOCK_Q2_K_SIZE, in_dim)?;
    dot_q2_k_row_from_accessor(&accessor, row_idx, input)
}

pub(crate) fn dot_q2_k_row_from_accessor(
    accessor: &QuantizedTensorAccessor<'_>,
    row_idx: usize,
    input: &[BlockQ8K],
) -> Option<f32> {
    let row = quantized_row(accessor, row_idx)?;
    let mut sum = 0.0f32;
    for (block_idx, x) in input.iter().enumerate() {
        let block = row.get(block_idx * BLOCK_Q2_K_SIZE..(block_idx + 1) * BLOCK_Q2_K_SIZE)?;
        let scales = &block[..16];
        let q2 = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let mut summs = 0i32;
        for j in 0..16 {
            summs += i32::from(x.bsums[j]) * i32::from(scales[j] >> 4);
        }

        let mut isum = 0i32;
        let mut is = 0usize;
        let mut q8_off = 0usize;
        for k in 0..(QK_K / 128) {
            let q2_chunk = &q2[k * 32..(k + 1) * 32];
            let mut shift = 0u8;
            for j in 0..4 {
                let scale_lo = i32::from(scales[is] & 0x0f);
                is += 1;
                isum += scale_lo * dot_q2_16(&q2_chunk[..16], &x.qs[q8_off..q8_off + 16], shift);

                let scale_hi = i32::from(scales[is] & 0x0f);
                is += 1;
                isum += scale_hi * dot_q2_16(&q2_chunk[16..32], &x.qs[q8_off + 16..q8_off + 32], shift);

                shift += 2;
                q8_off += 32;
                let _ = j;
            }
        }
        sum += x.d * d * isum as f32 - x.d * dmin * summs as f32;
    }
    Some(sum)
}

#[allow(dead_code)]
pub(crate) fn dot_iq2_xxs_row_prequant(
    model: &GgufModel,
    tensor: &BoundTensor,
    row_idx: usize,
    input: &[BlockQ8K],
    in_dim: usize,
) -> Option<f32> {
    let row = quant_tensor_row(model, tensor, row_idx, BLOCK_IQ2_XXS_SIZE, in_dim)?;
    let mut sumf = 0.0f32;
    for (block_idx, x) in input.iter().enumerate() {
        let block = row.get(block_idx * BLOCK_IQ2_XXS_SIZE..(block_idx + 1) * BLOCK_IQ2_XXS_SIZE)?;
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]])) * x.d;
        let mut q8_off = 0usize;
        let mut bsum = 0i32;
        for ib32 in 0..(QK_K / 32) {
            let base = 2 + ib32 * 8;
            let aux0 = u32::from_le_bytes(block[base..base + 4].try_into().ok()?);
            let aux1 = u32::from_le_bytes(block[base + 4..base + 8].try_into().ok()?);
            let aux8 = aux0.to_le_bytes();
            let ls = 2 * ((aux1 >> 28) as i32) + 1;
            let mut sumi = 0i32;
            for l in [0usize, 2usize] {
                let sign_idx0 = ((aux1 >> (7 * l)) & 127) as u8;
                let sign_idx1 = ((aux1 >> (7 * (l + 1))) & 127) as u8;
                sumi += dot_iq2_pair_16(
                    aux8[l],
                    sign_idx0,
                    aux8[l + 1],
                    sign_idx1,
                    &x.qs[q8_off..q8_off + 16],
                );
                q8_off += 16;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    Some(0.125 * sumf)
}

#[allow(dead_code)]
pub(crate) fn dot_iq2_xxs_pair_rows_prequant(
    model: &GgufModel,
    tensor0: &BoundTensor,
    row0_idx: usize,
    tensor1: &BoundTensor,
    row1_idx: usize,
    input: &[BlockQ8K],
    in_dim: usize,
) -> Option<(f32, f32)> {
    let accessor0 = quantized_tensor_accessor(model, tensor0, BLOCK_IQ2_XXS_SIZE, in_dim)?;
    let accessor1 = quantized_tensor_accessor(model, tensor1, BLOCK_IQ2_XXS_SIZE, in_dim)?;
    dot_iq2_xxs_pair_rows_from_accessors(&accessor0, row0_idx, &accessor1, row1_idx, input)
}

pub(crate) fn dot_iq2_xxs_pair_rows_from_accessors(
    accessor0: &QuantizedTensorAccessor<'_>,
    row0_idx: usize,
    accessor1: &QuantizedTensorAccessor<'_>,
    row1_idx: usize,
    input: &[BlockQ8K],
) -> Option<(f32, f32)> {
    let row0 = quantized_row(accessor0, row0_idx)?;
    let row1 = quantized_row(accessor1, row1_idx)?;
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for (block_idx, x) in input.iter().enumerate() {
        let block0 = row0.get(block_idx * BLOCK_IQ2_XXS_SIZE..(block_idx + 1) * BLOCK_IQ2_XXS_SIZE)?;
        let block1 = row1.get(block_idx * BLOCK_IQ2_XXS_SIZE..(block_idx + 1) * BLOCK_IQ2_XXS_SIZE)?;
        let d0 = f16_to_f32(u16::from_le_bytes([block0[0], block0[1]])) * x.d;
        let d1 = f16_to_f32(u16::from_le_bytes([block1[0], block1[1]])) * x.d;
        let mut q8_off = 0usize;
        let mut bsum0 = 0i32;
        let mut bsum1 = 0i32;
        for ib32 in 0..(QK_K / 32) {
            let base = 2 + ib32 * 8;
            let aux0_0 = u32::from_le_bytes(block0[base..base + 4].try_into().ok()?);
            let aux0_1 = u32::from_le_bytes(block0[base + 4..base + 8].try_into().ok()?);
            let aux1_0 = u32::from_le_bytes(block1[base..base + 4].try_into().ok()?);
            let aux1_1 = u32::from_le_bytes(block1[base + 4..base + 8].try_into().ok()?);
            let grid0 = aux0_0.to_le_bytes();
            let grid1 = aux1_0.to_le_bytes();
            let ls0 = 2 * ((aux0_1 >> 28) as i32) + 1;
            let ls1 = 2 * ((aux1_1 >> 28) as i32) + 1;
            let mut sumi0 = 0i32;
            let mut sumi1 = 0i32;
            for l in [0usize, 2usize] {
                let pair = &x.qs[q8_off..q8_off + 16];
                sumi0 += dot_iq2_pair_16(
                    grid0[l],
                    ((aux0_1 >> (7 * l)) & 127) as u8,
                    grid0[l + 1],
                    ((aux0_1 >> (7 * (l + 1))) & 127) as u8,
                    pair,
                );
                sumi1 += dot_iq2_pair_16(
                    grid1[l],
                    ((aux1_1 >> (7 * l)) & 127) as u8,
                    grid1[l + 1],
                    ((aux1_1 >> (7 * (l + 1))) & 127) as u8,
                    pair,
                );
                q8_off += 16;
            }
            bsum0 += sumi0 * ls0;
            bsum1 += sumi1 * ls1;
        }
        sum0 += d0 * bsum0 as f32;
        sum1 += d1 * bsum1 as f32;
    }
    Some((0.125 * sum0, 0.125 * sum1))
}

fn quant_tensor_row<'a>(
    model: &'a GgufModel,
    tensor: &BoundTensor,
    row_idx: usize,
    block_size: usize,
    in_dim: usize,
) -> Option<&'a [u8]> {
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    if in_dim == 0 || in_dim % QK_K != 0 {
        return None;
    }
    let row_bytes = in_dim.checked_div(QK_K)?.checked_mul(block_size)?;
    let start = row_idx.checked_mul(row_bytes)?;
    let end = start.checked_add(row_bytes)?;
    data.get(start..end)
}

pub(crate) fn quantized_tensor_accessor<'a>(
    model: &'a GgufModel,
    tensor: &BoundTensor,
    block_size: usize,
    in_dim: usize,
) -> Option<QuantizedTensorAccessor<'a>> {
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    if in_dim == 0 || in_dim % QK_K != 0 {
        return None;
    }
    let row_bytes = in_dim.checked_div(QK_K)?.checked_mul(block_size)?;
    Some(QuantizedTensorAccessor { data, row_bytes })
}

fn quantized_row<'a>(
    accessor: &'a QuantizedTensorAccessor<'_>,
    row_idx: usize,
) -> Option<&'a [u8]> {
    let start = row_idx.checked_mul(accessor.row_bytes)?;
    let end = start.checked_add(accessor.row_bytes)?;
    accessor.data.get(start..end)
}

fn dot_q2_16(q2: &[u8], q8: &[i8], shift: u8) -> i32 {
    // SAFETY: This function relies on aarch64 NEON instructions.
    // The caller must ensure that `q2` has at least 16 bytes and `q8` has at least 16 bytes.
    unsafe {
        let q2_raw = std::arch::aarch64::vld1q_u8(q2.as_ptr());
        let q8_vec = std::arch::aarch64::vld1q_s8(q8.as_ptr());
        let shift_vec = std::arch::aarch64::vdupq_n_s8(-(shift as i8));
        let q2_shifted = std::arch::aarch64::vshlq_u8(q2_raw, shift_vec);
        let m3 = std::arch::aarch64::vdupq_n_u8(3);
        let q2_masked = std::arch::aarch64::vandq_u8(q2_shifted, m3);
        let q2_vec = std::arch::aarch64::vreinterpretq_s8_u8(q2_masked);
        
        let mut sum_vec = std::arch::aarch64::vdupq_n_s32(0);
        std::arch::asm!(
            "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
            inout(vreg) sum_vec,
            in(vreg) q2_vec,
            in(vreg) q8_vec,
            options(pure, nomem, nostack)
        );
        std::arch::aarch64::vaddvq_s32(sum_vec)
    }
}

const fn build_iq2_xxs_decoded() -> [[[i8; 8]; 128]; 256] {
    let mut table = [[[0i8; 8]; 128]; 256];
    let mut g = 0;
    while g < 256 {
        let grid = IQ2_XXS_GRID[g].to_le_bytes();
        let mut s = 0;
        while s < 128 {
            let mut e = 0;
            while e < 8 {
                let val = grid[e] as i8;
                let sign_mask = KSIGNS_IQ2XS[s];
                let bit = KMASK_IQ2XS[e];
                if (sign_mask & bit) != 0 {
                    table[g][s][e] = -val;
                } else {
                    table[g][s][e] = val;
                }
                e += 1;
            }
            s += 1;
        }
        g += 1;
    }
    table
}

const IQ2_XXS_DECODED: [[[i8; 8]; 128]; 256] = build_iq2_xxs_decoded();

fn dot_iq2_pair_16(grid0_idx: u8, sign_idx0: u8, grid1_idx: u8, sign_idx1: u8, q8: &[i8]) -> i32 {
    // SAFETY: This function relies on aarch64 NEON instructions.
    // The indices are safely bounded by the table size, and `q8` must have at least 16 bytes.
    unsafe {
        let p0 = IQ2_XXS_DECODED[grid0_idx as usize][sign_idx0 as usize].as_ptr();
        let p1 = IQ2_XXS_DECODED[grid1_idx as usize][sign_idx1 as usize].as_ptr();
        
        let v0 = std::arch::aarch64::vld1_s8(p0);
        let v1 = std::arch::aarch64::vld1_s8(p1);
        let q2_vec = std::arch::aarch64::vcombine_s8(v0, v1);
        let q8_vec = std::arch::aarch64::vld1q_s8(q8.as_ptr());
        
        let mut sum_vec = std::arch::aarch64::vdupq_n_s32(0);
        std::arch::asm!(
            "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
            inout(vreg) sum_vec,
            in(vreg) q2_vec,
            in(vreg) q8_vec,
            options(pure, nomem, nostack)
        );
        std::arch::aarch64::vaddvq_s32(sum_vec)
    }
}

fn iq2_signed_grid_value(grid_idx: u8, sign_idx: u8, elem: usize) -> i8 {
    let grid = IQ2_XXS_GRID[grid_idx as usize].to_le_bytes();
    let value = grid[elem] as i8;
    if KSIGNS_IQ2XS[sign_idx as usize] & KMASK_IQ2XS[elem] != 0 {
        -value
    } else {
        value
    }
}

const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20,
    149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40,
    169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60,
    189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80,
    209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228,
    101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246,
    119, 120, 249, 250, 123, 252, 125, 126, 255,
];

const IQ2_XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];
