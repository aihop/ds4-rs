use crate::gguf::GgufModel;
use crate::kernels::matmul::{decode_tensor_1d, matvec_decoded_into, matvec_tensor};
use crate::weights::{BoundTensor, BoundWeights};

const DS4_N_HC: usize = 4;
const DS4_HC_EPS: f32 = 1.0e-6;
const DS4_N_HC_SINKHORN_ITER: usize = 20;
const DS4_HC_SPLIT_DIM: usize = 2 * DS4_N_HC + DS4_N_HC * DS4_N_HC;

pub(crate) struct HcPreResult {
    pub out: Vec<f32>,
    pub post: [f32; DS4_N_HC],
    pub comb: [f32; DS4_N_HC * DS4_N_HC],
}

#[allow(dead_code)]
pub(crate) struct HcPreBatchResult {
    pub out: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    pub n_tokens: usize,
    pub n_embd: usize,
}

pub(crate) fn supports_hc_path(weights: &BoundWeights) -> bool {
    weights.output_hc_fn.is_some()
        && weights.output_hc_scale.is_some()
        && weights.output_hc_base.is_some()
        && weights.blocks.iter().all(|block| {
            block.attention.hc_attn_fn.is_some()
                && block.attention.hc_attn_scale.is_some()
                && block.attention.hc_attn_base.is_some()
                && block.ffn.as_ref().is_none_or(|ffn| {
                    ffn.hc_ffn_fn.is_some()
                        && ffn.hc_ffn_scale.is_some()
                        && ffn.hc_ffn_base.is_some()
                })
        })
}

pub(crate) fn hc_from_plain_embedding(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len() * DS4_N_HC];
    for h in 0..DS4_N_HC {
        out[h * x.len()..(h + 1) * x.len()].copy_from_slice(x);
    }
    out
}

pub(crate) fn hc_pre_from_state(
    model: &GgufModel,
    fn_tensor: &BoundTensor,
    fn_tensor_data: Option<&[f32]>,
    scale_tensor: &BoundTensor,
    scale_tensor_data: Option<&[f32]>,
    base_tensor: &BoundTensor,
    base_tensor_data: Option<&[f32]>,
    residual_hc: &[f32],
) -> Option<HcPreResult> {
    if residual_hc.is_empty() || residual_hc.len() % DS4_N_HC != 0 {
        return None;
    }
    let n_embd = residual_hc.len() / DS4_N_HC;
    let norm_scale = rms_norm_scale_no_weight(residual_hc)?;
    let mix = match fn_tensor_data {
        Some(data) => {
            let width = usize::try_from(*fn_tensor.dims.first()?).ok()?;
            let rows = usize::try_from(*fn_tensor.dims.get(1)?).ok()?;
            let mut mix = [0.0; DS4_HC_SPLIT_DIM];
            matvec_decoded_into(&mut mix, data, rows, width, residual_hc)?;
            scale_in_place(&mut mix, norm_scale);
            mix
        }
        None => {
            let flat: Vec<f32> = residual_hc.iter().map(|x| x * norm_scale).collect();
            let owned = matvec_tensor(model, fn_tensor, &flat)?;
            let mut mix = [0.0; DS4_HC_SPLIT_DIM];
            if owned.len() != mix.len() {
                return None;
            }
            mix.copy_from_slice(&owned);
            mix
        }
    };
    let scale_owned = if scale_tensor_data.is_none() {
        decode_tensor_1d(model, scale_tensor)
    } else {
        None
    };
    let scale = scale_tensor_data.or(scale_owned.as_deref())?;
    let base_owned = if base_tensor_data.is_none() {
        decode_tensor_1d(model, base_tensor)
    } else {
        None
    };
    let base = base_tensor_data.or(base_owned.as_deref())?;
    let split = hc_split_sinkhorn_one(
        &mix,
        &scale,
        &base,
        DS4_N_HC,
        DS4_N_HC_SINKHORN_ITER,
        DS4_HC_EPS,
    )?;
    let out = hc_weighted_sum_one(residual_hc, &split[..DS4_N_HC], n_embd, DS4_N_HC);
    let mut post = [0.0; DS4_N_HC];
    post.copy_from_slice(&split[DS4_N_HC..2 * DS4_N_HC]);
    let mut comb = [0.0; DS4_N_HC * DS4_N_HC];
    comb.copy_from_slice(&split[2 * DS4_N_HC..]);
    Some(HcPreResult { out, post, comb })
}

#[allow(dead_code)]
pub(crate) fn hc_pre_from_states_batch(
    model: &GgufModel,
    fn_tensor: &BoundTensor,
    fn_tensor_data: Option<&[f32]>,
    scale_tensor: &BoundTensor,
    scale_tensor_data: Option<&[f32]>,
    base_tensor: &BoundTensor,
    base_tensor_data: Option<&[f32]>,
    residual_hc_batch: &[f32],
) -> Option<HcPreBatchResult> {
    if residual_hc_batch.is_empty() || residual_hc_batch.len() % DS4_N_HC != 0 {
        return None;
    }
    let n_embd = usize::try_from(*base_tensor.dims.first()?)
        .ok()?
        .checked_div(DS4_HC_SPLIT_DIM)?;
    if n_embd == 0 {
        return None;
    }
    let hc_dim = n_embd.checked_mul(DS4_N_HC)?;
    if residual_hc_batch.len() % hc_dim != 0 {
        return None;
    }
    let n_tokens = residual_hc_batch.len() / hc_dim;
    let mut out = Vec::with_capacity(n_tokens * n_embd);
    let mut post = Vec::with_capacity(n_tokens * DS4_N_HC);
    let mut comb = Vec::with_capacity(n_tokens * DS4_N_HC * DS4_N_HC);
    for residual_hc in residual_hc_batch.chunks_exact(hc_dim) {
        let one = hc_pre_from_state(
            model,
            fn_tensor,
            fn_tensor_data,
            scale_tensor,
            scale_tensor_data,
            base_tensor,
            base_tensor_data,
            residual_hc,
        )?;
        out.extend_from_slice(&one.out);
        post.extend_from_slice(&one.post);
        comb.extend_from_slice(&one.comb);
    }
    Some(HcPreBatchResult {
        out,
        post,
        comb,
        n_tokens,
        n_embd,
    })
}

pub(crate) fn hc_post_one(
    block_out: &[f32],
    residual_hc: &[f32],
    post: &[f32],
    comb: &[f32],
) -> Option<Vec<f32>> {
    if residual_hc.is_empty()
        || residual_hc.len() % DS4_N_HC != 0
        || post.len() != DS4_N_HC
        || comb.len() != DS4_N_HC * DS4_N_HC
    {
        return None;
    }
    let n_embd = residual_hc.len() / DS4_N_HC;
    if block_out.len() != n_embd {
        return None;
    }
    let mut out_hc = vec![0.0; residual_hc.len()];
    for dst in 0..DS4_N_HC {
        for d in 0..n_embd {
            let mut acc = block_out[d] * post[dst];
            for src in 0..DS4_N_HC {
                acc += comb[dst + src * DS4_N_HC] * residual_hc[src * n_embd + d];
            }
            out_hc[dst * n_embd + d] = acc;
        }
    }
    Some(out_hc)
}

#[allow(dead_code)]
pub(crate) fn hc_post_batch(
    block_out_batch: &[f32],
    residual_hc_batch: &[f32],
    post_batch: &[f32],
    comb_batch: &[f32],
    n_embd: usize,
) -> Option<Vec<f32>> {
    if n_embd == 0 || block_out_batch.len() % n_embd != 0 {
        return None;
    }
    let n_tokens = block_out_batch.len() / n_embd;
    let hc_dim = n_embd.checked_mul(DS4_N_HC)?;
    if residual_hc_batch.len() != n_tokens.checked_mul(hc_dim)?
        || post_batch.len() != n_tokens.checked_mul(DS4_N_HC)?
        || comb_batch.len() != n_tokens.checked_mul(DS4_N_HC * DS4_N_HC)?
    {
        return None;
    }

    let mut out = Vec::with_capacity(residual_hc_batch.len());
    for token_idx in 0..n_tokens {
        let block_out = &block_out_batch[token_idx * n_embd..(token_idx + 1) * n_embd];
        let residual_hc = &residual_hc_batch[token_idx * hc_dim..(token_idx + 1) * hc_dim];
        let post = &post_batch[token_idx * DS4_N_HC..(token_idx + 1) * DS4_N_HC];
        let comb = &comb_batch
            [token_idx * DS4_N_HC * DS4_N_HC..(token_idx + 1) * DS4_N_HC * DS4_N_HC];
        out.extend_from_slice(&hc_post_one(block_out, residual_hc, post, comb)?);
    }
    Some(out)
}

#[allow(dead_code)]
pub(crate) fn hc_post_sum_batch(
    lhs_batch: &[f32],
    rhs_batch: &[f32],
    residual_hc_batch: &[f32],
    post_batch: &[f32],
    comb_batch: &[f32],
    n_embd: usize,
) -> Option<Vec<f32>> {
    if lhs_batch.len() != rhs_batch.len() || lhs_batch.len() % n_embd != 0 {
        return None;
    }
    let mut summed = vec![0.0; lhs_batch.len()];
    for ((dst, lhs), rhs) in summed.iter_mut().zip(lhs_batch.iter()).zip(rhs_batch.iter()) {
        *dst = *lhs + *rhs;
    }
    hc_post_batch(&summed, residual_hc_batch, post_batch, comb_batch, n_embd)
}

pub(crate) fn output_hc_head(
    model: &GgufModel,
    weights: &BoundWeights,
    inp_hc: &[f32],
) -> Option<Vec<f32>> {
    let output_hc_fn = weights.output_hc_fn.as_ref()?;
    let output_hc_scale = weights.output_hc_scale.as_ref()?;
    let output_hc_base = weights.output_hc_base.as_ref()?;
    let norm_scale = rms_norm_scale_no_weight(inp_hc)?;
    let pre = match None {
        Some(data) => {
            let width = usize::try_from(*output_hc_fn.dims.first()?).ok()?;
            let rows = usize::try_from(*output_hc_fn.dims.get(1)?).ok()?;
            let mut pre = [0.0; DS4_N_HC];
            matvec_decoded_into(&mut pre, data, rows, width, inp_hc)?;
            scale_in_place(&mut pre, norm_scale);
            pre
        }
        None => {
            let flat: Vec<f32> = inp_hc.iter().map(|x| x * norm_scale).collect();
            let owned = matvec_tensor(model, output_hc_fn, &flat)?;
            let mut pre = [0.0; DS4_N_HC];
            if owned.len() != pre.len() {
                return None;
            }
            pre.copy_from_slice(&owned);
            pre
        }
    };
    let scale_owned = if true {
        decode_tensor_1d(model, output_hc_scale)
    } else {
        None
    };
    let scale = scale_owned.as_deref()?;
    let base_owned = if true {
        decode_tensor_1d(model, output_hc_base)
    } else {
        None
    };
    let base = base_owned.as_deref()?;
    if base.len() != DS4_N_HC || scale.is_empty() {
        return None;
    }
    let mut w = [0.0; DS4_N_HC];
    for i in 0..DS4_N_HC {
        w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }
    Some(hc_weighted_sum_one(inp_hc, &w, inp_hc.len() / DS4_N_HC, DS4_N_HC))
}

fn rms_norm_scale_no_weight(input: &[f32]) -> Option<f32> {
    if input.is_empty() {
        return None;
    }
    let mut ss = 0.0f64;
    for value in input {
        ss += f64::from(*value) * f64::from(*value);
    }
    Some(1.0f32 / ((ss as f32 / input.len() as f32) + 1e-6).sqrt())
}

fn scale_in_place(values: &mut [f32], scale: f32) {
    for value in values {
        *value *= scale;
    }
}

fn hc_weighted_sum_one(x: &[f32], weights: &[f32], n_embd: usize, n_hc: usize) -> Vec<f32> {
    let mut out = vec![0.0; n_embd];
    for d in 0..n_embd {
        let mut acc = 0.0f32;
        for h in 0..n_hc {
            acc += x[h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
    out
}

fn hc_split_sinkhorn_one(
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    n_hc: usize,
    iters: usize,
    eps: f32,
) -> Option<[f32; DS4_HC_SPLIT_DIM]> {
    if scale.len() < 3 || base.len() < 2 * n_hc + n_hc * n_hc || mix.len() < 2 * n_hc + n_hc * n_hc {
        return None;
    }
    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];
    let mut out = [0.0; DS4_HC_SPLIT_DIM];
    for i in 0..n_hc {
        let z = mix[i] * pre_scale + base[i];
        out[i] = sigmoid_stable(z) + eps;
    }
    for i in 0..n_hc {
        let off = n_hc + i;
        let z = mix[off] * post_scale + base[off];
        out[off] = 2.0 / (1.0 + (-z).exp());
    }

    let mut c = [0.0f32; DS4_N_HC * DS4_N_HC];
    for dst in 0..n_hc {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let off = 2 * n_hc + idx;
            let v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if v > row_max {
                row_max = v;
            }
        }
        let mut row_sum = 0.0f32;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let v = (c[idx] - row_max).exp();
            c[idx] = v;
            row_sum += v;
        }
        let inv = 1.0 / row_sum.max(eps);
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }
    for src in 0..n_hc {
        let mut sum = 0.0f32;
        for dst in 0..n_hc {
            sum += c[src + dst * n_hc];
        }
        let inv = 1.0 / (sum + eps);
        for dst in 0..n_hc {
            c[src + dst * n_hc] *= inv;
        }
    }
    for _ in 1..iters {
        for dst in 0..n_hc {
            let mut sum = 0.0f32;
            for src in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for src in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
        for src in 0..n_hc {
            let mut sum = 0.0f32;
            for dst in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for dst in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
    }
    out[2 * n_hc..].copy_from_slice(&c);
    Some(out)
}

fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_post_batch_matches_one_by_one() {
        let n_embd = 2;
        let block_out_batch = vec![1.0, 2.0, 3.0, 4.0];
        let residual_hc_batch = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0,
            71.0, 81.0,
        ];
        let post_batch = vec![1.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25];
        let comb_batch = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5,
        ];

        let batch = hc_post_batch(
            &block_out_batch,
            &residual_hc_batch,
            &post_batch,
            &comb_batch,
            n_embd,
        )
        .unwrap();

        let mut expected = Vec::new();
        for token_idx in 0..2 {
            let block_out = &block_out_batch[token_idx * n_embd..(token_idx + 1) * n_embd];
            let residual_hc =
                &residual_hc_batch[token_idx * 8..(token_idx + 1) * 8];
            let post = &post_batch[token_idx * 4..(token_idx + 1) * 4];
            let comb = &comb_batch[token_idx * 16..(token_idx + 1) * 16];
            expected.extend_from_slice(&hc_post_one(block_out, residual_hc, post, comb).unwrap());
        }

        assert_eq!(batch, expected);
    }

    #[test]
    fn hc_post_sum_batch_matches_manual_sum_then_post() {
        let n_embd = 2;
        let lhs_batch = vec![1.0, 2.0, 3.0, 4.0];
        let rhs_batch = vec![0.5, 1.5, -1.0, 2.0];
        let residual_hc_batch = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0,
            71.0, 81.0,
        ];
        let post_batch = vec![1.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25];
        let comb_batch = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5,
        ];

        let batch = hc_post_sum_batch(
            &lhs_batch,
            &rhs_batch,
            &residual_hc_batch,
            &post_batch,
            &comb_batch,
            n_embd,
        )
        .unwrap();

        let summed = vec![1.5, 3.5, 2.0, 6.0];
        let expected = hc_post_batch(
            &summed,
            &residual_hc_batch,
            &post_batch,
            &comb_batch,
            n_embd,
        )
        .unwrap();

        assert_eq!(batch, expected);
    }
}
