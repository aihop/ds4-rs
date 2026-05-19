use std::collections::BTreeMap;

use crate::gguf::GgufModel;
use crate::kernels::activation::{silu, softplus_stable, swiglu, swiglu_into};
use crate::kernels::decode_scratch::DecodeScratch;
use crate::kernels::hc::{hc_post_sum_batch, hc_pre_from_states_batch};
use crate::kernels::matmul::{
    matmul_q8_0_batch_into, matmul_q8_0_pair_batch_into, matvec_tensor, matvec_tensor_pair,
    matvec_tensor_pair_into, matvec_tensor_with_q8_0, matvec_tensor_with_q8_0_into,
    quantize_activation_q8_0_cached,
};
use crate::kernels::norm::{rms_norm_weight, rms_norm_with_decoded_weight};
use crate::kernels::quant::{
    dot_iq2_xxs_pair_rows_from_accessors, dot_q2_k_row_from_accessor, quantize_row_q8_k_into,
    quantized_tensor_accessor,
};
use crate::weights::BoundFfnBlock;

const SWIGLU_CLAMP_EXP: f32 = 7.0;
const EXPERT_WEIGHT_SCALE: f32 = 1.5;
const N_EXPERT_USED: usize = 6;

pub(crate) fn apply_ffn_block(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    scratch: &mut DecodeScratch,
    hidden: &[f32],
    token: i32,
) -> Option<Vec<f32>> {
    let block_out = ffn_block_output(model, layer, scratch, hidden, token)?;
    let mut combined = hidden.to_vec();
    for (dst, src) in combined.iter_mut().zip(block_out.iter()) {
        *dst += *src;
    }
    Some(combined)
}

pub(crate) fn ffn_block_output(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    scratch: &mut DecodeScratch,
    hidden: &[f32],
    token: i32,
) -> Option<Vec<f32>> {
    let norm = rms_norm_with_decoded_weight(&layer.ffn_norm_data, hidden)
        .or_else(|| rms_norm_weight(model, &layer.ffn_norm, hidden))?;
    let routed = routed_moe(model, layer, scratch, &norm, token)
        .unwrap_or_else(|| vec![0.0; hidden.len()]);
    let shared = shared_ffn(model, layer, scratch, &norm)?;
    if routed.len() != shared.len() {
        return None;
    }
    let mut block_out = vec![0.0; shared.len()];
    for ((dst, shared), routed) in block_out.iter_mut().zip(shared.iter()).zip(routed.iter()) {
        *dst = *shared + *routed;
    }
    Some(block_out)
}

#[allow(dead_code)]
pub(crate) fn ffn_block_output_batch(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    residual_hc_batch: &[f32],
    tokens: &[i32],
) -> Option<Vec<f32>> {
    if tokens.is_empty() {
        return None;
    }
    let hc_pre = hc_pre_from_states_batch(
        model,
        layer.hc_ffn_fn.as_ref()?,
        layer.hc_ffn_fn_data.as_deref(),
        layer.hc_ffn_scale.as_ref()?,
        layer.hc_ffn_scale_data.as_deref(),
        layer.hc_ffn_base.as_ref()?,
        layer.hc_ffn_base_data.as_deref(),
        residual_hc_batch,
    )?;
    if hc_pre.n_tokens != tokens.len() {
        return None;
    }

    let mut norm_batch = Vec::with_capacity(hc_pre.out.len());
    for hidden in hc_pre.out.chunks_exact(hc_pre.n_embd) {
        norm_batch.extend(
            rms_norm_with_decoded_weight(&layer.ffn_norm_data, hidden)
                .or_else(|| rms_norm_weight(model, &layer.ffn_norm, hidden))?,
        );
    }

    let routed = routed_moe_batch(model, layer, &norm_batch, tokens, hc_pre.n_embd)
        .unwrap_or_else(|| vec![0.0; norm_batch.len()]);
    let shared = shared_ffn_batch(model, layer, &norm_batch, hc_pre.n_embd)?;
    hc_post_sum_batch(
        &routed,
        &shared,
        residual_hc_batch,
        &hc_pre.post,
        &hc_pre.comb,
        hc_pre.n_embd,
    )
}

fn shared_ffn<'a>(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    scratch: &'a mut DecodeScratch,
    norm: &[f32],
) -> Option<&'a [f32]> {
    if layer.ffn_gate_shexp.tensor_type == 8 && layer.ffn_up_shexp.tensor_type == 8 {
        matvec_tensor_pair_into(
            model,
            &layer.ffn_gate_shexp,
            &layer.ffn_up_shexp,
            norm,
            &mut scratch.shared_gate,
            &mut scratch.shared_up,
        )?;
        swiglu_into(
            &mut scratch.shared_mid,
            &scratch.shared_gate,
            &scratch.shared_up,
            SWIGLU_CLAMP_EXP,
        )?;
        let mid_q8 = if layer.ffn_down_shexp.tensor_type == 8 {
            quantize_activation_q8_0_cached(&scratch.shared_mid)
        } else {
            None
        };
        match mid_q8.as_ref() {
            Some(quant) => matvec_tensor_with_q8_0_into(
                model,
                &layer.ffn_down_shexp,
                &scratch.shared_mid,
                quant,
                &mut scratch.shared_out,
            )?,
            None => {
                scratch.shared_out =
                    matvec_tensor(model, &layer.ffn_down_shexp, &scratch.shared_mid)?
            }
        }
        return Some(scratch.shared_out.as_slice());
    }
    let (gate, up) = matvec_tensor_pair(model, &layer.ffn_gate_shexp, &layer.ffn_up_shexp, norm)?;
    let mid = swiglu(&gate, &up, SWIGLU_CLAMP_EXP)?;
    let mid_q8 = if layer.ffn_down_shexp.tensor_type == 8 {
        quantize_activation_q8_0_cached(&mid)
    } else {
        None
    };
    scratch.shared_out = match mid_q8.as_ref() {
        Some(quant) => matvec_tensor_with_q8_0(model, &layer.ffn_down_shexp, &mid, quant)?,
        None => matvec_tensor(model, &layer.ffn_down_shexp, &mid)?,
    };
    Some(scratch.shared_out.as_slice())
}

#[allow(dead_code)]
fn shared_ffn_batch(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    norm_batch: &[f32],
    n_embd: usize,
) -> Option<Vec<f32>> {
    if n_embd == 0 || norm_batch.len() % n_embd != 0 {
        return None;
    }
    let n_tok = norm_batch.len() / n_embd;
    if layer.ffn_gate_shexp.tensor_type == 8
        && layer.ffn_up_shexp.tensor_type == 8
        && layer.ffn_down_shexp.tensor_type == 8
    {
        let hidden = usize::try_from(*layer.ffn_gate_shexp.dims.get(1)?).ok()?;
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut mid = vec![0.0; n_tok.checked_mul(hidden)?];
        let mut out = Vec::new();
        matmul_q8_0_pair_batch_into(
            model,
            &layer.ffn_gate_shexp,
            &layer.ffn_up_shexp,
            norm_batch,
            n_tok,
            &mut gate,
            &mut up,
        )?;
        for token_idx in 0..n_tok {
            let src_gate = &gate[token_idx * hidden..(token_idx + 1) * hidden];
            let src_up = &up[token_idx * hidden..(token_idx + 1) * hidden];
            let dst_mid = &mut mid[token_idx * hidden..(token_idx + 1) * hidden];
            for ((dst, &g), &u) in dst_mid.iter_mut().zip(src_gate.iter()).zip(src_up.iter()) {
                let gate_clamped = if SWIGLU_CLAMP_EXP > 1.0e-6 {
                    g.min(SWIGLU_CLAMP_EXP)
                } else {
                    g
                };
                let up_clamped = if SWIGLU_CLAMP_EXP > 1.0e-6 {
                    u.clamp(-SWIGLU_CLAMP_EXP, SWIGLU_CLAMP_EXP)
                } else {
                    u
                };
                *dst = silu(gate_clamped) * up_clamped;
            }
        }
        matmul_q8_0_batch_into(model, &layer.ffn_down_shexp, &mid, n_tok, &mut out)?;
        return Some(out);
    }
    let mut scratch = DecodeScratch::default();
    let mut out = vec![0.0; norm_batch.len()];
    for (norm, out_chunk) in norm_batch.chunks_exact(n_embd).zip(out.chunks_exact_mut(n_embd)) {
        out_chunk.copy_from_slice(shared_ffn(model, layer, &mut scratch, norm)?);
    }
    Some(out)
}

fn routed_moe(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    scratch: &mut DecodeScratch,
    norm: &[f32],
    token: i32,
) -> Option<Vec<f32>> {
    if layer.ffn_gate_exps.tensor_type != 16 || layer.ffn_up_exps.tensor_type != 16 {
        return None;
    }
    if layer.ffn_down_exps.tensor_type != 10 {
        return None;
    }
    let selected = select_experts(model, layer, norm, token)?;
    let expert_hidden_dim = usize::try_from(*layer.ffn_gate_exps.dims.get(1)?).ok()?;
    let down_dim = usize::try_from(*layer.ffn_down_exps.dims.get(1)?).ok()?;
    let mid_blocks = expert_hidden_dim.checked_div(crate::kernels::quant::QK_K)?;
    let gate_accessor = quantized_tensor_accessor(model, &layer.ffn_gate_exps, 66, norm.len())?;
    let up_accessor = quantized_tensor_accessor(model, &layer.ffn_up_exps, 66, norm.len())?;
    let down_accessor =
        quantized_tensor_accessor(model, &layer.ffn_down_exps, 84, expert_hidden_dim)?;
    let xq_blocks = norm.len().checked_div(crate::kernels::quant::QK_K)?;
    scratch.routed_xq.resize(xq_blocks, crate::kernels::quant::BlockQ8K::default());
    quantize_row_q8_k_into(&mut scratch.routed_xq, norm)?;
    scratch
        .routed_mid_all
        .resize(selected.len() * expert_hidden_dim, 0.0);
    scratch
        .routed_midq
        .resize(selected.len() * mid_blocks, crate::kernels::quant::BlockQ8K::default());
    scratch
        .routed_pair_out
        .resize(selected.len() * down_dim, 0.0);
    scratch.routed_out.resize(down_dim, 0.0);
    scratch.routed_out.fill(0.0);

    let input_q8 = scratch.routed_xq.as_slice();
    let mid_all = &mut scratch.routed_mid_all;
    let midq_all = &mut scratch.routed_midq;
    let pair_out_all = &mut scratch.routed_pair_out;

    // Decode uses only a handful of selected experts; keeping this path
    // allocation-free and synchronous avoids per-token thread spawn overhead.
    for (((mid, midq), out), (expert, expert_weight)) in mid_all
        .chunks_mut(expert_hidden_dim)
        .zip(midq_all.chunks_mut(mid_blocks))
        .zip(pair_out_all.chunks_mut(down_dim))
        .zip(selected.into_iter())
    {
        compute_routed_expert_output_into(
            input_q8,
            &gate_accessor,
            &up_accessor,
            &down_accessor,
            expert,
            expert_weight,
            expert_hidden_dim,
            down_dim,
            mid,
            midq,
            out,
        )?;
    }
    for pair_out in pair_out_all.chunks_exact(down_dim) {
        for (dst, value) in scratch.routed_out.iter_mut().zip(pair_out.iter()) {
            *dst += *value;
        }
    }
    Some(scratch.routed_out.clone())
}

#[allow(dead_code)]
fn routed_moe_batch(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    norm_batch: &[f32],
    tokens: &[i32],
    n_embd: usize,
) -> Option<Vec<f32>> {
    if n_embd == 0 || tokens.is_empty() || norm_batch.len() != tokens.len().checked_mul(n_embd)? {
        return None;
    }
    if layer.ffn_gate_exps.tensor_type != 16 || layer.ffn_up_exps.tensor_type != 16 {
        return None;
    }
    if layer.ffn_down_exps.tensor_type != 10 {
        return None;
    }

    let expert_hidden_dim = usize::try_from(*layer.ffn_gate_exps.dims.get(1)?).ok()?;
    let down_dim = usize::try_from(*layer.ffn_down_exps.dims.get(1)?).ok()?;
    let mid_blocks = expert_hidden_dim.checked_div(crate::kernels::quant::QK_K)?;
    let xq_blocks = n_embd.checked_div(crate::kernels::quant::QK_K)?;
    let gate_accessor = quantized_tensor_accessor(model, &layer.ffn_gate_exps, 66, n_embd)?;
    let up_accessor = quantized_tensor_accessor(model, &layer.ffn_up_exps, 66, n_embd)?;
    let down_accessor = quantized_tensor_accessor(model, &layer.ffn_down_exps, 84, expert_hidden_dim)?;

    let mut batch_xq = vec![crate::kernels::quant::BlockQ8K::default(); tokens.len() * xq_blocks];
    let mut groups: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
    for (token_idx, (norm, token)) in norm_batch
        .chunks_exact(n_embd)
        .zip(tokens.iter().copied())
        .enumerate()
    {
        quantize_row_q8_k_into(
            &mut batch_xq[token_idx * xq_blocks..(token_idx + 1) * xq_blocks],
            norm,
        )?;
        for (expert, weight) in select_experts(model, layer, norm, token)? {
            groups.entry(expert).or_default().push((token_idx, weight));
        }
    }

    let mut routed_out = vec![0.0; tokens.len() * down_dim];
    let mut mid = vec![0.0; expert_hidden_dim];
    let mut midq = vec![crate::kernels::quant::BlockQ8K::default(); mid_blocks];
    let mut pair_out = vec![0.0; down_dim];
    for (expert, items) in groups {
        for (token_idx, expert_weight) in items {
            pair_out.fill(0.0);
            compute_routed_expert_output_into(
                &batch_xq[token_idx * xq_blocks..(token_idx + 1) * xq_blocks],
                &gate_accessor,
                &up_accessor,
                &down_accessor,
                expert,
                expert_weight,
                expert_hidden_dim,
                down_dim,
                &mut mid,
                &mut midq,
                &mut pair_out,
            )?;
            let dst = &mut routed_out[token_idx * down_dim..(token_idx + 1) * down_dim];
            for (dst, value) in dst.iter_mut().zip(pair_out.iter()) {
                *dst += *value;
            }
        }
    }
    Some(routed_out)
}

fn select_experts(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    norm: &[f32],
    token: i32,
) -> Option<Vec<(usize, f32)>> {
    if layer.ffn_gate_tid2eid_data.is_some() {
        return hash_selected_experts(layer, token)
            .and_then(|selected| hash_router_weights(model, layer, norm, &selected));
    }
    topk_selected_experts(model, layer, norm)
}

fn hash_selected_experts(layer: &BoundFfnBlock, token: i32) -> Option<Vec<usize>> {
    let table = layer.ffn_gate_tid2eid.as_ref()?;
    let data = layer.ffn_gate_tid2eid_data.as_deref()?;
    if table.tensor_type != 26 || table.dims.len() != 2 || token < 0 {
        return None;
    }
    let cols = usize::try_from(*table.dims.first()?).ok()?;
    let rows = usize::try_from(*table.dims.get(1)?).ok()?;
    let token = usize::try_from(token).ok()?;
    if cols == 0 || token >= rows {
        return None;
    }
    let row_start = token.checked_mul(cols)?;
    let row = data.get(row_start..row_start.checked_add(cols)?)?;
    Some(
        row.iter()
            .copied()
            .filter_map(|value| usize::try_from(value).ok())
            .collect(),
    )
}

fn hash_router_weights(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    norm: &[f32],
    selected: &[usize],
) -> Option<Vec<(usize, f32)>> {
    let probs = router_probs(model, layer, norm)?;
    let mut sum = 0.0;
    let mut out = Vec::with_capacity(selected.len());
    for &expert in selected {
        let weight = *probs.get(expert)?;
        sum += weight;
        out.push((expert, weight));
    }
    normalize_router_weights(&mut out, sum);
    Some(out)
}

fn topk_selected_experts(
    model: &GgufModel,
    layer: &BoundFfnBlock,
    norm: &[f32],
) -> Option<Vec<(usize, f32)>> {
    let probs = router_probs(model, layer, norm)?;
    let mut selection = probs.clone();
    if let Some(bias) = layer.ffn_exp_probs_b_data.as_deref() {
        if bias.len() != selection.len() {
            return None;
        }
        for (value, bias) in selection.iter_mut().zip(bias.iter()) {
            *value += *bias;
        }
    }
    let ranked = topk_desc_indices(&selection, N_EXPERT_USED);
    let mut out: Vec<(usize, f32)> = ranked.into_iter().map(|idx| (idx, probs[idx])).collect();
    let sum: f32 = out.iter().map(|(_, weight)| *weight).sum();
    normalize_router_weights(&mut out, sum);
    Some(out)
}

fn topk_desc_indices(score: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(score.len());
    let mut idx = vec![usize::MAX; k];
    for (candidate, value) in score.iter().copied().enumerate() {
        for slot in 0..k {
            if idx[slot] == usize::MAX || value > score[idx[slot]] {
                for shift in (slot + 1..k).rev() {
                    idx[shift] = idx[shift - 1];
                }
                idx[slot] = candidate;
                break;
            }
        }
    }
    idx.into_iter().filter(|idx| *idx != usize::MAX).collect()
}

fn router_probs(model: &GgufModel, layer: &BoundFfnBlock, norm: &[f32]) -> Option<Vec<f32>> {
    let logits = matvec_tensor(model, &layer.ffn_gate_inp, norm)?;
    Some(
        logits
            .into_iter()
            .map(|value| softplus_stable(value).sqrt())
            .collect(),
    )
}

fn normalize_router_weights(weights: &mut [(usize, f32)], sum: f32) {
    let denom = sum.max(6.103_515_6e-5);
    for (_, weight) in weights.iter_mut() {
        *weight = *weight / denom * EXPERT_WEIGHT_SCALE;
    }
}

fn compute_routed_expert_output_into(
    input_q8: &[crate::kernels::quant::BlockQ8K],
    gate_accessor: &crate::kernels::quant::QuantizedTensorAccessor<'_>,
    up_accessor: &crate::kernels::quant::QuantizedTensorAccessor<'_>,
    down_accessor: &crate::kernels::quant::QuantizedTensorAccessor<'_>,
    expert: usize,
    expert_weight: f32,
    expert_hidden_dim: usize,
    down_dim: usize,
    mid: &mut [f32],
    mid_q8: &mut [crate::kernels::quant::BlockQ8K],
    out: &mut [f32],
) -> Option<()> {
    let gate_base = expert.checked_mul(expert_hidden_dim)?;
    let up_base = expert.checked_mul(expert_hidden_dim)?;
    if mid.len() != expert_hidden_dim || out.len() != down_dim {
        return None;
    }
    for row in 0..expert_hidden_dim {
        let row_idx = gate_base + row;
        let up_row_idx = up_base + row;
        let (gate_value, up_value) = dot_iq2_xxs_pair_rows_from_accessors(
            gate_accessor,
            row_idx,
            up_accessor,
            up_row_idx,
            input_q8,
        )?;
        let gate_clamped = if SWIGLU_CLAMP_EXP > 1.0e-6 {
            gate_value.min(SWIGLU_CLAMP_EXP)
        } else {
            gate_value
        };
        let up_clamped = if SWIGLU_CLAMP_EXP > 1.0e-6 {
            up_value.clamp(-SWIGLU_CLAMP_EXP, SWIGLU_CLAMP_EXP)
        } else {
            up_value
        };
        mid[row] = silu(gate_clamped) * up_clamped * expert_weight;
    }
    quantize_row_q8_k_into(mid_q8, mid)?;
    let down_base = expert.checked_mul(down_dim)?;
    for (row, dst) in out.iter_mut().enumerate() {
        *dst = dot_q2_k_row_from_accessor(down_accessor, down_base + row, mid_q8)?;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::gguf::{GgufData, GgufTensor};
    use crate::weights::BoundTensor;

    #[test]
    fn shared_ffn_adds_residual_output() {
        let mut data = vec![0u8; 4 * (2 + 4 + 4 + 4)];
        write_f32s(&mut data, 0, &[1.0, 1.0]);
        write_f32s(&mut data, 8, &[1.0, 0.0, 0.0, 1.0]);
        write_f32s(&mut data, 24, &[1.0, 0.0, 0.0, 1.0]);
        write_f32s(&mut data, 40, &[1.0, 0.0, 0.0, 1.0]);

        let model = test_model(
            data,
            vec![
                tensor("ffn_norm", vec![2], 0, 0, 8),
                tensor("gate", vec![2, 2], 0, 8, 16),
                tensor("up", vec![2, 2], 0, 24, 16),
                tensor("down", vec![2, 2], 0, 40, 16),
            ],
        );
        let layer = BoundFfnBlock {
            ffn_norm: bound("ffn_norm", vec![2], 0, 0),
            ffn_norm_data: vec![1.0, 1.0],
            ffn_gate_inp: bound("gate_inp", vec![2, 2], 0, 0),
            ffn_gate_exps: bound("gate_exps", vec![2, 2, 2], 16, 0),
            ffn_up_exps: bound("up_exps", vec![2, 2, 2], 16, 0),
            ffn_down_exps: bound("down_exps", vec![2, 2, 2], 10, 0),
            ffn_gate_shexp: bound("gate", vec![2, 2], 0, 8),
            ffn_up_shexp: bound("up", vec![2, 2], 0, 24),
            ffn_down_shexp: bound("down", vec![2, 2], 0, 40),
            hc_ffn_fn: None,
            hc_ffn_fn_data: None,
            hc_ffn_scale: None,
            hc_ffn_scale_data: None,
            hc_ffn_base: None,
            hc_ffn_base_data: None,
            ffn_exp_probs_b: None,
            ffn_exp_probs_b_data: None,
            ffn_gate_tid2eid: None,
            ffn_gate_tid2eid_data: None,
        };

        let mut scratch = DecodeScratch::default();
        let out = apply_ffn_block(&model, &layer, &mut scratch, &[1.0, 2.0], 0).unwrap();

        assert_eq!(out.len(), 2);
        assert!(out[0] > 1.1);
        assert!(out[1] > 2.6);
    }

    #[test]
    fn shared_ffn_batch_matches_one_by_one() {
        let mut data = vec![0u8; 4 * (2 + 4 + 4 + 4)];
        write_f32s(&mut data, 0, &[1.0, 1.0]);
        write_f32s(&mut data, 8, &[1.0, 0.0, 0.0, 1.0]);
        write_f32s(&mut data, 24, &[1.0, 0.0, 0.0, 1.0]);
        write_f32s(&mut data, 40, &[1.0, 0.0, 0.0, 1.0]);

        let model = test_model(
            data,
            vec![
                tensor("ffn_norm", vec![2], 0, 0, 8),
                tensor("gate", vec![2, 2], 0, 8, 16),
                tensor("up", vec![2, 2], 0, 24, 16),
                tensor("down", vec![2, 2], 0, 40, 16),
            ],
        );
        let layer = BoundFfnBlock {
            ffn_norm: bound("ffn_norm", vec![2], 0, 0),
            ffn_norm_data: vec![1.0, 1.0],
            ffn_gate_inp: bound("gate_inp", vec![2, 2], 0, 0),
            ffn_gate_exps: bound("gate_exps", vec![2, 2, 2], 16, 0),
            ffn_up_exps: bound("up_exps", vec![2, 2, 2], 16, 0),
            ffn_down_exps: bound("down_exps", vec![2, 2, 2], 10, 0),
            ffn_gate_shexp: bound("gate", vec![2, 2], 0, 8),
            ffn_up_shexp: bound("up", vec![2, 2], 0, 24),
            ffn_down_shexp: bound("down", vec![2, 2], 0, 40),
            hc_ffn_fn: None,
            hc_ffn_fn_data: None,
            hc_ffn_scale: None,
            hc_ffn_scale_data: None,
            hc_ffn_base: None,
            hc_ffn_base_data: None,
            ffn_exp_probs_b: None,
            ffn_exp_probs_b_data: None,
            ffn_gate_tid2eid: None,
            ffn_gate_tid2eid_data: None,
        };

        let norm_batch = vec![1.0, 2.0, 2.0, 1.0];
        let batch = shared_ffn_batch(&model, &layer, &norm_batch, 2).unwrap();

        let mut scratch = DecodeScratch::default();
        let mut expected = Vec::new();
        for norm in norm_batch.chunks_exact(2) {
            expected.extend_from_slice(shared_ffn(&model, &layer, &mut scratch, norm).unwrap());
        }

        assert_eq!(batch, expected);
    }

    #[test]
    fn topk_desc_indices_matches_descending_scores() {
        let score = vec![0.1, 0.9, 0.3, 0.7, 0.8];
        assert_eq!(topk_desc_indices(&score, 3), vec![1, 4, 3]);
    }

    fn bound(name: &str, dims: Vec<u64>, tensor_type: u32, abs_offset: u64) -> BoundTensor {
        BoundTensor {
            name: name.to_string(),
            dims,
            tensor_type,
            abs_offset,
        }
    }

    fn tensor(
        name: &str,
        dims: Vec<u64>,
        tensor_type: u32,
        abs_offset: u64,
        bytes: u64,
    ) -> GgufTensor {
        GgufTensor {
            name: name.to_string(),
            ndim: dims.len() as u32,
            dims: dims.clone(),
            tensor_type,
            rel_offset: abs_offset,
            abs_offset,
            elements: dims.iter().product(),
            bytes,
        }
    }

    fn test_model(data: Vec<u8>, tensors: Vec<GgufTensor>) -> GgufModel {
        let mut by_name = HashMap::new();
        for (idx, tensor) in tensors.iter().enumerate() {
            by_name.insert(tensor.name.clone(), idx);
        }
        GgufModel {
            version: 3,
            n_tensors: tensors.len() as u64,
            n_kv: 0,
            alignment: 32,
            file_size: data.len() as u64,
            tensor_data_pos: 0,
            architecture: Some("deepseek4".to_string()),
            vocab_size: Some(2),
            tokenizer_tokens: Vec::new(),
            tokenizer_merges: Vec::new(),
            tensors,
            tensors_by_name: by_name,
            data: Arc::new(GgufData::Owned(data)),
        }
    }

    fn write_f32s(dst: &mut [u8], offset: usize, values: &[f32]) {
        for (idx, value) in values.iter().enumerate() {
            let start = offset + idx * 4;
            dst[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}
