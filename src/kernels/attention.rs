use crate::gguf::GgufModel;
use crate::kv::TransformerKvCache;
use crate::kernels::decode_scratch::DecodeScratch;
use crate::kernels::ffn::{apply_ffn_block, ffn_block_output, ffn_block_output_batch};
use crate::kernels::hc::{
    hc_from_plain_embedding, hc_post_batch, hc_post_one, hc_pre_from_state,
    hc_pre_from_states_batch, output_hc_head, supports_hc_path,
};
use crate::kernels::matmul::{
    decode_token_embedding, dot_output_rows, matvec_tensor, matvec_tensor_rows,
    matvec_q8_0_grouped_rows_into, matvec_tensor_with_q8_0,
    quantize_activation_q8_0_cached,
};
use crate::kernels::norm::{
    rms_norm_weight, rms_norm_with_decoded_weight, rms_norm_with_decoded_weight_into,
};
use crate::model::{Ds4ModelShape, DS4_ROPE_FREQ_BASE};
use crate::types::Tokens;
use crate::weights::{BoundAttentionBlock, BoundWeights};
use std::time::Instant;

const DEFAULT_ROPE_ROTARY_DIMS: usize = 64;
const HC_LAYER_MAJOR_PREFILL_MIN_TOKENS: usize = 256;

pub(crate) fn try_infer_logits_from_blocks(
    model: &GgufModel,
    weights: &BoundWeights,
    tokens: &Tokens,
) -> Option<Vec<f32>> {
    if tokens.0.is_empty() {
        return None;
    }
    let mut kv_cache = TransformerKvCache::with_layers(weights.block_count());
    let mut scratch = DecodeScratch::default();
    let mut logits = None;
    for (pos, token) in tokens.0.iter().copied().enumerate() {
        logits = Some(try_decode_next_logits_from_blocks(
            model,
            weights,
            &mut kv_cache,
            &mut scratch,
            token,
            pos,
        )?);
    }
    logits
}

pub(crate) fn try_prefill_logits_from_blocks(
    model: &GgufModel,
    weights: &BoundWeights,
    kv_cache: &mut TransformerKvCache,
    scratch: &mut DecodeScratch,
    tokens: &[i32],
    start_pos: usize,
) -> Option<Vec<f32>> {
    if tokens.is_empty() {
        return None;
    }
    // #region debug-point B:prefill-branch
    debug_attention_event(
        "B",
        "src/kernels/attention.rs:try_prefill_logits_from_blocks:start",
        "[DEBUG] attention prefill branch start",
        format!(
            "{{\"tokens\":{},\"start_pos\":{},\"supports_hc\":{}}}",
            tokens.len(),
            start_pos,
            supports_hc_path(weights)
        ),
    );
    // #endregion
    // C keeps short suffixes on the ordinary decode path and only pays the
    // layer-major prefill cost once the suffix is large enough to amortize it.
    if supports_hc_path(weights) && tokens.len() >= HC_LAYER_MAJOR_PREFILL_MIN_TOKENS {
        let started = Instant::now();
        if let Some(logits) = forward_tokens_through_blocks_hc_layer_major(
            model,
            weights,
            kv_cache,
            scratch,
            tokens,
            start_pos,
        ) {
            // #region debug-point B:prefill-branch
            debug_attention_event(
                "B",
                "src/kernels/attention.rs:try_prefill_logits_from_blocks:hc-batch",
                "[DEBUG] attention prefill used hc layer-major batch",
                format!(
                    "{{\"tokens\":{},\"elapsed_ms\":{}}}",
                    tokens.len(),
                    started.elapsed().as_millis()
                ),
            );
            // #endregion
            return Some(logits);
        }
        // #region debug-point B:prefill-branch
        debug_attention_event(
            "B",
            "src/kernels/attention.rs:try_prefill_logits_from_blocks:hc-batch-fallback",
            "[DEBUG] attention prefill hc batch returned none",
            format!(
                "{{\"tokens\":{},\"elapsed_ms\":{}}}",
                tokens.len(),
                started.elapsed().as_millis()
            ),
        );
        // #endregion
    }
    let mut logits = None;
    for (offset, token) in tokens.iter().copied().enumerate() {
        logits = Some(try_decode_next_logits_from_blocks(
            model,
            weights,
            kv_cache,
            scratch,
            token,
            start_pos + offset,
        )?);
    }
    logits
}

pub(crate) fn try_decode_next_logits_from_blocks(
    model: &GgufModel,
    weights: &BoundWeights,
    kv_cache: &mut TransformerKvCache,
    scratch: &mut DecodeScratch,
    token: i32,
    pos: usize,
) -> Option<Vec<f32>> {
    let final_hidden = forward_token_through_blocks(model, weights, kv_cache, scratch, token, pos)?;
    let final_hidden = match (&weights.output_norm, &weights.output_norm_data) {
        (Some(_), Some(output_norm_data)) => {
            rms_norm_with_decoded_weight(output_norm_data, &final_hidden)?
        }
        (Some(output_norm), None) => rms_norm_weight(model, output_norm, &final_hidden)?,
        _ => final_hidden,
    };
    dot_output_rows(model, &weights.output, &final_hidden)
}

fn forward_token_through_blocks(
    model: &GgufModel,
    weights: &BoundWeights,
    kv_cache: &mut TransformerKvCache,
    scratch: &mut DecodeScratch,
    token: i32,
    pos: usize,
) -> Option<Vec<f32>> {
    if supports_hc_path(weights) {
        return forward_token_through_blocks_hc(model, weights, kv_cache, scratch, token, pos);
    }
    forward_token_through_blocks_plain(model, weights, kv_cache, scratch, token, pos)
}

fn forward_token_through_blocks_plain(
    model: &GgufModel,
    weights: &BoundWeights,
    kv_cache: &mut TransformerKvCache,
    scratch: &mut DecodeScratch,
    token: i32,
    pos: usize,
) -> Option<Vec<f32>> {
    let hidden_width = usize::try_from(*weights.token_embd.dims.first()?).ok()?;
    kv_cache.ensure_layers(weights.block_count());
    let mut hidden = decode_token_embedding(model, &weights.token_embd, token)?;
    if hidden.len() != hidden_width {
        return None;
    }
    for block in &weights.blocks {
        hidden = forward_attention_block(
            model,
            &block.attention,
            kv_cache.layer_mut(block.index)?,
            scratch,
            &hidden,
            pos,
        )?;
        if let Some(ffn) = &block.ffn {
            hidden = apply_ffn_block(model, ffn, scratch, &hidden, token)?;
        }
    }
    Some(hidden)
}

fn forward_token_through_blocks_hc(
    model: &GgufModel,
    weights: &BoundWeights,
    kv_cache: &mut TransformerKvCache,
    scratch: &mut DecodeScratch,
    token: i32,
    pos: usize,
) -> Option<Vec<f32>> {
    kv_cache.ensure_layers(weights.block_count());
    let hidden = decode_token_embedding(model, &weights.token_embd, token)?;
    let mut state = hc_from_plain_embedding(&hidden);
    for block in &weights.blocks {
        let attn_pre = hc_pre_from_state(
            model,
            block.attention.hc_attn_fn.as_ref()?,
            block.attention.hc_attn_fn_data.as_deref(),
            block.attention.hc_attn_scale.as_ref()?,
            block.attention.hc_attn_scale_data.as_deref(),
            block.attention.hc_attn_base.as_ref()?,
            block.attention.hc_attn_base_data.as_deref(),
            &state,
        )?;
        let attn_out = attention_block_output(
            model,
            &block.attention,
            kv_cache.layer_mut(block.index)?,
            scratch,
            &attn_pre.out,
            pos,
        )?;
        state = hc_post_one(&attn_out, &state, &attn_pre.post, &attn_pre.comb)?;
        if let Some(ffn) = &block.ffn {
            let ffn_pre = hc_pre_from_state(
                model,
                ffn.hc_ffn_fn.as_ref()?,
                ffn.hc_ffn_fn_data.as_deref(),
                ffn.hc_ffn_scale.as_ref()?,
                ffn.hc_ffn_scale_data.as_deref(),
                ffn.hc_ffn_base.as_ref()?,
                ffn.hc_ffn_base_data.as_deref(),
                &state,
            )?;
            let ffn_out = ffn_block_output(model, ffn, scratch, &ffn_pre.out, token)?;
            state = hc_post_one(&ffn_out, &state, &ffn_pre.post, &ffn_pre.comb)?;
        }
    }
    output_hc_head(model, weights, &state)
}

fn forward_tokens_through_blocks_hc_layer_major(
    model: &GgufModel,
    weights: &BoundWeights,
    kv_cache: &mut TransformerKvCache,
    scratch: &mut DecodeScratch,
    tokens: &[i32],
    start_pos: usize,
) -> Option<Vec<f32>> {
    let n_embd = usize::try_from(*weights.token_embd.dims.first()?).ok()?;
    let hc_dim = n_embd.checked_mul(4)?;
    kv_cache.ensure_layers(weights.block_count());

    let mut cur_states = vec![0.0; tokens.len() * hc_dim];
    let mut attn_states = vec![0.0; tokens.len() * hc_dim];
    for (token_idx, token) in tokens.iter().copied().enumerate() {
        let hidden = decode_token_embedding(model, &weights.token_embd, token)?;
        let hc = hc_from_plain_embedding(&hidden);
        cur_states[token_idx * hc_dim..(token_idx + 1) * hc_dim].copy_from_slice(&hc);
    }

    for block in &weights.blocks {
        let attn_pre = hc_pre_from_states_batch(
            model,
            block.attention.hc_attn_fn.as_ref()?,
            block.attention.hc_attn_fn_data.as_deref(),
            block.attention.hc_attn_scale.as_ref()?,
            block.attention.hc_attn_scale_data.as_deref(),
            block.attention.hc_attn_base.as_ref()?,
            block.attention.hc_attn_base_data.as_deref(),
            &cur_states,
        )?;
        let mut attn_out_batch = Vec::with_capacity(tokens.len() * n_embd);
        for token_idx in 0..tokens.len() {
            let attn_hidden = &attn_pre.out[token_idx * n_embd..(token_idx + 1) * n_embd];
            let attn_out = attention_block_output(
                model,
                &block.attention,
                kv_cache.layer_mut(block.index)?,
                scratch,
                attn_hidden,
                start_pos + token_idx,
            )?;
            attn_out_batch.extend_from_slice(&attn_out);
        }
        attn_states = hc_post_batch(
            &attn_out_batch,
            &cur_states,
            &attn_pre.post,
            &attn_pre.comb,
            n_embd,
        )?;

        if let Some(ffn) = &block.ffn {
            cur_states = ffn_block_output_batch(model, ffn, &attn_states, tokens)?;
        } else {
            cur_states.copy_from_slice(&attn_states);
        }
    }

    let final_state = &cur_states[(tokens.len() - 1) * hc_dim..tokens.len() * hc_dim];
    output_hc_head(model, weights, final_state)
}

fn debug_attention_event(hypothesis_id: &str, location: &str, msg: &str, data_json: String) {
    // #region debug-point B:network-report
    let event = format!(
        "{{\"sessionId\":\"slow-prefill-startup\",\"runId\":\"pre-fix\",\"hypothesisId\":\"{}\",\"location\":{},\"msg\":{},\"data\":{},\"ts\":{}}}",
        hypothesis_id,
        debug_json_string(location),
        debug_json_string(msg),
        data_json,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default()
    );
    let _ = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import pathlib, urllib.request, sys; p=pathlib.Path('.dbg/slow-prefill-startup.env'); u='http://127.0.0.1:7777/event';\n\
try:\n\
 c=p.read_text();\n\
 u=next((line.split('=',1)[1].strip() for line in c.splitlines() if line.startswith('DEBUG_SERVER_URL=')), u)\n\
except Exception:\n\
 pass\n\
urllib.request.urlopen(urllib.request.Request(u, data=sys.argv[1].encode(), headers={'Content-Type':'application/json'}), timeout=1).read()",
        )
        .arg(event)
        .output();
    // #endregion
}

fn debug_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn forward_attention_block(
    model: &GgufModel,
    layer: &BoundAttentionBlock,
    kv_layer: &mut crate::kv::TransformerKvLayer,
    scratch: &mut DecodeScratch,
    hidden: &[f32],
    pos: usize,
) -> Option<Vec<f32>> {
    let attn_out = attention_block_output(model, layer, kv_layer, scratch, hidden, pos)?;
    let mut combined = hidden.to_vec();
    for (dst, src) in combined.iter_mut().zip(attn_out.iter()) {
        *dst += *src;
    }
    Some(combined)
}

fn attention_block_output(
    model: &GgufModel,
    layer: &BoundAttentionBlock,
    kv_layer: &mut crate::kv::TransformerKvLayer,
    scratch: &mut DecodeScratch,
    hidden: &[f32],
    pos: usize,
) -> Option<Vec<f32>> {
    let shape = infer_shape(layer)?;
    rms_norm_with_decoded_weight_into(&mut scratch.attn_norm, &layer.attn_norm_data, hidden)?;
    let norm = scratch.attn_norm.as_slice();
    let norm_q8 = if layer.attn_q_a.tensor_type == 8 || layer.attn_kv.tensor_type == 8 {
        quantize_activation_q8_0_cached(norm)
    } else {
        None
    };
    let mut q = project_q(
        model,
        layer,
        &mut scratch.attn_qr,
        &mut scratch.attn_qr_norm,
        &shape,
        norm,
        norm_q8.as_ref(),
    )?;
    apply_rope_tail(&mut q, &shape, pos, false);

    let mut kv = project_kv(
        model,
        layer,
        &mut scratch.attn_kv_raw,
        &shape,
        norm,
        norm_q8.as_ref(),
    )?;
    apply_rope_tail(&mut kv, &Ds4ModelShape { n_head: 1, ..shape }, pos, false);
    kv_layer.push(&kv, &kv);

    let heads = attention_rows(&q, kv_layer, &shape, layer, model)?;
    let mut rotated_heads = heads;
    apply_rope_tail(&mut rotated_heads, &shape, pos, true);
    grouped_output(model, layer, scratch, &shape, &rotated_heads)
}

fn project_q(
    model: &GgufModel,
    layer: &BoundAttentionBlock,
    qr_buf: &mut Vec<f32>,
    qr_norm_buf: &mut Vec<f32>,
    shape: &Ds4ModelShape,
    norm: &[f32],
    norm_q8: Option<&crate::kernels::matmul::QuantizedActivationQ8_0>,
) -> Option<Vec<f32>> {
    let qr = match norm_q8 {
        Some(quant) if layer.attn_q_a.tensor_type == 8 => {
            crate::kernels::matmul::matvec_tensor_with_q8_0_into(
                model,
                &layer.attn_q_a,
                norm,
                quant,
                qr_buf,
            )?;
            qr_buf.as_slice()
        }
        _ => {
            *qr_buf = matvec_tensor(model, &layer.attn_q_a, norm)?;
            qr_buf.as_slice()
        }
    };
    if qr.len() != shape.n_lora_q {
        return None;
    }
    rms_norm_with_decoded_weight_into(qr_norm_buf, &layer.attn_q_a_norm_data, qr)?;
    let qr_norm = qr_norm_buf.as_slice();
    let qr_norm_q8 = if layer.attn_q_b.tensor_type == 8 {
        quantize_activation_q8_0_cached(qr_norm)
    } else {
        None
    };
    let mut q = match qr_norm_q8.as_ref() {
        Some(quant) => matvec_tensor_with_q8_0(model, &layer.attn_q_b, &qr_norm, quant)?,
        None => matvec_tensor(model, &layer.attn_q_b, &qr_norm)?,
    };
    if q.len() != shape.head_width() {
        return None;
    }
    head_rms_norm_inplace(&mut q, shape);
    Some(q)
}

fn project_kv(
    model: &GgufModel,
    layer: &BoundAttentionBlock,
    kv_raw_buf: &mut Vec<f32>,
    shape: &Ds4ModelShape,
    norm: &[f32],
    norm_q8: Option<&crate::kernels::matmul::QuantizedActivationQ8_0>,
) -> Option<Vec<f32>> {
    let kv_raw = match norm_q8 {
        Some(quant) if layer.attn_kv.tensor_type == 8 => {
            crate::kernels::matmul::matvec_tensor_with_q8_0_into(
                model,
                &layer.attn_kv,
                norm,
                quant,
                kv_raw_buf,
            )?;
            kv_raw_buf.as_slice()
        }
        _ => {
            *kv_raw_buf = matvec_tensor(model, &layer.attn_kv, norm)?;
            kv_raw_buf.as_slice()
        }
    };
    if kv_raw.len() != shape.n_head_dim {
        return None;
    }
    rms_norm_with_decoded_weight(&layer.attn_kv_a_norm_data, &kv_raw)
}

fn grouped_output(
    model: &GgufModel,
    layer: &BoundAttentionBlock,
    scratch: &mut DecodeScratch,
    shape: &Ds4ModelShape,
    heads: &[f32],
) -> Option<Vec<f32>> {
    let group_heads = shape.n_head / shape.n_out_group;
    let group_dim = shape.n_head_dim * group_heads;
    let low = if layer.attn_output_a.tensor_type == 8 {
        matvec_q8_0_grouped_rows_into(
            model,
            &layer.attn_output_a,
            heads,
            shape.n_out_group,
            group_dim,
            shape.n_lora_o,
            &mut scratch.attn_low,
        )?;
        scratch.attn_low.as_slice()
    } else {
        scratch.attn_low.resize(shape.n_out_group * shape.n_lora_o, 0.0);
        for group in 0..shape.n_out_group {
            let input = &heads[group * group_dim..(group + 1) * group_dim];
            let row_idx = group * shape.n_lora_o;
            let group_low =
                matvec_tensor_rows(model, &layer.attn_output_a, input, row_idx, shape.n_lora_o)?;
            scratch.attn_low[row_idx..row_idx + shape.n_lora_o].copy_from_slice(&group_low);
        }
        scratch.attn_low.as_slice()
    };
    let low_q8 = if layer.attn_output_b.tensor_type == 8 {
        quantize_activation_q8_0_cached(&low)
    } else {
        None
    };
    match low_q8.as_ref() {
        Some(quant) => matvec_tensor_with_q8_0(model, &layer.attn_output_b, &low, quant),
        None => matvec_tensor(model, &layer.attn_output_b, &low),
    }
}

fn head_rms_norm_inplace(x: &mut [f32], shape: &Ds4ModelShape) {
    if shape.n_head_dim == 0 || x.len() != shape.head_width() {
        return;
    }
    for head in x.chunks_exact_mut(shape.n_head_dim) {
        let mut ss = 0.0f32;
        for value in head.iter().copied() {
            ss += value * value;
        }
        let scale = 1.0 / ((ss / shape.n_head_dim as f32) + 1e-6).sqrt();
        for value in head {
            *value *= scale;
        }
    }
}

fn apply_rope_tail(x: &mut [f32], shape: &Ds4ModelShape, pos: usize, inverse: bool) {
    let n_nope = shape.rope_tail_width();
    let theta_scale = DS4_ROPE_FREQ_BASE.powf(-2.0 / shape.n_rot as f32);
    let sin_sign = if inverse { -1.0 } else { 1.0 };
    for head in 0..shape.n_head {
        let tail = &mut x[head * shape.n_head_dim + n_nope..(head + 1) * shape.n_head_dim];
        let mut theta = pos as f32;
        for pair in 0..(shape.n_rot / 2) {
            let idx = pair * 2;
            let (sin_t, cos_t) = theta.sin_cos();
            let a = tail[idx];
            let b = tail[idx + 1];
            tail[idx] = a * cos_t - sin_sign * b * sin_t;
            tail[idx + 1] = b * cos_t + sin_sign * a * sin_t;
            theta *= theta_scale;
        }
    }
}

fn infer_shape(layer: &BoundAttentionBlock) -> Option<Ds4ModelShape> {
    let n_head = usize::try_from(*layer.attn_sinks.dims.first()?).ok()?;
    let n_head_dim = usize::try_from(*layer.attn_kv_a_norm.dims.first()?).ok()?;
    let n_lora_q = usize::try_from(*layer.attn_q_a_norm.dims.first()?).ok()?;
    let out_a_width = usize::try_from(*layer.attn_output_a.dims.first()?).ok()?;
    let out_a_rows = usize::try_from(*layer.attn_output_a.dims.get(1)?).ok()?;
    if n_head == 0 || n_head_dim == 0 || out_a_width == 0 || out_a_rows == 0 {
        return None;
    }
    let group_heads = out_a_width.checked_div(n_head_dim)?;
    if group_heads == 0 || n_head % group_heads != 0 {
        return None;
    }
    let n_out_group = n_head / group_heads;
    let n_lora_o = out_a_rows.checked_div(n_out_group)?;
    Some(Ds4ModelShape {
        n_head,
        n_head_dim,
        n_rot: DEFAULT_ROPE_ROTARY_DIMS.min(n_head_dim),
        n_out_group,
        n_lora_q,
        n_lora_o,
    })
}

struct TransformerKvCacheLayerRef<'a> {
    layer: &'a crate::kv::TransformerKvLayer,
}

impl<'a> TransformerKvCacheLayerRef<'a> {
    fn len(&self) -> usize {
        self.layer.len()
    }

    fn key(&self, index: usize, dim: usize) -> Option<&[f32]> {
        let start = index * dim;
        self.layer.keys.get(start..start + dim)
    }

    fn value(&self, index: usize, dim: usize) -> Option<&[f32]> {
        let start = index * dim;
        self.layer.values.get(start..start + dim)
    }
}

fn attention_rows(
    q: &[f32],
    kv_layer: &crate::kv::TransformerKvLayer,
    shape: &Ds4ModelShape,
    layer: &BoundAttentionBlock,
    model: &GgufModel,
) -> Option<Vec<f32>> {
    attention_rows_impl(
        q,
        &TransformerKvCacheLayerRef { layer: kv_layer },
        shape,
        layer,
        model,
    )
}

fn attention_rows_impl(
    q: &[f32],
    kv_layer: &TransformerKvCacheLayerRef<'_>,
    shape: &Ds4ModelShape,
    layer: &BoundAttentionBlock,
    _model: &GgufModel,
) -> Option<Vec<f32>> {
    let n_kv = kv_layer.len();
    if layer.attn_sinks_data.len() != shape.n_head || n_kv == 0 {
        return None;
    }
    let scale = 1.0f32 / (shape.n_head_dim as f32).sqrt();
    let mut out_heads = vec![0.0; shape.head_width()];
    let mut scores = vec![0.0; n_kv];

    for h in 0..shape.n_head {
        let qh = &q[h * shape.n_head_dim..(h + 1) * shape.n_head_dim];
        let mut max_score = layer.attn_sinks_data[h];
        let kv_dim = shape.n_head_dim;
        for r in 0..n_kv {
            let kv = kv_layer.key(r, kv_dim)?;
            let score = dot_f32(qh, kv) * scale;
            scores[r] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let oh = &mut out_heads[h * shape.n_head_dim..(h + 1) * shape.n_head_dim];
        let mut denom = (layer.attn_sinks_data[h] - max_score).exp();
        for r in 0..n_kv {
            let kv = kv_layer.value(r, kv_dim)?;
            let weight = (scores[r] - max_score).exp();
            denom += weight;
            axpy_f32(oh, kv, weight);
        }
        scale_f32(oh, 1.0 / denom);
    }
    Some(out_heads)
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn axpy_f32(y: &mut [f32], x: &[f32], a: f32) {
    for (dst, src) in y.iter_mut().zip(x.iter()) {
        *dst += a * src;
    }
}

fn scale_f32(x: &mut [f32], a: f32) {
    for value in x {
        *value *= a;
    }
}
