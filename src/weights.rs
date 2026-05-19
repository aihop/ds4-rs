use crate::error::{Ds4Error, Result};
use crate::gguf::GgufModel;

#[derive(Clone, Debug)]
pub(crate) struct BoundTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub tensor_type: u32,
    pub abs_offset: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundAttentionBlock {
    pub attn_norm: BoundTensor,
    pub hc_attn_fn: Option<BoundTensor>,
    pub hc_attn_scale: Option<BoundTensor>,
    pub hc_attn_base: Option<BoundTensor>,
    pub attn_q_a: BoundTensor,
    pub attn_q_a_norm: BoundTensor,
    pub attn_q_b: BoundTensor,
    pub attn_kv: BoundTensor,
    pub attn_kv_a_norm: BoundTensor,
    pub attn_sinks: BoundTensor,
    pub attn_output_a: BoundTensor,
    pub attn_output_b: BoundTensor,
        pub attn_compressor_kv: Option<BoundTensor>,
    pub attn_compressor_gate: Option<BoundTensor>,
    pub attn_compressor_ape: Option<BoundTensor>,
    pub attn_compressor_norm: Option<BoundTensor>,
    pub indexer_compressor_kv: Option<BoundTensor>,
    pub indexer_compressor_gate: Option<BoundTensor>,
    pub indexer_compressor_ape: Option<BoundTensor>,
    pub indexer_compressor_norm: Option<BoundTensor>,
    pub indexer_attn_q_b: Option<BoundTensor>,
    pub indexer_proj: Option<BoundTensor>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct BoundFfnBlock {
    pub ffn_norm: BoundTensor,
    pub ffn_gate_inp: BoundTensor,
    pub ffn_gate_exps: BoundTensor,
    pub ffn_up_exps: BoundTensor,
    pub ffn_down_exps: BoundTensor,
    pub ffn_gate_shexp: BoundTensor,
    pub ffn_up_shexp: BoundTensor,
    pub ffn_down_shexp: BoundTensor,
    pub hc_ffn_fn: Option<BoundTensor>,
    pub hc_ffn_scale: Option<BoundTensor>,
    pub hc_ffn_base: Option<BoundTensor>,
    pub ffn_exp_probs_b: Option<BoundTensor>,
    pub ffn_gate_tid2eid: Option<BoundTensor>,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundBlock {
    pub index: usize,
    pub attention: BoundAttentionBlock,
    #[allow(dead_code)]
    pub ffn: Option<BoundFfnBlock>,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundWeights {
    pub token_embd: BoundTensor,
    pub output: BoundTensor,
    pub output_hc_fn: Option<BoundTensor>,
    pub output_hc_scale: Option<BoundTensor>,
    pub output_hc_base: Option<BoundTensor>,
    pub output_norm: Option<BoundTensor>,
    pub blocks: Vec<BoundBlock>,
}

pub(crate) fn bind_weights(model: &GgufModel) -> Result<BoundWeights> {
    let output_norm = bind_tensor(model, "output_norm.weight").ok();
    let output_hc_fn = bind_tensor(model, "output_hc_fn.weight").ok();
    let output_hc_scale = bind_tensor(model, "output_hc_scale.weight").ok();
    let output_hc_base = bind_tensor(model, "output_hc_base.weight").ok();
    Ok(BoundWeights {
        token_embd: bind_tensor(model, "token_embd.weight")
            .or_else(|_| bind_tensor(model, "tok_embeddings.weight"))?,
        output: bind_tensor(model, "output.weight")?,
        output_hc_fn,
        output_hc_scale,
        output_hc_base,
        output_norm,
        blocks: bind_blocks(model),
    })
}

pub(crate) fn bind_tensor(model: &GgufModel, name: &str) -> Result<BoundTensor> {
    let tensor = model
        .tensor(name)
        .ok_or_else(|| Ds4Error::Protocol(format!("required tensor is missing: {name}")))?;
    Ok(BoundTensor {
        name: name.to_string(),
        dims: tensor.dims.clone(),
        tensor_type: tensor.tensor_type,
        abs_offset: tensor.abs_offset,
    })
}

pub(crate) fn checksum_prefix(model: &GgufModel, tensor: &BoundTensor, max_bytes: usize) -> u64 {
    let Some(source) = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t)) else {
        return 0;
    };
    source
        .iter()
        .take(max_bytes)
        .fold(0u64, |acc, byte| acc.wrapping_mul(131).wrapping_add(*byte as u64))
}

impl BoundWeights {
    pub(crate) fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

fn bind_blocks(model: &GgufModel) -> Vec<BoundBlock> {
    let mut blocks = Vec::new();
    for index in block_indexes(model) {
        let attention = match bind_attention_block(model, index) {
            Ok(attn) => attn,
            Err(e) => {
                tracing::warn!("failed to bind attention block {}: {}", index, e);
                continue;
            }
        };
        let ffn = match bind_ffn_block(model, index) {
            Ok(ffn) => Some(ffn),
            Err(e) => {
                tracing::warn!("failed to bind ffn block {} (optional, may be expected): {}", index, e);
                None
            }
        };
        blocks.push(BoundBlock {
            index,
            attention,
            ffn,
        });
    }
    blocks
}

fn block_indexes(model: &GgufModel) -> Vec<usize> {
    let mut indexes = Vec::new();
    for tensor in &model.tensors {
        let Some(rest) = tensor.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((index, _)) = rest.split_once('.') else {
            continue;
        };
        let Ok(index) = index.parse::<usize>() else {
            continue;
        };
        if !indexes.contains(&index) {
            indexes.push(index);
        }
    }
    indexes.sort_unstable();
    indexes
}

fn bind_attention_block(model: &GgufModel, index: usize) -> Result<BoundAttentionBlock> {
    let attn_norm = bind_tensor(model, &format!("blk.{index}.attn_norm.weight"))?;
    let attn_q_a_norm = bind_tensor(model, &format!("blk.{index}.attn_q_a_norm.weight"))?;
    let attn_kv_a_norm = bind_tensor(model, &format!("blk.{index}.attn_kv_a_norm.weight"))?;
    let attn_sinks = bind_tensor(model, &format!("blk.{index}.attn_sinks.weight"))?;
    let hc_attn_fn = bind_tensor(model, &format!("blk.{index}.hc_attn_fn.weight")).ok();
    let hc_attn_scale = bind_tensor(model, &format!("blk.{index}.hc_attn_scale.weight")).ok();
    let hc_attn_base = bind_tensor(model, &format!("blk.{index}.hc_attn_base.weight")).ok();
    Ok(BoundAttentionBlock {
        attn_norm,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_q_a: bind_tensor(model, &format!("blk.{index}.attn_q_a.weight"))?,
        attn_q_a_norm,
        attn_q_b: bind_tensor(model, &format!("blk.{index}.attn_q_b.weight"))?,
        attn_kv: bind_tensor(model, &format!("blk.{index}.attn_kv.weight"))?,
        attn_kv_a_norm,
        attn_sinks,
        attn_output_a: bind_tensor(model, &format!("blk.{index}.attn_output_a.weight"))?,
        attn_output_b: bind_tensor(model, &format!("blk.{index}.attn_output_b.weight"))?,
        attn_compressor_kv: None,
        attn_compressor_gate: None,
        attn_compressor_ape: None,
        attn_compressor_norm: None,
        indexer_compressor_kv: None,
        indexer_compressor_gate: None,
        indexer_compressor_ape: None,
        indexer_compressor_norm: None,
        indexer_attn_q_b: None,
        indexer_proj: None,
    })
}

fn bind_ffn_block(model: &GgufModel, index: usize) -> Result<BoundFfnBlock> {
    let ffn_norm = bind_tensor(model, &format!("blk.{index}.ffn_norm.weight"))?;
    let hc_ffn_fn = bind_tensor(model, &format!("blk.{index}.hc_ffn_fn.weight")).ok();
    let hc_ffn_scale = bind_tensor(model, &format!("blk.{index}.hc_ffn_scale.weight")).ok();
    let hc_ffn_base = bind_tensor(model, &format!("blk.{index}.hc_ffn_base.weight")).ok();
    let ffn_exp_probs_b = bind_tensor(model, &format!("blk.{index}.exp_probs_b.bias")).ok();
    let ffn_gate_tid2eid = bind_tensor(model, &format!("blk.{index}.ffn_gate_tid2eid.weight")).ok();
    Ok(BoundFfnBlock {
        ffn_norm,
        ffn_gate_inp: bind_tensor(model, &format!("blk.{index}.ffn_gate_inp.weight"))?,
        ffn_gate_exps: bind_tensor(model, &format!("blk.{index}.ffn_gate_exps.weight"))?,
        ffn_up_exps: bind_tensor(model, &format!("blk.{index}.ffn_up_exps.weight"))?,
        ffn_down_exps: bind_tensor(model, &format!("blk.{index}.ffn_down_exps.weight"))?,
        ffn_gate_shexp: bind_tensor(model, &format!("blk.{index}.ffn_gate_shexp.weight"))?,
        ffn_up_shexp: bind_tensor(model, &format!("blk.{index}.ffn_up_shexp.weight"))?,
        ffn_down_shexp: bind_tensor(model, &format!("blk.{index}.ffn_down_shexp.weight"))?,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        ffn_exp_probs_b,
        ffn_gate_tid2eid,
    })
}

fn decode_tensor_1d_or_zeros(_model: &GgufModel, tensor: &BoundTensor) -> Vec<f32> {
    tensor
        .dims
        .first()
        .and_then(|dim| usize::try_from(*dim).ok())
        .map(|len| vec![0.0; len])
        .unwrap_or_default()
}

fn decode_tensor_2d_or_zeros(_model: &GgufModel, tensor: &BoundTensor) -> Vec<f32> {
    let width = tensor
        .dims
        .first()
        .and_then(|dim| usize::try_from(*dim).ok())
        .unwrap_or_default();
    let rows = tensor
        .dims
        .get(1)
        .and_then(|dim| usize::try_from(*dim).ok())
        .unwrap_or_default();
    vec![0.0; width.saturating_mul(rows)]
}

fn decode_tensor_i32_2d(model: &GgufModel, tensor: &BoundTensor) -> Option<Vec<i32>> {
    if tensor.tensor_type != 26 || tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let expected = width.checked_mul(rows)?.checked_mul(4)?;
    if data.len() < expected {
        return None;
    }
    Some(
        data[..expected]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}
