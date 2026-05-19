use std::sync::Arc;

use super::*;
use crate::gguf::GgufData;

#[test]
fn default_model_path_uses_crate_dir() {
    let opts = EngineOptions::default();
    assert!(opts.model_path.ends_with("ds4flash.gguf"));
    assert!(opts.model_path.is_absolute());
}

#[test]
fn uses_tensor_backed_logits_when_supported() {
    let model = inference_test_model();
    let tokenizer = Tokenizer::preview();
    let weights = bind_weights(&model).unwrap();
    let engine = Engine {
        options: EngineOptions::default(),
        model: Some(model),
        tokenizer,
        weights: Some(weights),
    };

    let logits = engine.infer_logits(&Tokens(vec![1]));

    assert_eq!(logits.len(), 3);
    assert!((logits[0] - 0.8484).abs() < 0.01);
    assert!((logits[1] - 2.2625).abs() < 0.01);
    assert!((logits[2] - 3.1110).abs() < 0.01);
}

#[test]
fn uses_reference_logits_for_large_output_heads_by_default() {
    let model = large_head_test_model();
    let tokenizer = Tokenizer::preview();
    let weights = bind_weights(&model).unwrap();
    let engine = Engine {
        options: EngineOptions::default(),
        model: Some(model),
        tokenizer,
        weights: Some(weights),
    };

    assert!(engine.should_use_reference_logits(engine.weights.as_ref().unwrap()));
    let logits = engine.infer_logits(&Tokens(vec![1]));
    assert_eq!(logits.len(), 64);
}

#[test]
fn quality_still_allows_reference_logits_for_large_output_heads() {
    let model = large_head_test_model();
    let tokenizer = Tokenizer::preview();
    let weights = bind_weights(&model).unwrap();
    let engine = Engine {
        options: EngineOptions {
            quality: true,
            ..EngineOptions::default()
        },
        model: Some(model),
        tokenizer,
        weights: Some(weights),
    };

    assert!(engine.should_use_reference_logits(engine.weights.as_ref().unwrap()));
}

#[test]
fn bind_weights_scans_all_blocks_and_optional_ffn() {
    let model = multi_block_binding_test_model();
    let weights = bind_weights(&model).unwrap();

    assert_eq!(weights.block_count(), 2);
    assert_eq!(weights.blocks[0].index, 0);
    assert_eq!(weights.blocks[1].index, 1);
    assert!(weights.blocks[0].ffn.is_some());
    assert!(weights.blocks[1].ffn.is_none());
}

fn inference_test_model() -> GgufModel {
    let mut data = vec![0u8; 64];
    write_f16s(&mut data, 0, &[0x3c00, 0x4000, 0x4200, 0x4400, 0xbc00, 0x3c00]);
    write_f32s(&mut data, 16, &[1.0, 2.0]);
    write_f32s(&mut data, 32, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let tensors = vec![
        GgufTensor {
            name: "token_embd.weight".to_string(),
            ndim: 2,
            dims: vec![2, 3],
            tensor_type: 1,
            rel_offset: 0,
            abs_offset: 0,
            elements: 6,
            bytes: 12,
        },
        GgufTensor {
            name: "output_norm.weight".to_string(),
            ndim: 1,
            dims: vec![2],
            tensor_type: 0,
            rel_offset: 16,
            abs_offset: 16,
            elements: 2,
            bytes: 8,
        },
        GgufTensor {
            name: "output.weight".to_string(),
            ndim: 2,
            dims: vec![2, 3],
            tensor_type: 0,
            rel_offset: 32,
            abs_offset: 32,
            elements: 6,
            bytes: 24,
        },
    ];
    let mut by_name = std::collections::HashMap::new();
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
        vocab_size: Some(3),
        tokenizer_tokens: Vec::new(),
        tokenizer_merges: Vec::new(),
        tensors,
        tensors_by_name: by_name,
        data: Arc::new(GgufData::Owned(data)),
    }
}

fn multi_block_binding_test_model() -> GgufModel {
    let mut tensors = vec![
        minimal_tensor("token_embd.weight", vec![2, 4], 0),
        minimal_tensor("output.weight", vec![2, 4], 0),
    ];
    for block in 0..2 {
        tensors.extend([
            minimal_tensor(&format!("blk.{block}.attn_norm.weight"), vec![2], 0),
            minimal_tensor(&format!("blk.{block}.attn_q_a.weight"), vec![2, 2], 0),
            minimal_tensor(&format!("blk.{block}.attn_q_a_norm.weight"), vec![2], 0),
            minimal_tensor(&format!("blk.{block}.attn_q_b.weight"), vec![2, 2], 0),
            minimal_tensor(&format!("blk.{block}.attn_kv.weight"), vec![2, 2], 0),
            minimal_tensor(&format!("blk.{block}.attn_kv_a_norm.weight"), vec![2], 0),
            minimal_tensor(&format!("blk.{block}.attn_sinks.weight"), vec![1], 0),
            minimal_tensor(&format!("blk.{block}.attn_output_a.weight"), vec![2, 1], 0),
            minimal_tensor(&format!("blk.{block}.attn_output_b.weight"), vec![1, 2], 0),
        ]);
    }
    tensors.extend([
        minimal_tensor("blk.0.ffn_norm.weight", vec![2], 0),
        minimal_tensor("blk.0.ffn_gate_inp.weight", vec![2, 2], 0),
        minimal_tensor("blk.0.ffn_gate_exps.weight", vec![2, 2, 2], 0),
        minimal_tensor("blk.0.ffn_up_exps.weight", vec![2, 2, 2], 0),
        minimal_tensor("blk.0.ffn_down_exps.weight", vec![2, 2, 2], 0),
        minimal_tensor("blk.0.ffn_gate_shexp.weight", vec![2, 2], 0),
        minimal_tensor("blk.0.ffn_up_shexp.weight", vec![2, 2], 0),
        minimal_tensor("blk.0.ffn_down_shexp.weight", vec![2, 2], 0),
    ]);

    let mut by_name = std::collections::HashMap::new();
    for (idx, tensor) in tensors.iter().enumerate() {
        by_name.insert(tensor.name.clone(), idx);
    }
    GgufModel {
        version: 3,
        n_tensors: tensors.len() as u64,
        n_kv: 0,
        alignment: 32,
        file_size: 1,
        tensor_data_pos: 0,
        architecture: Some("deepseek4".to_string()),
        vocab_size: Some(4),
        tokenizer_tokens: Vec::new(),
        tokenizer_merges: Vec::new(),
        tensors,
        tensors_by_name: by_name,
        data: Arc::new(GgufData::Owned(vec![0u8; 1])),
    }
}

fn large_head_test_model() -> GgufModel {
    let tensors = vec![
        GgufTensor {
            name: "token_embd.weight".to_string(),
            ndim: 2,
            dims: vec![2, 64],
            tensor_type: 1,
            rel_offset: 0,
            abs_offset: 0,
            elements: 128,
            bytes: 256,
        },
        GgufTensor {
            name: "output.weight".to_string(),
            ndim: 2,
            dims: vec![4096, 4096],
            tensor_type: 0,
            rel_offset: 256,
            abs_offset: 256,
            elements: 16_777_216,
            bytes: 67_108_864,
        },
    ];
    let mut by_name = std::collections::HashMap::new();
    for (idx, tensor) in tensors.iter().enumerate() {
        by_name.insert(tensor.name.clone(), idx);
    }
    GgufModel {
        version: 3,
        n_tensors: tensors.len() as u64,
        n_kv: 0,
        alignment: 32,
        file_size: 512,
        tensor_data_pos: 0,
        architecture: Some("deepseek4".to_string()),
        vocab_size: Some(64),
        tokenizer_tokens: Vec::new(),
        tokenizer_merges: Vec::new(),
        tensors,
        tensors_by_name: by_name,
        data: Arc::new(GgufData::Owned(vec![0u8; 512])),
    }
}

fn minimal_tensor(name: &str, dims: Vec<u64>, tensor_type: u32) -> GgufTensor {
    let elements = dims.iter().copied().product();
    GgufTensor {
        name: name.to_string(),
        ndim: dims.len() as u32,
        dims,
        tensor_type,
        rel_offset: 0,
        abs_offset: 0,
        elements,
        bytes: 0,
    }
}

fn write_f16s(dst: &mut [u8], offset: usize, values: &[u16]) {
    for (idx, value) in values.iter().enumerate() {
        let start = offset + idx * 2;
        dst[start..start + 2].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_f32s(dst: &mut [u8], offset: usize, values: &[f32]) {
    for (idx, value) in values.iter().enumerate() {
        let start = offset + idx * 4;
        dst[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
}
