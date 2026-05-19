use crate::gguf::GgufModel;
use crate::weights::BoundTensor;

// Latency matters more than peak throughput for short first-token requests.
// Be conservative before fanning out scoped threads for row matvec work.
const PARALLEL_ROW_THRESHOLD: usize = 8192;
const MIN_ROWS_PER_THREAD: usize = 2048;
const Q8_0_BLOCK: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

#[derive(Clone, Debug)]
pub(crate) struct QuantizedActivationQ8_0 {
    q: Vec<i8>,
    scale: Vec<f32>,
    blocks: usize,
    in_dim: usize,
}

#[derive(Clone, Debug)]
struct QuantizedActivationBatchQ8_0 {
    q: Vec<i8>,
    scale: Vec<f32>,
    blocks: usize,
    in_dim: usize,
    n_rows: usize,
}

pub(crate) fn decode_token_embedding(
    model: &GgufModel,
    tensor: &BoundTensor,
    token: i32,
) -> Option<Vec<f32>> {
    if token < 0 || tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    let token = usize::try_from(token).ok()?;
    if token >= rows || width == 0 {
        return None;
    }
    decode_tensor_row(model, tensor, token, width)
}

pub(crate) fn decode_tensor_1d(model: &GgufModel, tensor: &BoundTensor) -> Option<Vec<f32>> {
    if tensor.dims.len() != 1 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    decode_tensor_row(model, tensor, 0, width)
}

pub(crate) fn decode_tensor_2d(model: &GgufModel, tensor: &BoundTensor) -> Option<Vec<f32>> {
    if tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    if width == 0 || rows == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(width.checked_mul(rows)?);
    for row in 0..rows {
        out.extend(decode_tensor_row(model, tensor, row, width)?);
    }
    Some(out)
}

pub(crate) fn dot_output_rows(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
) -> Option<Vec<f32>> {
    matvec_tensor_rows(model, tensor, input, 0, usize::try_from(*tensor.dims.get(1)?).ok()?)
}

pub(crate) fn matvec_tensor(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
) -> Option<Vec<f32>> {
    matvec_tensor_rows(model, tensor, input, 0, usize::try_from(*tensor.dims.get(1)?).ok()?)
}

pub(crate) fn matvec_tensor_pair(
    model: &GgufModel,
    tensor0: &BoundTensor,
    tensor1: &BoundTensor,
    input: &[f32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    if tensor0.dims.len() != 2 || tensor1.dims.len() != 2 {
        return None;
    }
    let width0 = usize::try_from(*tensor0.dims.first()?).ok()?;
    let rows0 = usize::try_from(*tensor0.dims.get(1)?).ok()?;
    let width1 = usize::try_from(*tensor1.dims.first()?).ok()?;
    let rows1 = usize::try_from(*tensor1.dims.get(1)?).ok()?;
    if width0 != input.len() || width1 != input.len() || rows0 != rows1 || width0 != width1 {
        return None;
    }
    if tensor0.tensor_type == 8 && tensor1.tensor_type == 8 {
        let quant = quantize_activation_q8_0(input)?;
        return parallel_matvec_q8_0_pair(model, tensor0, tensor1, &quant);
    }
    Some((matvec_tensor(model, tensor0, input)?, matvec_tensor(model, tensor1, input)?))
}

pub(crate) fn matvec_tensor_pair_into(
    model: &GgufModel,
    tensor0: &BoundTensor,
    tensor1: &BoundTensor,
    input: &[f32],
    out0: &mut Vec<f32>,
    out1: &mut Vec<f32>,
) -> Option<()> {
    if tensor0.dims.len() != 2 || tensor1.dims.len() != 2 {
        return None;
    }
    let width0 = usize::try_from(*tensor0.dims.first()?).ok()?;
    let rows0 = usize::try_from(*tensor0.dims.get(1)?).ok()?;
    let width1 = usize::try_from(*tensor1.dims.first()?).ok()?;
    let rows1 = usize::try_from(*tensor1.dims.get(1)?).ok()?;
    if width0 != input.len() || width1 != input.len() || rows0 != rows1 || width0 != width1 {
        return None;
    }
    if tensor0.tensor_type == 8 && tensor1.tensor_type == 8 {
        let quant = quantize_activation_q8_0(input)?;
        return parallel_matvec_q8_0_pair_into(model, tensor0, tensor1, &quant, out0, out1);
    }
    let (v0, v1) = matvec_tensor_pair(model, tensor0, tensor1, input)?;
    assign_output(out0, rows0);
    assign_output(out1, rows1);
    out0.copy_from_slice(&v0);
    out1.copy_from_slice(&v1);
    Some(())
}

pub(crate) fn matmul_q8_0_batch_into(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
    n_tok: usize,
    out: &mut Vec<f32>,
) -> Option<()> {
    if tensor.tensor_type != 8 || tensor.dims.len() != 2 || n_tok == 0 {
        return None;
    }
    let in_dim = usize::try_from(*tensor.dims.first()?).ok()?;
    let out_dim = usize::try_from(*tensor.dims.get(1)?).ok()?;
    if in_dim == 0 || out_dim == 0 || input.len() != n_tok.checked_mul(in_dim)? {
        return None;
    }
    let quant = quantize_activation_q8_0_batch(input, n_tok, in_dim)?;
    parallel_matmul_q8_0_batch_into(model, tensor, &quant, out)
}

pub(crate) fn matmul_q8_0_pair_batch_into(
    model: &GgufModel,
    tensor0: &BoundTensor,
    tensor1: &BoundTensor,
    input: &[f32],
    n_tok: usize,
    out0: &mut Vec<f32>,
    out1: &mut Vec<f32>,
) -> Option<()> {
    if tensor0.tensor_type != 8 || tensor1.tensor_type != 8 || tensor0.dims.len() != 2 || tensor1.dims.len() != 2 || n_tok == 0 {
        return None;
    }
    let in_dim0 = usize::try_from(*tensor0.dims.first()?).ok()?;
    let out_dim0 = usize::try_from(*tensor0.dims.get(1)?).ok()?;
    let in_dim1 = usize::try_from(*tensor1.dims.first()?).ok()?;
    let out_dim1 = usize::try_from(*tensor1.dims.get(1)?).ok()?;
    if in_dim0 == 0
        || out_dim0 == 0
        || in_dim0 != in_dim1
        || out_dim0 != out_dim1
        || input.len() != n_tok.checked_mul(in_dim0)?
    {
        return None;
    }
    let quant = quantize_activation_q8_0_batch(input, n_tok, in_dim0)?;
    parallel_matmul_q8_0_pair_batch_into(model, tensor0, tensor1, &quant, out0, out1)
}

pub(crate) fn quantize_activation_q8_0_cached(input: &[f32]) -> Option<QuantizedActivationQ8_0> {
    quantize_activation_q8_0(input)
}

pub(crate) fn matvec_tensor_with_q8_0(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
    quant: &QuantizedActivationQ8_0,
) -> Option<Vec<f32>> {
    if tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    if width != input.len() || width != quant.in_dim || width == 0 || rows == 0 {
        return None;
    }
    if tensor.tensor_type == 8 {
        return parallel_matvec_q8_0_rows(model, tensor, quant, 0, rows);
    }
    parallel_dot_tensor_rows(model, tensor, input, 0, rows)
}

pub(crate) fn matvec_tensor_with_q8_0_into(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
    quant: &QuantizedActivationQ8_0,
    out: &mut Vec<f32>,
) -> Option<()> {
    if tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    if width != input.len() || width != quant.in_dim || width == 0 || rows == 0 {
        return None;
    }
    if tensor.tensor_type == 8 {
        return parallel_matvec_q8_0_rows_into(model, tensor, quant, 0, rows, out);
    }
    let v = parallel_dot_tensor_rows(model, tensor, input, 0, rows)?;
    assign_output(out, rows);
    out.copy_from_slice(&v);
    Some(())
}

#[allow(dead_code)]
pub(crate) fn matvec_q8_0_grouped_rows(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
    n_groups: usize,
    group_dim: usize,
    rank: usize,
) -> Option<Vec<f32>> {
    if tensor.tensor_type != 8 || tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    if width != group_dim
        || rank == 0
        || n_groups == 0
        || input.len() != n_groups.checked_mul(group_dim)?
        || rows < n_groups.checked_mul(rank)?
    {
        return None;
    }
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let mut quantized = Vec::with_capacity(n_groups);
    for input_group in input.chunks_exact(group_dim) {
        quantized.push(quantize_activation_q8_0(input_group)?);
    }
    let quantized = std::sync::Arc::new(quantized);
    let row_bytes = quantized.first()?.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let total_rows = n_groups.checked_mul(rank)?;
    let thread_count = parallel_row_threads(total_rows);
    if thread_count > 1 {
        let chunk_size = total_rows.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for start in (0..total_rows).step_by(chunk_size) {
                let end = (start + chunk_size).min(total_rows);
                let quantized = std::sync::Arc::clone(&quantized);
                handles.push(scope.spawn(move || -> Option<(usize, Vec<f32>)> {
                    let mut chunk = Vec::with_capacity(end - start);
                    for idx in start..end {
                        let group = idx / rank;
                        let row_in_group = idx - group * rank;
                        let tensor_row = group * rank + row_in_group;
                        let value =
                            dot_q8_0_tensor_row(data, row_bytes, tensor_row, &quantized[group])?;
                        chunk.push(value);
                    }
                    Some((start, chunk))
                }));
            }
            let mut out = vec![0.0; total_rows];
            for handle in handles {
                let (start, chunk) = handle.join().ok()??;
                out[start..start + chunk.len()].copy_from_slice(&chunk);
            }
            Some(out)
        });
    }
    let mut out = Vec::with_capacity(total_rows);
    for idx in 0..total_rows {
        let group = idx / rank;
        let row_in_group = idx - group * rank;
        let tensor_row = group * rank + row_in_group;
        out.push(dot_q8_0_tensor_row(data, row_bytes, tensor_row, &quantized[group])?);
    }
    Some(out)
}

pub(crate) fn matvec_q8_0_grouped_rows_into(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
    n_groups: usize,
    group_dim: usize,
    rank: usize,
    out: &mut Vec<f32>,
) -> Option<()> {
    if tensor.tensor_type != 8 || tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    let total_rows = n_groups.checked_mul(rank)?;
    if width != group_dim
        || rank == 0
        || n_groups == 0
        || input.len() != n_groups.checked_mul(group_dim)?
        || rows < total_rows
    {
        return None;
    }
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let mut quantized = Vec::with_capacity(n_groups);
    for input_group in input.chunks_exact(group_dim) {
        quantized.push(quantize_activation_q8_0(input_group)?);
    }
    let quantized = std::sync::Arc::new(quantized);
    let row_bytes = quantized.first()?.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let thread_count = parallel_row_threads(total_rows);
    assign_output(out, total_rows);
    if thread_count > 1 {
        let chunk_size = total_rows.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            let mut start = 0usize;
            for chunk in out.chunks_mut(chunk_size) {
                let chunk_start = start;
                start += chunk.len();
                let quantized = std::sync::Arc::clone(&quantized);
                handles.push(scope.spawn(move || -> Option<()> {
                    for (offset, dst) in chunk.iter_mut().enumerate() {
                        let idx = chunk_start + offset;
                        let group = idx / rank;
                        let row_in_group = idx - group * rank;
                        let tensor_row = group * rank + row_in_group;
                        *dst = dot_q8_0_tensor_row(data, row_bytes, tensor_row, &quantized[group])?;
                    }
                    Some(())
                }));
            }
            for handle in handles {
                handle.join().ok()??;
            }
            Some(())
        });
    }
    for idx in 0..total_rows {
        let group = idx / rank;
        let row_in_group = idx - group * rank;
        let tensor_row = group * rank + row_in_group;
        out[idx] = dot_q8_0_tensor_row(data, row_bytes, tensor_row, &quantized[group])?;
    }
    Some(())
}

#[allow(dead_code)]
pub(crate) fn matvec_decoded(
    data: &[f32],
    rows: usize,
    width: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    if width == 0 || rows == 0 || input.len() != width || data.len() != rows.checked_mul(width)? {
        return None;
    }
    let mut out = Vec::with_capacity(rows);
    for row in data.chunks_exact(width) {
        let mut acc = 0.0f32;
        for (w, x) in row.iter().zip(input.iter()) {
            acc += *w * *x;
        }
        out.push(acc);
    }
    Some(out)
}

pub(crate) fn matvec_decoded_into(
    out: &mut [f32],
    data: &[f32],
    rows: usize,
    width: usize,
    input: &[f32],
) -> Option<()> {
    if width == 0
        || rows == 0
        || input.len() != width
        || out.len() != rows
        || data.len() != rows.checked_mul(width)?
    {
        return None;
    }
    for (dst, row) in out.iter_mut().zip(data.chunks_exact(width)) {
        let mut acc = 0.0f32;
        for (w, x) in row.iter().zip(input.iter()) {
            acc += *w * *x;
        }
        *dst = acc;
    }
    Some(())
}

pub(crate) fn matvec_tensor_rows(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
    row0: usize,
    n_rows: usize,
) -> Option<Vec<f32>> {
    if tensor.dims.len() != 2 {
        return None;
    }
    let width = usize::try_from(*tensor.dims.first()?).ok()?;
    let rows = usize::try_from(*tensor.dims.get(1)?).ok()?;
    if width != input.len() || width == 0 || rows == 0 || row0 > rows || n_rows > rows - row0 {
        return None;
    }
    if tensor.tensor_type == 8 {
        let quant = quantize_activation_q8_0(input)?;
        return parallel_matvec_q8_0_rows(model, tensor, &quant, row0, n_rows);
    }

    parallel_dot_tensor_rows(model, tensor, input, row0, n_rows)
}

fn parallel_dot_tensor_rows(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
    row0: usize,
    n_rows: usize,
) -> Option<Vec<f32>> {
    let thread_count = parallel_row_threads(n_rows);
    if thread_count > 1 {
        let chunk_size = n_rows.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for local_start in (0..n_rows).step_by(chunk_size) {
                let local_end = (local_start + chunk_size).min(n_rows);
                let start = row0 + local_start;
                let end = row0 + local_end;
                handles.push(scope.spawn(move || -> Option<(usize, Vec<f32>)> {
                    let mut chunk = Vec::with_capacity(local_end - local_start);
                    for row in start..end {
                        chunk.push(dot_tensor_row(model, tensor, row, input)?);
                    }
                    Some((local_start, chunk))
                }));
            }
            let mut out = vec![0.0; n_rows];
            for handle in handles {
                let (local_start, chunk) = handle.join().ok()??;
                out[local_start..local_start + chunk.len()].copy_from_slice(&chunk);
            }
            Some(out)
        });
    }

    let mut out = Vec::with_capacity(n_rows);
    for row in row0..row0 + n_rows {
        out.push(dot_tensor_row(model, tensor, row, input)?);
    }
    Some(out)
}

fn parallel_matvec_q8_0_rows(
    model: &GgufModel,
    tensor: &BoundTensor,
    quant: &QuantizedActivationQ8_0,
    row0: usize,
    n_rows: usize,
) -> Option<Vec<f32>> {
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = quant.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let thread_count = parallel_row_threads(n_rows);
    if thread_count > 1 {
        let chunk_size = n_rows.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for local_start in (0..n_rows).step_by(chunk_size) {
                let local_end = (local_start + chunk_size).min(n_rows);
                handles.push(scope.spawn(move || -> Option<(usize, Vec<f32>)> {
                    let mut chunk = Vec::with_capacity(local_end - local_start);
                    for local_row in local_start..local_end {
                        chunk.push(dot_q8_0_tensor_row(data, row_bytes, row0 + local_row, quant)?);
                    }
                    Some((local_start, chunk))
                }));
            }
            let mut out = vec![0.0; n_rows];
            for handle in handles {
                let (local_start, chunk) = handle.join().ok()??;
                out[local_start..local_start + chunk.len()].copy_from_slice(&chunk);
            }
            Some(out)
        });
    }

    let mut out = Vec::with_capacity(n_rows);
    for row in row0..row0 + n_rows {
        out.push(dot_q8_0_tensor_row(data, row_bytes, row, quant)?);
    }
    Some(out)
}

fn parallel_matvec_q8_0_rows_into(
    model: &GgufModel,
    tensor: &BoundTensor,
    quant: &QuantizedActivationQ8_0,
    row0: usize,
    n_rows: usize,
    out: &mut Vec<f32>,
) -> Option<()> {
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = quant.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let thread_count = parallel_row_threads(n_rows);
    assign_output(out, n_rows);
    if thread_count > 1 {
        let chunk_size = n_rows.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            let mut local_start = 0usize;
            for chunk in out.chunks_mut(chunk_size) {
                let start = local_start;
                local_start += chunk.len();
                handles.push(scope.spawn(move || {
                    for (offset, dst) in chunk.iter_mut().enumerate() {
                        *dst = dot_q8_0_tensor_row(data, row_bytes, row0 + start + offset, quant)?;
                    }
                    Some(())
                }));
            }
            for handle in handles {
                handle.join().ok()??;
            }
            Some(())
        });
    }

    for local_row in 0..n_rows {
        out[local_row] = dot_q8_0_tensor_row(data, row_bytes, row0 + local_row, quant)?;
    }
    Some(())
}

fn parallel_matvec_q8_0_pair(
    model: &GgufModel,
    tensor0: &BoundTensor,
    tensor1: &BoundTensor,
    quant: &QuantizedActivationQ8_0,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let rows = usize::try_from(*tensor0.dims.get(1)?).ok()?;
    let data0 = model.tensor(&tensor0.name).and_then(|t| model.tensor_bytes(t))?;
    let data1 = model.tensor(&tensor1.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = quant.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let thread_count = parallel_row_threads(rows);
    if thread_count > 1 {
        let chunk_size = rows.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for start in (0..rows).step_by(chunk_size) {
                let end = (start + chunk_size).min(rows);
                handles.push(scope.spawn(move || -> Option<(usize, Vec<f32>, Vec<f32>)> {
                    let mut chunk0 = Vec::with_capacity(end - start);
                    let mut chunk1 = Vec::with_capacity(end - start);
                    for row in start..end {
                        let (v0, v1) = dot_q8_0_tensor_row_pair(data0, data1, row_bytes, row, quant)?;
                        chunk0.push(v0);
                        chunk1.push(v1);
                    }
                    Some((start, chunk0, chunk1))
                }));
            }
            let mut out0 = vec![0.0; rows];
            let mut out1 = vec![0.0; rows];
            for handle in handles {
                let (start, chunk0, chunk1) = handle.join().ok()??;
                out0[start..start + chunk0.len()].copy_from_slice(&chunk0);
                out1[start..start + chunk1.len()].copy_from_slice(&chunk1);
            }
            Some((out0, out1))
        });
    }

    let mut out0 = Vec::with_capacity(rows);
    let mut out1 = Vec::with_capacity(rows);
    for row in 0..rows {
        let (v0, v1) = dot_q8_0_tensor_row_pair(data0, data1, row_bytes, row, quant)?;
        out0.push(v0);
        out1.push(v1);
    }
    Some((out0, out1))
}

fn parallel_matvec_q8_0_pair_into(
    model: &GgufModel,
    tensor0: &BoundTensor,
    tensor1: &BoundTensor,
    quant: &QuantizedActivationQ8_0,
    out0: &mut Vec<f32>,
    out1: &mut Vec<f32>,
) -> Option<()> {
    let rows = usize::try_from(*tensor0.dims.get(1)?).ok()?;
    let data0 = model.tensor(&tensor0.name).and_then(|t| model.tensor_bytes(t))?;
    let data1 = model.tensor(&tensor1.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = quant.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let thread_count = parallel_row_threads(rows);
    assign_output(out0, rows);
    assign_output(out1, rows);
    if thread_count > 1 {
        let chunk_size = rows.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            let mut start = 0usize;
            for (chunk0, chunk1) in out0.chunks_mut(chunk_size).zip(out1.chunks_mut(chunk_size)) {
                let row_start = start;
                start += chunk0.len();
                handles.push(scope.spawn(move || {
                    for (offset, (dst0, dst1)) in chunk0.iter_mut().zip(chunk1.iter_mut()).enumerate() {
                        let (v0, v1) =
                            dot_q8_0_tensor_row_pair(data0, data1, row_bytes, row_start + offset, quant)?;
                        *dst0 = v0;
                        *dst1 = v1;
                    }
                    Some(())
                }));
            }
            for handle in handles {
                handle.join().ok()??;
            }
            Some(())
        });
    }

    for row in 0..rows {
        let (v0, v1) = dot_q8_0_tensor_row_pair(data0, data1, row_bytes, row, quant)?;
        out0[row] = v0;
        out1[row] = v1;
    }
    Some(())
}

fn parallel_matmul_q8_0_batch_into(
    model: &GgufModel,
    tensor: &BoundTensor,
    quant: &QuantizedActivationBatchQ8_0,
    out: &mut Vec<f32>,
) -> Option<()> {
    let out_dim = usize::try_from(*tensor.dims.get(1)?).ok()?;
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = quant.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let total = quant.n_rows.checked_mul(out_dim)?;
    assign_output(out, total);
    let thread_count = parallel_row_threads(out_dim);
    if thread_count > 1 {
        let chunk_size = out_dim.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for row_start in (0..out_dim).step_by(chunk_size) {
                let row_end = (row_start + chunk_size).min(out_dim);
                handles.push(scope.spawn(move || -> Option<(usize, Vec<f32>)> {
                    let mut chunk = vec![0.0; quant.n_rows.checked_mul(row_end - row_start)?];
                    for token_idx in 0..quant.n_rows {
                        let dst = &mut chunk[token_idx * (row_end - row_start)..(token_idx + 1) * (row_end - row_start)];
                        for (local_row, dst) in dst.iter_mut().enumerate() {
                            *dst = dot_q8_0_tensor_row_batch(
                                data,
                                row_bytes,
                                row_start + local_row,
                                quant,
                                token_idx,
                            )?;
                        }
                    }
                    Some((row_start, chunk))
                }));
            }
            for handle in handles {
                let (row_start, chunk) = handle.join().ok()??;
                let chunk_rows = chunk.len() / quant.n_rows;
                for token_idx in 0..quant.n_rows {
                    let src = &chunk[token_idx * chunk_rows..(token_idx + 1) * chunk_rows];
                    let dst_start = token_idx * out_dim + row_start;
                    out[dst_start..dst_start + chunk_rows].copy_from_slice(src);
                }
            }
            Some(())
        });
    }
    for token_idx in 0..quant.n_rows {
        let dst = &mut out[token_idx * out_dim..(token_idx + 1) * out_dim];
        for (row, dst) in dst.iter_mut().enumerate() {
            *dst = dot_q8_0_tensor_row_batch(data, row_bytes, row, quant, token_idx)?;
        }
    }
    Some(())
}

fn parallel_matmul_q8_0_pair_batch_into(
    model: &GgufModel,
    tensor0: &BoundTensor,
    tensor1: &BoundTensor,
    quant: &QuantizedActivationBatchQ8_0,
    out0: &mut Vec<f32>,
    out1: &mut Vec<f32>,
) -> Option<()> {
    let out_dim = usize::try_from(*tensor0.dims.get(1)?).ok()?;
    let data0 = model.tensor(&tensor0.name).and_then(|t| model.tensor_bytes(t))?;
    let data1 = model.tensor(&tensor1.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = quant.blocks.checked_mul(Q8_0_BLOCK_BYTES)?;
    let total = quant.n_rows.checked_mul(out_dim)?;
    assign_output(out0, total);
    assign_output(out1, total);
    let thread_count = parallel_row_threads(out_dim);
    if thread_count > 1 {
        let chunk_size = out_dim.div_ceil(thread_count);
        return std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for row_start in (0..out_dim).step_by(chunk_size) {
                let row_end = (row_start + chunk_size).min(out_dim);
                handles.push(scope.spawn(move || -> Option<(usize, Vec<f32>, Vec<f32>)> {
                    let chunk_len = quant.n_rows.checked_mul(row_end - row_start)?;
                    let mut chunk0 = vec![0.0; chunk_len];
                    let mut chunk1 = vec![0.0; chunk_len];
                    for token_idx in 0..quant.n_rows {
                        let start = token_idx * (row_end - row_start);
                        for local_row in 0..(row_end - row_start) {
                            let (v0, v1) = dot_q8_0_tensor_row_pair_batch(
                                data0,
                                data1,
                                row_bytes,
                                row_start + local_row,
                                quant,
                                token_idx,
                            )?;
                            chunk0[start + local_row] = v0;
                            chunk1[start + local_row] = v1;
                        }
                    }
                    Some((row_start, chunk0, chunk1))
                }));
            }
            for handle in handles {
                let (row_start, chunk0, chunk1) = handle.join().ok()??;
                let chunk_rows = chunk0.len() / quant.n_rows;
                for token_idx in 0..quant.n_rows {
                    let src0 = &chunk0[token_idx * chunk_rows..(token_idx + 1) * chunk_rows];
                    let src1 = &chunk1[token_idx * chunk_rows..(token_idx + 1) * chunk_rows];
                    let dst_start = token_idx * out_dim + row_start;
                    out0[dst_start..dst_start + chunk_rows].copy_from_slice(src0);
                    out1[dst_start..dst_start + chunk_rows].copy_from_slice(src1);
                }
            }
            Some(())
        });
    }
    for token_idx in 0..quant.n_rows {
        let dst0 = &mut out0[token_idx * out_dim..(token_idx + 1) * out_dim];
        let dst1 = &mut out1[token_idx * out_dim..(token_idx + 1) * out_dim];
        for row in 0..out_dim {
            let (v0, v1) = dot_q8_0_tensor_row_pair_batch(data0, data1, row_bytes, row, quant, token_idx)?;
            dst0[row] = v0;
            dst1[row] = v1;
        }
    }
    Some(())
}

pub(crate) fn decode_tensor_row(
    model: &GgufModel,
    tensor: &BoundTensor,
    row_idx: usize,
    row_width: usize,
) -> Option<Vec<f32>> {
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = tensor_row_bytes(tensor.tensor_type, row_width)?;
    let row_start = row_idx.checked_mul(row_bytes)?;
    let row_end = row_start.checked_add(row_bytes)?;
    let row = data.get(row_start..row_end)?;

    let mut out = Vec::with_capacity(row_width);
    match tensor.tensor_type {
        0 => {
            for chunk in row.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        1 => {
            for chunk in row.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        30 => {
            for chunk in row.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        8 => {
            for (block_idx, block) in row.chunks_exact(34).enumerate() {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let start = block_idx * 32;
                let take = row_width.saturating_sub(start).min(32);
                for byte in &block[2..2 + take] {
                    out.push(scale * (*byte as i8) as f32);
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

pub(crate) fn dot_tensor_row(
    model: &GgufModel,
    tensor: &BoundTensor,
    row_idx: usize,
    input: &[f32],
) -> Option<f32> {
    let data = model.tensor(&tensor.name).and_then(|t| model.tensor_bytes(t))?;
    let row_bytes = tensor_row_bytes(tensor.tensor_type, input.len())?;
    let row_start = row_idx.checked_mul(row_bytes)?;
    let row_end = row_start.checked_add(row_bytes)?;
    let row = data.get(row_start..row_end)?;

    let mut acc = 0.0f32;
    match tensor.tensor_type {
        0 => {
            for (idx, chunk) in row.chunks_exact(4).enumerate() {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                acc += value * input[idx];
            }
        }
        1 => {
            for (idx, chunk) in row.chunks_exact(2).enumerate() {
                let value = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                acc += value * input[idx];
            }
        }
        30 => {
            for (idx, chunk) in row.chunks_exact(2).enumerate() {
                let value = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                acc += value * input[idx];
            }
        }
        8 => {
            for (block_idx, block) in row.chunks_exact(34).enumerate() {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let start = block_idx * 32;
                let take = input.len().saturating_sub(start).min(32);
                for (offset, byte) in block[2..2 + take].iter().enumerate() {
                    acc += scale * ((*byte as i8) as f32) * input[start + offset];
                }
            }
        }
        _ => return None,
    }
    Some(acc)
}

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;

    let out = match exp {
        0 if frac == 0 => sign,
        0 => {
            let mut mant = frac;
            let mut exponent = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exponent -= 1;
            }
            mant &= 0x03ff;
            let exp32 = ((exponent + 127) as u32) << 23;
            sign | exp32 | (mant << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = ((exp - 15 + 127) as u32) << 23;
            sign | exp32 | (frac << 13)
        }
    };
    f32::from_bits(out)
}

fn tensor_row_bytes(tensor_type: u32, row_width: usize) -> Option<usize> {
    match tensor_type {
        0 => row_width.checked_mul(4),
        1 | 30 => row_width.checked_mul(2),
        8 => row_width
            .checked_add(Q8_0_BLOCK - 1)?
            .checked_div(Q8_0_BLOCK)?
            .checked_mul(Q8_0_BLOCK_BYTES),
        _ => None,
    }
}

fn parallel_row_threads(rows: usize) -> usize {
    if rows < PARALLEL_ROW_THRESHOLD {
        return 1;
    }
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    available.min(rows.div_ceil(MIN_ROWS_PER_THREAD)).max(1)
}

fn assign_output(out: &mut Vec<f32>, len: usize) {
    if out.len() != len {
        out.resize(len, 0.0);
    }
}

fn quantize_activation_q8_0(input: &[f32]) -> Option<QuantizedActivationQ8_0> {
    if input.is_empty() {
        return None;
    }
    let blocks = input.len().div_ceil(Q8_0_BLOCK);
    let mut q = vec![0i8; blocks * Q8_0_BLOCK];
    let mut scale = vec![0.0f32; blocks];
    for (block_idx, chunk) in input.chunks(Q8_0_BLOCK).enumerate() {
        let mut amax = 0.0f32;
        let mut vmax = 0.0f32;
        for &value in chunk {
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                vmax = value;
            }
        }
        if amax == 0.0 {
            continue;
        }
        let iscale = -127.0 / vmax;
        scale[block_idx] = 1.0 / iscale;
        let q_block = &mut q[block_idx * Q8_0_BLOCK..(block_idx + 1) * Q8_0_BLOCK];
        for (dst, &value) in q_block.iter_mut().zip(chunk.iter()) {
            *dst = (iscale * value).round().clamp(-128.0, 127.0) as i8;
        }
    }
    Some(QuantizedActivationQ8_0 {
        q,
        scale,
        blocks,
        in_dim: input.len(),
    })
}

fn quantize_activation_q8_0_batch(
    input: &[f32],
    n_rows: usize,
    in_dim: usize,
) -> Option<QuantizedActivationBatchQ8_0> {
    if n_rows == 0 || in_dim == 0 || input.len() != n_rows.checked_mul(in_dim)? {
        return None;
    }
    let blocks = in_dim.div_ceil(Q8_0_BLOCK);
    let mut q = vec![0i8; n_rows.checked_mul(blocks)?.checked_mul(Q8_0_BLOCK)?];
    let mut scale = vec![0.0f32; n_rows.checked_mul(blocks)?];
    for row_idx in 0..n_rows {
        let src = &input[row_idx * in_dim..(row_idx + 1) * in_dim];
        for (block_idx, chunk) in src.chunks(Q8_0_BLOCK).enumerate() {
            let mut amax = 0.0f32;
            let mut vmax = 0.0f32;
            for &value in chunk {
                let abs = value.abs();
                if abs > amax {
                    amax = abs;
                    vmax = value;
                }
            }
            if amax == 0.0 {
                continue;
            }
            let iscale = -127.0 / vmax;
            scale[row_idx * blocks + block_idx] = 1.0 / iscale;
            let q_start = (row_idx * blocks + block_idx) * Q8_0_BLOCK;
            let q_block = &mut q[q_start..q_start + Q8_0_BLOCK];
            for (dst, &value) in q_block.iter_mut().zip(chunk.iter()) {
                *dst = (iscale * value).round().clamp(-128.0, 127.0) as i8;
            }
        }
    }
    Some(QuantizedActivationBatchQ8_0 {
        q,
        scale,
        blocks,
        in_dim,
        n_rows,
    })
}

fn dot_q8_0_tensor_row(
    data: &[u8],
    row_bytes: usize,
    row_idx: usize,
    quant: &QuantizedActivationQ8_0,
) -> Option<f32> {
    let row_start = row_idx.checked_mul(row_bytes)?;
    let row_end = row_start.checked_add(row_bytes)?;
    let row = data.get(row_start..row_end)?;
    let mut acc = 0.0f32;
    for block_idx in 0..quant.blocks {
        let block = row.get(
            block_idx * Q8_0_BLOCK_BYTES..(block_idx + 1) * Q8_0_BLOCK_BYTES,
        )?;
        let w_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let x_scale = quant.scale[block_idx];
        if w_scale == 0.0 || x_scale == 0.0 {
            continue;
        }
        let q_input = &quant.q[block_idx * Q8_0_BLOCK..(block_idx + 1) * Q8_0_BLOCK];
        let start = block_idx * Q8_0_BLOCK;
        let take = quant.in_dim.saturating_sub(start).min(Q8_0_BLOCK);
        let mut block_sum = 0i32;
        for idx in 0..take {
            block_sum += i32::from(block[2 + idx] as i8) * i32::from(q_input[idx]);
        }
        acc += w_scale * x_scale * block_sum as f32;
    }
    Some(acc)
}

fn dot_q8_0_tensor_row_batch(
    data: &[u8],
    row_bytes: usize,
    row_idx: usize,
    quant: &QuantizedActivationBatchQ8_0,
    batch_idx: usize,
) -> Option<f32> {
    let row_start = row_idx.checked_mul(row_bytes)?;
    let row_end = row_start.checked_add(row_bytes)?;
    let row = data.get(row_start..row_end)?;
    let mut acc = 0.0f32;
    for block_idx in 0..quant.blocks {
        let block = row.get(block_idx * Q8_0_BLOCK_BYTES..(block_idx + 1) * Q8_0_BLOCK_BYTES)?;
        let w_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let x_scale = quant.scale[batch_idx * quant.blocks + block_idx];
        if w_scale == 0.0 || x_scale == 0.0 {
            continue;
        }
        let q_start = (batch_idx * quant.blocks + block_idx) * Q8_0_BLOCK;
        let q_input = &quant.q[q_start..q_start + Q8_0_BLOCK];
        let start = block_idx * Q8_0_BLOCK;
        let take = quant.in_dim.saturating_sub(start).min(Q8_0_BLOCK);
        let mut block_sum = 0i32;
        for idx in 0..take {
            block_sum += i32::from(block[2 + idx] as i8) * i32::from(q_input[idx]);
        }
        acc += w_scale * x_scale * block_sum as f32;
    }
    Some(acc)
}

fn dot_q8_0_tensor_row_pair_batch(
    data0: &[u8],
    data1: &[u8],
    row_bytes: usize,
    row_idx: usize,
    quant: &QuantizedActivationBatchQ8_0,
    batch_idx: usize,
) -> Option<(f32, f32)> {
    Some((
        dot_q8_0_tensor_row_batch(data0, row_bytes, row_idx, quant, batch_idx)?,
        dot_q8_0_tensor_row_batch(data1, row_bytes, row_idx, quant, batch_idx)?,
    ))
}

fn dot_q8_0_tensor_row_pair(
    data0: &[u8],
    data1: &[u8],
    row_bytes: usize,
    row_idx: usize,
    quant: &QuantizedActivationQ8_0,
) -> Option<(f32, f32)> {
    let row_start = row_idx.checked_mul(row_bytes)?;
    let row_end = row_start.checked_add(row_bytes)?;
    let row0 = data0.get(row_start..row_end)?;
    let row1 = data1.get(row_start..row_end)?;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    for block_idx in 0..quant.blocks {
        let block0 = row0.get(
            block_idx * Q8_0_BLOCK_BYTES..(block_idx + 1) * Q8_0_BLOCK_BYTES,
        )?;
        let block1 = row1.get(
            block_idx * Q8_0_BLOCK_BYTES..(block_idx + 1) * Q8_0_BLOCK_BYTES,
        )?;
        let x_scale = quant.scale[block_idx];
        if x_scale == 0.0 {
            continue;
        }
        let w0_scale = f16_to_f32(u16::from_le_bytes([block0[0], block0[1]]));
        let w1_scale = f16_to_f32(u16::from_le_bytes([block1[0], block1[1]]));
        let q_input = &quant.q[block_idx * Q8_0_BLOCK..(block_idx + 1) * Q8_0_BLOCK];
        let start = block_idx * Q8_0_BLOCK;
        let take = quant.in_dim.saturating_sub(start).min(Q8_0_BLOCK);
        let mut block_sum0 = 0i32;
        let mut block_sum1 = 0i32;
        for idx in 0..take {
            let x = i32::from(q_input[idx]);
            block_sum0 += i32::from(block0[2 + idx] as i8) * x;
            block_sum1 += i32::from(block1[2 + idx] as i8) * x;
        }
        acc0 += w0_scale * x_scale * block_sum0 as f32;
        acc1 += w1_scale * x_scale * block_sum1 as f32;
    }
    Some((acc0, acc1))
}
