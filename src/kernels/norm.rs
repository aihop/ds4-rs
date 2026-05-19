use crate::gguf::GgufModel;
use crate::kernels::matmul::decode_tensor_1d;
use crate::weights::BoundTensor;

pub(crate) fn rms_norm_with_decoded_weight(weight: &[f32], input: &[f32]) -> Option<Vec<f32>> {
    if weight.len() != input.len() || input.is_empty() {
        return None;
    }
    let mut out = vec![0.0; input.len()];
    rms_norm_with_decoded_weight_into(&mut out, weight, input)?;
    Some(out)
}

pub(crate) fn rms_norm_with_decoded_weight_into(
    out: &mut Vec<f32>,
    weight: &[f32],
    input: &[f32],
) -> Option<()> {
    if weight.len() != input.len() || input.is_empty() {
        return None;
    }
    let mut ss = 0.0f64;
    for value in input {
        ss += f64::from(*value) * f64::from(*value);
    }
    let scale = 1.0f32 / ((ss as f32 / input.len() as f32) + 1e-6).sqrt();
    if out.len() != input.len() {
        out.resize(input.len(), 0.0);
    }
    for ((dst, x), w) in out.iter_mut().zip(input.iter()).zip(weight.iter()) {
        *dst = x * scale * w;
    }
    Some(())
}

pub(crate) fn rms_norm_weight(
    model: &GgufModel,
    tensor: &BoundTensor,
    input: &[f32],
) -> Option<Vec<f32>> {
    if tensor.dims.len() != 1 || usize::try_from(*tensor.dims.first()?).ok()? != input.len() {
        return None;
    }
    let weight = decode_tensor_1d(model, tensor)?;
    rms_norm_with_decoded_weight(&weight, input)
}
