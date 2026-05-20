import os

def replace_in_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)

reps_attn = [
    ('block.attention.hc_attn_fn_data.as_deref(),', 'None,'),
    ('block.attention.hc_attn_scale_data.as_deref(),', 'None,'),
    ('block.attention.hc_attn_base_data.as_deref(),', 'None,'),
    ('ffn.hc_ffn_fn_data.as_deref(),', 'None,'),
    ('ffn.hc_ffn_scale_data.as_deref(),', 'None,'),
    ('ffn.hc_ffn_base_data.as_deref(),', 'None,'),
    ('(&weights.output_norm, &weights.output_norm_data)', '(&weights.output_norm, &None::<Vec<f32>> /* output_norm_data */)'),
    ('rms_norm_with_decoded_weight_into(&mut scratch.attn_norm, &layer.attn_norm_data, hidden)?;', 'let norm_weight = crate::kernels::matmul::decode_tensor_1d(model, &layer.attn_norm)?;\n    rms_norm_with_decoded_weight_into(&mut scratch.attn_norm, &norm_weight, hidden)?;'),
    ('rms_norm_with_decoded_weight_into(qr_norm_buf, &layer.attn_q_a_norm_data, qr)?;', 'let qr_norm_weight = crate::kernels::matmul::decode_tensor_1d(model, &layer.attn_q_a_norm)?;\n    rms_norm_with_decoded_weight_into(qr_norm_buf, &qr_norm_weight, qr)?;'),
    ('rms_norm_with_decoded_weight(&layer.attn_kv_a_norm_data, &kv_raw)', 'rms_norm_with_decoded_weight(&crate::kernels::matmul::decode_tensor_1d(model, &layer.attn_kv_a_norm)?, &kv_raw)'),
    ('layer.attn_sinks_data.len()', 'usize::try_from(*layer.attn_sinks.dims.first()?).unwrap_or(0)'),
    ('layer.attn_sinks_data[h]', 'crate::kernels::matmul::decode_tensor_1d(model, &layer.attn_sinks)?[h]'),
]

replace_in_file('src/kernels/attention.rs', reps_attn)

reps_ffn = [
    ('layer.hc_ffn_fn_data.as_deref(),', 'None,'),
    ('layer.hc_ffn_scale_data.as_deref(),', 'None,'),
    ('layer.hc_ffn_base_data.as_deref(),', 'None,'),
    ('rms_norm_with_decoded_weight(&layer.ffn_norm_data, hidden)', 'rms_norm_with_decoded_weight(&crate::kernels::matmul::decode_tensor_1d(model, &layer.ffn_norm)?, hidden)'),
    ('layer.ffn_gate_tid2eid_data.is_some()', 'layer.ffn_gate_tid2eid.is_some()'),
    ('let data = layer.ffn_gate_tid2eid_data.as_deref()?;', 'let data_vec = crate::kernels::matmul::decode_tensor_1d(model, layer.ffn_gate_tid2eid.as_ref()?);\n    let data = Some(data_vec.as_slice())?;'),
    ('if let Some(bias) = layer.ffn_exp_probs_b_data.as_deref() {', 'if let Some(bias) = layer.ffn_exp_probs_b.as_ref().and_then(|t| crate::kernels::matmul::decode_tensor_1d(model, t)) {'),
]

replace_in_file('src/kernels/ffn.rs', reps_ffn)

reps_hc = [
    ('weights.output_hc_fn_data.as_deref()', 'None'),
    ('weights.output_hc_scale_data.is_none()', 'true'),
    ('weights.output_hc_scale_data.as_deref().or(scale_owned.as_deref())?', 'scale_owned.as_deref()?'),
    ('weights.output_hc_base_data.is_none()', 'true'),
    ('weights.output_hc_base_data.as_deref().or(base_owned.as_deref())?', 'base_owned.as_deref()?'),
]
replace_in_file('src/kernels/hc.rs', reps_hc)
