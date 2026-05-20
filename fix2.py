import os

def replace_in_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)

reps_attn = [
    ('_model: &GgufModel,', 'model: &GgufModel,'),
]

replace_in_file('src/kernels/attention.rs', reps_attn)

reps_ffn = [
    ('hash_selected_experts(layer, token)', 'hash_selected_experts(model, layer, token)'),
    ('fn hash_selected_experts(layer: &BoundFfnBlock, token: i32) -> Option<Vec<usize>> {', 'fn hash_selected_experts(model: &GgufModel, layer: &BoundFfnBlock, token: i32) -> Option<Vec<usize>> {'),
    ('let data_vec = crate::kernels::matmul::decode_tensor_1d(model, layer.ffn_gate_tid2eid.as_ref()?);\n    let data = Some(data_vec.as_slice())?;', 'let data_vec = crate::kernels::matmul::decode_tensor_1d(model, layer.ffn_gate_tid2eid.as_ref()?)?;\n    let data = data_vec.as_slice();'),
    ('.filter_map(|value| usize::try_from(value).ok())', '.map(|value| value as usize)'),
    ('.copied()', '.copied()'), # no op just to keep formatting
]

replace_in_file('src/kernels/ffn.rs', reps_ffn)
