pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub(crate) fn swiglu(gate: &[f32], up: &[f32], clamp: f32) -> Option<Vec<f32>> {
    if gate.len() != up.len() {
        return None;
    }
    let mut out = vec![0.0; gate.len()];
    swiglu_into(&mut out, gate, up, clamp)?;
    Some(out)
}

pub(crate) fn swiglu_into(dst: &mut Vec<f32>, gate: &[f32], up: &[f32], clamp: f32) -> Option<()> {
    if gate.len() != up.len() {
        return None;
    }
    if dst.len() != gate.len() {
        dst.resize(gate.len(), 0.0);
    }
    for ((dst, &g), &u) in dst.iter_mut().zip(gate.iter()).zip(up.iter()) {
        let gate_clamped = if clamp > 1.0e-6 { g.min(clamp) } else { g };
        let up_clamped = if clamp > 1.0e-6 {
            u.clamp(-clamp, clamp)
        } else {
            u
        };
        *dst = silu(gate_clamped) * up_clamped;
    }
    Some(())
}
