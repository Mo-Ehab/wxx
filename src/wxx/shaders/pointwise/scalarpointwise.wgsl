struct ScalarOpMeta {
    numel: u32,       // Total elements in tensor X
    op_type: u32,     // 0 = Add, 1 = Sub, 2 = Mul, 3 = Pow, 4 = Div
    scalar_val: f32,  // The scalar operand 'c'
    reverse: u32,     // 0 = X op c (e.g. X - c), 1 = c op X (e.g. c - X)
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> metamatrix: ScalarOpMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_elements = metamatrix.numel;
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    let c = metamatrix.scalar_val;

    var idx = global_id.x;
    while (idx < total_elements) {
        if (idx >= arrayLength(&Y)) { break; }
        let x_val = X[idx];
        var res: f32 = 0.0;

        switch (metamatrix.op_type) {
            case 0u: { // ADD: X + c
                res = x_val + c;
            }
            case 1u: { // SUB: X - c or c - X
                if (metamatrix.reverse == 1u) {
                    res = c - x_val;
                } else {
                    res = x_val - c;
                }
            }
            case 2u: { // MUL: X * c
                res = x_val * c;
            }
            case 3u: { // POW: X ^ c or c ^ X
                if (metamatrix.reverse == 1u) {
                    // c ^ X; the host rejects non-positive c.
                    res = pow(c, x_val);
                } else if (x_val < 0.0) {
                    // WGSL pow is undefined for a negative base.  The real
                    // value only exists for integer exponents: odd exponents
                    // keep the sign, even ones drop it.  Fractional
                    // exponents return NaN, matching NumPy/PyTorch.
                    if (c == trunc(c)) {
                        let odd = fract(abs(c) * 0.5) > 0.25;
                        let magnitude = pow(-x_val, c);
                        res = select(magnitude, -magnitude, odd);
                    } else {
                        res = bitcast<f32>(0x7fc00000u);
                    }
                } else {
                    res = pow(x_val, c);
                }
            }
            case 4u: { // DIV: X / c or c / X
                if (metamatrix.reverse == 1u) {
                    res = c / x_val;
                } else {
                    res = x_val / c;
                }
            }
            default: {
                res = x_val;
            }
        }

        Y[idx] = res;
        idx += stride;
    }
}