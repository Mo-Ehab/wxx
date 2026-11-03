struct ScalarOpMeta {
    numel: u32,       // Total elements in tensor X
    op_type: u32,     // 0 = Add, 1 = Sub, 2 = Mul, 3 = pow
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
                res = x_val - c;

            }
            case 2u: { // MUL: X * c
                res = x_val * c;
            }
            case 3u: { // MUL: X * c
                res = pow(x_val, c);
            }
            default: {
                res = x_val;
            }
        }

        Y[idx] = res;
        idx += stride;
    }
}