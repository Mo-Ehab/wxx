struct DropoutMeta {
    numel: u32,
    seed: u32,
    p: f32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read_write> dX: array<f32>;
@group(0) @binding(2) var<uniform> metamatrix: DropoutMeta;

const WORKGROUP_SIZE: u32 = 256u;

fn hash_u32(input_val: u32) -> u32 {
    var x = input_val;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    var idx = global_id.x;

    while (idx < metamatrix.numel) {
        let random_value = f32(hash_u32(idx ^ metamatrix.seed) >> 8u) * (1.0 / 16777216.0);
        if (random_value >= metamatrix.p) {
            dX[idx] = dY[idx] * metamatrix.scale;
        } else {
            dX[idx] = 0.0;
        }
        idx += stride;
    }
}
