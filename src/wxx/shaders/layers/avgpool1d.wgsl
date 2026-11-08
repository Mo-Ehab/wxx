struct PoolMeta {
    batch: u32,
    channels: u32,
    in_l: u32,
    out_l: u32,
    kernel: u32,
    stride: u32,
    pad: u32,
    write_indices: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;              // (B, C, Lin)
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;        // (B, C, Lout)
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;  // (B, C, Lout)
@group(0) @binding(3) var<uniform> metamatrix: PoolMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_out = metamatrix.batch * metamatrix.channels * metamatrix.out_l;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total_out) {
        var rem = idx;
        let l_out = rem % metamatrix.out_l; rem /= metamatrix.out_l;
        let c     = rem % metamatrix.channels;
        let b     = rem / metamatrix.channels;

        let base_in = (b * metamatrix.channels + c) * metamatrix.in_l;
        let l_start = i32(l_out * metamatrix.stride) - i32(metamatrix.pad);

        var sum: f32 = 0.0;
        var count: f32 = 0.0;
        var max_val: f32 = -1e38;
        var max_idx: u32 = 0xFFFFFFFFu;

        for (var k = 0u; k < metamatrix.kernel; k++) {
            let il = l_start + i32(k);
            if (il >= 0 && u32(il) < metamatrix.in_l) {
                let in_idx = base_in + u32(il);
                let x_val = X[in_idx];
                sum += x_val;
                count += 1.0;
                if (x_val > max_val) {
                    max_val = x_val;
                    max_idx = in_idx;
                }
            }
        }

        Y[idx] = select(0.0, sum / count, count > 0.0);
        if (metamatrix.write_indices != 0u) {
            indices[idx] = max_idx;
        }

        idx += stride;
    }
}
