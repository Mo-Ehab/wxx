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

@group(0) @binding(0) var<storage, read> dY: array<f32>;       // (B, C, Lout)
@group(0) @binding(1) var<storage, read> indices: array<u32>;  // (B, C, Lout)
@group(0) @binding(2) var<storage, read_write> dX: array<f32>; // (B, C, Lin)
@group(0) @binding(3) var<uniform> metamatrix: PoolMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_in = metamatrix.batch * metamatrix.channels * metamatrix.in_l;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total_in) {
        var rem = idx;
        let l_in = rem % metamatrix.in_l; rem /= metamatrix.in_l;
        let c    = rem % metamatrix.channels;
        let b    = rem / metamatrix.channels;

        var dx_acc: f32 = 0.0;

        let k_i32 = i32(metamatrix.kernel);
        let s_i32 = i32(metamatrix.stride);
        let p_i32 = i32(metamatrix.pad);

        let l_padded = i32(l_in) + p_i32;

        // Signed bounds check prevents integer underflow.
        let l_out_min_i = max(0, (l_padded - k_i32 + s_i32) / s_i32);
        let l_out_max_i = min(i32(metamatrix.out_l) - 1, l_padded / s_i32);

        if (l_out_min_i <= l_out_max_i) {
            let base_out = (b * metamatrix.channels + c) * metamatrix.out_l;

            for (var lo = u32(l_out_min_i); lo <= u32(l_out_max_i); lo++) {
                if (indices[base_out + lo] == idx) {
                    dx_acc += dY[base_out + lo];
                }
            }
        }

        dX[idx] = dx_acc;
        idx += stride;
    }
}
