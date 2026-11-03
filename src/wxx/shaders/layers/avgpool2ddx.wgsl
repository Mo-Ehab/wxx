struct PoolMeta {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;       // (B, C, Hout, Wout)
@group(0) @binding(1) var<storage, read_write> dX: array<f32>; // (B, C, Hin, Win)
@group(0) @binding(2) var<uniform> metamatrix: PoolMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_in = metamatrix.batch * metamatrix.channels * metamatrix.in_h * metamatrix.in_w;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total_in) {
        var rem = idx;
        let w_in = rem % metamatrix.in_w; rem /= metamatrix.in_w;
        let h_in = rem % metamatrix.in_h; rem /= metamatrix.in_h;
        let c    = rem % metamatrix.channels;
        let b    = rem / metamatrix.channels;

        var dx_acc: f32 = 0.0;

        let kh_i32 = i32(metamatrix.kernel_h);
        let kw_i32 = i32(metamatrix.kernel_w);
        let sh_i32 = i32(metamatrix.stride_h);
        let sw_i32 = i32(metamatrix.stride_w);
        let ph_i32 = i32(metamatrix.pad_h);
        let pw_i32 = i32(metamatrix.pad_w);

        let h_padded = i32(h_in) + ph_i32;
        let w_padded = i32(w_in) + pw_i32;

        let h_out_min_i = max(0, (h_padded - kh_i32 + sh_i32) / sh_i32);
        let h_out_max_i = min(i32(metamatrix.out_h) - 1, h_padded / sh_i32);

        let w_out_min_i = max(0, (w_padded - kw_i32 + sw_i32) / sw_i32);
        let w_out_max_i = min(i32(metamatrix.out_w) - 1, w_padded / sw_i32);

        if (h_out_min_i <= h_out_max_i && w_out_min_i <= w_out_max_i) {
            let base_out = (b * metamatrix.channels + c) * (metamatrix.out_h * metamatrix.out_w);

            for (var ho = u32(h_out_min_i); ho <= u32(h_out_max_i); ho++) {
                let h_start = i32(ho * metamatrix.stride_h) - ph_i32;
                let row_offset = base_out + ho * metamatrix.out_w;

                for (var wo = u32(w_out_min_i); wo <= u32(w_out_max_i); wo++) {
                    let w_start = i32(wo * metamatrix.stride_w) - pw_i32;

                    // O(1) Analytic area calculation (replaces 2D nested loop)
                    let valid_h_start = max(0, h_start);
                    let valid_h_end   = min(i32(metamatrix.in_h), h_start + kh_i32);
                    let valid_w_start = max(0, w_start);
                    let valid_w_end   = min(i32(metamatrix.in_w), w_start + kw_i32);

                    let valid_count = f32((valid_h_end - valid_h_start) * (valid_w_end - valid_w_start));

                    if (valid_count > 0.0) {
                        dx_acc += dY[row_offset + wo] / valid_count;
                    }
                }
            }
        }

        dX[idx] = dx_acc;
        idx += stride;
    }
}