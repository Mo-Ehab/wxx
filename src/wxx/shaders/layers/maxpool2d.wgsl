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
    write_indices: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;              // (B, C, Hin, Win)
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;        // (B, C, Hout, Wout)
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;  // (B, C, Hout, Wout)
@group(0) @binding(3) var<uniform> metamatrix: PoolMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_out = metamatrix.batch * metamatrix.channels * metamatrix.out_h * metamatrix.out_w;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total_out) {
        // Unpack 1D index -> (b, c, h_out, w_out)
        var rem = idx;
        let w_out = rem % metamatrix.out_w; rem /= metamatrix.out_w;
        let h_out = rem % metamatrix.out_h; rem /= metamatrix.out_h;
        let c     = rem % metamatrix.channels;
        let b     = rem / metamatrix.channels;

        let base_in = (b * metamatrix.channels + c) * (metamatrix.in_h * metamatrix.in_w);
        let h_start = i32(h_out * metamatrix.stride_h) - i32(metamatrix.pad_h);
        let w_start = i32(w_out * metamatrix.stride_w) - i32(metamatrix.pad_w);

        var max_val: f32 = -1e38;
        var max_idx: u32 = 0xFFFFFFFFu;

        for (var kh = 0u; kh < metamatrix.kernel_h; kh++) {
            let ih = h_start + i32(kh);
            if (ih >= 0 && u32(ih) < metamatrix.in_h) {
                let ih_offset = base_in + u32(ih) * metamatrix.in_w;
                for (var kw = 0u; kw < metamatrix.kernel_w; kw++) {
                    let iw = w_start + i32(kw);
                    if (iw >= 0 && u32(iw) < metamatrix.in_w) {
                        let in_idx = ih_offset + u32(iw);
                        let x_val = X[in_idx];
                        if (x_val > max_val) {
                            max_val = x_val;
                            max_idx = in_idx;
                        }
                    }
                }
            }
        }

        Y[idx] = max_val;
        if (metamatrix.write_indices != 0u) {
            indices[idx] = max_idx;
        }

        idx += stride;
    }
}
