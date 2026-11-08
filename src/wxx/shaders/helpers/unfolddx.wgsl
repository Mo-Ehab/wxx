struct UnfoldMeta {
    prefix: u32,   // product of dimensions before `dimension`
    length: u32,   // size of `dimension` in the input
    suffix: u32,   // product of dimensions after `dimension`
    out_len: u32,  // number of windows
    window: u32,   // unfold size
    step: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;       // (P, Lout, S, window)
@group(0) @binding(1) var<storage, read_write> dX: array<f32>; // (P, length, S)
@group(0) @binding(2) var<uniform> metamatrix: UnfoldMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_in = metamatrix.prefix * metamatrix.length * metamatrix.suffix;
    let grid_stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total_in) {
        let s = idx % metamatrix.suffix;
        let n = (idx / metamatrix.suffix) % metamatrix.length;
        let p = idx / (metamatrix.suffix * metamatrix.length);

        var acc: f32 = 0.0;
        for (var w = 0u; w < metamatrix.window; w++) {
            if (n >= w) {
                let pos = n - w;
                if (pos % metamatrix.step == 0u) {
                    let l = pos / metamatrix.step;
                    if (l < metamatrix.out_len) {
                        let out_idx =
                            ((p * metamatrix.out_len + l) * metamatrix.suffix + s)
                            * metamatrix.window + w;
                        acc += dY[out_idx];
                    }
                }
            }
        }

        dX[idx] = acc;
        idx += grid_stride;
    }
}
