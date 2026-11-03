struct PoolMeta {
    batch: u32,
    channels: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolMeta;

const WORKGROUP_SIZE: u32 = 256u;

// Each input element independently sums the output cells whose adaptive
// window contains it, so no atomics (or write races) are required.
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.batch * params.channels * params.h_in * params.w_in;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        let iw = idx % params.w_in;
        let t0 = idx / params.w_in;
        let ih = t0 % params.h_in;
        let t1 = t0 / params.h_in;
        let c = t1 % params.channels;
        let b = t1 / params.channels;

        let oh_start = (ih * params.h_out) / params.h_in;
        let oh_end = ((ih + 1u) * params.h_out - 1u) / params.h_in;
        let ow_start = (iw * params.w_out) / params.w_in;
        let ow_end = ((iw + 1u) * params.w_out - 1u) / params.w_in;

        var acc: f32 = 0.0;
        for (var oh = oh_start; oh <= oh_end; oh = oh + 1u) {
            let h_start = (oh * params.h_in) / params.h_out;
            let h_end = ((oh + 1u) * params.h_in + params.h_out - 1u) / params.h_out;
            for (var ow = ow_start; ow <= ow_end; ow = ow + 1u) {
                let w_start = (ow * params.w_in) / params.w_out;
                let w_end = ((ow + 1u) * params.w_in + params.w_out - 1u) / params.w_out;
                let count = (h_end - h_start) * (w_end - w_start);
                let out_idx = ((b * params.channels + c) * params.h_out + oh)
                    * params.w_out + ow;
                acc += grad_output[out_idx] / f32(count);
            }
        }

        grad_input[idx] = acc;
        idx += stride;
    }
}
