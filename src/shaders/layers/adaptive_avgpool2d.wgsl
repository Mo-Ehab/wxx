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

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.batch * params.channels * params.h_out * params.w_out;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        let ow = idx % params.w_out;
        let t0 = idx / params.w_out;
        let oh = t0 % params.h_out;
        let t1 = t0 / params.h_out;
        let c = t1 % params.channels;
        let b = t1 / params.channels;

        let h_start = (oh * params.h_in) / params.h_out;
        let h_end = ((oh + 1u) * params.h_in + params.h_out - 1u) / params.h_out;
        let w_start = (ow * params.w_in) / params.w_out;
        let w_end = ((ow + 1u) * params.w_in + params.w_out - 1u) / params.w_out;

        var acc: f32 = 0.0;
        for (var h = h_start; h < h_end; h = h + 1u) {
            for (var w = w_start; w < w_end; w = w + 1u) {
                let in_idx = ((b * params.channels + c) * params.h_in + h)
                    * params.w_in + w;
                acc += input_data[in_idx];
            }
        }

        let count = (h_end - h_start) * (w_end - w_start);
        output_data[idx] = acc / f32(count);
        idx += stride;
    }
}
