// Materialise a strided complex tensor into a contiguous buffer.

struct ComplexContiguousMeta {
    numel: u32,
    rank: u32,
    offset: u32,
    _pad: u32,
    shape: array<u32, 6>,
    strides: array<u32, 6>,
};

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> params: ComplexContiguousMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.numel;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        var temp = idx;
        var offset = params.offset;

        for (var i = 0u; i < params.rank; i = i + 1u) {
            let dim_idx = params.rank - 1u - i;
            let dim_size = params.shape[dim_idx];
            let coord = temp % dim_size;
            temp = temp / dim_size;
            offset += coord * params.strides[dim_idx];
        }

        dst[idx] = src[offset];
        idx += stride;
    }
}
