// Scatter a slice into a freshly allocated, zero-filled tensor.
//
//   out[..., start : start + length, ...] = grad      (along ``dim``)
//   out[everything else]                  = 0
//
// This fuses the "allocate zeros + assign into a narrow view" pattern used by
// the backward pass of ``narrow`` and basic slicing into a single dispatch,
// which also avoids uploading a host-side zero array.

struct ScatterMeta {
    out_numel: u32,
    rank: u32,
    dim: u32,
    start: u32,
    length: u32,
    grad_offset: u32,
    out_shape: array<u32, 6>,
    grad_strides: array<u32, 6>,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> params: ScatterMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < params.out_numel) {
        var temp_idx = idx;
        var in_slice = true;
        var grad_offset = params.grad_offset;

        for (var i = 0u; i < params.rank; i = i + 1u) {
            let dim_idx = params.rank - 1u - i;
            let dim_size = params.out_shape[dim_idx];
            let coord = temp_idx % dim_size;
            temp_idx = temp_idx / dim_size;

            if (dim_idx == params.dim) {
                if (coord < params.start || coord >= params.start + params.length) {
                    in_slice = false;
                } else {
                    grad_offset += (coord - params.start) * params.grad_strides[dim_idx];
                }
            } else {
                grad_offset += coord * params.grad_strides[dim_idx];
            }
        }

        if (in_slice) {
            Out[idx] = Grad[grad_offset];
        } else {
            Out[idx] = 0.0;
        }
        idx += stride;
    }
}
