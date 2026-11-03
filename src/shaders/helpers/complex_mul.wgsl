// Broadcast elementwise complex multiplication.
//
// Complex tensors are stored as interleaved vec2<f32> (real, imag).  The
// strides are expressed in complex elements, so this kernel also works on
// transposed / strided views without materialising them first.

struct ComplexMulMeta {
    numel: u32,
    rank: u32,
    offset_A: u32,
    offset_B: u32,
    out_shape: array<u32, 6>,
    strides_A: array<u32, 6>,
    strides_B: array<u32, 6>,
    conj_A: u32,
    conj_B: u32,
};

@group(0) @binding(0) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> params: ComplexMulMeta;

const WORKGROUP_SIZE: u32 = 256u;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

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
        var offset_A = params.offset_A;
        var offset_B = params.offset_B;

        for (var i = 0u; i < params.rank; i = i + 1u) {
            let dim_idx = params.rank - 1u - i;
            let dim_size = params.out_shape[dim_idx];
            let coord = temp % dim_size;
            temp = temp / dim_size;

            offset_A += coord * params.strides_A[dim_idx];
            offset_B += coord * params.strides_B[dim_idx];
        }

        var a = A[offset_A];
        var b = B[offset_B];
        if (params.conj_A != 0u) { a.y = -a.y; }
        if (params.conj_B != 0u) { b.y = -b.y; }
        C[idx] = complex_mul(a, b);
        idx += stride;
    }
}
