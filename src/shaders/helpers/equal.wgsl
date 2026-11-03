struct EqualMeta {
    mode: u32,      // 0: f32/f32, 1: u32/u32, 2: f32/scalar, 3: u32/scalar
    number: u32,    // scalar bits (f32 bit pattern or u32 value)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> dataA : array<u32>;
@group(0) @binding(1) var<storage, read> dataB : array<u32>;
@group(0) @binding(2) var<storage, read_write> dataOut : array<f32>;
@group(0) @binding(3) var<uniform> params : EqualMeta;

const WORKGROUP_SIZE : u32 = 256u;

fn is_equal(a: u32, b: u32, mode: u32) -> bool {
    switch (mode) {
        case 0u: {
            return bitcast<f32>(a) == bitcast<f32>(b);
        }
        case 1u: {
            return a == b;
        }
        case 2u: {
            return bitcast<f32>(a) == bitcast<f32>(params.number);
        }
        default: {
            return a == params.number;
        }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>
) {
    let total_elements = arrayLength(&dataOut);
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var i = global_id.x;
    while (i < total_elements) {
        let a = dataA[i];
        // For scalar modes dataB is unused, but keeping the binding uniform
        // lets all four comparison modes share one pipeline layout.
        let b = dataB[i];
        dataOut[i] = select(0.0, 1.0, is_equal(a, b, params.mode));
        i += stride;
    }
}
