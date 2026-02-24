struct Params {
    len: u32,
    d: u32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> result_array: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let tid = global_id.x;
    let stride = 1u << params.d;
    let index = (tid + 1u) * stride * 2u - 1u;

    if (index < params.len) {
        result_array[index] += result_array[index - stride];
    }
}
