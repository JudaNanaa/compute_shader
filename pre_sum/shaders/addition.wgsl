struct Params {
    len: u32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> input_array: array<u32>;

@group(0) @binding(2)
var<storage, read_write> result_array: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let thread_id = global_id.x;
    if (thread_id < params.len) {
     result_array[thread_id] = input_array[thread_id] + result_array[thread_id]; 
  }
}
