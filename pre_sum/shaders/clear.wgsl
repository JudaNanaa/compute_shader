@group(0) @binding(0)
var<uniform> len: u32;

@group(0) @binding(1)
var<storage, read_write> result_array: array<u32>;

@compute @workgroup_size(64)
fn main() {
  result_array[len - 1] = 0; 
}
