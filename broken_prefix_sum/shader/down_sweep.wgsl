struct Params {
  block_size: u32
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> buffer: array<u32>;


@group(0) @binding(2)
var<storage> input_buffer: array<u32>;


@group(0) @binding(3)
var<storage, read_write> sum_buffer: array<u32>;


var<workgroup> sharedd: array<u32, 64>;

@compute @workgroup_size(64)
fn cs_main(
  @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
  @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

  let local_id = local_invocation_id.x;
  let group_id = workgroup_id.x;
  let block_size = params.block_size;

sharedd[local_id] = buffer[group_id * block_size + local_id];
workgroupBarrier();

if (local_id == 0u) {
    sum_buffer[group_id] = sharedd[block_size - 1u];
    sharedd[block_size - 1u] = 0u;
}
workgroupBarrier();

var stride = block_size >> 1u;

while (stride > 0u) {

    let index = (local_id + 1u) * 2u * stride - 1u;

    if (index < block_size) {
        let temp = sharedd[index - stride];
        sharedd[index - stride] = sharedd[index];
        sharedd[index] += temp;
    }

    workgroupBarrier();
    stride >>= 1u;
}

// Synchronisation finale
workgroupBarrier();

buffer[group_id * block_size + local_id] = sharedd[local_id] + input_buffer[group_id * block_size + local_id];
}
