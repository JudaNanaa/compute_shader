struct Params {
  block_size: u32
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> buffer: array<u32>;


@group(0) @binding(2)
var<storage, read_write> sum_buffer: array<u32>;

@compute @workgroup_size(64)
fn cs_main(
  @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
  @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

  let local_id = local_invocation_id.x;
  var group_id = workgroup_id.x;
  let block_size = params.block_size;

  if (group_id == 0) {
    return;
  }
 
	var prev_sum = 0u;
	while (group_id > 0) {
		prev_sum += sum_buffer[group_id - 1u];
		group_id--;
	}

	group_id = workgroup_id.x;

	buffer[group_id * block_size + local_id] += prev_sum;
}
