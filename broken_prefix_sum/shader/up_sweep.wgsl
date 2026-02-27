struct Params {
  block_size: u32
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> buffer: array<u32>;

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
  
  var stride = 1u;
  while (stride < block_size) {
	workgroupBarrier();
	
	let index = (local_id + 1u) * 2u * stride - 1u;
	if (index < block_size) {
		sharedd[index] += sharedd[index - stride];
	}
	stride <<= 1;
  }
  buffer[group_id * block_size + local_id] = sharedd[local_id];
}

