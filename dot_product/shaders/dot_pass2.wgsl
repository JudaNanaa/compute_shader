struct Params {
    len: u32,
}

@group(0) @binding(0)
var<storage, read_write> input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

var<workgroup> g_shared: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let index = global_id.x;


	g_shared[tid] = input[index];

    workgroupBarrier();

    var stride = 32u;
    loop {
        if (tid < stride) {
            g_shared[tid] += g_shared[tid + stride];
        }

        workgroupBarrier();

        if (stride == 1u) {
            break;
        }

        stride = stride / 2u;
    }

    if (tid == 0u) {
        output[workgroup_id.x] = g_shared[0];
    }
}