struct Params {
    len: u32,
}

@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> result: u32;

@group(0) @binding(2)
var<uniform> params: Params;



var<workgroup> g_shared: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.len) {
        g_shared[tid] = data[gid];
    } else {
        g_shared[tid] = 0u;
    }

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
        result = g_shared[0];
    }
}