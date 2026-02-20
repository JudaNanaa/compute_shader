struct Params {
    len: u32,
}

@group(0) @binding(0)
var<storage, read> array_a: array<u32>;

@group(0) @binding(1)
var<storage, read> array_b: array<u32>;

@group(0) @binding(2)
var<storage, read_write> partial_sums: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> g_shared: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.len) {
        g_shared[tid] = array_a[gid] * array_b[gid];
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

    // Thread 0 écrit la somme du workgroup
    if (tid == 0u) {
        partial_sums[group_id.x] = g_shared[0];
    }
}