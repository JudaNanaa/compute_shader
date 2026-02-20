use flume::bounded;
use rand::RngExt;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;
const SIZE: u32 = 1000000;

fn create_random_vec(size: u32) -> Vec<u32> {
    let mut rng = rand::rng();
    let mut v = Vec::with_capacity(size as usize);

    for _ in 0..size {
        v.push(rng.random_range(0..3));
    }

    return v;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
}

struct ReductionPass {
    bind_group: wgpu::BindGroup,
    dispatch_count: u32,
}

fn create_reduction_passes(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    mut current_len: u32,
    mut input_buffer: wgpu::Buffer,
) -> (Vec<ReductionPass>, wgpu::Buffer) {
    let mut passes = Vec::new();

    while current_len > 1 {
        let dispatch_count = current_len.div_ceil(WORKGROUP_SIZE);

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction Output"),
            size: dispatch_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduction BindGroup"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        passes.push(ReductionPass {
            bind_group,
            dispatch_count,
        });

        input_buffer = output_buffer;
        current_len = dispatch_count;
    }

    return (passes, input_buffer);
}

async fn dot_product_gpu(a: &Vec<u32>, b: &Vec<u32>) -> anyhow::Result<u32> {
    assert_eq!(a.len(), b.len());

    // ========================
    // GPU INIT
    // ========================

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    // ========================
    // BUFFERS
    // ========================

    let params = Params {
        len: a.len() as u32,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ========================
    // PASS 1 (multiply + local reduce)
    // ========================

    let nb_workgroups = (a.len() as u32).div_ceil(WORKGROUP_SIZE);

    let partial_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Partial Buffer"),
        size: nb_workgroups as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader1 = device.create_shader_module(wgpu::include_wgsl!("../shaders/dot_pass1.wgsl"));

    let pipeline1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dot Pass 1"),
        layout: None,
        module: &shader1,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline1.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: partial_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // ========================
    // PASS N (reduction)
    // ========================

    let shader2 = device.create_shader_module(wgpu::include_wgsl!("../shaders/dot_pass2.wgsl"));

    let reduction_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Reduction Pipeline"),
        layout: None,
        module: &shader2,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let (reduction_passes, final_buffer) =
        create_reduction_passes(&device, &reduction_pipeline, nb_workgroups, partial_buffer);

    // ========================
    // ENCODER
    // ========================

    let mut encoder = device.create_command_encoder(&Default::default());

    // Pass 1
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline1);
        pass.set_bind_group(0, &bind_group1, &[]);
        pass.dispatch_workgroups(nb_workgroups, 1, 1);
    }

    // Reduction passes
    for rp in reduction_passes {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&reduction_pipeline);
        pass.set_bind_group(0, &rp.bind_group, &[]);
        pass.dispatch_workgroups(rp.dispatch_count, 1, 1);
    }

    // ========================
    // READBACK
    // ========================

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&final_buffer, 0, &readback, 0, 4);

    queue.submit(Some(encoder.finish()));

	let result;
    {
        let (tx, rx) = bounded(1);

        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            tx.send(r).unwrap();
        });

        device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;

        let data = readback.get_mapped_range(..);
        result = bytemuck::cast_slice::<u8, u32>(&data)[0];
    }

    readback.unmap();

    return Ok(result);
}

fn main() {
    let a = create_random_vec(SIZE);
    let b = create_random_vec(SIZE);

    env_logger::init();

    let gpu_result = pollster::block_on(dot_product_gpu(&a, &b)).unwrap();

    let cpu_result: u32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    println!("CPU result: {}", cpu_result);
    println!("GPU result: {}", gpu_result);

    if cpu_result == gpu_result {
        println!("✅ GPU dot product is correct!");
    } else {
        println!("❌ Mismatch detected!");
    }
}
