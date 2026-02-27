use pollster::FutureExt;
use rand::RngExt;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: usize = 64;

fn create_random_vec(size: u32) -> Vec<u32> {
    let mut rng = rand::rng();
    let mut v = Vec::with_capacity(size as usize);

    for _ in 0..size {
        v.push(rng.random_range(0..3));
    }

    v
}

fn pad_to_power_of_two(input: &[u32]) -> Vec<u32> {
    let next_pow2 = input.len().next_power_of_two();
    let mut padded = input.to_vec();
    padded.resize(next_pow2, 0);
    padded
}

struct ComputePipelineStruct {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

fn up_sweep_pipeline(device: &wgpu::Device, buffer: &wgpu::Buffer) -> ComputePipelineStruct {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/up_sweep.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("up_sweep"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let buffer_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer size buffer"),
        contents: bytemuck::bytes_of(&(WORKGROUP_SIZE as u32)),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
        ],
    });

    ComputePipelineStruct {
        pipeline,
        bind_group,
    }
}

fn down_sweep_pipeline(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    sum_buffer: &wgpu::Buffer,
    tab: &[u32],
) -> ComputePipelineStruct {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/down_sweep.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("down_sweep"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input buffer"),
        contents: bytemuck::cast_slice(tab),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer size buffer"),
        contents: bytemuck::bytes_of(&(WORKGROUP_SIZE as u32)),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sum_buffer.as_entire_binding(),
            },
        ],
    });

    ComputePipelineStruct {
        pipeline,
        bind_group,
    }
}

fn add_pipeline(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    sum_buffer: &wgpu::Buffer,
) -> ComputePipelineStruct {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/addition.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("add_sweep"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let buffer_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer size buffer"),
        contents: bytemuck::bytes_of(&(WORKGROUP_SIZE as u32)),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sum_buffer.as_entire_binding(),
            },
        ],
    });

    ComputePipelineStruct {
        pipeline,
        bind_group,
    }
}

async fn run(tab: &[u32]) -> anyhow::Result<Vec<u32>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    let padded = pad_to_power_of_two(tab);

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input buffer"),
        contents: bytemuck::cast_slice(&padded),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

	let nb_blocks = padded.len().div_ceil(WORKGROUP_SIZE);

    let sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sum buffer"),
        size: (nb_blocks * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let up_sweep = up_sweep_pipeline(&device, &buffer);
    let down_sweep = down_sweep_pipeline(&device, &buffer, &sum_buffer, &padded);
    let addition = add_pipeline(&device, &buffer, &sum_buffer);

    let mut encoder = device.create_command_encoder(&Default::default());

    let dispatch_nb = padded.len().div_ceil(WORKGROUP_SIZE) as u32;

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&up_sweep.pipeline);
        pass.set_bind_group(0, &up_sweep.bind_group, &[]);
        pass.dispatch_workgroups(dispatch_nb, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&down_sweep.pipeline);
        pass.set_bind_group(0, &down_sweep.bind_group, &[]);
        pass.dispatch_workgroups(dispatch_nb, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&addition.pipeline);
        pass.set_bind_group(0, &addition.bind_group, &[]);
        pass.dispatch_workgroups(dispatch_nb, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback"),
        size: (padded.len() * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, readback.size());

    queue.submit(Some(encoder.finish()));

    readback.map_async(wgpu::MapMode::Read, .., |_| {});
    device.poll(wgpu::PollType::wait_indefinitely())?;

    let data = readback.get_mapped_range(..);
    let mut result = bytemuck::cast_slice::<u8, u32>(&data).to_vec();

    result.truncate(tab.len());
    Ok(result)
}

fn cpu_prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut output = Vec::with_capacity(input.len());

    let mut sum = 0;
    for &value in input {
        sum += value;
        output.push(sum);
    }

    output
}

use std::time::Instant;

const SIZE: u32 = 1000000;
const BENCH_ITERS: usize = 5;

fn main() {
    let a = create_random_vec(SIZE);

    // -------------------------
    // CPU BENCH
    // -------------------------
    let mut cpu_time = 0.0;

    for _ in 0..BENCH_ITERS {
        let input = a.clone();
        let start = Instant::now();
        let _ = cpu_prefix_sum(&input);
        cpu_time += start.elapsed().as_secs_f64();
    }

    cpu_time /= BENCH_ITERS as f64;
    println!("CPU avg time: {} sec", cpu_time);

    // -------------------------
    // GPU WARMUP
    // -------------------------
    run(&a).block_on().unwrap();

    // -------------------------
    // GPU BENCH
    // -------------------------
    let mut gpu_time = 0.0;
    let mut gpu_result = Vec::new();

    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        gpu_result = run(&a).block_on().unwrap();
        gpu_time += start.elapsed().as_secs_f64();
    }

    gpu_time /= BENCH_ITERS as f64;
    println!("GPU avg time: {} sec", gpu_time);

    // -------------------------
    // VALIDATION
    // -------------------------
    let cpu_result = cpu_prefix_sum(&a);

    if cpu_result == gpu_result {
        println!("✅ Results match!");
    } else {
        println!("❌ Results DO NOT match!");
    }
    assert_eq!(cpu_result, gpu_result);
}
