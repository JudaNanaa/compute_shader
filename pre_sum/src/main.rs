use flume::bounded;
use rand::RngExt;
use wgpu::util::DeviceExt;

mod blelloch;
mod init_wgpu;

const WORKGROUP_SIZE: u32 = 64;
const SIZE: u32 = 100000;

fn create_random_vec(size: u32) -> Vec<u32> {
    let mut rng = rand::rng();
    let mut v = Vec::with_capacity(size as usize);

    for _ in 0..size {
        v.push(rng.random_range(0..3));
    }

    v
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
}

async fn naive_prefix_sum(a: &[u32]) -> anyhow::Result<Vec<u32>> {
    // ========================
    // GPU INIT
    // ========================

    let (device, queue) = init_wgpu::init_wgpu().await;

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

    // ========================
    // PASS 1
    // ========================

    let nb_workgroups = (a.len() as u32).div_ceil(WORKGROUP_SIZE);

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Partial Buffer"),
        size: a.len() as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader1 = device.create_shader_module(wgpu::include_wgsl!("../shaders/pre_sum.wgsl"));

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
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });

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

    // ========================
    // READBACK
    // ========================

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback"),
        size: a.len() as u64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback, 0, readback.size());

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
        result = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    }

    readback.unmap();

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

const BENCH_ITERS: usize = 5;

fn main() {
    env_logger::init();

    let a = create_random_vec(SIZE);

    println!("a == {:?}", a);
    println!("🔥 Warming up GPU...");
    pollster::block_on(blelloch::blelloch_prefix_sum(&a)).unwrap();
    // pollster::block_on(naive_prefix_sum(&a)).unwrap();

    println!("🚀 Benchmarking ({} iterations)...", BENCH_ITERS);

    // =========================
    // BLELLOCH
    // =========================
    let start = Instant::now();

    let mut blelloch_result = Vec::new();
    for _ in 0..BENCH_ITERS {
        blelloch_result = pollster::block_on(blelloch::blelloch_prefix_sum(&a)).unwrap();
    }

    let blelloch_time = start.elapsed().as_secs_f64() / BENCH_ITERS as f64;

    // =========================
    // CPU optimal
    // =========================
    let start = Instant::now();

    let mut cpu_result = Vec::new();
    for _ in 0..BENCH_ITERS {
        cpu_result = cpu_prefix_sum(&a);
    }

    let cpu_time = start.elapsed().as_secs_f64() / BENCH_ITERS as f64;

    // =========================
    // NAIVE
    // =========================
    // let start = Instant::now();
    //
    // let mut naive_result = Vec::new();
    // for _ in 0..BENCH_ITERS {
    //     naive_result = pollster::block_on(naive_prefix_sum(&a)).unwrap();
    // }
    //
    // let naive_time = start.elapsed().as_secs_f64() / BENCH_ITERS as f64;

    // =========================
    // VALIDATION
    // =========================
    // assert_eq!(naive_result, cpu_result);
    assert_eq!(blelloch_result, cpu_result);

    println!();
    println!("========== RESULTS ==========");
    println!("Blelloch avg : {:.6} sec", blelloch_time);
    // println!("Naive    avg : {:.6} sec", naive_time);
    println!("CPU    avg : {:.6} sec", cpu_time);
    // println!("Speedup      : {:.2}x", naive_time / blelloch_time);
    println!("Speedup (CPU/GPU): {:.2}x", cpu_time / blelloch_time);
}
