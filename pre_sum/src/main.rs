use flume::bounded;
use rand::RngExt;
use wgpu::util::DeviceExt;

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

async fn dot_product_gpu(a: &[u32]) -> anyhow::Result<Vec<u32>> {
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

    println!("size a_buffer == {}", a_buffer.size());

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

    println!("result Buffer size == {}", result_buffer.size());
    println!("a.len == {}", a.len() as u64 * 4);

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

use std::time::Instant;

const ITERATIONS: usize = 5;

fn main() {
    env_logger::init();

    let a = create_random_vec(SIZE);

    println!("🔥 Warming up GPU...");
    let result = pollster::block_on(dot_product_gpu(&a)).unwrap();
    println!("vec au debut = {:?}", a);
    println!("vec a la fin = {:?}", result);
}
