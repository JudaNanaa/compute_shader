use flume::bounded;
use rand::RngExt;
use wgpu::util::DeviceExt;

fn create_random_vec(size: u32) -> Vec<u32> {
    let mut dest = Vec::with_capacity(size as usize);
    let mut rng = rand::rng();

    for _ in 0..size {
        let rng_value = rng.random_range(0..size);
        dest.push(rng_value);
    }
    return dest;
}

// #[repr(C)]
// #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// struct Params {
//     len: u32,
// }

// impl Params {
//     fn init(len: u32) -> Self {
//         Self { len }
//     }
// }

async fn dot_product_gpu(first_vec: &Vec<u32>, second_vec: &Vec<u32>) -> anyhow::Result<u32> {
    assert_eq!(first_vec.len(), second_vec.len());

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    // ========================
    // PARAMS
    // ========================
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        len: u32,
    }

    let params = Params {
        len: first_vec.len() as u32,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ========================
    // INPUT BUFFERS
    // ========================
    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A buffer"),
        contents: bytemuck::cast_slice(first_vec),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B buffer"),
        contents: bytemuck::cast_slice(second_vec),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ========================
    // PASS 1
    // ========================
    let workgroup_size = 64u32;
    let nb_workgroups = (first_vec.len() as u32).div_ceil(workgroup_size);

    let partial_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Partial buffer"),
        size: nb_workgroups as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader1 = device.create_shader_module(wgpu::include_wgsl!("../shaders/dot_pass1.wgsl"));

    let pipeline1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dot pass 1"),
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
    // PASS 2
    // ========================
    let final_params = Params { len: nb_workgroups };

    let final_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Final params"),
        contents: bytemuck::bytes_of(&final_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result buffer"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader2 = device.create_shader_module(wgpu::include_wgsl!("../shaders/dot_pass2.wgsl"));

    let pipeline2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dot pass 2"),
        layout: None,
        module: &shader2,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline2.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: partial_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: final_params_buffer.as_entire_binding(),
            },
        ],
    });

    // ========================
    // ENCODER
    // ========================
    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline1);
        pass.set_bind_group(0, &bind_group1, &[]);
        pass.dispatch_workgroups(nb_workgroups, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline2);
        pass.set_bind_group(0, &bind_group2, &[]);
        pass.dispatch_workgroups(1, 1, 1); // 1 workgroup suffit
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback, 0, 4);

    queue.submit(Some(encoder.finish()));

    let result: u32;

    {
        // The mapping process is async, so we'll need to create a channel to get
        // the success flag for our mapping
        let (tx, rx) = bounded(1);

        // We send the success or failure of our mapping via a callback
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap()
        });

        // The callback we submitted to map async will only get called after the
        // device is polled or the queue submitted
        device.poll(wgpu::PollType::wait_indefinitely())?;

        // We check if the mapping was successful here
        rx.recv()??;

        // We then get the bytes that were stored in the buffer
        let output_data = readback.get_mapped_range(..);

        result = bytemuck::cast_slice(&output_data).to_vec()[0];
    }

    // We need to unmap the buffer to be able to use it again
    readback.unmap();

    Ok(result)
}

const SIZE: u32 = 100;

fn main() {
    let first_vec = create_random_vec(SIZE);
    let second_vec = create_random_vec(SIZE);

    env_logger::init();

    // Appel GPU
    let gpu_result = pollster::block_on(dot_product_gpu(&first_vec, &second_vec));

    if let Ok(gpu_value) = gpu_result {
        // Calcul CPU
        let cpu_value: u32 = first_vec
            .iter()
            .zip(second_vec.iter())
            .map(|(a, b)| a * b)
            .sum();

        println!("CPU result: {}", cpu_value);
        println!("GPU result: {}", gpu_value);

        if cpu_value == gpu_value {
            println!("✅ GPU dot product is correct!");
        } else {
            println!("❌ Mismatch detected!");
        }
    } else {
        println!("❌ GPU computation failed.");
    }
}
