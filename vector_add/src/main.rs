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

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
}

impl Params {
    fn init(len: u32) -> Self {
        Self { len }
    }
}

async fn add_two_vec(first_vec: &Vec<u32>, second_vec: &Vec<u32>) -> anyhow::Result<Vec<u32>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = instance.request_adapter(&Default::default()).await.unwrap();

    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/vector_add.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Introduction Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

	let params = Params::init(first_vec.len() as u32);

    let len_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let first_input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("first input"),
        contents: bytemuck::cast_slice(&first_vec),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let second_input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("second input"),
        contents: bytemuck::cast_slice(&second_vec),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: first_input_buffer.size(),
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("temp"),
        size: first_input_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: first_input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: second_input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: len_uniform.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        // We specified 64 threads per workgroup in the shader, so we need to compute how many
        // workgroups we need to dispatch.
        let num_dispatches = first_vec.len().div_ceil(64) as u32;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_dispatches, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &temp_buffer, 0, output_buffer.size());

    queue.submit([encoder.finish()]);

    let sum_vec: Vec<u32>;

    {
        // The mapping process is async, so we'll need to create a channel to get
        // the success flag for our mapping
        let (tx, rx) = bounded(1);

        // We send the success or failure of our mapping via a callback
        temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap()
        });

        // The callback we submitted to map async will only get called after the
        // device is polled or the queue submitted
        device.poll(wgpu::PollType::wait_indefinitely())?;

        // We check if the mapping was successful here
        rx.recv_async().await??;

        // We then get the bytes that were stored in the buffer
        let output_data = temp_buffer.get_mapped_range(..);

        sum_vec = bytemuck::cast_slice(&output_data).to_vec();
    }

    // We need to unmap the buffer to be able to use it again
    temp_buffer.unmap();

    println!("Success!");

    return Ok(sum_vec);
}

const SIZE: u32 = 1000000;

fn main() {
    let first_vec = create_random_vec(SIZE);
    let second_vec = create_random_vec(SIZE);

    env_logger::init();

    let sum_vec = pollster::block_on(add_two_vec(&first_vec, &second_vec));

    if let Ok(tab) = sum_vec {
        // Somme CPU
        let cpu_sum: Vec<u32> = first_vec
            .iter()
            .zip(second_vec.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Vérification
        if cpu_sum == tab {
            println!("✅ GPU result is correct!");
        } else {
            println!("❌ Mismatch detected!");
        }
    }
}
