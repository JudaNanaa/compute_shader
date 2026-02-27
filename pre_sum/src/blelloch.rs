use wgpu::util::DeviceExt;

use crate::init_wgpu;

const WORKGROUP_SIZE: usize = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
    d: u32,
}

fn pad_to_power_of_two(input: &[u32]) -> Vec<u32> {
    let next_pow2 = input.len().next_power_of_two();
    let mut padded = input.to_vec();
    padded.resize(next_pow2, 0);
    padded
}

fn create_buffer_init(
    device: &wgpu::Device,
    label: Option<&str>,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label,
        contents,
        usage,
    })
}

struct ComputePipelineParams {
    bind_group: wgpu::BindGroup,
    dispatch_count: u32,
}

fn up_sweep_pipeline(
    device: &wgpu::Device,
    tab: &[u32],
) -> (
    wgpu::ComputePipeline,
    Vec<ComputePipelineParams>,
    wgpu::Buffer,
) {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/up_sweep.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("up_sweep"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let buffer = create_buffer_init(
        device,
        Some("scan_buffer"),
        bytemuck::cast_slice(tab),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let levels = tab.len().ilog2();
    let mut passes = Vec::with_capacity(levels as usize);

    for d in 0..levels {
        let stride = 1 << d;
        let active_threads = tab.len() / (2 * stride);
        let dispatch_count = active_threads.div_ceil(WORKGROUP_SIZE) as u32;

        let params = Params {
            len: tab.len() as u32,
            d,
        };

        let params_buffer = create_buffer_init(
            device,
            Some("up_params"),
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("up_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });

        passes.push(ComputePipelineParams {
            bind_group,
            dispatch_count,
        });
    }

    (pipeline, passes, buffer)
}

fn clear_pipeline(
    device: &wgpu::Device,
    buffer: wgpu::Buffer,
    tab: &[u32],
) -> (wgpu::ComputePipeline, ComputePipelineParams, wgpu::Buffer) {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/clear.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("clear"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let len_buffer = create_buffer_init(
        device,
        Some("Params buffer"),
        bytemuck::bytes_of(&tab.len()),
        wgpu::BufferUsages::UNIFORM,
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clear BindGroup"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: len_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
        ],
    });

    let compute_pipeline_params = ComputePipelineParams {
        dispatch_count: 1,
        bind_group,
    };
    (pipeline, compute_pipeline_params, buffer)
}

fn down_sweep_pipeline(
    device: &wgpu::Device,
    buffer: wgpu::Buffer,
    tab: &[u32],
) -> (
    wgpu::ComputePipeline,
    Vec<ComputePipelineParams>,
    wgpu::Buffer,
) {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/down_sweep.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("down_sweep"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let levels = tab.len().ilog2();
    let mut passes = Vec::with_capacity(levels as usize);

    for d in (0..levels).rev() {
        let stride = 1 << d;
        let active_threads = tab.len() / (2 * stride);
        let dispatch_count = active_threads.div_ceil(WORKGROUP_SIZE) as u32;

        let params = Params {
            len: tab.len() as u32,
            d,
        };

        let params_buffer = create_buffer_init(
            device,
            Some("down_params"),
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("down_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });

        passes.push(ComputePipelineParams {
            bind_group,
            dispatch_count,
        });
    }

    (pipeline, passes, buffer)
}

fn addition_pipeline(
    device: &wgpu::Device,
    buffer: wgpu::Buffer,
    tab: &[u32],
) -> (wgpu::ComputePipeline, ComputePipelineParams, wgpu::Buffer) {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/addition.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("addition"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let len_buffer = create_buffer_init(
        device,
        Some("Params buffer"),
        bytemuck::bytes_of(&(tab.len() as u32)),
        wgpu::BufferUsages::UNIFORM,
    );

    let output_buffer = create_buffer_init(
        device,
        Some("output buffer"),
        bytemuck::cast_slice(tab),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("addition BindGroup"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: len_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let compute_pipeline_params = ComputePipelineParams {
        dispatch_count: tab.len().div_ceil(WORKGROUP_SIZE) as u32,
        bind_group,
    };
    (pipeline, compute_pipeline_params, output_buffer)
}

pub async fn blelloch_prefix_sum(tab: &[u32]) -> anyhow::Result<Vec<u32>> {
    let (device, queue) = init_wgpu::init_wgpu().await;

    let padded = pad_to_power_of_two(tab);

    let (up_sweep_pipeline, up_sweep_pipeline_parms, output_buffer) =
        up_sweep_pipeline(&device, &padded);

    let (clear_pipeline, clear_pipeline_parms, output_buffer) =
        clear_pipeline(&device, output_buffer, &padded);

    let (down_sweep_pipeline, down_sweep_pipeline_parms, output_buffer) =
        down_sweep_pipeline(&device, output_buffer, &padded);

    let (addition_pipeline, addition_pipeline_parms, output_buffer) =
        addition_pipeline(&device, output_buffer, &padded);

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&up_sweep_pipeline);
        for elem in &up_sweep_pipeline_parms {
            pass.set_bind_group(0, &elem.bind_group, &[]);
            pass.dispatch_workgroups(elem.dispatch_count, 1, 1);
        }
    }

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&clear_pipeline);
        pass.set_bind_group(0, &clear_pipeline_parms.bind_group, &[]);
        pass.dispatch_workgroups(clear_pipeline_parms.dispatch_count, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&down_sweep_pipeline);
        for elem in &down_sweep_pipeline_parms {
            pass.set_bind_group(0, &elem.bind_group, &[]);
            pass.dispatch_workgroups(elem.dispatch_count, 1, 1);
        }
    }

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&addition_pipeline);
        pass.set_bind_group(0, &addition_pipeline_parms.bind_group, &[]);
        pass.dispatch_workgroups(addition_pipeline_parms.dispatch_count, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback"),
        size: (padded.len() * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, readback.size());

    queue.submit(Some(encoder.finish()));

    readback.map_async(wgpu::MapMode::Read, .., |_| {});
    device.poll(wgpu::PollType::wait_indefinitely())?;

    let data = readback.get_mapped_range(..);
    let mut result = bytemuck::cast_slice::<u8, u32>(&data).to_vec();

    result.truncate(tab.len());
    Ok(result)
}
