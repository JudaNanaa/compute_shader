use std::sync::Arc;

use anyhow::Result;
use image::EncodableLayout;
use wgpu::util::DeviceExt;

use crate::{
    gpu_context::{self, GpuContext},
    image_exportrer::ImageExporter,
    point::Point,
    texture,
    texture_builder::TextureBuilder,
};

const WORKGROUP_SIZE: u32 = 8;

struct BoundPipeline {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

impl BoundPipeline {
    fn new(
        device: &wgpu::Device,
        shader_module: &wgpu::ShaderModule,
        bind_group_layout: &wgpu::BindGroupLayout,
        bind_group: wgpu::BindGroup,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute pipeline"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        Self {
            bind_group,
            pipeline,
        }
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass, width: u32, height: u32) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            width.div_ceil(WORKGROUP_SIZE),
            height.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }
}

pub struct ImagePipeline {
    pub ctx: Arc<gpu_context::GpuContext>,
    input_texture: texture::Texture,
    output_texture: texture::Texture,
    overlay_texture: texture::Texture,
    points_buffer: wgpu::Buffer,
    file_output_name: String,
    overlay_bound_pipeline: BoundPipeline,
    blend_bound_pipeline: BoundPipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    color: [f32; 4],
    size: f32,
    shape: u32,
    _padding: [u32; 2],
}

impl Params {
    pub fn new(color: [f32; 4], size: f32, shape: u32) -> Self {
        Self {
            color,
            size,
            shape,
            _padding: [0, 0],
        }
    }
}

impl ImagePipeline {
    pub async fn new(ctx: Arc<GpuContext>) -> Result<Self> {
        let device = &ctx.device;
        let queue = &ctx.queue;

        let file_input = ctx.debug_config.input_file.clone();
        let file_output = ctx.debug_config.output_file.clone();

        let input_texture = texture::Texture::from_bytes(
            &device,
            &queue,
            std::fs::read(file_input)?.as_bytes(),
            "input file texture",
        )?;

        let size = input_texture.texture.size();
        let output_texture = TextureBuilder::new(&device, size).build(
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            "output texture",
        );
        let overlay_texture = TextureBuilder::new(&device, size).build(
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            "overlay texture",
        );

        let points = Point::random_points(1000, 1000.0, 1000.0);

        let params = Params::new(
            ctx.debug_config.get_color(),
            ctx.debug_config.get_size(),
            ctx.debug_config.get_current_shape_u32(),
        );

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Points buffer"),
            contents: bytemuck::cast_slice(&points),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let overlay_bound_pipeline = Self::create_overlay_pipeline(
            &device,
            &params_buffer,
            &overlay_texture,
            &points_buffer,
        );
        let blend_bound_pipeline =
            Self::create_blend_pipeline(&device, &input_texture, &overlay_texture, &output_texture);

        Ok(Self {
            ctx,
            input_texture,
            output_texture,
            overlay_texture,
            points_buffer,
            file_output_name: file_output.to_string(),
            overlay_bound_pipeline,
            blend_bound_pipeline,
        })
    }

    pub async fn image_dot(&self) -> Result<()> {
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());

        let image_width = self.input_texture.texture.width();
        let image_height = self.input_texture.texture.height();

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            // pass.set_bind_group(0, &self.overlay_bound_pipeline.bind_group, &[]);
            self.overlay_bound_pipeline
                .dispatch(&mut pass, image_width, image_height);
        }

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.blend_bound_pipeline
                .dispatch(&mut pass, image_width, image_height);
        }

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = image_width * bytes_per_pixel;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;

        let output_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buffer"),
            size: padded_bytes_per_row as u64 * image_height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(image_height),
                },
            },
            wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth_or_array_layers: 1,
            },
        );

        self.ctx.queue.submit(Some(encoder.finish()));
        output_buffer.map_async(wgpu::MapMode::Read, .., |_| {});
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely())?;

        let data = output_buffer.get_mapped_range(..);
        let mut pixels =
            Vec::with_capacity((image_width * image_height * bytes_per_pixel) as usize);

        for chunk in data.chunks(padded_bytes_per_row as usize) {
            pixels.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }

        drop(data);
        output_buffer.unmap();

        ImageExporter::save(pixels, image_width, image_height, &self.file_output_name)?;

        Ok(())
    }

    fn create_overlay_pipeline(
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        overlay: &texture::Texture,
        points_buffer: &wgpu::Buffer,
    ) -> BoundPipeline {
        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/overlay.wgsl"));

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            wgpu::BufferSize::new(size_of::<Point>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&overlay.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: points_buffer.as_entire_binding(),
                },
            ],
            label: Some("bind_group_overlay"),
        });

        BoundPipeline::new(device, &shader_module, &layout, bind_group)
    }

    fn create_blend_pipeline(
        device: &wgpu::Device,
        input: &texture::Texture,
        overlay: &texture::Texture,
        output: &texture::Texture,
    ) -> BoundPipeline {
        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/blend.wgsl"));

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blend_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&overlay.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output.view),
                },
            ],
            label: Some("bind_group_blend"),
        });

        BoundPipeline::new(device, &shader_module, &layout, bind_group)
    }
}
