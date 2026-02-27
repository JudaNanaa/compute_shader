use anyhow::{Ok, Result};
use image::EncodableLayout;

use crate::texture;

const WORKGROUP_SIZE: u32 = 8;

pub struct ImagePipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    input_texture: texture::Texture,
    output_texture: texture::Texture,
    file_output_name: String,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl ImagePipeline {
    pub async fn new(file_input: &str, file_output: &str) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        let input_texture = texture::Texture::from_bytes(
            &device,
            &queue,
            std::fs::read(file_input)?.as_bytes(),
            "input file texture",
        )?;

        let output_texture = Self::create_output_texture(&device, input_texture.texture.size());

        let bind_group_layout = Self::create_bind_group_layout(&device);
        let bind_group = Self::create_bind_group(&device, &bind_group_layout, &input_texture, &output_texture);
        let pipeline = Self::create_pipeline(&device, &bind_group_layout);

        Ok(Self {
            device,
            queue,
            input_texture,
            output_texture,
            file_output_name: file_output.to_string(),
            pipeline,
            bind_group,
        })
    }

    pub async fn black_and_white(&self) -> Result<()> {
        let mut encoder = self.device.create_command_encoder(&Default::default());

        let image_width = self.input_texture.texture.width();
        let image_height = self.input_texture.texture.height();

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                image_width.div_ceil(WORKGROUP_SIZE),
                image_height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = image_width * bytes_per_pixel;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let buffer_size = padded_bytes_per_row as u64 * image_height as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buffer"),
            size: buffer_size,
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

        self.queue.submit(Some(encoder.finish()));
        output_buffer.map_async(wgpu::MapMode::Read, .., |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely())?;

        let data = output_buffer.get_mapped_range(..);
        let mut pixels = Vec::with_capacity((image_width * image_height * bytes_per_pixel) as usize);

        for chunk in data.chunks(padded_bytes_per_row as usize) {
            pixels.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }

        drop(data);
        output_buffer.unmap();

        let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(image_width, image_height, pixels).unwrap();

        if self.file_output_name.ends_with(".jpg") {
            image::DynamicImage::ImageRgba8(img).to_rgb8().save(&self.file_output_name)?;
        } else {
            img.save(&self.file_output_name)?;
        }

        Ok(())
    }

    fn create_output_texture(device: &wgpu::Device, size: wgpu::Extent3d) -> texture::Texture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        texture::Texture { texture, view }
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        input: &texture::Texture,
        output: &texture::Texture,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output.view),
                },
            ],
            label: Some("bind_group"),
        })
    }

    fn create_pipeline(device: &wgpu::Device, bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
        let shader_module = device.create_shader_module(wgpu::include_wgsl!("../shaders/black_and_white.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        })
    }
}