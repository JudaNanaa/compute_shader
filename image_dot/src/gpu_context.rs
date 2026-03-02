use crate::toml_parse::{DebugGpuContext, debug_file_parse};

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub debug_config: DebugGpuContext,
}

impl GpuContext {
    pub async fn new() -> Self {

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });


        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .unwrap();

        let debug_config = match debug_file_parse() {
            Ok(conf) => conf,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
        Self {
            device,
            queue,
            debug_config,
        }
    }
}
