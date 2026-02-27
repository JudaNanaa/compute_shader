use crate::texture;

pub struct TextureBuilder<'a> {
    device: &'a wgpu::Device,
    size: wgpu::Extent3d,
}

impl<'a> TextureBuilder<'a> {
    pub fn new(device: &'a wgpu::Device, size: wgpu::Extent3d) -> Self {
        Self { device, size }
    }

    pub fn build(self, usage: wgpu::TextureUsages, label: &str) -> texture::Texture {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: self.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        texture::Texture { texture, view }
    }
}