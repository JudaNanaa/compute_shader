use anyhow::Result;

pub struct ImageExporter;

impl ImageExporter {
    pub fn save(pixels: Vec<u8>, width: u32, height: u32, path: &str) -> Result<()> {
        let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

        if path.ends_with(".jpg") {
            image::DynamicImage::ImageRgba8(img).to_rgb8().save(path)?;
        } else {
            img.save(path)?;
        }

        Ok(())
    }
}