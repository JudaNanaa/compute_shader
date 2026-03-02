use std::sync::Arc;

use pollster::FutureExt;


mod image_pipeline;
mod texture;
mod gpu_context;
mod image_exportrer;
mod texture_builder;
mod toml_parse;
mod point;

use rand::RngExt;

use crate::point::Point;

pub fn random_points(count: usize, width: f32, height: f32) -> Vec<Point> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| Point::new(rng.random_range(0.0..width), rng.random_range(0.0..height)))
        .collect()
}

fn main() {
	let ctx = Arc::new(gpu_context::GpuContext::new().block_on());

    let image_pipeline = image_pipeline::ImagePipeline::new(ctx.clone())
        .block_on()
        .unwrap();

    image_pipeline.image_dot().block_on().unwrap();

    println!("l'image a ete modifie");
}
