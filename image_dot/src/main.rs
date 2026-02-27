use std::sync::Arc;

use pollster::FutureExt;

use crate::image_pipeline::Point;

mod image_pipeline;
mod texture;
mod gpu_context;
mod image_exportrer;
mod texture_builder;

use rand::RngExt;

pub fn random_points(count: usize, width: f32, height: f32) -> Vec<Point> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| Point::new(rng.random_range(0.0..width), rng.random_range(0.0..height)))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        eprintln!("Ya besoin que de 2 arguments!");
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    let points = random_points(100, 1000.0, 1000.0);

	let ctx = Arc::new(gpu_context::GpuContext::new().block_on());



    let image_pipeline = image_pipeline::ImagePipeline::new(Arc::clone(&ctx), &input_file, &output_file, &points)
        .block_on()
        .unwrap();

    image_pipeline.image_dot().block_on().unwrap();

    println!("le noir et blanc est fini!");
}
