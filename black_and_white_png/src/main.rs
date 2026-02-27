use pollster::FutureExt;

mod image_pipeline;
mod texture;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        eprintln!("Ya besoin que de 2 arguments!");
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    let image_pipeline = image_pipeline::ImagePipeline::new(&input_file, &output_file)
        .block_on()
        .unwrap();

    image_pipeline.black_and_white().block_on().unwrap();

    println!("le noir et blanc est fini!");
}
