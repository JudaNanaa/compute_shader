use rand::RngExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Point {
    x: f32,
    y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn random_points(count: usize, width: f32, height: f32) -> Vec<Point> {
        let mut rng = rand::rng();
        (0..count)
            .map(|_| Point::new(rng.random_range(0.0..width), rng.random_range(0.0..height)))
            .collect()
    }
}
