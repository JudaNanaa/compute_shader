const SHAPE_CIRCLE: u32 = 1u;
const SHAPE_TRIANGLE: u32 = 2u;
const SHAPE_SQUARE: u32 = 3u;
const SHAPE_CROSS: u32 = 4u;

struct Params {
	color: vec4<f32>,
	size: f32,
	shape: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var overlay_tex : texture_storage_2d<rgba8unorm, write>;

struct Point {
    position : vec2<f32>,
};

@group(0) @binding(2)
var<storage, read> points : array<Point>;

fn point_in_triangle(pos: vec2<f32>, center: vec2<f32>, size: f32) -> bool {

    let h = size * 1.5;
    let p0 = center + vec2<f32>(0.0, -h);
    let p1 = center + vec2<f32>(-size, size);
    let p2 = center + vec2<f32>(size, size);

    return point_in_barycentric(pos, p0, p1, p2);
}

fn point_in_barycentric(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {

    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    return (u >= 0.0 && v >= 0.0 && w >= 0.0);
}

fn point_in_square(pos: vec2<f32>, center: vec2<f32>, size: f32) -> bool {

    let half = size;

    let min = center - vec2<f32>(half, half);
    let max = center + vec2<f32>(half, half);

    return (pos.x >= min.x && pos.x <= max.x &&
            pos.y >= min.y && pos.y <= max.y);
}

fn point_in_cross(pos: vec2<f32>, center: vec2<f32>, size: f32) -> bool {

    let dx = abs(pos.x - center.x);
    let dy = abs(pos.y - center.y);

    let thickness = 2.0;
    let arm = size;

    let horizontal = (dx < arm) && (dy < thickness);
    let vertical   = (dy < arm) && (dx < thickness);

    return horizontal || vertical;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id : vec3<u32>) {

    let dims = textureDimensions(overlay_tex);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    let pixel_pos = vec2<f32>(id.xy);

    var draw = false;

    for (var i: u32 = 0u; i < arrayLength(&points); i = i + 1u) {

        let p = points[i].position;

        let dist = distance(pixel_pos, p);

		var hit = false;
		if (params.shape == SHAPE_CIRCLE) {
			let dist = distance(pixel_pos, p);
			hit = dist < params.size;
		}
		else if (params.shape == SHAPE_TRIANGLE) {
			hit = point_in_triangle(pixel_pos, p, params.size);
		}
		else if (params.shape == SHAPE_SQUARE) {
			hit = point_in_square(pixel_pos, p, params.size);
		}
		else if (params.shape == SHAPE_CROSS) {
    		hit = point_in_cross(pixel_pos, p, params.size);
		}
		if (hit) {
 		   draw = true;
		   break;
		}
    }

    if (draw) {
        textureStore(
            overlay_tex,
            vec2<i32>(id.xy),
			params.color
        );
    } else {
        textureStore(
            overlay_tex,
            vec2<i32>(id.xy),
            vec4<f32>(0.0, 0.0, 0.0, 0.0)
        );
    }
}