@group(0) @binding(0)
var overlay_tex : texture_storage_2d<rgba8unorm, write>;

struct Point {
    position : vec2<f32>,
};

@group(0) @binding(1)
var<storage, read> points : array<Point>;

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

        if (dist < 10.0) {
            draw = true;
            break;
        }
    }

    if (draw) {
        textureStore(
            overlay_tex,
            vec2<i32>(id.xy),
            vec4<f32>(1.0, 0.0, 0.0, 1.0)
        );
    } else {
        textureStore(
            overlay_tex,
            vec2<i32>(id.xy),
            vec4<f32>(0.0, 0.0, 0.0, 0.0)
        );
    }
}