@group(0) @binding(0)
var input_tex : texture_2d<f32>;

@group(0) @binding(1)
var overlay_tex : texture_2d<f32>;

@group(0) @binding(2)
var output_tex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {

    let dims = textureDimensions(input_tex);

    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(id.xy);

    let base = textureLoad(input_tex, coords, 0);

    let overlay = textureLoad(overlay_tex, coords, 0);

    let alpha = overlay.a;

    let final_rgb = base.rgb * (1.0 - alpha) + overlay.rgb * alpha;

    let final_color = vec4<f32>(final_rgb, 1.0);

    textureStore(output_tex, coords, final_color);
}