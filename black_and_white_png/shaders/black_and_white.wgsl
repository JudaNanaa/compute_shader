@group(0) @binding(0)
var input_tex : texture_2d<f32>;

@group(0) @binding(1)
var output_tex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id : vec3<u32>) {

    let dims = textureDimensions(input_tex);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(id.xy);

    let pixel = textureLoad(input_tex, coords, 0);

    let gray = dot(pixel.rgb, vec3<f32>(0.299, 0.587, 0.114));

    textureStore(
        output_tex,
        coords,
        vec4<f32>(gray, gray, gray, 1.0)
    );
}