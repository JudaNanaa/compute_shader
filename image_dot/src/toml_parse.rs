use std::{fs::File, io::Read};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Deserializer};

//
// ─────────────────────────────
// COLOR
// ─────────────────────────────
//

#[derive(Debug, Clone, Copy)]
pub struct Color(pub [f32; 4]);

#[derive(Deserialize)]
#[serde(untagged)]
enum ColorInput {
    Named(String),
    Rgba([f32; 4]),
}

impl TryFrom<ColorInput> for Color {
    type Error = anyhow::Error;

    fn try_from(input: ColorInput) -> Result<Self> {
        match input {
            ColorInput::Rgba(rgba) => Ok(Color(rgba)),
            ColorInput::Named(name) => {
                let rgba = match name.to_lowercase().as_str() {
                    "red" => [1.0, 0.0, 0.0, 1.0],
                    "green" => [0.0, 1.0, 0.0, 1.0],
                    "blue" => [0.0, 0.0, 1.0, 1.0],
                    "white" => [1.0, 1.0, 1.0, 1.0],
                    "black" => [0.0, 0.0, 0.0, 1.0],
                    _ => {
                        return Err(anyhow!("Unknown color name: {}", name));
                    }
                };

                Ok(Color(rgba))
            }
        }
    }
}

//
// ─────────────────────────────
// SHAPE CONFIG
// ─────────────────────────────
//

#[derive(Debug)]
pub struct ShapeConfig {
    pub color: [f32; 4],
    pub size: f32,
}

#[derive(Deserialize)]
struct ShapeConfigHelper {
    color: ColorInput,
    size: f32,
}

impl<'de> Deserialize<'de> for ShapeConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper =
            ShapeConfigHelper::deserialize(deserializer).map_err(serde::de::Error::custom)?;

        let color = Color::try_from(helper.color).map_err(serde::de::Error::custom)?;

        Ok(ShapeConfig {
            color: color.0,
            size: helper.size,
        })
    }
}

//
// ─────────────────────────────
// SHAPE TYPE
// ─────────────────────────────
//

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum ShapeType {
    Circle = 1,
    Triangle = 2,
    Square = 3,
    Cross = 4,
    Line = 5,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ShapeTypeRaw {
    Triangle,
    Circle,
    Square,
    Cross,
    Line,
}

impl From<ShapeTypeRaw> for ShapeType {
    fn from(raw: ShapeTypeRaw) -> Self {
        match raw {
            ShapeTypeRaw::Triangle => ShapeType::Triangle,
            ShapeTypeRaw::Cross => ShapeType::Cross,
            ShapeTypeRaw::Square => ShapeType::Square,
            ShapeTypeRaw::Circle => ShapeType::Circle,
            ShapeTypeRaw::Line => ShapeType::Line,
        }
    }
}

//
// ─────────────────────────────
// GPU CONTEXT
// ─────────────────────────────
//

#[derive(Debug)]
pub struct DebugGpuContext {
    pub input_file: String,
    pub output_file: String,
    pub shape: ShapeType,
    pub triangle: ShapeConfig,
    pub circle: ShapeConfig,
    pub line: ShapeConfig,
    square: ShapeConfig,
    cross: ShapeConfig,
}

impl DebugGpuContext {
    pub fn get_color(&self) -> [f32; 4] {
        self.get_shape_infos().color
    }

    pub fn get_size(&self) -> f32 {
        self.get_shape_infos().size
    }

    pub fn get_current_shape(&self) -> ShapeType {
        self.shape
    }

    pub fn get_current_shape_u32(&self) -> u32 {
        self.get_current_shape() as u32
    }

    pub fn get_shape_infos(&self) -> &ShapeConfig {
        match self.get_current_shape() {
            ShapeType::Cross => &self.cross,
            ShapeType::Square => &self.square,
            ShapeType::Circle => &self.circle,
            ShapeType::Triangle => &self.triangle,
            ShapeType::Line => &self.line,
        }
    }
}

#[derive(Deserialize)]
struct DebugGpuContextHelper {
    input_file: String,
    #[serde(default)]
    output_file: String,
    shape: ShapeTypeRaw,
    triangle: ShapeConfig,
    circle: ShapeConfig,
    line: ShapeConfig,
    square: ShapeConfig,
    cross: ShapeConfig,
}

impl<'de> Deserialize<'de> for DebugGpuContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper =
            DebugGpuContextHelper::deserialize(deserializer).map_err(serde::de::Error::custom)?;

        Ok(DebugGpuContext {
            input_file: helper.input_file,
            output_file: helper.output_file,
            shape: helper.shape.into(),
            triangle: helper.triangle,
            circle: helper.circle,
            line: helper.line,
            square: helper.square,
			cross: helper.cross
        })
    }
}

//
// ─────────────────────────────
// PARSE FUNCTION
// ─────────────────────────────
//

pub fn debug_file_parse() -> Result<DebugGpuContext> {
    let mut file = File::open("./debug_config.toml")?;

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let config: DebugGpuContext = toml::from_str(&content)?;

    Ok(config)
}
