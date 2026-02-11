use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub struct DynamicDimension {
    pub name: String,
    pub max_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(untagged)]
pub enum Dimension {
    Static(u32),
    Dynamic(DynamicDimension),
}

pub fn to_dimension_vector(shape: &[u32]) -> Vec<Dimension> {
    shape.iter().copied().map(Dimension::Static).collect()
}

pub fn get_static_or_max_size(dim: &Dimension) -> u32 {
    match dim {
        Dimension::Static(v) => *v,
        Dimension::Dynamic(d) => d.max_size,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphJson {
    pub format: String, // "webnn-graph-json"
    pub version: u32,   // 1 or 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub quantized: bool,
    pub inputs: BTreeMap<String, OperandDesc>,
    #[serde(default)]
    pub consts: BTreeMap<String, ConstDecl>,
    pub nodes: Vec<Node>,
    // output_name -> value reference name
    pub outputs: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperandDesc {
    #[serde(rename = "dataType")]
    pub data_type: DataType,
    pub shape: Vec<Dimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "int4")]
    Int4,
    #[serde(rename = "uint4")]
    Uint4,
    #[serde(rename = "int32")]
    Int32,
    #[serde(rename = "uint32")]
    Uint32,
    #[serde(rename = "int64")]
    Int64,
    #[serde(rename = "uint64")]
    Uint64,
    #[serde(rename = "int8")]
    Int8,
    #[serde(rename = "uint8")]
    Uint8,
}

impl DataType {
    pub fn from_wg(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::Float32),
            "f16" => Some(Self::Float16),
            "i4" => Some(Self::Int4),
            "u4" => Some(Self::Uint4),
            "i32" => Some(Self::Int32),
            "u32" => Some(Self::Uint32),
            "i64" => Some(Self::Int64),
            "u64" => Some(Self::Uint64),
            "i8" => Some(Self::Int8),
            "u8" => Some(Self::Uint8),
            _ => None,
        }
    }

    pub fn to_wg_text(&self) -> &'static str {
        match self {
            Self::Float32 => "f32",
            Self::Float16 => "f16",
            Self::Int4 => "i4",
            Self::Uint4 => "u4",
            Self::Int32 => "i32",
            Self::Uint32 => "u32",
            Self::Int64 => "i64",
            Self::Uint64 => "u64",
            Self::Int8 => "i8",
            Self::Uint8 => "u8",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstDecl {
    #[serde(rename = "dataType")]
    pub data_type: DataType,
    pub shape: Vec<u32>,
    pub init: ConstInit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ConstInit {
    Weights { r#ref: String },
    Scalar { value: serde_json::Value },
    InlineBytes { bytes: Vec<u8> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub op: String,
    pub inputs: Vec<String>,
    #[serde(default)]
    pub options: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub outputs: Option<Vec<String>>,
}

pub fn new_graph_json() -> GraphJson {
    GraphJson {
        format: "webnn-graph-json".to_string(),
        version: 2,
        name: None,
        quantized: false,
        inputs: BTreeMap::new(),
        consts: BTreeMap::new(),
        nodes: Vec::new(),
        outputs: BTreeMap::new(),
    }
}

impl OperandDesc {
    pub fn static_shape(&self) -> Option<Vec<u32>> {
        let mut shape = Vec::with_capacity(self.shape.len());
        for dim in &self.shape {
            match dim {
                Dimension::Static(v) => shape.push(*v),
                Dimension::Dynamic(_) => return None,
            }
        }
        Some(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datatype_from_wg() {
        assert_eq!(DataType::from_wg("f32"), Some(DataType::Float32));
        assert_eq!(DataType::from_wg("f16"), Some(DataType::Float16));
        assert_eq!(DataType::from_wg("i32"), Some(DataType::Int32));
        assert_eq!(DataType::from_wg("u32"), Some(DataType::Uint32));
        assert_eq!(DataType::from_wg("i64"), Some(DataType::Int64));
        assert_eq!(DataType::from_wg("u64"), Some(DataType::Uint64));
        assert_eq!(DataType::from_wg("i8"), Some(DataType::Int8));
        assert_eq!(DataType::from_wg("u8"), Some(DataType::Uint8));
        assert_eq!(DataType::from_wg("invalid"), None);
        assert_eq!(DataType::from_wg("float32"), None);
    }

    #[test]
    fn test_new_graph_json() {
        let graph = new_graph_json();
        assert_eq!(graph.format, "webnn-graph-json");
        assert_eq!(graph.version, 2);
        assert!(graph.inputs.is_empty());
        assert!(graph.consts.is_empty());
        assert!(graph.nodes.is_empty());
        assert!(graph.outputs.is_empty());
    }

    #[test]
    fn test_operand_desc_equality() {
        let desc1 = OperandDesc {
            data_type: DataType::Float32,
            shape: to_dimension_vector(&[1, 2, 3]),
        };
        let desc2 = OperandDesc {
            data_type: DataType::Float32,
            shape: to_dimension_vector(&[1, 2, 3]),
        };
        let desc3 = OperandDesc {
            data_type: DataType::Float16,
            shape: to_dimension_vector(&[1, 2, 3]),
        };
        assert_eq!(desc1, desc2);
        assert_ne!(desc1, desc3);
    }

    #[test]
    fn test_const_init_variants() {
        let weights_init = ConstInit::Weights {
            r#ref: "W".to_string(),
        };
        let scalar_init = ConstInit::Scalar {
            value: serde_json::json!(1.0),
        };
        let bytes_init = ConstInit::InlineBytes {
            bytes: vec![1, 2, 3, 4],
        };

        // Test that they're different variants
        assert!(matches!(weights_init, ConstInit::Weights { .. }));
        assert!(matches!(scalar_init, ConstInit::Scalar { .. }));
        assert!(matches!(bytes_init, ConstInit::InlineBytes { .. }));
    }
}
