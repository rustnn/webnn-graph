// Type conversion and constant operators: Cast, Constant

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ConversionHandler;

fn dtype_to_webnn_string(dt: &crate::ast::DataType) -> &'static str {
    match dt {
        crate::ast::DataType::Float32 => "float32",
        crate::ast::DataType::Float16 => "float16",
        crate::ast::DataType::Int4 => "int4",
        crate::ast::DataType::Uint4 => "uint4",
        crate::ast::DataType::Int32 => "int32",
        crate::ast::DataType::Uint32 => "uint32",
        crate::ast::DataType::Int64 => "int64",
        crate::ast::DataType::Uint64 => "uint64",
        crate::ast::DataType::Int8 => "int8",
        crate::ast::DataType::Uint8 => "uint8",
    }
}

impl OpHandler for ConversionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Cast" | "Constant")
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Cast" => self.convert_cast(node, &node_name, context),
            "Constant" => self.convert_constant(node, &node_name),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ConversionHandler {
    /// Convert ONNX Cast to WebNN cast
    /// ONNX Cast converts tensor data type
    fn convert_cast(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Cast expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract 'to' attribute (target data type)
        let mut to_type: Option<i64> = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "to" && attr.i != 0 {
                to_type = Some(attr.i);
            }
        }

        if to_type.is_none() {
            return Err(OnnxError::MissingAttribute {
                attr: "to".to_string(),
                op: "Cast".to_string(),
            });
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        // Map ONNX type to WebNN DataType
        let target_type = crate::onnx::convert::map_onnx_data_type(to_type.unwrap() as i32)?;

        let mut options = Map::new();
        options.insert(
            "to".to_string(),
            serde_json::json!(dtype_to_webnn_string(&target_type)),
        );

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "cast".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Constant to WebNN constant
    /// ONNX Constant creates an inline constant tensor
    fn convert_constant(
        &self,
        node: &NodeProto,
        node_name: &str,
    ) -> Result<ConversionResult, OnnxError> {
        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        // Extract 'value' attribute (TensorProto)
        let tensor = node
            .attribute
            .as_slice()
            .iter()
            .find_map(|attr| {
                if attr.name.as_str() == "value" {
                    attr.t.as_ref()
                } else {
                    None
                }
            })
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "value".to_string(),
                op: "Constant".to_string(),
            })?;
        let onnx_type = tensor.data_type;
        let data_type = crate::onnx::convert::map_onnx_data_type(onnx_type)?;

        let shape: Vec<i64> = tensor.dims.as_slice().to_vec();
        let raw_data = tensor.raw_data.as_slice().to_vec();

        let mut options = Map::new();
        options.insert(
            "dataType".to_string(),
            serde_json::json!(dtype_to_webnn_string(&data_type)),
        );
        options.insert("shape".to_string(), serde_json::json!(shape));

        // Store raw bytes as base64 for now (WebNN implementation can decode)
        let b64_data =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &raw_data);
        options.insert("data".to_string(), serde_json::json!(b64_data));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "constant".to_string(),
            inputs: vec![],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        let attr = AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    #[test]
    fn test_conversion_handler_supports() {
        let handler = ConversionHandler;
        assert!(handler.supports("Cast"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_cast() {
        let handler = ConversionHandler;
        let mut node = create_test_node("Cast", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "to", 7); // INT64
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "cast");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("to"));
        assert_eq!(
            result.nodes[0].options.get("to"),
            Some(&serde_json::json!("int64"))
        );
    }

    #[test]
    fn test_convert_constant_uses_lowercase_dtype_and_base64_data() {
        let handler = ConversionHandler;
        let mut node = create_test_node("Constant", vec![], vec!["c0"]);
        let tensor = crate::protos::onnx::TensorProto {
            data_type: crate::protos::onnx::TensorProto_DataType::Float as i32,
            dims: vec![1],
            raw_data: vec![0, 0, 128, 63], // 1.0f32
            ..Default::default()
        };
        node.attribute.push(AttributeProto {
            name: "value".to_string(),
            t: Some(tensor),
            ..Default::default()
        });

        let result = handler
            .convert(
                &node,
                &ConversionContext {
                    initializers: &std::collections::HashMap::new(),
                    value_shapes: &std::collections::HashMap::new(),
                    value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
                    const_values: &std::collections::HashMap::new(),
                    value_ids: &std::collections::HashMap::new(),
                    value_types: &std::collections::HashMap::new(),
                },
            )
            .unwrap();

        assert_eq!(result.nodes.len(), 1);
        assert_eq!(
            result.nodes[0].options.get("dataType"),
            Some(&serde_json::json!("float32"))
        );
        assert!(result.nodes[0].options.get("data").is_some());
    }
}
