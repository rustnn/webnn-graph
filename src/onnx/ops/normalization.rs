// Normalization operators: LayerNormalization, Softmax

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{
    normalize_axis_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct NormalizationHandler;

impl OpHandler for NormalizationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "LayerNormalization" | "Softmax")
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
            "LayerNormalization" => self.convert_layer_norm(node, &node_name, context),
            "Softmax" => self.convert_softmax(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl NormalizationHandler {
    /// Convert ONNX LayerNormalization to WebNN layerNormalization
    fn convert_layer_norm(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "LayerNormalization expects at least 1 input".to_string(),
            ));
        }

        // Extract attributes
        let mut epsilon = 1e-5f32;
        let mut axis = -1i64;

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "epsilon" => {
                    if attr.f != 0.0 {
                        epsilon = attr.f;
                    }
                }
                "axis" => {
                    if attr.i != 0 {
                        axis = attr.i;
                    }
                }
                _ => {}
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let mut options = Map::new();
        options.insert("epsilon".to_string(), serde_json::json!(epsilon));

        // WebNN layerNormalization uses positive axes.
        if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            let normalized_axis = normalize_axis_best_effort(axis, rank);
            options.insert("axes".to_string(), serde_json::json!([normalized_axis]));
        } else if axis != -1 {
            options.insert("axes".to_string(), serde_json::json!([axis]));
        }

        // LayerNormalization can have scale and bias as inputs
        let webnn_inputs = if inputs.len() >= 3 {
            // Input, scale, bias
            let input0 = context.resolve_input(&inputs[0]);
            let input1 = context.resolve_input(&inputs[1]);
            let input2 = context.resolve_input(&inputs[2]);
            vec![input0, input1, input2]
        } else if inputs.len() == 2 {
            // Input, scale
            let input0 = context.resolve_input(&inputs[0]);
            let input1 = context.resolve_input(&inputs[1]);
            vec![input0, input1]
        } else {
            // Just input
            let input0 = context.resolve_input(&inputs[0]);
            vec![input0]
        };

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "layerNormalization".to_string(),
            inputs: webnn_inputs,
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

    /// Convert ONNX Softmax to WebNN softmax
    fn convert_softmax(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Softmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute
        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank)
        } else {
            axis
        };

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "softmax".to_string(),
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
    fn test_normalization_handler_supports() {
        let handler = NormalizationHandler;
        assert!(handler.supports("LayerNormalization"));
        assert!(handler.supports("Softmax"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_softmax() {
        let handler = NormalizationHandler;
        let mut node = create_test_node("Softmax", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 128, 384]);
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
        assert_eq!(result.nodes[0].op, "softmax");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert_eq!(result.nodes[0].id, "y");
        assert!(result.nodes[0].options.contains_key("axis"));
        assert_eq!(
            result.nodes[0].options.get("axis"),
            Some(&serde_json::json!(2))
        );
    }

    #[test]
    fn test_convert_layer_norm() {
        let handler = NormalizationHandler;
        let mut node =
            create_test_node("LayerNormalization", vec!["x", "scale", "bias"], vec!["y"]);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 128, 384]);
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
        assert_eq!(result.nodes[0].op, "layerNormalization");
        assert_eq!(result.nodes[0].inputs.len(), 3);
        assert!(result.nodes[0].options.contains_key("epsilon"));
        assert_eq!(
            result.nodes[0].options.get("axes"),
            Some(&serde_json::json!([2]))
        );
    }
}
