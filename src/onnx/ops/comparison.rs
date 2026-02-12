// Comparison operators: Greater, Less, Equal, GreaterOrEqual, LessOrEqual

use crate::ast::{DataType, Node};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ComparisonHandler;

impl OpHandler for ComparisonHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Greater" | "Less" | "Equal" | "GreaterOrEqual" | "LessOrEqual"
        )
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

        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 2 inputs, got {}",
                op_type,
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        // Resolve input names (respecting prior mappings)
        let input0 = context.resolve_input(&inputs[0]);
        let input1 = context.resolve_input(&inputs[1]);

        // Map ONNX operator to WebNN operator
        let webnn_op = match op_type {
            "Greater" => "greater",
            "Less" => "lesser",
            "Equal" => "equal",
            "GreaterOrEqual" => "greaterOrEqual",
            "LessOrEqual" => "lesserOrEqual",
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: op_type.to_string(),
                    node: node_name,
                })
            }
        };

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: webnn_op.to_string(),
            inputs: vec![input0, input1],
            options: Map::new(),
            outputs: None,
        }]);

        // Comparison operations output uint8 (boolean) tensors
        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            result
                .output_types
                .insert(output.to_string(), DataType::Uint8);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::NodeProto;
    use std::collections::HashMap;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_comparison_handler_supports() {
        let handler = ComparisonHandler;
        assert!(handler.supports("Greater"));
        assert!(handler.supports("Less"));
        assert!(handler.supports("Equal"));
        assert!(handler.supports("GreaterOrEqual"));
        assert!(handler.supports("LessOrEqual"));
        assert!(!handler.supports("Add"));
        assert!(!handler.supports("Relu"));
    }

    #[test]
    fn test_greater_conversion() {
        let handler = ComparisonHandler;
        let node = create_test_node("Greater", vec!["x", "y"], vec!["output"]);
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
        let converted_node = &result.nodes[0];
        assert_eq!(converted_node.op, "greater");
        assert_eq!(converted_node.inputs.len(), 2);
        assert_eq!(converted_node.inputs[0], "x");
        assert_eq!(converted_node.inputs[1], "y");

        // Check output type is uint8 (boolean)
        assert_eq!(result.output_types.get("output"), Some(&DataType::Uint8));
    }

    #[test]
    fn test_less_conversion() {
        let handler = ComparisonHandler;
        let node = create_test_node("Less", vec!["a", "b"], vec!["result"]);
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
        assert_eq!(result.nodes[0].op, "lesser");
    }

    #[test]
    fn test_equal_conversion() {
        let handler = ComparisonHandler;
        let node = create_test_node("Equal", vec!["p", "q"], vec!["eq"]);
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
        assert_eq!(result.nodes[0].op, "equal");
    }

    #[test]
    fn test_greater_or_equal_conversion() {
        let handler = ComparisonHandler;
        let node = create_test_node("GreaterOrEqual", vec!["x", "threshold"], vec!["mask"]);
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
        assert_eq!(result.nodes[0].op, "greaterOrEqual");
    }

    #[test]
    fn test_less_or_equal_conversion() {
        let handler = ComparisonHandler;
        let node = create_test_node("LessOrEqual", vec!["x", "max"], vec!["valid"]);
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
        assert_eq!(result.nodes[0].op, "lesserOrEqual");
    }

    #[test]
    fn test_comparison_invalid_inputs() {
        let handler = ComparisonHandler;
        let node = create_test_node("Greater", vec!["x"], vec!["output"]); // Only 1 input
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context);
        assert!(result.is_err());
    }
}
