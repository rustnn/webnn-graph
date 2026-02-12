// Elementwise binary operators: Add, Sub, Mul, Div, Pow

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ElementwiseHandler;

impl OpHandler for ElementwiseHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max"
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

        // Map ONNX operator to WebNN operator (lowercase)
        let webnn_op = match op_type {
            "Add" => "add",
            "Sub" => "sub",
            "Mul" => "mul",
            "Div" => "div",
            "Pow" => "pow",
            "Min" => "min",
            "Max" => "max",
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
    use crate::protos::onnx::NodeProto;

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
    fn test_elementwise_handler_supports() {
        let handler = ElementwiseHandler;
        assert!(handler.supports("Add"));
        assert!(handler.supports("Sub"));
        assert!(handler.supports("Mul"));
        assert!(handler.supports("Div"));
        assert!(handler.supports("Pow"));
        assert!(handler.supports("Min"));
        assert!(handler.supports("Max"));
        assert!(!handler.supports("MatMul"));
    }

    #[test]
    fn test_convert_add() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Add", vec!["a", "b"], vec!["c"]);
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
        assert_eq!(result.nodes[0].op, "add");
        assert_eq!(result.nodes[0].inputs, vec!["a", "b"]);
        assert_eq!(result.nodes[0].id, "c");
    }

    #[test]
    fn test_convert_mul() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Mul", vec!["x", "y"], vec!["z"]);
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
        assert_eq!(result.nodes[0].op, "mul");
        assert_eq!(result.nodes[0].inputs, vec!["x", "y"]);
    }

    #[test]
    fn test_convert_div() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Div", vec!["a", "b"], vec!["c"]);
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
        assert_eq!(result.nodes[0].op, "div");
    }

    #[test]
    fn test_convert_min() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Min", vec!["x", "y"], vec!["z"]);
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
        assert_eq!(result.nodes[0].op, "min");
        assert_eq!(result.nodes[0].inputs, vec!["x", "y"]);
    }

    #[test]
    fn test_convert_max() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Max", vec!["a", "b"], vec!["c"]);
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
        assert_eq!(result.nodes[0].op, "max");
        assert_eq!(result.nodes[0].inputs, vec!["a", "b"]);
    }
}
