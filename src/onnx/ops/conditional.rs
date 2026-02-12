// Conditional operators: Where

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ConditionalHandler;

impl OpHandler for ConditionalHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Where")
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
        if inputs.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 3 inputs (condition, x, y), got {}",
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
        let condition = context.resolve_input(&inputs[0]);
        let true_value = context.resolve_input(&inputs[1]);
        let false_value = context.resolve_input(&inputs[2]);

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "where".to_string(),
            inputs: vec![condition, true_value, false_value],
            options: Map::new(),
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            // Where output type matches the input data type (x and y), not condition
            if let Some(dtype) = context.value_types.get(&inputs[1]) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DataType;
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
    fn test_conditional_handler_supports() {
        let handler = ConditionalHandler;
        assert!(handler.supports("Where"));
        assert!(!handler.supports("Add"));
        assert!(!handler.supports("Greater"));
    }

    #[test]
    fn test_where_conversion() {
        let handler = ConditionalHandler;
        let node = create_test_node("Where", vec!["condition", "x", "y"], vec!["output"]);
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let mut value_types = HashMap::new();
        value_types.insert("x".to_string(), DataType::Float32);
        value_types.insert("y".to_string(), DataType::Float32);
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
        assert_eq!(converted_node.op, "where");
        assert_eq!(converted_node.inputs.len(), 3);
        assert_eq!(converted_node.inputs[0], "condition");
        assert_eq!(converted_node.inputs[1], "x");
        assert_eq!(converted_node.inputs[2], "y");

        // Check output type matches input data type
        assert_eq!(result.output_types.get("output"), Some(&DataType::Float32));
    }

    #[test]
    fn test_where_invalid_inputs() {
        let handler = ConditionalHandler;
        let node = create_test_node("Where", vec!["condition", "x"], vec!["output"]); // Only 2 inputs
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
        if let Err(OnnxError::InvalidShape(msg)) = result {
            assert!(msg.contains("expects 3 inputs"));
        } else {
            panic!("Expected InvalidShape error");
        }
    }
}
