// Reduction operators: ReduceMean, ReduceSum, ReduceMax, ReduceMin

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{
    normalize_axes_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ReductionHandler;

impl OpHandler for ReductionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin"
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

        match op_type {
            "ReduceMean" => self.convert_reduce(node, &node_name, "reduceMean", context),
            "ReduceSum" => self.convert_reduce(node, &node_name, "reduceSum", context),
            "ReduceMax" => self.convert_reduce(node, &node_name, "reduceMax", context),
            "ReduceMin" => self.convert_reduce(node, &node_name, "reduceMin", context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReductionHandler {
    /// Convert ONNX reduce operations to WebNN reduce operations
    fn convert_reduce(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects at least 1 input",
                webnn_op
            )));
        }

        // Extract attributes
        let mut axes: Option<Vec<i64>> = None;
        let mut keepdims = 1i64; // ONNX default is 1 (keep dimensions)

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axes" => {
                    axes = Some(attr.ints.clone());
                }
                "keepdims" if attr.i != 0 => {
                    keepdims = attr.i;
                }
                _ => {}
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let mut options = Map::new();

        // Add axes if specified
        if let Some(axes_values) = axes {
            let axes_values = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
                normalize_axes_best_effort(&axes_values, rank)
            } else {
                axes_values
            };
            options.insert("axes".to_string(), serde_json::json!(axes_values));
        }

        // Add keepDims option (WebNN uses keepDimensions)
        options.insert(
            "keepDimensions".to_string(),
            serde_json::json!(keepdims != 0),
        );

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: webnn_op.to_string(),
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

    fn add_ints_attribute(node: &mut NodeProto, name: &str, values: Vec<i64>) {
        let attr = AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    #[test]
    fn test_reduction_handler_supports() {
        let handler = ReductionHandler;
        assert!(handler.supports("ReduceMean"));
        assert!(handler.supports("ReduceSum"));
        assert!(handler.supports("ReduceMax"));
        assert!(handler.supports("ReduceMin"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_reduce_mean() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceMean", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1, 2]);
        add_int_attribute(&mut node, "keepdims", 1);
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
        assert_eq!(result.nodes[0].op, "reduceMean");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("axes"));
        assert!(result.nodes[0].options.contains_key("keepDimensions"));
    }

    #[test]
    fn test_convert_reduce_sum() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceSum", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![-1]);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![2, 3, 4]);
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
        assert_eq!(result.nodes[0].op, "reduceSum");
        assert_eq!(
            result.nodes[0].options.get("axes"),
            Some(&serde_json::json!([2]))
        );
    }
}
