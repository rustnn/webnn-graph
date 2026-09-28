// Activation and unary math operators: Relu, Gelu, Tanh, Sigmoid, Sqrt, Exp, Log, Abs, Neg, Erf

use crate::ast::{ConstDecl, ConstInit, DataType, Node};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::{json, Map};
use std::collections::HashSet;

pub struct ActivationHandler;

impl OpHandler for ActivationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Relu"
                | "Gelu"
                | "Tanh"
                | "Sigmoid"
                | "Sqrt"
                | "Exp"
                | "Log"
                | "Abs"
                | "Neg"
                | "Erf"
                | "Cos"
                | "Sin"
                | "Identity"
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

        if op_type == "Gelu" {
            if !matches!(node.domain.as_str(), "" | "ai.onnx" | "com.microsoft") {
                return Err(OnnxError::UnsupportedOp {
                    op: format!("{}::Gelu", node.domain),
                    node: node_name,
                });
            }
            if node.input.len() != 1
                || node.output.len() != 1
                || node.input[0].is_empty()
                || node.output[0].is_empty()
            {
                return Err(OnnxError::InvalidShape(format!(
                    "Gelu '{}' expects one input and one output",
                    node_name
                )));
            }
            if context
                .value_types
                .get(&node.input[0])
                .is_some_and(|dtype| !matches!(dtype, DataType::Float16 | DataType::Float32))
            {
                return Err(OnnxError::UnsupportedOp {
                    op: "Gelu requires float16 or float32 input".to_string(),
                    node: node_name,
                });
            }
            if Self::gelu_uses_tanh(node, &node_name)? {
                return self.convert_tanh_gelu(node, &node_name, context);
            }
        }

        // Map ONNX operator to WebNN operation name
        let webnn_op = match op_type {
            "Relu" => "relu",
            "Gelu" => "gelu",
            "Tanh" => "tanh",
            "Sigmoid" => "sigmoid",
            "Sqrt" => "sqrt",
            "Exp" => "exp",
            "Log" => "log",
            "Abs" => "abs",
            "Neg" => "neg",
            "Erf" => "erf",
            "Cos" => "cos",
            "Sin" => "sin",
            "Identity" => "identity",
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: op_type.to_string(),
                    node: node_name,
                })
            }
        };

        self.convert_unary(node, &node_name, webnn_op, context)
    }
}

impl ActivationHandler {
    fn gelu_uses_tanh(node: &NodeProto, node_name: &str) -> Result<bool, OnnxError> {
        let invalid = |reason: &str| OnnxError::InvalidAttribute {
            attr: "approximate".to_string(),
            op: "Gelu".to_string(),
            node: node_name.to_string(),
            reason: reason.to_string(),
        };
        let mut attributes = node.attribute.iter().filter(|a| a.name == "approximate");
        let Some(attribute) = attributes.next() else {
            return Ok(false);
        };
        if node.domain == "com.microsoft" {
            return Err(invalid("com.microsoft Gelu does not define this attribute"));
        }
        if attributes.next().is_some() {
            return Err(invalid("attribute must not be repeated"));
        }
        if attribute.r#type != crate::protos::onnx::attribute_proto::AttributeType::String as i32 {
            return Err(invalid("expected a string"));
        }
        match attribute.s.as_slice() {
            b"none" => Ok(false),
            b"tanh" => Ok(true),
            _ => Err(invalid("expected 'none' or 'tanh'")),
        }
    }

    fn convert_tanh_gelu(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let input = context.resolve_input(&node.input[0]);
        let output = sanitize_identifier(&node.output[0]);
        let dtype = context
            .value_types
            .get(&node.input[0])
            .or_else(|| context.value_types.get(&input))
            .ok_or_else(|| {
                OnnxError::InvalidShape(format!("Gelu '{}' requires a known input type", node_name))
            })?;
        if !matches!(dtype, DataType::Float16 | DataType::Float32) {
            return Err(OnnxError::UnsupportedOp {
                op: format!("Gelu with {:?} input", dtype),
                node: node_name.to_string(),
            });
        }

        let mut used: HashSet<String> = context.value_ids.values().cloned().collect();
        used.insert(input.clone());
        used.insert(output.clone());
        let mut private_values = Vec::new();
        let mut fresh = |suffix: &str| {
            let base = format!("{}__gelu_{}", output, suffix);
            let mut id = base.clone();
            let mut index = 1;
            while !used.insert(id.clone()) {
                id = format!("{}_{}", base, index);
                index += 1;
            }
            private_values.push(id.clone());
            id
        };
        let mut result = ConversionResult::default();
        let x = if *dtype == DataType::Float16 {
            let id = fresh("float32");
            result.nodes.push(Node {
                id: id.clone(),
                op: "cast".to_string(),
                inputs: vec![input],
                options: Map::from_iter([("to".to_string(), json!("float32"))]),
                outputs: None,
            });
            id
        } else {
            input
        };

        // WebNN gelu is erf-based. Preserve ONNX's separate tanh formula using
        // primitive operations. Promote half inputs for the polynomial and
        // cancellation near the negative tail, then round only the result.
        let mut scalar = |suffix: &str, value: f32| {
            let id = fresh(suffix);
            result.consts.push((
                id.clone(),
                ConstDecl {
                    data_type: DataType::Float32,
                    shape: vec![],
                    init: ConstInit::InlineBytes {
                        bytes: value.to_le_bytes().to_vec(),
                    },
                },
            ));
            id
        };
        let half = scalar("half", 0.5);
        let one = scalar("one", 1.0);
        let coefficient = scalar("coefficient", 0.044715);
        // Round the mathematical coefficient once, not pi and the division
        // separately; the latter gives the preceding float32 value.
        let scale = scalar("scale", (2.0_f64 / std::f64::consts::PI).sqrt() as f32);
        let mut operation = |suffix: &str, op: &str, inputs: Vec<String>| {
            let id = fresh(suffix);
            result.nodes.push(Node {
                id: id.clone(),
                op: op.to_string(),
                inputs,
                options: Map::new(),
                outputs: None,
            });
            id
        };
        let square = operation("square", "mul", vec![x.clone(), x.clone()]);
        let cube = operation("cube", "mul", vec![square, x.clone()]);
        let cubic = operation("cubic", "mul", vec![coefficient, cube]);
        let polynomial = operation("polynomial", "add", vec![x.clone(), cubic]);
        let scaled = operation("scaled", "mul", vec![scale, polynomial]);
        let tanh = operation("tanh", "tanh", vec![scaled]);
        let gate = operation("gate", "add", vec![one, tanh]);
        let half_x = operation("half_x", "mul", vec![half, x]);
        let result_id = if *dtype == DataType::Float16 {
            fresh("result")
        } else {
            output.clone()
        };
        result.nodes.push(Node {
            id: result_id.clone(),
            op: "mul".to_string(),
            inputs: vec![half_x, gate],
            options: Map::new(),
            outputs: None,
        });
        if *dtype == DataType::Float16 {
            result.nodes.push(Node {
                id: output.clone(),
                op: "cast".to_string(),
                inputs: vec![result_id],
                options: Map::from_iter([("to".to_string(), json!("float16"))]),
                outputs: None,
            });
        }
        result
            .output_mappings
            .insert(node.output[0].clone(), output);
        result
            .output_types
            .insert(node.output[0].clone(), dtype.clone());
        result.private_values = private_values;
        Ok(result)
    }

    /// Convert ONNX unary/activation operation to WebNN
    fn convert_unary(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                webnn_op,
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let options = Map::new();

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
    fn test_activation_handler_supports() {
        let handler = ActivationHandler;
        assert!(handler.supports("Relu"));
        assert!(handler.supports("Gelu"));
        assert!(handler.supports("Tanh"));
        assert!(handler.supports("Sigmoid"));
        assert!(handler.supports("Sqrt"));
        assert!(handler.supports("Exp"));
        assert!(handler.supports("Log"));
        assert!(handler.supports("Abs"));
        assert!(handler.supports("Neg"));
        assert!(handler.supports("Erf"));
        assert!(handler.supports("Cos"));
        assert!(handler.supports("Sin"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_relu() {
        let handler = ActivationHandler;
        let node = create_test_node("Relu", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "relu");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_sqrt() {
        let handler = ActivationHandler;
        let node = create_test_node("Sqrt", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "sqrt");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_gelu() {
        let handler = ActivationHandler;
        let node = create_test_node("Gelu", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "gelu");
    }

    fn convert_gelu_attributes(
        attributes: Vec<crate::protos::onnx::AttributeProto>,
    ) -> Result<ConversionResult, OnnxError> {
        let mut node = create_test_node("Gelu", vec!["x"], vec!["y"]);
        node.attribute = attributes;
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::from([("x".to_string(), DataType::Float32)]);
        ActivationHandler.convert(
            &node,
            &ConversionContext {
                initializers: &initializers,
                value_shapes: &value_shapes,
                value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
                const_values: &const_values,
                value_ids: &value_ids,
                value_types: &value_types,
            },
        )
    }

    fn approximate(value: &[u8]) -> crate::protos::onnx::AttributeProto {
        crate::protos::onnx::AttributeProto {
            name: "approximate".to_string(),
            r#type: 3, // AttributeProto::STRING
            s: value.to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn test_gelu_default_and_none_preserve_exact_operation() {
        for attributes in [vec![], vec![approximate(b"none")]] {
            let result = convert_gelu_attributes(attributes).expect("exact GELU");
            assert_eq!(result.nodes.len(), 1);
            assert_eq!(result.nodes[0].op, "gelu");
            assert_eq!(result.nodes[0].inputs, ["x"]);
            assert!(result.nodes[0].options.is_empty());
            assert_eq!(result.output_mappings.get("y"), Some(&"y".to_string()));
        }
    }

    #[test]
    fn test_gelu_tanh_is_not_silently_replaced_with_exact_gelu() {
        let result = convert_gelu_attributes(vec![approximate(b"tanh")]).unwrap();
        assert!(result.nodes.iter().any(|node| node.op == "tanh"));
        assert!(result.nodes.iter().all(|node| node.op != "gelu"));
        assert_eq!(result.consts.len(), 4);
    }

    #[test]
    fn test_gelu_rejects_invalid_or_malformed_approximation() {
        let mut wrong_type = approximate(b"none");
        wrong_type.r#type = 2; // INT, even if the string field is populated.
        let mut missing_type = approximate(b"none");
        missing_type.r#type = 0;
        for attributes in [
            vec![approximate(b"invalid")],
            vec![approximate(b"")],
            vec![approximate(b"TANH")],
            vec![approximate(&[0xff])],
            vec![wrong_type],
            vec![missing_type],
            vec![approximate(b"none"), approximate(b"tanh")],
        ] {
            let error = convert_gelu_attributes(attributes)
                .expect_err("invalid approximation must not become exact GELU");
            assert!(matches!(
                &error,
                OnnxError::InvalidAttribute { attr, op, node, .. }
                    if attr == "approximate" && op == "Gelu" && node == "test_gelu"
            ));
            let message = error.to_string();
            assert!(message.contains("approximate"), "{message}");
            assert!(message.contains("Gelu"), "{message}");
            assert!(message.contains("test_gelu"), "{message}");
        }
    }

    #[test]
    fn test_convert_cos() {
        let handler = ActivationHandler;
        let node = create_test_node("Cos", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "cos");
    }

    #[test]
    fn test_convert_sin() {
        let handler = ActivationHandler;
        let node = create_test_node("Sin", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "sin");
    }
}
