// MatMul and Gemm operator handlers

use crate::ast::{ConstDecl, ConstInit, DataType, Node};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct MatMulHandler;

impl OpHandler for MatMulHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "MatMul" | "Gemm")
    }

    fn convert(
        &self,
        node: &NodeProto,
        _context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "MatMul" => self.convert_matmul(node, &node_name, _context),
            "Gemm" => self.convert_gemm(node, &node_name, _context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl MatMulHandler {
    /// Convert ONNX MatMul to WebNN matmul
    /// MatMul: Y = A @ B
    fn convert_matmul(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);
        let input1 = context.resolve_input(&inputs[1]);

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "matmul".to_string(),
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

    /// Convert ONNX Gemm to WebNN operations
    /// Gemm: Y = alpha * A' @ B' + beta * C
    /// where A' = transpose(A) if transA else A
    fn convert_gemm(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gemm expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        // Extract attributes
        let mut alpha = 1.0f32;
        let mut beta = 1.0f32;
        let mut trans_a = false;
        let mut trans_b = false;

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "alpha" if attr.f != 0.0 => {
                    alpha = attr.f;
                }
                "beta" if attr.f != 0.0 => {
                    beta = attr.f;
                }
                "transA" if attr.i != 0 => {
                    trans_a = attr.i != 0;
                }
                "transB" if attr.i != 0 => {
                    trans_b = attr.i != 0;
                }
                _ => {}
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0_raw = inputs[0].to_string();
        let input1_raw = inputs[1].to_string();
        let input2_raw = inputs.get(2).map(|s| s.to_string());

        let input0 = context.resolve_input(&input0_raw);
        let input1 = context.resolve_input(&input1_raw);
        let input2 = input2_raw
            .as_ref()
            .map(|name| context.resolve_input(name))
            .unwrap_or_default();

        let mut nodes = Vec::new();
        let mut consts = Vec::new();
        let mut current_result = sanitize_identifier(&format!("{}_matmul", node_name));

        let build_transpose_perm = |input_name: &str,
                                    value_shapes: &std::collections::HashMap<String, Vec<i64>>|
         -> Result<Vec<i64>, OnnxError> {
            if let Some(shape) = value_shapes.get(input_name) {
                if shape.len() < 2 {
                    return Err(OnnxError::InvalidShape(format!(
                        "Gemm transpose requires rank >= 2 for '{}', got {:?}",
                        input_name, shape
                    )));
                }
                let rank = shape.len();
                let mut perm: Vec<i64> = (0..rank as i64).collect();
                perm.swap(rank - 1, rank - 2);
                Ok(perm)
            } else {
                Err(OnnxError::InvalidShape(format!(
                    "Gemm transpose requires known shape for '{}'",
                    input_name
                )))
            }
        };

        // Handle transpose A if needed
        let input_a = if trans_a {
            let trans_a_name = sanitize_identifier(&format!("{}_transposeA", node_name));
            let perm = build_transpose_perm(&input0_raw, context.value_shapes)?;
            nodes.push(Node {
                id: trans_a_name.clone(),
                op: "transpose".to_string(),
                inputs: vec![input0.clone()],
                options: {
                    let mut opts = Map::new();
                    opts.insert("permutation".to_string(), serde_json::json!(perm));
                    opts
                },
                outputs: None,
            });
            trans_a_name
        } else {
            input0.clone()
        };

        // Handle transpose B if needed
        let input_b = if trans_b {
            let trans_b_name = sanitize_identifier(&format!("{}_transposeB", node_name));
            let perm = build_transpose_perm(&input1_raw, context.value_shapes)?;
            nodes.push(Node {
                id: trans_b_name.clone(),
                op: "transpose".to_string(),
                inputs: vec![input1.clone()],
                options: {
                    let mut opts = Map::new();
                    opts.insert("permutation".to_string(), serde_json::json!(perm));
                    opts
                },
                outputs: None,
            });
            trans_b_name
        } else {
            input1.clone()
        };

        // MatMul: A @ B
        nodes.push(Node {
            id: current_result.clone(),
            op: "matmul".to_string(),
            inputs: vec![input_a, input_b],
            options: Map::new(),
            outputs: None,
        });

        // Scale by alpha if not 1.0
        if (alpha - 1.0).abs() > f32::EPSILON {
            let scaled = sanitize_identifier(&format!("{}_scaled", node_name));
            let alpha_const_id = sanitize_identifier(&format!("{}_alpha", node_name));
            consts.push((
                alpha_const_id.clone(),
                ConstDecl {
                    data_type: DataType::Float32,
                    shape: vec![],
                    init: ConstInit::Scalar {
                        value: serde_json::json!(alpha),
                    },
                },
            ));
            nodes.push(Node {
                id: scaled.clone(),
                op: "mul".to_string(),
                inputs: vec![current_result.clone(), alpha_const_id],
                options: Map::new(),
                outputs: None,
            });
            current_result = scaled;
        }

        // Add beta * C if C is provided
        if inputs.len() > 2 {
            let bias_input = if (beta - 1.0).abs() > f32::EPSILON {
                // Scale C by beta
                let scaled_c = sanitize_identifier(&format!("{}_scaled_c", node_name));
                let beta_const_id = sanitize_identifier(&format!("{}_beta", node_name));
                consts.push((
                    beta_const_id.clone(),
                    ConstDecl {
                        data_type: DataType::Float32,
                        shape: vec![],
                        init: ConstInit::Scalar {
                            value: serde_json::json!(beta),
                        },
                    },
                ));
                nodes.push(Node {
                    id: scaled_c.clone(),
                    op: "mul".to_string(),
                    inputs: vec![input2.clone(), beta_const_id],
                    options: Map::new(),
                    outputs: None,
                });
                scaled_c
            } else {
                input2.clone()
            };

            // Add the bias
            nodes.push(Node {
                id: output_name.clone(),
                op: "add".to_string(),
                inputs: vec![current_result, bias_input],
                options: Map::new(),
                outputs: None,
            });
        } else {
            // No bias, just rename the final result
            if current_result != output_name {
                // If we have intermediate nodes, the last one should output to output_name
                if let Some(last_node) = nodes.last_mut() {
                    last_node.id = output_name.clone();
                }
            }
        }

        let mut result = ConversionResult {
            nodes,
            consts,
            output_mappings: std::collections::HashMap::new(),
            output_types: std::collections::HashMap::new(),
        };

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
    fn test_matmul_handler_supports() {
        let handler = MatMulHandler;
        assert!(handler.supports("MatMul"));
        assert!(handler.supports("Gemm"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_matmul() {
        let handler = MatMulHandler;
        let node = create_test_node("MatMul", vec!["a", "b"], vec!["c"]);
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
        assert_eq!(result.nodes[0].op, "matmul");
        assert_eq!(result.nodes[0].inputs, vec!["a", "b"]);
        assert_eq!(result.nodes[0].id, "c");
    }

    #[test]
    fn test_convert_gemm_simple() {
        let handler = MatMulHandler;
        let node = create_test_node("Gemm", vec!["a", "b"], vec!["c"]);
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
        // Simple Gemm without C, alpha=1, beta=1 should produce just matmul
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "matmul");
    }
}
