use crate::ast::Node;
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ScatterHandler;

impl ScatterHandler {
    fn get_string_attr(node: &NodeProto, name: &str) -> Option<String> {
        for a in node.attribute.as_slice() {
            if a.name.as_str() != name {
                continue;
            }
            // AttributeProto::STRING is stored as bytes in protobuf; Proto3 field is Vec<u8>
            let raw = a.s.clone();
            if raw.is_empty() {
                return None;
            }
            return String::from_utf8(raw).ok();
        }
        None
    }
}

impl OpHandler for ScatterHandler {
    fn supports(&self, op_type: &str) -> bool {
        op_type == "ScatterND"
    }

    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError> {
        // ONNX ScatterND inputs: data, indices, updates
        // WebNN scatterND(input, indices, updates)
        //
        // ONNX has optional "reduction" attr: none|add|mul|max|min (default: none).
        // WebNN scatterND has no reduction option => only support "none".
        let reduction =
            Self::get_string_attr(node, "reduction").unwrap_or_else(|| "none".to_string());
        if reduction != "none" {
            let node_name = if !node.name.is_empty() {
                node.name.as_str().to_string()
            } else {
                "".to_string()
            };
            return Err(OnnxError::UnsupportedOp {
                op: format!("ScatterND(reduction={})", reduction),
                node: node_name,
            });
        }

        let inputs = node.input.as_slice();
        if inputs.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "ScatterND expects 3 inputs (data, indices, updates), got {}",
                inputs.len()
            )));
        }

        let data_id = context.resolve_input(&inputs[0]);
        let indices_id = context.resolve_input(&inputs[1]);
        let updates_id = context.resolve_input(&inputs[2]);

        let outputs = node.output.as_slice();
        if outputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "ScatterND expects 1 output, got {}",
                outputs.len()
            )));
        }
        let out_name = outputs[0].to_string();
        let out_id = crate::onnx::convert::sanitize_identifier(&out_name);

        // WebNN scatterND has optional behaviors re: clamping/negative indices in the spec,
        // but this converter currently doesn’t expose scatterND options elsewhere, so we emit
        // the plain call. (If you later add options plumbing, this is where they'd go.)
        let n = Node {
            id: out_name.clone(),
            op: "scatterND".to_string(),
            inputs: vec![data_id, indices_id, updates_id],
            options: Map::new(),
            outputs: Some(vec![out_id.clone()]),
        };

        let mut res = ConversionResult::new(vec![n]);

        // Propagate dtype: ScatterND output dtype matches input dtype in both ONNX and WebNN.
        if let Some(dt) = context.value_types.get(&inputs[0].to_string()).cloned() {
            res.output_types.insert(out_name.clone(), dt);
        }
        res.output_mappings.insert(out_name, out_id);

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DataType;
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_string_attr(node: &mut NodeProto, name: &str, value: &str) {
        // Protobuf stores STRING attrs in the `s` bytes field.
        let attr = AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    // Own the backing maps so the ConversionContext can safely borrow them.
    //
    // NOTE: ConversionContext expects initializers as &TensorProto values, so we keep an
    // empty map of references by default. (These tests don't require initializers.)
    struct TestContext {
        initializers: std::collections::HashMap<String, &'static TensorProto>,
        value_shapes: std::collections::HashMap<String, Vec<i64>>,
        const_values: std::collections::HashMap<String, Vec<i64>>,
        value_ids: std::collections::HashMap<String, String>,
        value_types: std::collections::HashMap<String, DataType>,
    }

    impl TestContext {
        fn new() -> Self {
            Self {
                initializers: std::collections::HashMap::new(),
                value_shapes: std::collections::HashMap::new(),
                const_values: std::collections::HashMap::new(),
                value_ids: std::collections::HashMap::new(),
                value_types: std::collections::HashMap::new(),
            }
        }

        #[allow(dead_code)]
        fn add_initializer(&mut self, name: &str, tensor: TensorProto) {
            // Unit-test convenience: leak the tensor to obtain a &'static reference.
            // Fine in tests because the process ends immediately after.
            let leaked: &'static TensorProto = Box::leak(Box::new(tensor));
            self.initializers.insert(name.to_string(), leaked);
        }

        fn ctx(&self) -> ConversionContext<'_> {
            ConversionContext {
                initializers: &self.initializers,
                value_shapes: &self.value_shapes,
                value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
                const_values: &self.const_values,
                value_ids: &self.value_ids,
                value_types: &self.value_types,
            }
        }
    }

    #[test]
    fn test_scatter_handler_supports() {
        let handler = ScatterHandler;
        assert!(handler.supports("ScatterND"));
        assert!(!handler.supports("MatMul"));
        assert!(!handler.supports("ScatterElements"));
    }

    #[test]
    fn test_convert_scatter_nd_basic() {
        let handler = ScatterHandler;
        let node = create_test_node("ScatterND", vec!["data", "indices", "updates"], vec!["y"]);
        let tc = TestContext::new();
        let context = tc.ctx();

        let result = handler.convert(&node, &context).unwrap();

        assert_eq!(result.nodes.len(), 1);
        let n = &result.nodes[0];
        assert_eq!(n.op, "scatterND");
        assert_eq!(n.inputs, vec!["data", "indices", "updates"]);
        assert_eq!(n.id, "y");
        assert_eq!(n.outputs, Some(vec!["y".to_string()]));

        assert_eq!(result.output_mappings.get("y").unwrap(), "y");
    }

    #[test]
    fn test_convert_scatter_nd_reduction_unsupported() {
        let handler = ScatterHandler;
        let mut node = create_test_node("ScatterND", vec!["data", "indices", "updates"], vec!["y"]);
        add_string_attr(&mut node, "reduction", "add");
        let tc = TestContext::new();
        let context = tc.ctx();

        // Avoid unwrap_err() because it requires ConversionResult: Debug.
        match handler.convert(&node, &context) {
            Ok(_) => panic!("expected UnsupportedOp, got Ok"),
            Err(err) => match err {
                OnnxError::UnsupportedOp { op, .. } => {
                    assert!(op.contains("ScatterND(reduction=add)"));
                }
                other => panic!("expected UnsupportedOp, got {:?}", other),
            },
        }
    }

    #[test]
    fn test_convert_scatter_nd_invalid_input_count() {
        let handler = ScatterHandler;
        let node = create_test_node("ScatterND", vec!["data", "indices"], vec!["y"]);
        let tc = TestContext::new();
        let context = tc.ctx();

        match handler.convert(&node, &context) {
            Ok(_) => panic!("expected InvalidShape, got Ok"),
            Err(err) => match err {
                OnnxError::InvalidShape(msg) => assert!(msg.contains("expects 3 inputs")),
                other => panic!("expected InvalidShape, got {:?}", other),
            },
        }
    }

    #[test]
    fn test_convert_scatter_nd_invalid_output_count() {
        let handler = ScatterHandler;
        let node = create_test_node(
            "ScatterND",
            vec!["data", "indices", "updates"],
            vec!["y0", "y1"],
        );
        let tc = TestContext::new();
        let context = tc.ctx();

        match handler.convert(&node, &context) {
            Ok(_) => panic!("expected InvalidShape, got Ok"),
            Err(err) => match err {
                OnnxError::InvalidShape(msg) => assert!(msg.contains("expects 1 output")),
                other => panic!("expected InvalidShape, got {:?}", other),
            },
        }
    }
}
