use crate::ast::{ConstInit, Dimension, GraphJson, Node};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SerializeError {
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),
}

/// Serialization options.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerializeOptions {
    /// When true, emits `@quantized` annotation on the graph header to signal
    /// that tensors/constants are already quantized and should be preserved as-is.
    /// Downstream tools can use this hint to avoid dequantizing or re-quantizing.
    pub quantized: bool,
}

pub fn serialize_graph_to_wg_text(
    graph: &GraphJson,
    opts: SerializeOptions,
) -> Result<String, SerializeError> {
    let mut output = String::new();

    // Validate format
    if graph.format != "webnn-graph-json" {
        return Err(SerializeError::InvalidFormat(graph.format.clone()));
    }
    if graph.version != 1 && graph.version != 2 {
        return Err(SerializeError::UnsupportedVersion(graph.version));
    }

    // Header
    let name = graph.name.as_deref().unwrap_or("graph");
    let quantized_flag = if opts.quantized || graph.quantized {
        " @quantized"
    } else {
        ""
    };
    output.push_str(&format!(
        "webnn_graph \"{}\" v{}{} {{\n",
        escape_string(name),
        graph.version,
        quantized_flag
    ));

    // Inputs block
    if !graph.inputs.is_empty() {
        output.push_str("  inputs {\n");
        for (name, desc) in &graph.inputs {
            let dtype = desc.data_type.to_wg_text();
            let shape = serialize_shape(&desc.shape)?;
            output.push_str(&format!("    {}: {}{};\n", name, dtype, shape));
        }
        output.push_str("  }\n\n");
    }

    // Consts block
    if !graph.consts.is_empty() {
        output.push_str("  consts {\n");
        for (name, const_decl) in &graph.consts {
            let dtype = const_decl.data_type.to_wg_text();
            let shape = serialize_shape_u32(&const_decl.shape);
            let annotation = serialize_const_init(&const_decl.init);
            output.push_str(&format!("    {}: {}{}{}", name, dtype, shape, annotation));
            output.push_str(";\n");
        }
        output.push_str("  }\n\n");
    }

    // Nodes block
    if !graph.nodes.is_empty() {
        output.push_str("  nodes {\n");
        for node in &graph.nodes {
            output.push_str(&format!("    {}\n", serialize_node(node)));
        }
        output.push_str("  }\n\n");
    }

    // Outputs block
    output.push_str("  outputs {");
    if !graph.outputs.is_empty() {
        let outputs: Vec<String> = graph.outputs.keys().map(|k| format!(" {};", k)).collect();
        output.push_str(&outputs.join(""));
        output.push(' ');
    }
    output.push_str("}\n");

    output.push_str("}\n");
    Ok(output)
}

fn serialize_shape(shape: &[Dimension]) -> Result<String, SerializeError> {
    let mut dims: Vec<String> = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            Dimension::Static(v) => dims.push(v.to_string()),
            Dimension::Dynamic(d) => {
                dims.push(format!(
                    "dyn(\"{}\", {})",
                    escape_string(&d.name),
                    d.max_size
                ));
            }
        }
    }
    Ok(format!("[{}]", dims.join(", ")))
}

fn serialize_shape_u32(shape: &[u32]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", dims.join(", "))
}

fn serialize_const_init(init: &ConstInit) -> String {
    match init {
        ConstInit::Weights { r#ref } => {
            format!(" @weights(\"{}\")", escape_string(r#ref))
        }
        ConstInit::Scalar { value } => {
            format!(" @scalar({})", serialize_json_value(value))
        }
        ConstInit::InlineBytes { bytes } => {
            let nums: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
            format!(" @bytes([{}])", nums.join(", "))
        }
    }
}

fn serialize_node(node: &Node) -> String {
    let call = serialize_call(&node.op, &node.inputs, &node.options);

    if let Some(outputs) = &node.outputs {
        // Multi-output node: [a, b] = op(...)
        let out_list = outputs.join(", ");
        format!("[{}] = {};", out_list, call)
    } else {
        // Single output node: id = op(...)
        format!("{} = {};", node.id, call)
    }
}

fn serialize_call(
    op: &str,
    inputs: &[String],
    options: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let mut args = Vec::new();

    // Add positional inputs
    for input in inputs {
        args.push(input.clone());
    }

    // Add named options
    for (key, value) in options {
        args.push(format!("{}={}", key, serialize_json_value(value)));
    }

    format!("{}({})", op, args.join(", "))
}

fn serialize_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => format!("\"{}\"", escape_string(s)),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(serialize_json_value).collect();
            format!("[{}]", items.join(", "))
        }
        serde_json::Value::Object(_) => {
            // Objects are not typically used in the WG text format
            value.to_string()
        }
    }
}

fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        new_graph_json, to_dimension_vector, ConstDecl, ConstInit, DataType, Node, OperandDesc,
    };
    use crate::parser::parse_wg_text;

    #[test]
    fn test_serialize_simple_graph() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1, 10]),
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains(&format!("webnn_graph \"test\" v{}", g.version)));
        assert!(text.contains("inputs {"));
        assert!(text.contains("x: f32[1, 10];"));
        assert!(text.contains("nodes {"));
        assert!(text.contains("result = relu(x);"));
        assert!(text.contains("outputs { result; }"));
    }

    #[test]
    fn test_serialize_dynamic_input_shape() {
        let mut g = new_graph_json();
        g.name = Some("dyn".to_string());
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![
                    Dimension::Dynamic(crate::ast::DynamicDimension {
                        name: "batch_size".to_string(),
                        max_size: 8,
                    }),
                    Dimension::Static(128),
                ],
            },
        );
        g.outputs.insert("x".to_string(), "x".to_string());
        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains("x: f32[dyn(\"batch_size\", 8), 128];"));
    }

    #[test]
    fn test_serialize_weights_annotation() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());
        g.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );
        g.outputs.insert("W".to_string(), "W".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains("W: f32[10, 5] @weights(\"W\");"));
    }

    #[test]
    fn test_serialize_scalar_annotation() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());
        g.consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![1],
                init: ConstInit::Scalar {
                    value: serde_json::json!(3.5),
                },
            },
        );
        g.outputs.insert("scale".to_string(), "scale".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains("scale: f32[1] @scalar(3.5);"));
    }

    #[test]
    fn test_serialize_multi_output_node() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[10]),
            },
        );
        g.nodes.push(Node {
            id: "a".to_string(),
            op: "split".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: Some(vec!["a".to_string(), "b".to_string()]),
        });
        g.outputs.insert("a".to_string(), "a".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains("[a, b] = split(x);"));
    }

    #[test]
    fn test_serialize_node_options() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1, 10]),
            },
        );

        let mut options = serde_json::Map::new();
        options.insert("axis".to_string(), serde_json::json!(1));
        options.insert("keepdims".to_string(), serde_json::json!(true));

        g.nodes.push(Node {
            id: "result".to_string(),
            op: "softmax".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains("softmax(x,"));
        assert!(text.contains("axis=1"));
        assert!(text.contains("keepdims=true"));
    }

    #[test]
    fn test_serialize_various_dtypes() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());

        let dtypes = vec![
            ("f32_input", DataType::Float32),
            ("f16_input", DataType::Float16),
            ("i32_input", DataType::Int32),
            ("u32_input", DataType::Uint32),
            ("i64_input", DataType::Int64),
            ("u64_input", DataType::Uint64),
            ("i8_input", DataType::Int8),
            ("u8_input", DataType::Uint8),
        ];

        for (name, dtype) in dtypes {
            g.inputs.insert(
                name.to_string(),
                OperandDesc {
                    data_type: dtype,
                    shape: to_dimension_vector(&[1]),
                },
            );
        }
        g.outputs
            .insert("f32_input".to_string(), "f32_input".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains("f32_input: f32[1];"));
        assert!(text.contains("f16_input: f16[1];"));
        assert!(text.contains("i32_input: i32[1];"));
        assert!(text.contains("u32_input: u32[1];"));
        assert!(text.contains("i64_input: i64[1];"));
        assert!(text.contains("u64_input: u64[1];"));
        assert!(text.contains("i8_input: i8[1];"));
        assert!(text.contains("u8_input: u8[1];"));
    }

    #[test]
    fn test_roundtrip() {
        let input = r#"
webnn_graph "resnet_head" v1 {
  inputs {
    x: f32[1, 2048];
  }
  consts {
    W: f32[2048, 1000] @weights("W");
    b: f32[1000] @weights("b");
  }
  nodes {
    logits0 = matmul(x, W);
    logits = add(logits0, b);
    probs = softmax(logits, axis=1);
  }
  outputs { probs; }
}
"#;
        // Parse the text
        let graph = parse_wg_text(input).unwrap();

        // Serialize back to text
        let serialized = serialize_graph_to_wg_text(&graph, SerializeOptions::default()).unwrap();

        // Parse again to verify structure is preserved
        let graph2 = parse_wg_text(&serialized).unwrap();

        // Verify key properties
        assert_eq!(graph.name, graph2.name);
        assert_eq!(graph.inputs.len(), graph2.inputs.len());
        assert_eq!(graph.consts.len(), graph2.consts.len());
        assert_eq!(graph.nodes.len(), graph2.nodes.len());
        assert_eq!(graph.outputs.len(), graph2.outputs.len());
    }

    #[test]
    fn test_default_graph_name() {
        let mut g = new_graph_json();
        // No name set (None)
        g.outputs.insert("x".to_string(), "x".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains(&format!("webnn_graph \"graph\" v{}", g.version)));
    }

    #[test]
    fn test_string_escaping() {
        let mut g = new_graph_json();
        g.name = Some("test\"with\\quotes".to_string());
        g.outputs.insert("x".to_string(), "x".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains(&format!(
            "webnn_graph \"test\\\"with\\\\quotes\" v{}",
            g.version
        )));
    }

    #[test]
    fn test_value_types() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
            },
        );

        let mut options = serde_json::Map::new();
        options.insert("int_val".to_string(), serde_json::json!(42));
        options.insert("float_val".to_string(), serde_json::json!(3.5));
        options.insert("bool_val".to_string(), serde_json::json!(true));
        options.insert("null_val".to_string(), serde_json::json!(null));
        options.insert("array_val".to_string(), serde_json::json!([1, 2, 3]));

        g.nodes.push(Node {
            id: "result".to_string(),
            op: "test_op".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let text = serialize_graph_to_wg_text(&g, SerializeOptions::default()).unwrap();
        assert!(text.contains("int_val=42"));
        assert!(text.contains("float_val=3.5"));
        assert!(text.contains("bool_val=true"));
        assert!(text.contains("null_val=null"));
        assert!(text.contains("array_val=[1, 2, 3]"));
    }
}
