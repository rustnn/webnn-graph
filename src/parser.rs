use crate::ast::{
    new_graph_json, ConstDecl, ConstInit, DataType, Dimension, DynamicDimension, GraphJson, Node,
    OperandDesc,
};
use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "wg.pest"]
struct WGParser;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("parse error: {0}")]
    Pest(Box<pest::error::Error<Rule>>),
    #[error("invalid dtype: {0}")]
    BadDType(String),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("constant shapes must be static")]
    DynamicConstShape,
}

impl From<pest::error::Error<Rule>> for ParseError {
    fn from(err: pest::error::Error<Rule>) -> Self {
        ParseError::Pest(Box::new(err))
    }
}

type ParsedExpr = (String, Vec<String>, Map<String, Value>, Option<Vec<String>>);

pub fn parse_wg_text(input: &str) -> Result<GraphJson, ParseError> {
    let mut pairs = WGParser::parse(Rule::file, input)?;
    let file = pairs
        .next()
        .ok_or_else(|| ParseError::Internal("missing file".into()))?;

    let mut g = new_graph_json();
    let mut nodes: Vec<Node> = Vec::new();

    for p in file.into_inner() {
        match p.as_rule() {
            Rule::header => {
                // Extract graph name, version, quantized flag from header
                for inner in p.into_inner() {
                    match inner.as_rule() {
                        Rule::string => g.name = Some(unquote(inner.as_str())),
                        Rule::int => {
                            let version: u32 = inner
                                .as_str()
                                .parse()
                                .map_err(|e| ParseError::Internal(format!("bad version: {}", e)))?;
                            g.version = version;
                        }
                        Rule::quantized => g.quantized = true,
                        _ => {}
                    }
                }
            }
            Rule::inputs_block => parse_inputs_block(p, &mut g.inputs)?,
            Rule::consts_block => parse_consts_block(p, &mut g.consts)?,
            Rule::nodes_block => parse_nodes_block(p, &mut nodes)?,
            Rule::outputs_block => parse_outputs_block(p, &mut g.outputs)?,
            _ => {}
        }
    }

    g.nodes = nodes;
    Ok(g)
}

fn parse_inputs_block(
    p: Pair<Rule>,
    out: &mut BTreeMap<String, OperandDesc>,
) -> Result<(), ParseError> {
    for inner in p.into_inner() {
        if inner.as_rule() == Rule::input_decl {
            let mut it = inner.into_inner();
            let name = it.next().unwrap().as_str().to_string();
            let (dt, shape) = parse_ty(it.next().unwrap())?;
            out.insert(
                name,
                OperandDesc {
                    data_type: dt,
                    shape,
                },
            );
        }
    }
    Ok(())
}

fn parse_consts_block(
    p: Pair<Rule>,
    out: &mut BTreeMap<String, ConstDecl>,
) -> Result<(), ParseError> {
    for inner in p.into_inner() {
        if inner.as_rule() == Rule::const_decl {
            let mut it = inner.into_inner();
            let name = it.next().unwrap().as_str().to_string();
            let (dt, shape) = parse_ty(it.next().unwrap())?;

            let mut init: Option<ConstInit> = None;
            for ann in it {
                if ann.as_rule() == Rule::const_annot {
                    let text = ann.as_str();
                    if text.starts_with("@weights") {
                        let s = ann
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::string)
                            .map(|p| unquote(p.as_str()))
                            .unwrap_or_else(|| name.clone());
                        init = Some(ConstInit::Weights { r#ref: s });
                    } else if text.starts_with("@scalar") {
                        let n = ann
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::number)
                            .map(|p| parse_number_value(p.as_str()))
                            .unwrap_or(Value::Null);
                        init = Some(ConstInit::Scalar { value: n });
                    } else if text.starts_with("@bytes") {
                        let bytes = ann
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::byte_array)
                            .map(|pair| {
                                pair.into_inner()
                                    .filter(|p| p.as_rule() == Rule::int)
                                    .filter_map(|p| p.as_str().parse::<u32>().ok())
                                    .map(|v| v as u8)
                                    .collect::<Vec<u8>>()
                            })
                            .unwrap_or_default();
                        init = Some(ConstInit::InlineBytes { bytes });
                    }
                }
            }

            let init = init.unwrap_or(ConstInit::Weights {
                r#ref: name.clone(),
            });
            out.insert(
                name,
                ConstDecl {
                    data_type: dt,
                    shape: dims_to_static_shape(&shape)?,
                    init,
                },
            );
        }
    }
    Ok(())
}

fn parse_nodes_block(p: Pair<Rule>, out: &mut Vec<Node>) -> Result<(), ParseError> {
    for inner in p.into_inner() {
        if inner.as_rule() != Rule::stmt {
            continue;
        }
        let stmt = inner.into_inner().next().unwrap();
        match stmt.as_rule() {
            Rule::assign => out.push(parse_assign(stmt)?),
            Rule::multi_assign => out.push(parse_multi_assign(stmt)?),
            _ => {}
        }
    }
    Ok(())
}

fn parse_assign(p: Pair<Rule>) -> Result<Node, ParseError> {
    let mut it = p.into_inner();
    let id = it.next().unwrap().as_str().to_string();
    let (op, inputs, options, outputs) = parse_expr(it.next().unwrap())?;
    Ok(Node {
        id,
        op,
        inputs,
        options,
        outputs,
    })
}

fn parse_multi_assign(p: Pair<Rule>) -> Result<Node, ParseError> {
    let mut it = p.into_inner();
    let mut outs: Vec<String> = Vec::new();

    // first items are idents inside [...]
    // We receive them as a flat sequence of ident tokens due to grammar.
    // Collect until we hit expr.
    while let Some(next) = it.peek() {
        if next.as_rule() == Rule::expr {
            break;
        }
        let t = it.next().unwrap();
        if t.as_rule() == Rule::ident {
            outs.push(t.as_str().to_string());
        }
    }

    let expr = it
        .next()
        .ok_or_else(|| ParseError::Internal("missing expr in multi_assign".into()))?;
    let (op, inputs, options, _outputs_unused) = parse_expr(expr)?;
    // Use the first output name as the node id for uniqueness; keep real outputs in Node.outputs.
    let id = outs.first().cloned().unwrap_or_else(|| "tmp".into());
    Ok(Node {
        id,
        op,
        inputs,
        options,
        outputs: Some(outs),
    })
}

fn parse_expr(p: Pair<Rule>) -> Result<ParsedExpr, ParseError> {
    match p.as_rule() {
        Rule::expr => parse_expr(p.into_inner().next().unwrap()),
        Rule::call => parse_call(p),
        Rule::ident => Ok((
            String::new(),
            vec![p.as_str().to_string()],
            Map::new(),
            None,
        )),
        _ => Err(ParseError::Internal(format!(
            "unexpected expr rule: {:?}",
            p.as_rule()
        ))),
    }
}

fn parse_call(p: Pair<Rule>) -> Result<ParsedExpr, ParseError> {
    let mut it = p.into_inner();
    let op = it.next().unwrap().as_str().to_string();
    let mut inputs: Vec<String> = Vec::new();
    let mut options: Map<String, Value> = Map::new();

    // Debug: trace concat operations
    let is_concat = op == "concat";
    if is_concat {
        crate::debug_println!("[PARSER DEBUG] Parsing concat operation");
    }

    if let Some(args) = it.next() {
        if args.as_rule() == Rule::args {
            for (arg_idx, arg) in args.into_inner().enumerate() {
                if arg.as_rule() != Rule::arg {
                    continue;
                }
                let mut a = arg.into_inner().peekable();

                // Check if this is a named argument: ident '=' value
                let first = match a.next() {
                    Some(f) => f,
                    None => continue,
                };

                if is_concat {
                    crate::debug_println!(
                        "[PARSER DEBUG]   arg[{}]: first.rule={:?}, first.as_str()={}, has_next={}",
                        arg_idx,
                        first.as_rule(),
                        first.as_str(),
                        a.peek().is_some()
                    );
                    if let Some(next) = a.peek() {
                        crate::debug_println!(
                            "[PARSER DEBUG]   arg[{}]: next.rule={:?}, next.as_str()={}",
                            arg_idx,
                            next.as_rule(),
                            next.as_str()
                        );
                    }
                }

                if first.as_rule() == Rule::ident
                    && a.peek().is_some()
                    && a.peek().unwrap().as_rule() == Rule::value
                {
                    // Named argument
                    let key = first.as_str().to_string();
                    let val = parse_value(a.next().unwrap())?;
                    if is_concat {
                        crate::debug_println!("[PARSER DEBUG]   Named argument: {}={:?}", key, val);
                    }
                    options.insert(key, val);
                } else {
                    // Positional argument
                    if is_concat {
                        crate::debug_println!(
                            "[PARSER DEBUG]   Positional argument: rule={:?}",
                            first.as_rule()
                        );
                    }
                    // Handle bracketed input lists like `concat([a, b], ...)`: `parse_value`
                    // returns `Value::Array` for `[a, b]`, which we flatten into individual inputs.                    let v = parse_value(first)?;
                    match v {
                        Value::String(s) => inputs.push(s),
                        Value::Array(arr) => {
                            for item in arr {
                                match item {
                                    Value::String(s) => inputs.push(s),
                                    other => {
                                        if let Some(s) = other.as_str() {
                                            inputs.push(s.to_string());
                                        }
                                    }
                                }
                            }
                        }
                        other => {
                            if let Some(s) = other.as_str() {
                                inputs.push(s.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    if is_concat {
        crate::debug_println!(
            "[PARSER DEBUG] Concat parsed: inputs={:?}, options={:?}",
            inputs,
            options
        );
    }

    Ok((op, inputs, options, None))
}

fn parse_outputs_block(
    p: Pair<Rule>,
    out: &mut BTreeMap<String, String>,
) -> Result<(), ParseError> {
    // WG: outputs { probs }  OR outputs { a,b; }
    // We'll map each output name to itself.
    for inner in p.into_inner() {
        if inner.as_rule() == Rule::output_item {
            for item in inner.into_inner() {
                if item.as_rule() == Rule::ident {
                    let name = item.as_str().to_string();
                    out.insert(name.clone(), name);
                }
            }
        }
    }
    Ok(())
}

fn parse_ty(p: Pair<Rule>) -> Result<(DataType, Vec<Dimension>), ParseError> {
    let mut it = p.into_inner();
    let dt_s = it.next().unwrap().as_str();
    let dt = DataType::from_wg(dt_s).ok_or_else(|| ParseError::BadDType(dt_s.to_string()))?;
    let shape = parse_shape(it.next().unwrap())?;
    Ok((dt, shape))
}

fn parse_shape(p: Pair<Rule>) -> Result<Vec<Dimension>, ParseError> {
    let mut shape = Vec::new();
    for inner in p.into_inner() {
        if inner.as_rule() == Rule::shape_dim {
            let item = inner
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::Internal("shape_dim missing inner value".to_string()))?;
            match item.as_rule() {
                Rule::int => {
                    let v: u32 = item
                        .as_str()
                        .parse()
                        .map_err(|_| ParseError::Internal("bad int".into()))?;
                    shape.push(Dimension::Static(v));
                }
                Rule::dynamic_dim => {
                    let mut it = item.into_inner();
                    let name = it
                        .next()
                        .map(|p| unquote(p.as_str()))
                        .ok_or_else(|| ParseError::Internal("dynamic_dim missing name".into()))?;
                    let max_size: u32 = it
                        .next()
                        .ok_or_else(|| ParseError::Internal("dynamic_dim missing max".into()))?
                        .as_str()
                        .parse()
                        .map_err(|_| ParseError::Internal("dynamic_dim bad max".into()))?;
                    shape.push(Dimension::Dynamic(DynamicDimension { name, max_size }));
                }
                _ => return Err(ParseError::Internal("unexpected shape_dim rule".into())),
            }
        }
    }
    Ok(shape)
}

fn dims_to_static_shape(shape: &[Dimension]) -> Result<Vec<u32>, ParseError> {
    let mut out = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            Dimension::Static(v) => out.push(*v),
            Dimension::Dynamic(_) => return Err(ParseError::DynamicConstShape),
        }
    }
    Ok(out)
}

fn parse_value(p: Pair<Rule>) -> Result<Value, ParseError> {
    match p.as_rule() {
        Rule::value => parse_value(p.into_inner().next().unwrap()),
        Rule::literal => parse_value(p.into_inner().next().unwrap()),
        Rule::string => Ok(Value::String(unquote(p.as_str()))),
        Rule::number => Ok(parse_number_value(p.as_str())),
        Rule::boolean => Ok(Value::Bool(p.as_str() == "true")),
        Rule::null => Ok(Value::Null),
        Rule::array => {
            let mut arr = Vec::new();
            for inner in p.into_inner() {
                if inner.as_rule() == Rule::value {
                    arr.push(parse_value(inner)?);
                }
            }
            Ok(Value::Array(arr))
        }
        Rule::object => {
            let mut map = serde_json::Map::new();
            for inner in p.into_inner() {
                if inner.as_rule() == Rule::object_item {
                    let mut it = inner.into_inner();
                    let key_pair = it
                        .next()
                        .ok_or_else(|| ParseError::Internal("object key missing".into()))?;
                    let key = match key_pair.as_rule() {
                        Rule::string => unquote(key_pair.as_str()),
                        Rule::ident => key_pair.as_str().to_string(),
                        _ => {
                            return Err(ParseError::Internal(
                                "unexpected object key rule".to_string(),
                            ));
                        }
                    };
                    let value_pair = it
                        .next()
                        .ok_or_else(|| ParseError::Internal("object value missing".into()))?;
                    map.insert(key, parse_value(value_pair)?);
                }
            }
            Ok(Value::Object(map))
        }
        Rule::ident => Ok(Value::String(p.as_str().to_string())),
        _ => Err(ParseError::Internal(format!(
            "unexpected value rule: {:?}",
            p.as_rule()
        ))),
    }
}

fn parse_number_value(s: &str) -> Value {
    // Prefer i64 when exact, otherwise f64.
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        if let Ok(i) = s.parse::<i64>() {
            return Value::Number(i.into());
        }
    }
    Value::Number(serde_json::Number::from_f64(s.parse::<f64>().unwrap_or(0.0)).unwrap())
}

fn unquote(s: &str) -> String {
    let mut t = s.to_string();
    if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
        t.remove(0);
        t.pop();
    }
    t.replace("\\\"", "\"").replace("\\\\", "\\")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_graph() {
        let input = r#"
webnn_graph "test" v1 {
  inputs {
    x: f32[1, 10];
  }
  consts {
    W: f32[10, 5] @weights("W");
  }
  nodes {
    result = matmul(x, W);
  }
  outputs { result; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        assert_eq!(graph.format, "webnn-graph-json");
        assert_eq!(graph.version, 1);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.consts.len(), 1);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_parse_inputs() {
        let input = r#"
webnn_graph "test" v1 {
  inputs {
    x: f32[1, 10];
    y: i32[5];
  }
  nodes {}
  outputs { x; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        assert_eq!(graph.inputs.len(), 2);
        assert!(graph.inputs.contains_key("x"));
        assert!(graph.inputs.contains_key("y"));

        let x_desc = &graph.inputs["x"];
        assert_eq!(x_desc.data_type, DataType::Float32);
        assert_eq!(
            x_desc.shape,
            vec![Dimension::Static(1), Dimension::Static(10)]
        );

        let y_desc = &graph.inputs["y"];
        assert_eq!(y_desc.data_type, DataType::Int32);
        assert_eq!(y_desc.shape, vec![Dimension::Static(5)]);
    }

    #[test]
    fn test_parse_dynamic_input_shape() {
        let input = r#"
webnn_graph "test" v2 {
  inputs {
    x: f32[dyn("batch_size", 8), 128];
  }
  nodes {}
  outputs { x; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        let x_desc = &graph.inputs["x"];
        assert!(matches!(
            &x_desc.shape[0],
            Dimension::Dynamic(d) if d.name == "batch_size" && d.max_size == 8
        ));
        assert!(matches!(&x_desc.shape[1], Dimension::Static(128)));
    }

    #[test]
    fn test_parse_consts_with_weights() {
        let input = r#"
webnn_graph "test" v1 {
  inputs { x: f32[1]; }
  consts {
    W: f32[10, 5] @weights("W");
    b: f32[5] @weights("bias");
  }
  nodes {}
  outputs { x; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        assert_eq!(graph.consts.len(), 2);

        let w = &graph.consts["W"];
        assert_eq!(w.data_type, DataType::Float32);
        assert_eq!(w.shape, vec![10, 5]);
        assert!(matches!(&w.init, ConstInit::Weights { r#ref } if r#ref == "W"));

        let b = &graph.consts["b"];
        assert!(matches!(&b.init, ConstInit::Weights { r#ref } if r#ref == "bias"));
    }

    #[test]
    fn test_parse_consts_with_scalar() {
        let input = r#"
webnn_graph "test" v1 {
  inputs { x: f32[1]; }
  consts {
    scale: f32[1] @scalar(2.5);
  }
  nodes {}
  outputs { x; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        let scale = &graph.consts["scale"];
        match &scale.init {
            ConstInit::Scalar { value } => {
                assert_eq!(value.as_f64().unwrap(), 2.5);
            }
            _ => panic!("Expected scalar init"),
        }
    }

    #[test]
    fn test_parse_nodes() {
        let input = r#"
webnn_graph "test" v1 {
  inputs { x: f32[1, 2048]; }
  consts { W: f32[2048, 1000] @weights("W"); }
  nodes {
    result = matmul(x, W);
  }
  outputs { result; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        assert_eq!(graph.nodes.len(), 1);

        let node = &graph.nodes[0];
        assert_eq!(node.id, "result");
        assert_eq!(node.op, "matmul");
        assert_eq!(node.inputs, vec!["x", "W"]);
        assert!(node.options.is_empty());
    }

    #[test]
    fn test_parse_nodes_with_options() {
        let input = r#"
webnn_graph "test" v1 {
  inputs { x: f32[1, 10]; }
  nodes {
    result = softmax(x, axis=1);
  }
  outputs { result; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        let node = &graph.nodes[0];
        assert_eq!(node.op, "softmax");
        assert_eq!(node.inputs, vec!["x"]);
        assert_eq!(node.options.get("axis").unwrap().as_i64().unwrap(), 1);
    }

    #[test]
    fn test_parse_multi_assign() {
        let input = r#"
webnn_graph "test" v1 {
  inputs { x: f32[10]; }
  nodes {
    [a, b] = split(x);
  }
  outputs { a; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        let node = &graph.nodes[0];
        assert_eq!(node.id, "a");
        assert_eq!(node.op, "split");
        assert_eq!(node.outputs, Some(vec!["a".to_string(), "b".to_string()]));
    }

    #[test]
    fn test_parse_concat_bracket_input_list() {
        let input = r#"
webnn_graph "model" v1 {
  inputs {
    tensors_0: f32[2, 3, 4, 5];
    tensors_1: f32[2, 3, 4, 5];
  }
  nodes {
    [operand_1] = concat([tensors_0, tensors_1], axis=0);
  }
  outputs { operand_1; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        let node = &graph.nodes[0];
        assert_eq!(node.op, "concat");
        assert_eq!(node.inputs, vec!["tensors_0", "tensors_1"]);
        assert_eq!(node.options.get("axis").unwrap().as_i64().unwrap(), 0);
    }

    #[test]
    fn test_parse_outputs() {
        let input = r#"
webnn_graph "test" v1 {
  inputs { x: f32[1]; }
  nodes {
    a = relu(x);
    b = sigmoid(x);
  }
  outputs { a; b; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.outputs.get("a").unwrap(), "a");
        assert_eq!(graph.outputs.get("b").unwrap(), "b");
    }

    #[test]
    fn test_parse_invalid_dtype() {
        let input = r#"
webnn_graph "test" v1 {
  inputs { x: float32[1]; }
  nodes {}
  outputs { x; }
}
"#;
        let result = parse_wg_text(input);
        assert!(result.is_err());
        // The pest parser should fail because "float32" doesn't match the dtype rule
        match result {
            Err(ParseError::Pest(_)) => {}
            Err(e) => panic!("Expected Pest parse error, got: {:?}", e),
            Ok(_) => panic!("Expected error but parsing succeeded"),
        }
    }

    #[test]
    fn test_unquote() {
        assert_eq!(unquote(r#""hello""#), "hello");
        assert_eq!(unquote(r#""hello\"world""#), "hello\"world");
        assert_eq!(unquote(r#""path\\to\\file""#), "path\\to\\file");
        assert_eq!(unquote("no_quotes"), "no_quotes");
    }

    #[test]
    fn test_parse_number_value() {
        let int_val = parse_number_value("42");
        assert_eq!(int_val.as_i64().unwrap(), 42);

        let float_val = parse_number_value("3.12");
        assert_eq!(float_val.as_f64().unwrap(), 3.12);

        let sci_val = parse_number_value("1e-3");
        assert_eq!(sci_val.as_f64().unwrap(), 0.001);
    }

    #[test]
    fn test_parse_dollar_sign_identifiers() {
        let input = r#"
webnn_graph "test" v1 {
  inputs {
    x: f32[1, 10];
  }
  consts {
    $_weight: f32[10, 5] @weights("W");
  }
  nodes {
    $_temp = relu(x);
    result = matmul($_temp, $_weight);
  }
  outputs { result; }
}
"#;
        let graph = parse_wg_text(input).unwrap();
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.consts.len(), 1);
        assert!(graph.consts.contains_key("$_weight"));
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].id, "$_temp");
        assert_eq!(graph.nodes[1].id, "result");
        assert_eq!(graph.nodes[1].inputs, vec!["$_temp", "$_weight"]);
    }
}
