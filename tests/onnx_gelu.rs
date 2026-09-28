#![cfg(feature = "onnx")]

use prost::Message;
use std::collections::HashMap;
use webnn_graph::ast::{ConstInit, DataType, GraphJson};
use webnn_graph::emit_js::emit_builder_js;
use webnn_graph::onnx::convert::{convert_onnx, ConvertOptions, OnnxConverter, OnnxError};
use webnn_graph::protos::onnx::{
    tensor_shape_proto, type_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
};
use webnn_graph::validate::validate_graph;

fn model(domain: &str, opset: i64, dtype: i32, approximate: Option<&str>) -> ModelProto {
    let tensor_type = TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: dtype,
            shape: Some(TensorShapeProto {
                dim: vec![tensor_shape_proto::Dimension {
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(3)),
                    ..Default::default()
                }],
            }),
        })),
        ..Default::default()
    };
    let value = |name: &str| ValueInfoProto {
        name: name.to_string(),
        r#type: Some(tensor_type.clone()),
        ..Default::default()
    };
    let attribute = approximate
        .map(|value| AttributeProto {
            name: "approximate".to_string(),
            r#type: 3,
            s: value.as_bytes().to_vec(),
            ..Default::default()
        })
        .into_iter()
        .collect();
    ModelProto {
        ir_version: 9,
        graph: Some(GraphProto {
            name: "gelu_regression".to_string(),
            input: vec![value("x")],
            output: vec![value("y")],
            node: vec![NodeProto {
                name: "activation".to_string(),
                op_type: "Gelu".to_string(),
                domain: domain.to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute,
                ..Default::default()
            }],
            ..Default::default()
        }),
        opset_import: vec![OperatorSetIdProto {
            domain: domain.to_string(),
            version: opset,
        }],
        ..Default::default()
    }
}

#[test]
fn standard_gelu_opset20_converts_both_approximations() {
    for approximate in [None, Some("none"), Some("tanh")] {
        for optimize in [false, true] {
            let graph = OnnxConverter::new(model("", 20, 1, approximate))
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap();
            validate_graph(&graph).unwrap();
            if approximate == Some("tanh") {
                assert!(graph.nodes.iter().any(|node| node.op == "tanh"));
                assert!(graph.nodes.iter().all(|node| node.op != "gelu"));
            } else {
                assert_eq!(graph.nodes[0].op, "gelu");
            }
        }
    }
}

#[test]
fn legacy_exact_gelu_still_converts_validates_and_emits_js() {
    // The older com.microsoft Gelu is exact and has no approximate attribute.
    for (dtype, expected_type) in [(1, DataType::Float32), (10, DataType::Float16)] {
        for optimize in [false, true] {
            let graph = OnnxConverter::new(model("com.microsoft", 1, dtype, None))
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap();
            validate_graph(&graph).unwrap();
            assert_eq!(graph.inputs["x"].data_type, expected_type);
            assert_eq!(graph.nodes.len(), 1);
            assert_eq!(graph.nodes[0].op, "gelu");
            assert_eq!(graph.nodes[0].inputs, ["x"]);
            assert!(graph.nodes[0].options.is_empty());
            assert!(emit_builder_js(&graph).contains("builder[\"gelu\"](env.get(\"x\"), {})"));
        }
    }
}

#[test]
fn legacy_gelu_rejects_an_unsupported_approximation_attribute() {
    // Deliberately malformed legacy nodes used to reach the handler and have
    // their approximation discarded. Optimizing must not hide this error.
    for approximate in ["tanh", "invalid"] {
        for optimize in [false, true] {
            let error = OnnxConverter::new(model("com.microsoft", 1, 1, Some(approximate)))
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap_err();
            assert!(matches!(error, OnnxError::InvalidAttribute { .. }));
            assert!(error.to_string().contains("activation"));
        }
    }
}

fn convert(model: ModelProto, optimize: bool) -> GraphJson {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gelu.onnx");
    std::fs::write(&path, model.encode_to_vec()).unwrap();
    let graph = convert_onnx(
        &path,
        ConvertOptions {
            extract_weights: false,
            optimize,
            ..Default::default()
        },
    )
    .unwrap();
    validate_graph(&graph).unwrap();
    graph
}

fn evaluate(graph: &GraphJson, input: f32) -> f32 {
    let mut values = HashMap::from([("x".to_string(), input)]);
    for (name, constant) in &graph.consts {
        assert_eq!(constant.data_type, DataType::Float32);
        assert!(constant.shape.is_empty());
        let ConstInit::InlineBytes { bytes } = &constant.init else {
            panic!("inline scalar")
        };
        values.insert(
            name.clone(),
            f32::from_le_bytes(bytes.as_slice().try_into().unwrap()),
        );
    }
    for node in &graph.nodes {
        let x = values[&node.inputs[0]];
        let result = match node.op.as_str() {
            "mul" => x * values[&node.inputs[1]],
            "add" => x + values[&node.inputs[1]],
            "tanh" => x.tanh(),
            "identity" => x,
            "cast" => match node.options["to"].as_str().unwrap() {
                "float16" => half::f16::from_f32(x).to_f32(),
                "float32" => x,
                other => panic!("unexpected dtype {other}"),
            },
            other => panic!("unexpected operation {other}"),
        };
        values.insert(node.id.clone(), result);
    }
    values[&graph.outputs["y"]]
}

#[test]
fn tanh_gelu_matches_onnx_runtime_including_half_precision_tails() {
    // ONNX checker-valid Gelu-20, ORT 1.30 CPUExecutionProvider. Values at
    // +/-2.707 separate exact/tanh GELU; the far tails expose half overflow.
    let inputs = [
        -65504.0, -256.0, -20.0, -10.0, -5.0, -4.0, -3.0, -2.707, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1,
        0.5, 1.0, 2.0, 2.707, 3.0, 4.0, 5.0, 10.0, 20.0, 256.0, 65504.0,
    ];
    let expected: [f32; 25] = [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        -2.9802322e-7,
        -7.009506e-5,
        -0.0036375225,
        -0.008716196,
        -0.045402348,
        -0.15880796,
        -0.154286,
        -0.046017252,
        0.0,
        0.053982753,
        0.345714,
        0.841192,
        1.9545977,
        2.698284,
        2.9963627,
        3.99993,
        4.9999995,
        10.0,
        20.0,
        256.0,
        65504.0,
    ];
    let expected_half: [f32; 25] = [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        -2.9802322e-7,
        -7.021427e-5,
        -0.0036373138,
        -0.008712769,
        -0.045410156,
        -0.15881348,
        -0.15429688,
        -0.046020508,
        0.0,
        0.053955078,
        0.34570313,
        0.8413086,
        1.9550781,
        2.6992188,
        2.9960938,
        4.0,
        5.0,
        10.0,
        20.0,
        256.0,
        65504.0,
    ];
    for optimize in [false, true] {
        for (dtype, reference) in [(1, &expected), (10, &expected_half)] {
            let graph = convert(model("", 20, dtype, Some("tanh")), optimize);
            let scale = graph
                .consts
                .iter()
                .find(|(id, _)| id.contains("__gelu_scale"))
                .unwrap()
                .1;
            let ConstInit::InlineBytes { bytes } = &scale.init else {
                panic!("inline scale")
            };
            assert_eq!(
                u32::from_le_bytes(bytes.as_slice().try_into().unwrap()),
                0x3f4c422a
            );
            for (&input, &expected) in inputs.iter().zip(reference) {
                let input = if dtype == 10 {
                    half::f16::from_f32(input).to_f32()
                } else {
                    input
                };
                let actual = evaluate(&graph, input);
                let tolerance = if dtype == 10 {
                    2.0e-7 + expected.abs() * 1.0e-3
                } else {
                    2.0e-7 + expected.abs() * 1.0e-6
                };
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "dtype={dtype}, input={input}, actual={actual}, expected={expected}"
                );
            }
            assert!(evaluate(&graph, f32::NAN).is_nan());
            assert!(evaluate(&graph, f32::NEG_INFINITY).is_nan());
            assert_eq!(evaluate(&graph, f32::INFINITY), f32::INFINITY);
            assert_eq!(evaluate(&graph, -0.0).to_bits(), (-0.0_f32).to_bits());
        }
    }
}

#[test]
fn tanh_gelu_keeps_scalar_inputs_and_rejects_unknown_rank() {
    for optimize in [false, true] {
        for dtype in [1, 10] {
            let mut scalar = model("", 20, dtype, Some("tanh"));
            let graph = scalar.graph.as_mut().unwrap();
            for value in graph.input.iter_mut().chain(&mut graph.output) {
                let Some(type_proto::Value::TensorType(tensor)) =
                    &mut value.r#type.as_mut().unwrap().value
                else {
                    unreachable!()
                };
                tensor.shape.as_mut().unwrap().dim.clear();
            }
            let result = convert(scalar.clone(), optimize);
            assert!(result.inputs["x"].shape.is_empty());
            assert!((evaluate(&result, 1.0) - 0.8412).abs() < 0.0002);
            let Some(type_proto::Value::TensorType(tensor)) =
                &mut scalar.graph.as_mut().unwrap().input[0]
                    .r#type
                    .as_mut()
                    .unwrap()
                    .value
            else {
                unreachable!()
            };
            tensor.shape = None;
            assert!(OnnxConverter::new(scalar)
                .unwrap()
                .convert(&ConvertOptions::default())
                .is_err());
        }
    }
}

#[test]
fn tanh_gelu_private_values_cannot_shadow_future_outputs_or_inputs() {
    for optimize in [false, true] {
        let mut source = model("", 20, 1, Some("tanh"));
        let graph = source.graph.as_mut().unwrap();
        graph.node.push(NodeProto {
            op_type: "Identity".to_string(),
            input: vec!["y".to_string()],
            output: vec!["y__gelu_half".to_string()],
            ..Default::default()
        });
        let mut extra_input = graph.input[0].clone();
        extra_input.name = "y__gelu_square".to_string();
        graph.input.push(extra_input);
        let converted = convert(source, optimize);
        assert!(!converted.consts.contains_key("y__gelu_half"));
        assert_eq!(
            converted
                .nodes
                .iter()
                .filter(|node| node.id == "y__gelu_half")
                .count(),
            1
        );
        assert!(converted
            .nodes
            .iter()
            .all(|node| node.id != "y__gelu_square"));
        assert!((evaluate(&converted, 1.0) - 0.841192).abs() < 2.0e-7);
    }
}

#[test]
fn unsupported_opsets_types_and_invalid_gelu_modes_fail_closed() {
    for version in [10, 21] {
        assert!(matches!(
            OnnxConverter::new(model("", version, 1, None))
                .unwrap()
                .convert(&ConvertOptions::default()),
            Err(OnnxError::UnsupportedOpset { .. })
        ));
    }
    for dtype in [11, 16, 17, 18, 19, 20] {
        assert!(
            OnnxConverter::new(model("", 20, dtype, Some("tanh")))
                .unwrap()
                .convert(&ConvertOptions::default())
                .is_err(),
            "dtype {dtype}"
        );
    }
    for mode in ["invalid", "", "TANH"] {
        assert!(matches!(
            OnnxConverter::new(model("", 20, 1, Some(mode)))
                .unwrap()
                .convert(&ConvertOptions::default()),
            Err(OnnxError::InvalidAttribute { .. })
        ));
    }
    assert!(OnnxConverter::new(model("", 18, 1, None))
        .unwrap()
        .convert(&ConvertOptions::default())
        .is_err());
}

#[test]
fn emitted_tanh_gelu_executes_with_webnn_shaped_builder_signatures() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("Node.js unavailable: emitted JavaScript execution not tested");
        return;
    }
    for dtype in [1, 10] {
        let graph = convert(model("", 20, dtype, Some("tanh")), true);
        let mut child = Command::new("node")
            .args(["--input-type=module", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut script = include_str!("support/gelu_builder.mjs").to_string();
        script.push_str(&emit_builder_js(&graph));
        script.push_str("\nconst result = await buildGraph({x: [-2.707, 0, 2.707]}); console.log(JSON.stringify(result.y.data));\n");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(script.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual: Vec<f32> = serde_json::from_slice(&output.stdout).unwrap();
        for (&actual, input) in actual.iter().zip([-2.707_f32, 0.0, 2.707]) {
            let input = if dtype == 10 {
                half::f16::from_f32(input).to_f32()
            } else {
                input
            };
            assert!((actual - evaluate(&graph, input)).abs() < 1.0e-6);
        }
    }
}
