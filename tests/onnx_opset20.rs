#![cfg(feature = "onnx")]

use prost::Message;
use std::collections::HashMap;
use webnn_graph::onnx::convert::{convert_onnx, ConvertOptions, OnnxConverter, OnnxError};
use webnn_graph::onnx::shape_inference::infer_static_shapes;
use webnn_graph::protos::onnx::{
    tensor_shape_proto, type_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto,
};
use webnn_graph::validate::validate_graph;

fn value(name: &str, dtype: i32, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: dtype,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&dim| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(dim)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn model(op: &str, version: i64, input_shape: &[i64], output_shape: &[i64]) -> ModelProto {
    ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version,
        }],
        graph: Some(GraphProto {
            name: "opset_audit".to_string(),
            input: vec![value("x", 1, input_shape)],
            output: vec![value("y", 1, output_shape)],
            node: vec![NodeProto {
                op_type: op.to_string(),
                name: "audited_operator".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn integer(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: 2,
        i: value,
        ..Default::default()
    }
}

#[test]
fn opset19_and_20_preserve_average_pool_dilation() {
    for version in [19, 20] {
        let mut model = model("AveragePool", version, &[1, 1, 5, 5], &[1, 1, 3, 3]);
        model.graph.as_mut().unwrap().node[0].attribute = ["kernel_shape", "dilations"]
            .into_iter()
            .map(|name| AttributeProto {
                name: name.to_string(),
                r#type: 7,
                ints: vec![2, 2],
                ..Default::default()
            })
            .collect();
        let graph = OnnxConverter::new(model)
            .unwrap()
            .convert(&ConvertOptions::default())
            .unwrap();
        validate_graph(&graph).unwrap();
        assert_eq!(graph.nodes[0].op, "averagePool2d");
        assert_eq!(
            graph.nodes[0].options["dilations"],
            serde_json::json!([2, 2])
        );
    }
}

#[test]
fn modern_reductions_preserve_constant_axes_and_keepdims_zero() {
    for op in ["ReduceMean", "ReduceSum", "ReduceMin", "ReduceMax"] {
        for optimize in [false, true] {
            let mut model = model(op, 20, &[2, 3], &[2]);
            let graph = model.graph.as_mut().unwrap();
            graph.node[0].input.push("axes".to_string());
            graph.node[0].attribute = vec![integer("keepdims", 0)];
            graph.initializer.push(TensorProto {
                name: "axes".to_string(),
                data_type: 7,
                dims: vec![1],
                raw_data: (-1_i64).to_le_bytes().to_vec(),
                ..Default::default()
            });
            let inferred = infer_static_shapes(&model, &HashMap::new()).unwrap();
            assert_eq!(inferred.value_shapes["y"], [2]);
            let graph = OnnxConverter::new(model)
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap();
            validate_graph(&graph).unwrap();
            let reduction = graph.nodes.iter().find(|node| node.id == "y").unwrap();
            assert_eq!(reduction.options["axes"], serde_json::json!([1]));
            assert_eq!(reduction.options["keepDimensions"], false);
        }
    }
}

#[test]
fn modern_reductions_preserve_noop_and_reject_dynamic_axes() {
    for noop in [0, 1] {
        let output_shape = if noop == 0 { vec![1, 1] } else { vec![2, 3] };
        let mut model = model("ReduceMax", 20, &[2, 3], &output_shape);
        model.graph.as_mut().unwrap().node[0].attribute =
            vec![integer("noop_with_empty_axes", noop)];
        let inferred = infer_static_shapes(&model, &HashMap::new()).unwrap();
        assert_eq!(inferred.value_shapes["y"], output_shape);
        let graph = OnnxConverter::new(model)
            .unwrap()
            .convert(&ConvertOptions::default())
            .unwrap();
        assert_eq!(
            graph.nodes[0].op,
            if noop == 0 { "reduceMax" } else { "identity" }
        );
        assert!(!graph.nodes[0].options.contains_key("axes"));
    }
    let mut source = model("ReduceMin", 20, &[2, 3], &[2]);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].input.push("axes".to_string());
    graph.input.push(value("axes", 7, &[1]));
    assert!(matches!(
        OnnxConverter::new(source)
            .unwrap()
            .convert(&ConvertOptions::default()),
        Err(OnnxError::UnsupportedOp { .. })
    ));
}

#[test]
fn modern_reductions_accept_empty_axes_initializer() {
    for optimize in [false, true] {
        for noop in [0, 1] {
            let output_shape = if noop == 0 { vec![1, 1] } else { vec![2, 3] };
            let mut source = model("ReduceMean", 20, &[2, 3], &output_shape);
            let graph = source.graph.as_mut().unwrap();
            graph.node[0].input.push("axes".to_string());
            graph.node[0].attribute = vec![integer("noop_with_empty_axes", noop)];
            graph.initializer.push(TensorProto {
                name: "axes".to_string(),
                data_type: 7,
                dims: vec![0],
                ..Default::default()
            });
            let result = OnnxConverter::new(source)
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap();
            validate_graph(&result).unwrap();
            assert_eq!(
                result.nodes[0].op,
                if noop == 0 { "reduceMean" } else { "identity" }
            );
            assert!(!result.nodes[0].options.contains_key("axes"));
        }
    }
}

#[test]
fn cast19_saturation_does_not_admit_unsupported_float8_types() {
    for target in [10, 17, 18, 19, 20] {
        let mut source = model("Cast", 19, &[3], &[3]);
        source.graph.as_mut().unwrap().node[0].attribute =
            vec![integer("to", target), integer("saturate", 0)];
        let result = OnnxConverter::new(source)
            .unwrap()
            .convert(&ConvertOptions::default());
        if target == 10 {
            assert_eq!(result.unwrap().nodes[0].options["to"], "float16");
        } else {
            assert!(matches!(result, Err(OnnxError::TypeConversion(_))));
        }
    }
}

#[test]
fn opset20_does_not_silently_accept_unimplemented_operators() {
    for op in [
        "CastLike",
        "QuantizeLinear",
        "DequantizeLinear",
        "Resize",
        "GridSample",
        "AffineGrid",
        "DFT",
    ] {
        let error = OnnxConverter::new(model(op, 20, &[3], &[3]))
            .unwrap()
            .convert(&ConvertOptions::default())
            .unwrap_err();
        assert!(
            matches!(error, OnnxError::UnsupportedOp { .. }),
            "{op}: {error}"
        );
    }
}

#[test]
fn empty_axes_does_not_turn_unsupported_composite_reductions_into_identity() {
    // With noop=1, ONNX still applies log/abs/square for these operators.
    // Until their complete semantics are implemented, neither empty axes nor
    // optimization may turn them into a successful identity conversion.
    for op in [
        "ReduceLogSum",
        "ReduceSumSquare",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSumExp",
    ] {
        for optimize in [false, true] {
            for explicit_axes in [false, true] {
                let mut source = model(op, 20, &[6], &[6]);
                let graph = source.graph.as_mut().unwrap();
                graph.node[0].attribute = vec![integer("noop_with_empty_axes", 1)];
                if explicit_axes {
                    graph.node[0].input.push("axes".to_string());
                    graph.initializer.push(TensorProto {
                        name: "axes".to_string(),
                        data_type: 7,
                        dims: vec![0],
                        ..Default::default()
                    });
                }
                assert!(
                    matches!(
                        OnnxConverter::new(source)
                            .unwrap()
                            .convert(&ConvertOptions {
                                optimize,
                                ..Default::default()
                            }),
                        Err(OnnxError::UnsupportedOp { .. })
                    ),
                    "{op}"
                );
            }
        }
    }
}

#[test]
fn file_import_rejects_unsupported_types_before_outer_constant_folding() {
    use webnn_graph::onnx::constant_folding::{
        evaluators::get_evaluators, fold_constants_in_model,
    };
    let mut source = model("Cast", 20, &[], &[3]);
    let graph = source.graph.as_mut().unwrap();
    graph.input.clear();
    graph.node[0].input = vec!["double_value".to_string()];
    graph.node[0].attribute = vec![integer("to", 1)];
    graph.node.insert(
        0,
        NodeProto {
            op_type: "Constant".to_string(),
            output: vec!["double_value".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: 4,
                t: Some(TensorProto {
                    data_type: 11,
                    dims: vec![3],
                    double_data: vec![-4.0, 0.5, 4.0],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    let mut folded = source.clone();
    assert_eq!(
        fold_constants_in_model(&mut folded, &get_evaluators()).unwrap(),
        2
    );
    assert!(folded.graph.unwrap().node.is_empty());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("unsupported_foldable.onnx");
    std::fs::write(&path, source.encode_to_vec()).unwrap();
    for optimize in [false, true] {
        assert!(matches!(
            convert_onnx(
                &path,
                ConvertOptions {
                    extract_weights: false,
                    optimize,
                    ..Default::default()
                }
            ),
            Err(OnnxError::TypeConversion(_))
        ));
    }
}
