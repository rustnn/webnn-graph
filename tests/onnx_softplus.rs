#![cfg(feature = "onnx")]

use std::collections::HashMap;

use prost::Message;
use webnn_graph::ast::{to_dimension_vector, DataType};
use webnn_graph::emit_js::emit_builder_js;
use webnn_graph::onnx::convert::{convert_onnx, ConvertOptions};
use webnn_graph::onnx::shape_inference::infer_static_shapes;
use webnn_graph::protos::onnx::{
    tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorProto_DataType, TensorShapeProto, TypeProto, ValueInfoProto,
};
use webnn_graph::validate::validate_graph;

fn softplus_model(dtype: TensorProto_DataType, dims: &[i64]) -> ModelProto {
    let tensor_type = TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: dtype.into(),
            shape: Some(TensorShapeProto {
                dim: dims
                    .iter()
                    .map(|&d| tensor_shape_proto::Dimension {
                        value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                        ..Default::default()
                    })
                    .collect(),
            }),
        })),
        ..Default::default()
    };
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            ..Default::default()
        }],
        graph: Some(GraphProto {
            input: vec![ValueInfoProto {
                name: "x".into(),
                r#type: Some(tensor_type.clone()),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".into(),
                r#type: Some(tensor_type),
                ..Default::default()
            }],
            node: vec![NodeProto {
                op_type: "Softplus".into(),
                name: "delta_activation".into(),
                input: vec!["x".into()],
                output: vec!["y".into()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn softplus_imports_as_native_unary_op_with_and_without_optimization() {
    for (onnx_dtype, dtype) in [
        (TensorProto_DataType::Float, DataType::Float32),
        (TensorProto_DataType::Float16, DataType::Float16),
    ] {
        for dims in [vec![1], vec![2, 3]] {
            let model = softplus_model(onnx_dtype, &dims);
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("softplus.onnx");
            std::fs::write(&path, model.encode_to_vec()).unwrap();

            for optimize in [false, true] {
                let graph = convert_onnx(
                    &path,
                    ConvertOptions {
                        optimize,
                        extract_weights: false,
                        ..Default::default()
                    },
                )
                .unwrap();
                validate_graph(&graph).unwrap();
                assert_eq!(graph.inputs["x"].data_type, dtype);
                let expected_dims: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
                assert_eq!(graph.inputs["x"].shape, to_dimension_vector(&expected_dims));
                assert_eq!(graph.nodes.len(), 1);
                assert_eq!(graph.nodes[0].op, "softplus");
                assert_eq!(graph.nodes[0].inputs, ["x"]);
                assert!(graph.nodes[0].options.is_empty());
                assert_eq!(graph.outputs["y"], "y");
                assert!(
                    emit_builder_js(&graph).contains("builder[\"softplus\"](env.get(\"x\"), {})")
                );
            }
        }
    }
}

#[test]
fn softplus_static_inference_preserves_shape_and_type_without_value_info() {
    for (onnx_dtype, dtype) in [
        (TensorProto_DataType::Float, DataType::Float32),
        (TensorProto_DataType::Float16, DataType::Float16),
    ] {
        for dims in [vec![], vec![2, 3]] {
            let mut model = softplus_model(onnx_dtype, &dims);
            // Do not let annotated outputs conceal missing shape propagation.
            model.graph.as_mut().unwrap().output[0].r#type = None;
            let inferred = infer_static_shapes(&model, &HashMap::new()).unwrap();
            assert_eq!(inferred.value_shapes.get("y"), Some(&dims));
            assert_eq!(inferred.value_types.get("y"), Some(&dtype));
        }
    }
}

#[test]
fn softplus_rejects_invalid_input_count() {
    for inputs in [vec![], vec!["x".into(), "x".into()]] {
        let mut model = softplus_model(TensorProto_DataType::Float, &[2]);
        model.graph.as_mut().unwrap().node[0].input = inputs;
        let converter = webnn_graph::onnx::convert::OnnxConverter::new(model).unwrap();
        let error = converter.convert(&ConvertOptions::default()).unwrap_err();
        assert!(error.to_string().contains("softplus expects 1 input"));
    }
}
