// ConstantOfShape operation evaluator
// Creates a tensor filled with a constant value, with shape specified by input

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::{NodeProto, TensorProto_DataType};

pub struct ConstantOfShapeEvaluator;

impl EvaluatorTrait for ConstantOfShapeEvaluator {
    fn op_type(&self) -> &str {
        "ConstantOfShape"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "ConstantOfShape" {
            return false;
        }

        // ConstantOfShape requires the shape input to be a constant
        if let Some(input_name) = node.input.as_slice().first() {
            return ctx.is_constant(input_name.as_str());
        }

        false
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let input_name =
            node.input
                .as_slice()
                .first()
                .ok_or_else(|| OnnxError::MissingAttribute {
                    attr: "input (shape)".to_string(),
                    op: "ConstantOfShape".to_string(),
                })?;

        let shape_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!(
                "Shape tensor '{}' not found in constants",
                input_name
            ))
        })?;

        // Extract shape values (should be a 1D int64 tensor)
        let shape_values = match &shape_tensor.data {
            TensorData::Int64(v) => v.clone(),
            TensorData::Int32(v) => v.iter().map(|&x| x as i64).collect(),
            _ => {
                return Err(OnnxError::ShapeInference(
                    "ConstantOfShape shape input must be int64 or int32".to_string(),
                ))
            }
        };

        // Get the value attribute (default is 0.0f if not specified)
        let mut fill_value_i64 = 0i64;
        let mut fill_value_f32 = 0.0f32;
        let mut data_type = TensorProto_DataType::Float as i32;

        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "value" {
                if let Some(value_tensor) = attr.t.as_ref() {
                    data_type = value_tensor.data_type;

                    match data_type {
                        x if x == TensorProto_DataType::Int64 as i32 => {
                            let raw = value_tensor.raw_data.as_slice();
                            if !raw.is_empty() && raw.len() >= 8 {
                                fill_value_i64 = i64::from_le_bytes([
                                    raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                                ]);
                            } else if !value_tensor.int64_data.as_slice().is_empty() {
                                fill_value_i64 = value_tensor.int64_data.as_slice()[0];
                            }
                        }
                        x if x == TensorProto_DataType::Float as i32 => {
                            let raw = value_tensor.raw_data.as_slice();
                            if !raw.is_empty() && raw.len() >= 4 {
                                fill_value_f32 =
                                    f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                            } else if !value_tensor.float_data.as_slice().is_empty() {
                                fill_value_f32 = value_tensor.float_data.as_slice()[0];
                            }
                        }
                        _ => {
                            return Err(OnnxError::ShapeInference(format!(
                                "Unsupported data type for ConstantOfShape value: {:?}",
                                data_type
                            )))
                        }
                    }
                }
            }
        }

        // Calculate total number of elements
        let numel = if shape_values.is_empty() {
            1
        } else {
            shape_values.iter().product::<i64>()
        };

        if numel < 0 {
            return Err(OnnxError::ShapeInference(format!(
                "Invalid shape for ConstantOfShape: {:?}",
                shape_values
            )));
        }

        // Create output tensor filled with the value
        let output = match data_type {
            x if x == TensorProto_DataType::Int64 as i32 => ConstantTensor {
                data: TensorData::Int64(vec![fill_value_i64; numel as usize]),
                shape: shape_values,
                data_type,
            },
            x if x == TensorProto_DataType::Float as i32 => ConstantTensor {
                data: TensorData::Float32(vec![fill_value_f32; numel as usize]),
                shape: shape_values,
                data_type,
            },
            _ => {
                return Err(OnnxError::ShapeInference(format!(
                    "Unsupported data type: {:?}",
                    data_type
                )))
            }
        };

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, TensorProto};
    use std::collections::HashMap;

    #[test]
    fn test_constant_of_shape_int64() {
        // Create shape tensor [2, 3]
        let shape_tensor = TensorProto {
            name: "shape".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![2],
            raw_data: vec![
                2, 0, 0, 0, 0, 0, 0, 0, // 2
                3, 0, 0, 0, 0, 0, 0, 0, // 3
            ],
            ..Default::default()
        };

        let leaked_shape: &'static TensorProto = Box::leak(Box::new(shape_tensor));

        let mut init_map = HashMap::new();
        init_map.insert("shape".to_string(), leaked_shape);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConstantOfShapeEvaluator;

        // Create ConstantOfShape node with value=5
        let mut node = NodeProto {
            op_type: "ConstantOfShape".to_string(),
            input: vec!["shape".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };

        // Add value attribute
        let value_tensor = TensorProto {
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![1],
            raw_data: vec![5, 0, 0, 0, 0, 0, 0, 0], // 5
            ..Default::default()
        };

        let attr = AttributeProto {
            name: "value".to_string(),
            t: Some(value_tensor),
            ..Default::default()
        };

        node.attribute.push(attr);

        assert!(evaluator.can_evaluate(&node, &ctx));

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        let output = &result[0];
        assert_eq!(output.shape, vec![2, 3]);
        assert_eq!(output.data_type, TensorProto_DataType::Int64 as i32);

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values.len(), 6);
            assert!(values.iter().all(|&v| v == 5));
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_constant_of_shape_float32() {
        // Create shape tensor [4]
        let shape_tensor = TensorProto {
            name: "shape".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![1],
            raw_data: vec![4, 0, 0, 0, 0, 0, 0, 0], // 4
            ..Default::default()
        };

        let leaked_shape: &'static TensorProto = Box::leak(Box::new(shape_tensor));

        let mut init_map = HashMap::new();
        init_map.insert("shape".to_string(), leaked_shape);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConstantOfShapeEvaluator;

        // Create ConstantOfShape node with value=1.5
        let mut node = NodeProto {
            op_type: "ConstantOfShape".to_string(),
            input: vec!["shape".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };

        // Add value attribute (1.5f32 = 0x3FC00000)
        let value_tensor = TensorProto {
            data_type: TensorProto_DataType::Float.into(),
            dims: vec![1],
            raw_data: vec![0x00, 0x00, 0xC0, 0x3F], // 1.5
            ..Default::default()
        };

        let attr = AttributeProto {
            name: "value".to_string(),
            t: Some(value_tensor),
            ..Default::default()
        };

        node.attribute.push(attr);

        let result = evaluator.evaluate(&node, &ctx).unwrap();

        if let TensorData::Float32(ref values) = result[0].data {
            assert_eq!(values.len(), 4);
            assert!(values.iter().all(|&v| (v - 1.5).abs() < 0.001));
        } else {
            panic!("Expected Float32 data");
        }
    }
}
