// Constant operation evaluator
// Extracts inline constant values from Constant nodes

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor,
};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::NodeProto;

pub struct ConstantEvaluator;

impl EvaluatorTrait for ConstantEvaluator {
    fn op_type(&self) -> &str {
        "Constant"
    }

    fn can_evaluate(&self, node: &NodeProto, _ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Constant" {
            return false;
        }

        // Check if node has a 'value' attribute with tensor data
        node.attribute
            .as_slice()
            .iter()
            .any(|a| a.name.as_str() == "value" && a.t.is_some())
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        _ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let value_attr = node
            .attribute
            .as_slice()
            .iter()
            .find(|a| a.name.as_str() == "value" && a.t.is_some())
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "value".to_string(),
                op: "Constant".to_string(),
            })?;

        let tensor_proto = value_attr.t.as_ref().unwrap();

        // Convert TensorProto to ConstantTensor
        let constant_tensor = ConstantTensor::from_tensor_proto(tensor_proto)?;

        Ok(vec![constant_tensor])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::constant_folding::TensorData;
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto, TensorProto_DataType};
    use std::collections::HashMap;

    #[test]
    fn test_constant_evaluator() {
        let init_map = HashMap::new();
        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConstantEvaluator;

        // Create a Constant node with inline tensor
        let tensor = TensorProto {
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![3],
            int64_data: vec![1, 2, 3],
            ..Default::default()
        };

        let value_attr = AttributeProto {
            name: "value".to_string(),
            t: Some(tensor),
            ..Default::default()
        };

        let mut node = NodeProto {
            op_type: "Constant".to_string(),
            output: vec!["const_output".to_string()],
            ..Default::default()
        };
        node.attribute.push(value_attr);

        // Check can_evaluate
        assert!(evaluator.can_evaluate(&node, &ctx));

        // Evaluate
        let result = evaluator.evaluate(&node, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        let output = &result[0];
        assert_eq!(output.shape, vec![3]);
        assert_eq!(output.data_type, TensorProto_DataType::Int64 as i32);

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![1, 2, 3]);
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_constant_scalar() {
        let init_map = HashMap::new();
        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConstantEvaluator;

        // Create a scalar constant
        let tensor = TensorProto {
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![], // Scalar
            int64_data: vec![42],
            ..Default::default()
        };

        let value_attr = AttributeProto {
            name: "value".to_string(),
            t: Some(tensor),
            ..Default::default()
        };

        let mut node = NodeProto {
            op_type: "Constant".to_string(),
            output: vec!["scalar_const".to_string()],
            ..Default::default()
        };
        node.attribute.push(value_attr);

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.shape, Vec::<i64>::new()); // Scalar
        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![42]);
        } else {
            panic!("Expected Int64 data");
        }
    }
}
