// Operator handler trait and registry

use crate::ast::{ConstDecl, Node};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::{NodeProto, TensorProto};
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod activation;
pub mod comparison;
pub mod conditional;
pub mod conv;
pub mod conversion;
pub mod elementwise;
pub mod matmul;
pub mod normalization;
pub mod pool;
pub mod reduction;
pub mod reshape;
pub mod scatter;
pub mod utility;

use activation::ActivationHandler;
use comparison::ComparisonHandler;
use conditional::ConditionalHandler;
use conv::ConvHandler;
use conversion::ConversionHandler;
use elementwise::ElementwiseHandler;
use matmul::MatMulHandler;
use normalization::NormalizationHandler;
use pool::PoolHandler;
use reduction::ReductionHandler;
use reshape::ReshapeHandler;
use scatter::ScatterHandler;
use utility::UtilityHandler;

/// Context for operator conversion
pub struct ConversionContext<'a> {
    /// Map of initializer names to TensorProto (for resolving constant shapes)
    pub initializers: &'a HashMap<String, &'a TensorProto>,
    /// Map of value names to their shapes (for shape inference)
    pub value_shapes: &'a HashMap<String, Vec<i64>>,
    /// Map of value names to shape dimensions preserving ONNX dim_param where available.
    pub value_shape_dims: &'a HashMap<String, Vec<crate::ast::Dimension>>,
    /// Map of value names to constant integer contents (for const folding)
    pub const_values: &'a HashMap<String, Vec<i64>>,
    /// Map of ONNX value names to WebNN value identifiers
    pub value_ids: &'a HashMap<String, String>,
    /// Map of value names to data types
    pub value_types: &'a HashMap<String, crate::ast::DataType>,
}

impl<'a> ConversionContext<'a> {
    pub fn resolve_input(&self, name: &str) -> String {
        if let Some(mapped) = self.value_ids.get(name) {
            return mapped.clone();
        }

        let sanitized = crate::onnx::convert::sanitize_identifier(name);
        if let Some(mapped) = self.value_ids.get(&sanitized) {
            return mapped.clone();
        }

        sanitized
    }

    pub fn resolve_shape(&self, name: &str) -> Option<&Vec<i64>> {
        let sanitized = crate::onnx::convert::sanitize_identifier(name);
        let trimmed = name.trim_start_matches('/');
        self.value_shapes
            .get(name)
            .or_else(|| self.value_shapes.get(&sanitized))
            .or_else(|| self.value_shapes.get(trimmed))
    }

    pub fn input_rank(&self, name: &str) -> Option<usize> {
        self.resolve_shape(name).map(|s| s.len())
    }
}

pub fn normalize_axis(axis: i64, rank: usize) -> Result<i64, OnnxError> {
    let rank_i64 = rank as i64;
    let normalized = if axis < 0 { axis + rank_i64 } else { axis };
    if normalized < 0 || normalized >= rank_i64 {
        return Err(OnnxError::InvalidShape(format!(
            "axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }
    Ok(normalized)
}

pub fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<i64>, OnnxError> {
    axes.iter().map(|&a| normalize_axis(a, rank)).collect()
}

pub fn normalize_axis_best_effort(axis: i64, rank: usize) -> i64 {
    normalize_axis(axis, rank).unwrap_or(axis)
}

pub fn normalize_axes_best_effort(axes: &[i64], rank: usize) -> Vec<i64> {
    axes.iter()
        .map(|&a| normalize_axis_best_effort(a, rank))
        .collect()
}

pub fn empty_value_shape_dims() -> &'static HashMap<String, Vec<crate::ast::Dimension>> {
    static EMPTY: OnceLock<HashMap<String, Vec<crate::ast::Dimension>>> = OnceLock::new();
    EMPTY.get_or_init(HashMap::new)
}

/// Results of converting a single ONNX node
#[derive(Default, Debug)]
pub struct ConversionResult {
    pub nodes: Vec<Node>,
    pub consts: Vec<(String, ConstDecl)>,
    /// ONNX output name -> WebNN value id
    pub output_mappings: HashMap<String, String>,
    /// ONNX output name -> data type
    pub output_types: HashMap<String, crate::ast::DataType>,
}

impl ConversionResult {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            nodes,
            consts: Vec::new(),
            output_mappings: HashMap::new(),
            output_types: HashMap::new(),
        }
    }
}

/// Trait for handling ONNX operator conversion
pub trait OpHandler {
    /// Check if this handler supports the given operator type
    fn supports(&self, op_type: &str) -> bool;

    /// Convert an ONNX node to WebNN node(s)
    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError>;
}

/// Registry for operator handlers
pub struct OpRegistry {
    handlers: Vec<Box<dyn OpHandler>>,
}

impl OpRegistry {
    /// Create a new operator registry with all handlers
    pub fn new() -> Self {
        let handlers: Vec<Box<dyn OpHandler>> = vec![
            Box::new(MatMulHandler),
            Box::new(ConvHandler),
            Box::new(PoolHandler),
            Box::new(ElementwiseHandler),
            Box::new(ComparisonHandler),
            Box::new(ConditionalHandler),
            Box::new(NormalizationHandler),
            Box::new(ReshapeHandler),
            Box::new(ConversionHandler),
            Box::new(UtilityHandler),
            Box::new(ReductionHandler),
            Box::new(ActivationHandler),
            Box::new(ScatterHandler),
        ];

        OpRegistry { handlers }
    }

    /// Convert an ONNX node using the appropriate handler
    pub fn convert_node<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();

        for handler in &self.handlers {
            if handler.supports(op_type) {
                return handler.convert(node, context);
            }
        }

        // No handler found
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "<unnamed>".to_string()
        };

        Err(OnnxError::UnsupportedOp {
            op: op_type.to_string(),
            node: node_name,
        })
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}
