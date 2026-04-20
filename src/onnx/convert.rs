// Main ONNX to WebNN conversion logic

use crate::ast::{DataType, Dimension, DynamicDimension, GraphJson};
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    ModelProto, TensorProto, TensorProto_DataType,
};
use prost::Message;
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;
use webnn_onnx_utils::{data_types as utils_data_types, identifiers};

const MIN_SUPPORTED_OPSET: i64 = 11;
const MAX_SUPPORTED_OPSET: i64 = 18;

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("failed to read ONNX file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse ONNX protobuf: {0}")]
    ProtobufError(String),

    #[error("unsupported ONNX opset version {version} for domain '{domain}'")]
    UnsupportedOpset { domain: String, version: i64 },

    #[error("unsupported operator: {op} (node: {node})")]
    UnsupportedOp { op: String, node: String },

    #[error("missing required attribute: {attr} in {op}")]
    MissingAttribute { attr: String, op: String },

    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("type conversion error: {0}")]
    TypeConversion(#[from] webnn_onnx_utils::error::ConversionError),

    #[error("shape inference failed for node: {0}")]
    ShapeInference(String),
}

/// Sanitize ONNX identifiers for WebNN DSL compatibility
/// Replaces problematic characters that would confuse the parser
pub fn sanitize_identifier(name: &str) -> String {
    identifiers::sanitize_for_webnn(name)
}

/// Convert ONNX data type code to WebNN DataType using shared utilities
pub(crate) fn map_onnx_data_type(onnx_type: i32) -> Result<DataType, OnnxError> {
    if onnx_type == TensorProto_DataType::Bool as i32 {
        return Ok(DataType::Uint8);
    }

    let utils_dtype = utils_data_types::onnx_to_webnn(onnx_type)?;
    Ok(match utils_dtype {
        utils_data_types::DataType::Float32 => DataType::Float32,
        utils_data_types::DataType::Float16 => DataType::Float16,
        utils_data_types::DataType::Int32 => DataType::Int32,
        utils_data_types::DataType::Uint32 => DataType::Uint32,
        utils_data_types::DataType::Int64 => DataType::Int64,
        utils_data_types::DataType::Uint64 => DataType::Uint64,
        utils_data_types::DataType::Int8 => DataType::Int8,
        utils_data_types::DataType::Uint8 => DataType::Uint8,
    })
}

/// Infer output shape for an ONNX node based on its operation type and inputs
fn infer_shape(
    node: &crate::protos::onnx::NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let op = node.op_type.as_str();

    match op {
        // Unary operations that preserve shape
        "Cast" | "Relu" | "Tanh" | "Sigmoid" | "Erf" | "Softmax" | "Gelu" | "Exp" | "Log"
        | "Abs" | "Neg" | "Sqrt" | "LayerNormalization" | "Trilu" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            value_shapes.get(ins[0].as_str()).cloned()
        }

        // Binary operations with NumPy-style broadcasting semantics.
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let shape_a = value_shapes.get(ins[0].as_str());
            let shape_b = value_shapes.get(ins[1].as_str());

            match (shape_a, shape_b) {
                (Some(a), Some(b)) => {
                    let rank = a.len().max(b.len());
                    let mut out_rev = Vec::with_capacity(rank);
                    for i in 0..rank {
                        let da = a.get(a.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
                        let db = b.get(b.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
                        if da == db || da == 1 {
                            out_rev.push(db);
                        } else if db == 1 {
                            out_rev.push(da);
                        } else {
                            return None;
                        }
                    }
                    out_rev.reverse();
                    Some(out_rev)
                }
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                (None, None) => None,
            }
        }

        // MatMul (2D matrix multiplication)
        "MatMul" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let a_shape = value_shapes.get(ins[0].as_str())?;
            let b_shape = value_shapes.get(ins[1].as_str())?;

            // Handle 2D case: [M, K] @ [K, N] -> [M, N]
            if a_shape.len() >= 2 && b_shape.len() >= 2 {
                let m = a_shape[a_shape.len() - 2];
                let n = b_shape[b_shape.len() - 1];

                // For higher-dim inputs, preserve batch dimensions
                if a_shape.len() == 2 && b_shape.len() == 2 {
                    return Some(vec![m, n]);
                } else if a_shape.len() > 2 {
                    let mut result = a_shape[..a_shape.len() - 2].to_vec();
                    result.push(m);
                    result.push(n);
                    return Some(result);
                }
            }
            None
        }

        // Transpose preserves shape with permuted dimensions
        "Transpose" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Get perm attribute
            let perm: Vec<usize> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "perm")
                .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<usize>>())
                .unwrap_or_else(|| (0..input_shape.len()).rev().collect());

            // Apply permutation
            Some(perm.iter().map(|&i| input_shape[i]).collect())
        }

        // Reduce operations
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Check keepdims attribute (default is 1/true)
            let keepdims = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "keepdims")
                .and_then(|a| if a.i != 0 { Some(a.i != 0) } else { None })
                .unwrap_or(true);

            // Get axes attribute
            let axes: Vec<i64> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();

            if axes.is_empty() {
                // Reduce all dimensions
                if keepdims {
                    Some(vec![1; input_shape.len()])
                } else {
                    Some(vec![])
                }
            } else {
                // Reduce specific axes
                let mut output_shape = input_shape.clone();
                for &axis in &axes {
                    let idx = if axis < 0 {
                        (input_shape.len() as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    if idx < output_shape.len() {
                        if keepdims {
                            output_shape[idx] = 1;
                        } else {
                            output_shape[idx] = -1; // Mark for removal
                        }
                    }
                }
                if !keepdims {
                    output_shape.retain(|&d| d != -1);
                }
                Some(output_shape)
            }
        }

        // Gemm (generalized matrix multiplication)
        "Gemm" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let a_shape = value_shapes.get(ins[0].as_str())?;
            let b_shape = value_shapes.get(ins[1].as_str())?;

            if a_shape.len() != 2 || b_shape.len() != 2 {
                return None;
            }

            // Check transA and transB attributes
            let trans_a = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "transA")
                .and_then(|a| if a.i != 0 { Some(a.i != 0) } else { None })
                .unwrap_or(false);

            let trans_b = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "transB")
                .and_then(|a| if a.i != 0 { Some(a.i != 0) } else { None })
                .unwrap_or(false);

            let m = if trans_a { a_shape[1] } else { a_shape[0] };
            let n = if trans_b { b_shape[0] } else { b_shape[1] };

            Some(vec![m, n])
        }

        "Gather" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let data_shape = value_shapes.get(ins[0].as_str())?;
            let indices_shape = value_shapes.get(ins[1].as_str())?;

            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .and_then(|a| if a.i != 0 { Some(a.i) } else { None })
                .unwrap_or(0);

            if axis < 0 {
                axis += data_shape.len() as i64;
            }

            let axis_usize = axis as usize;
            if axis_usize > data_shape.len() {
                return None;
            }

            let mut output = Vec::new();
            output.extend_from_slice(&data_shape[..axis_usize]);
            output.extend(indices_shape.iter().cloned());
            if axis_usize < data_shape.len() {
                output.extend_from_slice(&data_shape[axis_usize + 1..]);
            }
            Some(output)
        }

        "Unsqueeze" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?.clone();
            let mut axes: Vec<i64> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();

            if axes.is_empty() {
                return None;
            }

            axes.sort();
            let mut output_shape = input_shape;
            for axis in axes {
                let idx = if axis < 0 {
                    (output_shape.len() as i64 + axis + 1) as usize
                } else {
                    axis as usize
                };
                if idx <= output_shape.len() {
                    output_shape.insert(idx, 1);
                }
            }
            Some(output_shape)
        }

        "Concat" => {
            let mut shapes = Vec::new();
            for inp in node.input.as_slice() {
                let shape = value_shapes.get(inp.as_str())?;
                shapes.push(shape.clone());
            }

            if shapes.is_empty() {
                return None;
            }

            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .and_then(|a| if a.i != 0 { Some(a.i) } else { None })
                .unwrap_or(0);

            if axis < 0 {
                axis += shapes[0].len() as i64;
            }
            let axis_usize = axis as usize;

            let mut output = shapes[0].clone();
            for shape in shapes.iter().skip(1) {
                if shape.len() != output.len() || axis_usize >= shape.len() {
                    return None;
                }
                output[axis_usize] += shape[axis_usize];
            }
            Some(output)
        }

        "Reshape" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?;
            let shape_input = ins[1].as_str();
            let mut target: Vec<i64> = if let Some(values) = const_values.get(shape_input) {
                values.clone()
            } else if let Some(shape_tensor) = initializers.get(shape_input) {
                if !shape_tensor.raw_data.as_slice().is_empty() {
                    if shape_tensor.data_type == TensorProto_DataType::Int32 as i32 {
                        shape_tensor
                            .raw_data
                            .as_slice()
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        shape_tensor
                            .raw_data
                            .as_slice()
                            .chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
                    }
                } else if !shape_tensor.int64_data.as_slice().is_empty() {
                    shape_tensor.int64_data.as_slice().to_vec()
                } else if !shape_tensor.int32_data.as_slice().is_empty() {
                    shape_tensor
                        .int32_data
                        .as_slice()
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            if target.is_empty() {
                return None;
            }

            if target.contains(&-1) {
                let total_input: i64 = input_shape.iter().product();
                let known: i64 = target.iter().filter(|&&d| d != -1).product();
                if known == 0 || total_input % known != 0 {
                    return None;
                }
                if let Some(idx) = target.iter().position(|&d| d == -1) {
                    target[idx] = total_input / known;
                }
            }

            Some(target)
        }

        "Slice" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?;

            let read_ints = |name: Option<&String>| -> Option<Vec<i64>> {
                if let Some(n) = name {
                    if let Some(v) = const_values.get(n) {
                        return Some(v.clone());
                    }
                    if let Some(t) = initializers.get(n) {
                        let raw = t.raw_data.as_slice();
                        if !raw.is_empty() {
                            if t.data_type == TensorProto_DataType::Int32 as i32 {
                                return Some(
                                    raw.chunks_exact(4)
                                        .map(|c| {
                                            i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64
                                        })
                                        .collect(),
                                );
                            } else {
                                return Some(
                                    raw.chunks_exact(8)
                                        .map(|c| {
                                            i64::from_le_bytes([
                                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                            ])
                                        })
                                        .collect(),
                                );
                            }
                        } else if !t.int64_data.as_slice().is_empty() {
                            return Some(t.int64_data.as_slice().to_vec());
                        } else if !t.int32_data.as_slice().is_empty() {
                            return Some(
                                t.int32_data.as_slice().iter().map(|&v| v as i64).collect(),
                            );
                        }
                    }
                }
                None
            };

            let starts = read_ints(ins.get(1))?;
            let ends = read_ints(ins.get(2))?;
            let axes =
                read_ints(ins.get(3)).unwrap_or_else(|| (0..input_shape.len() as i64).collect());
            let steps = read_ints(ins.get(4)).unwrap_or_else(|| vec![1; axes.len()]);

            if axes.len() != starts.len() || axes.len() != ends.len() || axes.len() != steps.len() {
                return None;
            }

            let mut output = input_shape.clone();
            for i in 0..axes.len() {
                let axis = if axes[i] < 0 {
                    (input_shape.len() as i64 + axes[i]) as usize
                } else {
                    axes[i] as usize
                };
                if axis >= output.len() {
                    return None;
                }

                let step = steps[i];
                if step != 1 {
                    return None;
                }

                let dim = input_shape[axis];
                let mut start = starts[i];
                let mut end = ends[i];

                if start < 0 {
                    start += dim;
                }
                if end < 0 {
                    end += dim;
                }

                start = start.max(0);
                end = end.min(dim);

                if end < start {
                    output[axis] = 0;
                } else {
                    output[axis] = end - start;
                }
            }

            Some(output)
        }

        _ => None,
    }
}

fn shape_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        if d < 0 {
            return None;
        }
        usize::try_from(d).ok().map(|v| acc.saturating_mul(v))
    })
}

fn const_shape_for_folding(
    name: &str,
    values: &[i64],
    value_shapes: &HashMap<String, Vec<i64>>,
) -> Vec<i64> {
    if let Some(shape) = value_shapes.get(name) {
        if shape_numel(shape) == Some(values.len()) {
            return shape.clone();
        }
    }

    if values.len() == 1 {
        Vec::new()
    } else {
        vec![values.len() as i64]
    }
}

fn broadcast_shape(shape_a: &[i64], shape_b: &[i64]) -> Option<Vec<i64>> {
    let rank = shape_a.len().max(shape_b.len());
    let mut out_rev = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = shape_a
            .get(shape_a.len().wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);
        let db = shape_b
            .get(shape_b.len().wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);
        if da <= 0 || db <= 0 {
            return None;
        }
        if da == db || da == 1 {
            out_rev.push(db);
        } else if db == 1 {
            out_rev.push(da);
        } else {
            return None;
        }
    }
    out_rev.reverse();
    Some(out_rev)
}

fn linear_index_for_broadcast_operand(
    out_linear_idx: usize,
    out_shape: &[i64],
    in_shape: &[i64],
) -> Option<usize> {
    if in_shape.is_empty() {
        return Some(0);
    }

    let in_rank = in_shape.len();
    let out_rank = out_shape.len();
    if in_rank > out_rank {
        return None;
    }

    let mut in_linear_idx = 0usize;
    let mut in_stride = 1usize;
    let mut rem = out_linear_idx;

    for out_axis_rev in 0..out_rank {
        let out_axis = out_rank - 1 - out_axis_rev;
        let out_dim = usize::try_from(out_shape[out_axis]).ok()?;
        if out_dim == 0 {
            return None;
        }
        let out_coord = rem % out_dim;
        rem /= out_dim;

        if out_axis_rev < in_rank {
            let in_axis = in_rank - 1 - out_axis_rev;
            let in_dim = usize::try_from(in_shape[in_axis]).ok()?;
            if in_dim == 0 {
                return None;
            }
            let in_coord = if in_dim == 1 { 0 } else { out_coord };
            in_linear_idx = in_linear_idx.saturating_add(in_coord.saturating_mul(in_stride));
            in_stride = in_stride.saturating_mul(in_dim);
        }
    }

    Some(in_linear_idx)
}

fn fold_binary_const_i64(
    op_type: &str,
    a_values: &[i64],
    b_values: &[i64],
    a_shape: &[i64],
    b_shape: &[i64],
) -> Option<(Vec<i64>, Vec<i64>)> {
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_numel = shape_numel(&out_shape)?;

    let mut out_values = Vec::with_capacity(out_numel);
    for out_idx in 0..out_numel {
        let a_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, a_shape)?;
        let b_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, b_shape)?;
        let av = *a_values.get(a_idx)?;
        let bv = *b_values.get(b_idx)?;
        let v = match op_type {
            "Add" => av + bv,
            "Sub" => av - bv,
            "Mul" => av * bv,
            "Div" => {
                if bv == 0 {
                    return None;
                }
                av / bv
            }
            "Equal" => {
                if av == bv {
                    1
                } else {
                    0
                }
            }
            _ => return None,
        };
        out_values.push(v);
    }

    Some((out_values, out_shape))
}

fn value_shape_dims_for<'a>(
    name: &str,
    value_shape_dims: &'a HashMap<String, Vec<Dimension>>,
) -> Option<&'a [Dimension]> {
    let sanitized = sanitize_identifier(name);
    let trimmed = name.trim_start_matches('/');
    value_shape_dims
        .get(name)
        .or_else(|| value_shape_dims.get(&sanitized))
        .or_else(|| value_shape_dims.get(trimmed))
        .map(Vec::as_slice)
}

fn dims_contain_dynamic(dims: &[Dimension]) -> bool {
    dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
}

pub(crate) fn parse_dynamic_dim_expr(dim_name: &str) -> (String, i64) {
    let s = dim_name.trim();
    if let Some((lhs, rhs)) = s.rsplit_once('+') {
        if let Ok(offset) = rhs.trim().parse::<i64>() {
            return (lhs.trim().to_string(), offset);
        }
    }
    if let Some((lhs, rhs)) = s.rsplit_once('-') {
        if let Ok(offset) = rhs.trim().parse::<i64>() {
            return (lhs.trim().to_string(), -offset);
        }
    }
    (s.to_string(), 0)
}

pub(crate) fn format_dynamic_dim_expr(base: &str, offset: i64) -> String {
    if offset > 0 {
        format!("{base} + {offset}")
    } else if offset < 0 {
        format!("{base} - {}", offset.abs())
    } else {
        base.to_string()
    }
}

fn parse_additive_dynamic_dim_expr(dim_name: &str) -> Option<(BTreeMap<String, i64>, i64)> {
    let expr = dim_name.trim();
    if expr.is_empty() {
        return None;
    }

    let normalized = expr.replace('+', " + ").replace('-', " - ");
    let mut terms = BTreeMap::new();
    let mut constant = 0i64;
    let mut sign = 1i64;
    let mut saw_term = false;

    for token in normalized.split_whitespace() {
        match token {
            "+" => sign = 1,
            "-" => sign = -1,
            _ => {
                saw_term = true;
                if let Ok(value) = token.parse::<i64>() {
                    constant += sign * value;
                } else {
                    *terms.entry(token.to_string()).or_insert(0) += sign;
                }
                sign = 1;
            }
        }
    }

    if !saw_term {
        return None;
    }

    terms.retain(|_, coeff| *coeff != 0);
    Some((terms, constant))
}

fn format_additive_dynamic_dim_expr(
    terms: &BTreeMap<String, i64>,
    constant: i64,
) -> Option<String> {
    if terms.is_empty() && constant == 0 {
        return None;
    }

    let mut out = String::new();
    for (name, coeff) in terms {
        for _ in 0..coeff.abs() {
            if out.is_empty() {
                if *coeff < 0 {
                    out.push_str("- ");
                }
                out.push_str(name);
            } else if *coeff < 0 {
                out.push_str(" - ");
                out.push_str(name);
            } else {
                out.push_str(" + ");
                out.push_str(name);
            }
        }
    }

    if constant != 0 {
        if out.is_empty() {
            out.push_str(&constant.to_string());
        } else if constant < 0 {
            out.push_str(" - ");
            out.push_str(&constant.abs().to_string());
        } else {
            out.push_str(" + ");
            out.push_str(&constant.to_string());
        }
    }

    Some(out)
}

fn is_runtime_resolvable_dynamic_dim_expr(dim_name: &str) -> bool {
    let s = dim_name.trim();
    if s.is_empty() || s.contains('*') || s.contains('/') {
        return false;
    }
    if let Some((lhs, rhs)) = s.rsplit_once('+') {
        return !lhs.trim().is_empty() && rhs.trim().parse::<i64>().is_ok();
    }
    if let Some((lhs, rhs)) = s.rsplit_once('-') {
        return !lhs.trim().is_empty() && rhs.trim().parse::<i64>().is_ok();
    }
    true
}

fn shift_dynamic_dimension(dim: &DynamicDimension, delta: i64) -> Option<DynamicDimension> {
    let (base, offset) = parse_dynamic_dim_expr(&dim.name);
    let name = format_dynamic_dim_expr(&base, offset.checked_add(delta)?);
    let shifted_max = (dim.max_size as i64).checked_add(delta)?.max(0);
    let max_size = u32::try_from(shifted_max).ok()?;
    Some(DynamicDimension { name, max_size })
}

pub(crate) fn dynamic_scalar_dimension_for_value(
    name: &str,
    value_shape_dims: &HashMap<String, Vec<Dimension>>,
) -> Option<DynamicDimension> {
    let dims = value_shape_dims_for(name, value_shape_dims)?;
    if dims.len() != 1 {
        return None;
    }
    match &dims[0] {
        Dimension::Dynamic(dim) => Some(dim.clone()),
        Dimension::Static(_) => None,
    }
}

fn dimension_vector_for_value(
    name: &str,
    const_values: &HashMap<String, Vec<i64>>,
    value_shape_dims: &HashMap<String, Vec<Dimension>>,
) -> Option<Vec<Dimension>> {
    if let Some(dims) = value_shape_dims_for(name, value_shape_dims) {
        return Some(dims.to_vec());
    }
    let values = const_values.get(name)?;
    values
        .iter()
        .map(|&v| u32::try_from(v).ok().map(Dimension::Static))
        .collect()
}

fn is_trivial_static_dimension_vector(dims: &[Dimension]) -> bool {
    dims.len() <= 3 && dims.iter().all(|d| matches!(d, Dimension::Static(1)))
}

fn combine_binary_dimension(
    op_type: &str,
    dynamic: &DynamicDimension,
    static_value: i64,
    dynamic_on_lhs: bool,
) -> Option<Dimension> {
    match op_type {
        "Add" => shift_dynamic_dimension(dynamic, static_value).map(Dimension::Dynamic),
        "Sub" if dynamic_on_lhs => {
            shift_dynamic_dimension(dynamic, -static_value).map(Dimension::Dynamic)
        }
        "Mul" if static_value == 0 => Some(Dimension::Static(0)),
        "Mul" if static_value == 1 => Some(Dimension::Dynamic(dynamic.clone())),
        "Mul" if static_value > 1 => Some(Dimension::Dynamic(DynamicDimension {
            name: if dynamic_on_lhs {
                format!("{} * {}", dynamic.name, static_value)
            } else {
                format!("{} * {}", static_value, dynamic.name)
            },
            max_size: dynamic.max_size.saturating_mul(static_value as u32),
        })),
        "Div" if dynamic_on_lhs && static_value == 1 => Some(Dimension::Dynamic(dynamic.clone())),
        "Div" if dynamic_on_lhs && static_value > 1 => Some(Dimension::Dynamic(DynamicDimension {
            name: format!("{} / {}", dynamic.name, static_value),
            max_size: dynamic.max_size / (static_value as u32),
        })),
        _ => None,
    }
}

fn combine_dynamic_dimensions(
    op_type: &str,
    lhs: &DynamicDimension,
    rhs: &DynamicDimension,
    lhs_value: i64,
    rhs_value: i64,
) -> Option<Dimension> {
    match op_type {
        "Add" | "Sub" => {
            let (mut terms, mut constant) = parse_additive_dynamic_dim_expr(&lhs.name)?;
            let (rhs_terms, rhs_constant) = parse_additive_dynamic_dim_expr(&rhs.name)?;
            let rhs_sign = if op_type == "Add" { 1 } else { -1 };

            for (name, coeff) in rhs_terms {
                *terms.entry(name).or_insert(0) += rhs_sign * coeff;
            }
            constant += rhs_sign * rhs_constant;
            terms.retain(|_, coeff| *coeff != 0);

            let value = if op_type == "Add" {
                lhs_value.checked_add(rhs_value)?
            } else {
                lhs_value.checked_sub(rhs_value)?
            };
            if terms.is_empty() {
                return u32::try_from(value).ok().map(Dimension::Static);
            }

            let name = format_additive_dynamic_dim_expr(&terms, constant)?;
            let max_size = u32::try_from(value).ok()?;
            Some(Dimension::Dynamic(DynamicDimension { name, max_size }))
        }
        _ => None,
    }
}

fn fold_binary_dynamic_dims(
    op_type: &str,
    a_values: &[i64],
    b_values: &[i64],
    a_shape: &[i64],
    b_shape: &[i64],
    a_dims: Option<&[Dimension]>,
    b_dims: Option<&[Dimension]>,
) -> Option<Vec<Dimension>> {
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_numel = shape_numel(&out_shape)?;
    let mut out_dims = Vec::with_capacity(out_numel);
    let mut has_dynamic = false;

    for out_idx in 0..out_numel {
        let a_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, a_shape)?;
        let b_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, b_shape)?;
        let av = *a_values.get(a_idx)?;
        let bv = *b_values.get(b_idx)?;
        let a_dim = a_dims.and_then(|dims| dims.get(a_idx));
        let b_dim = b_dims.and_then(|dims| dims.get(b_idx));

        let out_dim = match (a_dim, b_dim) {
            (Some(Dimension::Dynamic(dynamic)), Some(Dimension::Static(_)))
            | (Some(Dimension::Dynamic(dynamic)), None) => {
                let dim = combine_binary_dimension(op_type, dynamic, bv, true)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            (Some(Dimension::Static(_)), Some(Dimension::Dynamic(dynamic)))
            | (None, Some(Dimension::Dynamic(dynamic))) => {
                let dim = combine_binary_dimension(op_type, dynamic, av, false)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            (Some(Dimension::Dynamic(a_dynamic)), Some(Dimension::Dynamic(b_dynamic))) => {
                let dim = combine_dynamic_dimensions(op_type, a_dynamic, b_dynamic, av, bv)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            _ => {
                let value = match op_type {
                    "Add" => av + bv,
                    "Sub" => av - bv,
                    "Mul" => av * bv,
                    "Div" => {
                        if bv == 0 {
                            return None;
                        }
                        av / bv
                    }
                    _ => return None,
                };
                Dimension::Static(u32::try_from(value).ok()?)
            }
        };

        out_dims.push(out_dim);
    }

    has_dynamic.then_some(out_dims)
}

pub(crate) fn dynamic_range_length_dimension(
    start: i64,
    delta: i64,
    start_dim: Option<&DynamicDimension>,
    limit: &DynamicDimension,
) -> Option<DynamicDimension> {
    if delta != 1 {
        return None;
    }

    let (mut terms, mut constant) = parse_additive_dynamic_dim_expr(&limit.name)?;
    if let Some(start_dim) = start_dim {
        let (start_terms, start_constant) = parse_additive_dynamic_dim_expr(&start_dim.name)?;
        for (name, coeff) in start_terms {
            *terms.entry(name).or_insert(0) -= coeff;
        }
        constant -= start_constant;
    } else {
        constant -= start;
    }
    terms.retain(|_, coeff| *coeff != 0);
    if terms.is_empty() {
        return None;
    }

    let name = format_additive_dynamic_dim_expr(&terms, constant)?;
    if !is_runtime_resolvable_dynamic_dim_expr(&name) {
        return None;
    }

    let max_size = u32::try_from((limit.max_size as i64).checked_sub(start)?).ok()?;
    Some(DynamicDimension { name, max_size })
}

/// Conversion options for ONNX to WebNN
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Extract weights to external file (default: true)
    pub extract_weights: bool,
    /// Output file path for graph (.webnn or .json)
    pub output_path: String,
    /// Weights file path (.weights)
    pub weights_path: Option<String>,
    /// Manifest file path (.manifest.json)
    pub manifest_path: Option<String>,
    /// Override dynamic dimension values (e.g., batch_size=1, sequence_length=128)
    pub free_dim_overrides: HashMap<String, u32>,
    /// Enable constant folding and shape propagation optimizations
    pub optimize: bool,
    /// Experimental: preserve unresolved dynamic input dimensions in v2 graph metadata
    pub experimental_dynamic_inputs: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            extract_weights: true,
            output_path: "output.webnn".to_string(),
            weights_path: Some("output.weights".to_string()),
            manifest_path: Some("output.manifest.json".to_string()),
            free_dim_overrides: HashMap::new(),
            optimize: false,
            experimental_dynamic_inputs: false,
        }
    }
}

struct TensorInfo {
    _data_type: DataType,
    _shape: Vec<i64>,
}

/// Main converter structure
pub struct OnnxConverter {
    model: ModelProto,
    graph: GraphJson,
    _value_info: HashMap<String, TensorInfo>,
}

impl OnnxConverter {
    /// Create a new converter from an ONNX model
    pub fn new(model: ModelProto) -> Result<Self, OnnxError> {
        let graph_name = if let Some(graph) = &model.graph {
            if !graph.name.is_empty() {
                graph.name.as_str().to_string()
            } else {
                "graph".to_string()
            }
        } else {
            "graph".to_string()
        };

        let graph = GraphJson {
            format: "webnn-graph-json".to_string(),
            version: 1,
            name: Some(graph_name),
            quantized: false,
            inputs: BTreeMap::new(),
            consts: BTreeMap::new(),
            nodes: Vec::new(),
            outputs: BTreeMap::new(),
        };

        Ok(Self {
            model,
            graph,
            _value_info: HashMap::new(),
        })
    }

    /// Extract metadata from ONNX model
    pub fn extract_metadata(&self) -> Result<(), OnnxError> {
        if self.model.graph.is_none() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        let graph = self.model.graph.as_ref().unwrap();

        // Print basic info
        println!("Model name: {}", self.graph.name.as_ref().unwrap());
        println!("Inputs: {}", graph.input.as_slice().len());
        println!("Outputs: {}", graph.output.as_slice().len());
        println!("Nodes: {}", graph.node.as_slice().len());
        println!("Initializers: {}", graph.initializer.as_slice().len());

        Ok(())
    }

    /// Convert ONNX model to GraphJson
    pub fn convert(mut self, options: &ConvertOptions) -> Result<GraphJson, OnnxError> {
        if self.model.graph.is_none() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        // Validate opset imports
        for import in self.model.opset_import.as_slice() {
            let domain = import.domain.as_str();
            let version = import.version;
            let domain_name = if domain.is_empty() {
                "ai.onnx".to_string()
            } else {
                domain.to_string()
            };

            if (domain.is_empty() || domain == "ai.onnx")
                && !(MIN_SUPPORTED_OPSET..=MAX_SUPPORTED_OPSET).contains(&version)
            {
                return Err(OnnxError::UnsupportedOpset {
                    domain: domain_name,
                    version,
                });
            }
        }

        let onnx_graph = self.model.graph.as_ref().unwrap();
        let mut value_name_map: HashMap<String, String> = HashMap::new();
        let mut effective_overrides = options.free_dim_overrides.clone();
        let mut inference_overrides = effective_overrides.clone();
        let mut value_types: HashMap<String, DataType> = HashMap::new();

        // Merge overrides from model metadata if present
        for meta in self.model.metadata_props.as_slice() {
            if meta
                .key
                .as_str()
                .eq_ignore_ascii_case("freedimensionoverrides")
            {
                if let Ok(json) = serde_json::from_str::<JsonValue>(meta.value.as_str()) {
                    let obj = json
                        .get("freeDimensionOverrides")
                        .unwrap_or(&json)
                        .as_object()
                        .cloned();
                    if let Some(map) = obj {
                        for (name, value) in map {
                            if let Some(v) = value.as_u64() {
                                effective_overrides.entry(name.clone()).or_insert(v as u32);
                            }
                        }
                    }
                }
            }
        }

        // Process inputs (exclude initializers)
        let initializer_names: HashSet<String> = onnx_graph
            .initializer
            .as_slice()
            .iter()
            .map(|init| init.name.as_str().to_string())
            .collect();

        let default_dynamic_max_size: u32 = 65_535;
        let default_inference_dim_values: HashMap<&str, u32> =
            HashMap::from([("batch_size", 1), ("batch", 1), ("n", 1), ("b", 1)]);
        let dynamic_max_for_dim = |name: &str| -> u32 {
            let lower = name.to_ascii_lowercase();
            if lower.contains("past")
                || lower.contains("seq")
                || lower.contains("length")
                || lower == "s"
                || lower == "t"
            {
                4096
            } else if lower.contains("batch") || lower == "b" || lower == "n" {
                8
            } else {
                default_dynamic_max_size
            }
        };

        let resolve_dim_override =
            |dim_param: &str, overrides: &mut HashMap<String, u32>| -> Option<u32> {
                if let Some(v) = overrides.get(dim_param) {
                    return Some(*v);
                }

                let lower = dim_param.to_ascii_lowercase();
                if let Some(v) = overrides.get(&lower) {
                    return Some(*v);
                }
                None
            };
        let resolve_dim_for_inference =
            |dim_param: &str, overrides: &mut HashMap<String, u32>| -> Option<u32> {
                if let Some(v) = resolve_dim_override(dim_param, overrides) {
                    return Some(v);
                }
                let lower = dim_param.to_ascii_lowercase();
                if let Some(v) = default_inference_dim_values.get(lower.as_str()) {
                    overrides.insert(dim_param.to_string(), *v);
                    return Some(*v);
                }
                None
            };

        for input in onnx_graph.input.as_slice() {
            let raw_name = input.name.as_str().to_string();
            let name = sanitize_identifier(&raw_name);

            // Skip if this is an initializer (constant)
            if initializer_names.contains(&raw_name) {
                continue;
            }

            // Get type info
            if let Some(type_proto) = &input.r#type {
                if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                    let data_type = if tensor_type.elem_type != 0 {
                        let onnx_type = tensor_type.elem_type;
                        map_onnx_data_type(onnx_type)?
                    } else {
                        DataType::Float32 // Default
                    };

                    let shape = if let Some(shape_proto) = &tensor_type.shape {
                        let mut resolved: Vec<Dimension> = Vec::new();
                        for (idx, dim) in shape_proto.dim.iter().enumerate() {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            resolved.push(Dimension::Static(*v as u32));
                                        } else if options.experimental_dynamic_inputs {
                                            resolved.push(Dimension::Dynamic(DynamicDimension {
                                                name: format!("{}_dim{}", name, idx),
                                                max_size: default_dynamic_max_size,
                                            }));
                                        } else {
                                            let dim_hint = format!("{}_dim{}", name, idx);
                                            return Err(OnnxError::InvalidShape(format!(
                                                "Input '{}' has non-positive dim value ({}) at index {}. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                                raw_name,
                                                v,
                                                idx,
                                                dim_hint
                                            )));
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        if let Some(v) = resolve_dim_override(
                                            dim_param,
                                            &mut effective_overrides,
                                        ) {
                                            resolved.push(Dimension::Static(v));
                                        } else if options.experimental_dynamic_inputs {
                                            let max_size = dynamic_max_for_dim(dim_param);
                                            resolved.push(Dimension::Dynamic(DynamicDimension {
                                                name: dim_param.to_string(),
                                                max_size,
                                            }));
                                        } else if let Some(v) = resolve_dim_for_inference(
                                            dim_param,
                                            &mut inference_overrides,
                                        ) {
                                            effective_overrides
                                                .entry(dim_param.clone())
                                                .or_insert(v);
                                            resolved.push(Dimension::Static(v));
                                        } else {
                                            return Err(OnnxError::InvalidShape(format!(
                                                "Input '{}' has unresolved dynamic dimension '{}'. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                                raw_name, dim_param, dim_param
                                            )));
                                        }
                                    }
                                }
                            } else if options.experimental_dynamic_inputs {
                                resolved.push(Dimension::Dynamic(DynamicDimension {
                                    name: format!("{}_dim{}", name, idx),
                                    max_size: default_dynamic_max_size,
                                }));
                            } else {
                                let dim_hint = format!("{}_dim{}", name, idx);
                                return Err(OnnxError::InvalidShape(format!(
                                    "Input '{}' has unknown dimension at index {}. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                    raw_name, idx, dim_hint
                                )));
                            }
                        }
                        resolved
                    } else {
                        return Err(OnnxError::InvalidShape(format!(
                            "Input '{}' is missing shape information",
                            raw_name
                        )));
                    };

                    if shape.is_empty() {
                        continue;
                    }

                    self.graph.inputs.insert(
                        name.clone(),
                        crate::ast::OperandDesc {
                            data_type: data_type.clone(),
                            shape,
                        },
                    );

                    value_name_map.insert(raw_name.clone(), name.clone());
                    value_name_map.insert(name.clone(), name.clone());
                    value_types.insert(raw_name.clone(), data_type.clone());
                    value_types.insert(name.clone(), data_type);
                }
            }
        }

        // Process initializers (constants/weights)
        for initializer in onnx_graph.initializer.as_slice() {
            let name = sanitize_identifier(initializer.name.as_str());
            let raw_data = initializer.raw_data.as_slice();

            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !raw_data.is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                crate::debug_println!("Warning: Skipping initializer '{}' with no data", name);
                continue;
            }

            let onnx_type = initializer.data_type;
            let data_type = map_onnx_data_type(onnx_type)?;
            let shape: Vec<u32> = initializer
                .dims
                .as_slice()
                .iter()
                .map(|d| *d as u32)
                .collect();

            let init = if options.extract_weights {
                // External weights reference (use original name for weights file)
                crate::ast::ConstInit::Weights {
                    r#ref: sanitize_identifier(initializer.name.as_str()),
                }
            } else {
                // Inline bytes
                let bytes = raw_data.to_vec();
                crate::ast::ConstInit::InlineBytes { bytes }
            };

            self.graph
                .consts
                .entry(name.clone())
                .or_insert(crate::ast::ConstDecl {
                    data_type: data_type.clone(),
                    shape,
                    init,
                });

            value_name_map.insert(initializer.name.as_str().to_string(), name.clone());
            value_name_map.insert(name.clone(), name.clone());
            value_types.insert(initializer.name.as_str().to_string(), data_type.clone());
            value_types.insert(name, data_type);
        }

        // Process nodes using OpRegistry
        let registry = crate::onnx::ops::OpRegistry::new();

        // Build initializers map for resolving constant shapes
        let mut initializers_map = std::collections::HashMap::new();
        for initializer in onnx_graph.initializer.as_slice() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.raw_data.as_slice().is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                continue;
            }
            initializers_map.insert(initializer.name.as_str().to_string(), initializer);
        }

        // Build value_shapes map from value_info and inputs for shape inference
        let mut value_shapes = std::collections::HashMap::new();
        let mut value_shape_dims = std::collections::HashMap::new();

        // Add input shapes (already validated)
        for (raw_name, mapped_name) in value_name_map.clone() {
            if initializer_names.contains(&raw_name) {
                continue;
            }
            if let Some(input) = onnx_graph
                .input
                .as_slice()
                .iter()
                .find(|i| i.name.as_str() == raw_name)
            {
                if let Some(type_proto) = &input.r#type {
                    if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                        if let Some(shape_proto) = &tensor_type.shape {
                            let mut shape: Vec<i64> = Vec::new();
                            let mut unknown = false;
                            for dim in &shape_proto.dim {
                                if let Some(dim_value) = &dim.value {
                                    match dim_value {
                                        DimensionValue::DimValue(v) => {
                                            if *v > 0 {
                                                shape.push(*v);
                                            } else if options.experimental_dynamic_inputs {
                                                shape.push(default_dynamic_max_size as i64);
                                            } else {
                                                unknown = true;
                                                break;
                                            }
                                        }
                                        DimensionValue::DimParam(dim_param) => {
                                            if let Some(v) = resolve_dim_for_inference(
                                                dim_param,
                                                &mut inference_overrides,
                                            ) {
                                                shape.push(v as i64);
                                            } else if options.experimental_dynamic_inputs {
                                                shape.push(dynamic_max_for_dim(dim_param) as i64);
                                            } else {
                                                unknown = true;
                                                break;
                                            }
                                        }
                                    }
                                } else if options.experimental_dynamic_inputs {
                                    shape.push(default_dynamic_max_size as i64);
                                } else {
                                    unknown = true;
                                    break;
                                }
                            }
                            if !unknown && !shape.is_empty() {
                                value_shapes.insert(raw_name.clone(), shape.clone());
                                value_shapes.insert(mapped_name.clone(), shape);
                            }
                            let mut dims = Vec::new();
                            for dim in &shape_proto.dim {
                                if let Some(dim_value) = &dim.value {
                                    match dim_value {
                                        DimensionValue::DimValue(v) => {
                                            if *v > 0 {
                                                dims.push(crate::ast::Dimension::Static(*v as u32));
                                            }
                                        }
                                        DimensionValue::DimParam(dim_param) => {
                                            dims.push(crate::ast::Dimension::Dynamic(
                                                crate::ast::DynamicDimension {
                                                    name: dim_param.clone(),
                                                    max_size: dynamic_max_for_dim(dim_param),
                                                },
                                            ));
                                        }
                                    }
                                }
                            }
                            if !dims.is_empty() {
                                value_shape_dims.insert(raw_name.clone(), dims.clone());
                                value_shape_dims.insert(mapped_name.clone(), dims);
                            }
                        }
                    }
                }
            }
        }

        // Add initializer shapes
        for initializer in onnx_graph.initializer.as_slice() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.raw_data.as_slice().is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                continue;
            }
            let shape: Vec<i64> = initializer.dims.as_slice().to_vec();
            value_shapes.insert(initializer.name.as_str().to_string(), shape);
            let dims: Vec<crate::ast::Dimension> = initializer
                .dims
                .iter()
                .copied()
                .filter(|d| *d > 0)
                .map(|d| crate::ast::Dimension::Static(d as u32))
                .collect();
            if !dims.is_empty() {
                value_shape_dims.insert(initializer.name.as_str().to_string(), dims);
            }
        }

        // Add value_info shapes (intermediate tensors from shape inference)
        // Try to resolve dynamic dimensions using overrides
        for value_info in onnx_graph.value_info.as_slice() {
            if let Some(type_proto) = &value_info.r#type {
                if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                    if let Some(shape_proto) = &tensor_type.shape {
                        let mut shape: Vec<i64> = Vec::new();
                        let mut unknown = false;

                        for dim in &shape_proto.dim {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            shape.push(*v);
                                        } else if options.experimental_dynamic_inputs {
                                            shape.push(default_dynamic_max_size as i64);
                                        } else {
                                            unknown = true;
                                            break;
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        if let Some(v) = resolve_dim_for_inference(
                                            dim_param,
                                            &mut inference_overrides,
                                        ) {
                                            shape.push(v as i64);
                                        } else if options.experimental_dynamic_inputs {
                                            shape.push(dynamic_max_for_dim(dim_param) as i64);
                                        } else {
                                            unknown = true;
                                            break;
                                        }
                                    }
                                }
                            } else if options.experimental_dynamic_inputs {
                                shape.push(default_dynamic_max_size as i64);
                            } else {
                                unknown = true;
                                break;
                            }
                        }

                        if !unknown && !shape.is_empty() && shape.iter().all(|&d| d > 0) {
                            value_shapes.insert(value_info.name.as_str().to_string(), shape);
                        }
                        let mut dims = Vec::new();
                        for dim in &shape_proto.dim {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            dims.push(crate::ast::Dimension::Static(*v as u32));
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        dims.push(crate::ast::Dimension::Dynamic(
                                            crate::ast::DynamicDimension {
                                                name: dim_param.clone(),
                                                max_size: dynamic_max_for_dim(dim_param),
                                            },
                                        ));
                                    }
                                }
                            }
                        }
                        if !dims.is_empty() {
                            value_shape_dims.insert(value_info.name.as_str().to_string(), dims);
                        }
                    }
                }
            }
        }

        // Seed const values with integer initializers and Constant nodes
        let mut const_values: HashMap<String, Vec<i64>> = HashMap::new();
        for (name, initializer) in &initializers_map {
            if initializer.data_type == TensorProto_DataType::Int64 as i32
                || initializer.data_type == TensorProto_DataType::Int32 as i32
            {
                let raw = initializer.raw_data.as_slice();
                let values = if !raw.is_empty() {
                    if initializer.data_type == TensorProto_DataType::Int32 as i32 {
                        raw.chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
                    }
                } else if !initializer.int64_data.as_slice().is_empty() {
                    initializer.int64_data.as_slice().to_vec()
                } else if !initializer.int32_data.as_slice().is_empty() {
                    initializer
                        .int32_data
                        .as_slice()
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    Vec::new()
                };

                if !values.is_empty() {
                    const_values.insert(name.clone(), values);
                }
            }
        }

        for node in onnx_graph.node.as_slice() {
            if node.op_type.as_str() == "Constant" {
                if let Some(attr) = node
                    .attribute
                    .as_slice()
                    .iter()
                    .find(|a| a.name.as_str() == "value" && a.t.is_some())
                {
                    let tensor = attr.t.as_ref().unwrap();
                    if tensor.data_type == TensorProto_DataType::Int64 as i32
                        || tensor.data_type == TensorProto_DataType::Int32 as i32
                    {
                        let raw = tensor.raw_data.as_slice();
                        let values = if !raw.is_empty() {
                            if tensor.data_type == TensorProto_DataType::Int32 as i32 {
                                raw.chunks_exact(4)
                                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                    .collect()
                            } else {
                                raw.chunks_exact(8)
                                    .map(|c| {
                                        i64::from_le_bytes([
                                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                        ])
                                    })
                                    .collect()
                            }
                        } else if !tensor.int64_data.as_slice().is_empty() {
                            tensor.int64_data.as_slice().to_vec()
                        } else if !tensor.int32_data.as_slice().is_empty() {
                            tensor
                                .int32_data
                                .as_slice()
                                .iter()
                                .map(|&v| v as i64)
                                .collect()
                        } else {
                            Vec::new()
                        };

                        if let Some(out) = node.output.as_slice().first() {
                            if !values.is_empty() {
                                const_values.insert(out.to_string(), values);
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                }
            }
        }

        // Run the static shape/type inference scaffold to seed shapes/types/constants
        // before lowering. Errors surface early if dynamic dims remain.
        let mut dynamic_inference_attempts: HashSet<String> = HashSet::new();
        loop {
            match crate::onnx::shape_inference::infer_static_shapes(
                &self.model,
                &inference_overrides,
            ) {
                Ok(inferred) => {
                    // Initial seeding: use or_insert since these are the first values
                    // (no prior shapes to override)
                    for (k, v) in inferred.value_shapes {
                        value_shapes.entry(k).or_insert(v);
                    }
                    for (k, v) in inferred.value_types {
                        value_types.entry(k).or_insert(v);
                    }
                    for (k, v) in inferred.const_values {
                        // Use insert() instead of or_insert() to allow shape inference to correct
                        // earlier wrong values (e.g., Where operation heuristics)
                        if k.contains("rotary") && k.contains("Where") {
                            if let Some(old_val) = const_values.get(&k) {
                                crate::debug_println!(
                                    "[CONVERT] Overwriting {} from {:?} to {:?}",
                                    k,
                                    old_val,
                                    v
                                );
                            } else {
                                crate::debug_println!("[CONVERT] Inserting new {} = {:?}", k, v);
                            }
                        }
                        const_values.insert(k, v);
                    }
                    break;
                }
                Err(crate::onnx::shape_inference::ShapeInferenceError::DynamicDim {
                    input,
                    dim,
                }) => {
                    if options.experimental_dynamic_inputs
                        && !dynamic_inference_attempts.contains(dim.as_str())
                    {
                        let fallback = dynamic_max_for_dim(&dim);
                        inference_overrides.insert(dim.clone(), fallback);
                        dynamic_inference_attempts.insert(dim.clone());
                        crate::debug_println!(
                            "[CONVERT] Retrying static shape inference with inferred override {}={} \
                             (required by input '{}')",
                            dim,
                            fallback,
                            input
                        );
                        continue;
                    }
                    crate::debug_println!(
                        "[CONVERT] Skipping static shape inference due to unresolved dynamic dim '{}' on input '{}'",
                        dim,
                        input
                    );
                    break;
                }
                Err(e) => return Err(OnnxError::ShapeInference(e.to_string())),
            }
        }

        // Propagate shapes and fold constant shape expressions in a few passes
        for _ in 0..3 {
            if options.optimize {
                let max_iterations = 10;
                for iteration in 0..max_iterations {
                    let initial_count = value_shapes.len();

                    for onnx_node in onnx_graph.node.as_slice() {
                        let all_outputs_known = onnx_node
                            .output
                            .as_slice()
                            .iter()
                            .all(|out| value_shapes.contains_key(out.as_str()));
                        if all_outputs_known {
                            continue;
                        }

                        if let Some(inferred) =
                            infer_shape(onnx_node, &value_shapes, &initializers_map, &const_values)
                        {
                            if let Some(output_name) = onnx_node.output.as_slice().first() {
                                // Debug: track shape changes for layer 15 operations
                                if output_name.contains("layers_15_self_attn")
                                    && (output_name.contains("Reshape")
                                        || output_name.contains("Transpose"))
                                {
                                    crate::debug_println!(
                                        "[SHAPE DEBUG] {} {} -> {:?}",
                                        onnx_node.op_type.as_str(),
                                        output_name,
                                        inferred
                                    );
                                }
                                // Force the correct shape - shape inference computes exact output shape
                                value_shapes.insert(output_name.to_string(), inferred);
                            }
                        }
                    }

                    if value_shapes.len() == initial_count {
                        break;
                    }

                    if iteration == max_iterations - 1 {
                        crate::debug_println!(
                            "Warning: Shape propagation reached max iterations ({}/{})",
                            value_shapes.len(),
                            onnx_graph.node.as_slice().len()
                        );
                    }
                }
            }

            // If we know the input_ids shape (batch, seq), upgrade any lone hidden-dim
            // tensors (length-1 shapes) to [batch, seq, hidden] to unblock downstream
            // matmul/reshape resolution in decoder graphs that lost batch/seq dims.
            if let Some(ids_shape) = value_shapes.get("input_ids") {
                if ids_shape.len() == 2 {
                    let (batch, seq) = (ids_shape[0], ids_shape[1]);
                    let upgrades: Vec<(String, Vec<i64>)> = value_shapes
                        .iter()
                        .filter_map(|(k, v)| {
                            if v.len() == 1 && v[0] > 1 {
                                Some((k.clone(), vec![batch, seq, v[0]]))
                            } else {
                                None
                            }
                        })
                        .collect();
                    for (k, v) in upgrades {
                        value_shapes.insert(k, v);
                    }
                }
            }

            crate::debug_println!(
                "[debug] layer_norm shape {:?}",
                value_shapes.get("/decoder/block.0/layer.0/layer_norm/Mul_1_output_0")
            );
            crate::debug_println!(
                "[debug] matmul q shape {:?}",
                value_shapes.get("/decoder/block.0/layer.0/SelfAttention/q/MatMul_output_0")
            );
            crate::debug_println!(
                "[debug] input_ids shape {:?}",
                value_shapes.get("input_ids")
            );
            crate::debug_println!(
                "[debug] ln div shape {:?}",
                value_shapes.get("/decoder/block.0/layer.0/layer_norm/Div_output_0")
            );

            let consts_before = const_values.len();

            // DEBUG: Check value before propagation
            if let Some(val) = const_values.get("/model/rotary_emb/Where_output_0") {
                crate::debug_println!("[PROP BEFORE] /model/rotary_emb/Where_output_0 = {:?}", val);
            }

            // Extend const value map for const-foldable shapes
            for node in onnx_graph.node.as_slice() {
                let op_type = node.op_type.as_str();
                if op_type == "Shape" {
                    if let (Some(inp), Some(out)) = (
                        node.input.as_slice().first(),
                        node.output.as_slice().first(),
                    ) {
                        let out = out.to_string();
                        if let Some(shape) = value_shapes.get(inp).cloned() {
                            if shape.iter().all(|d| *d > 0) {
                                // Propagate dynamic dim metadata: Shape output is a 1-D
                                // tensor whose elements correspond to input dimensions.
                                if options.experimental_dynamic_inputs {
                                    let inp_s = inp.to_string();
                                    if let Some(dims) = value_shape_dims.get(&inp_s).or_else(|| {
                                        value_shape_dims.get(&sanitize_identifier(&inp_s))
                                    }) {
                                        // Each element of the Shape output corresponds to one
                                        // input dimension.  Build a 1-D dim vector where
                                        // dynamic input dims become Dynamic elements.
                                        let out_dims: Vec<crate::ast::Dimension> = dims
                                            .iter()
                                            .map(|d| match d {
                                                crate::ast::Dimension::Dynamic(dd) => {
                                                    crate::ast::Dimension::Dynamic(dd.clone())
                                                }
                                                crate::ast::Dimension::Static(v) => {
                                                    crate::ast::Dimension::Static(*v)
                                                }
                                            })
                                            .collect();
                                        value_shape_dims.insert(out.clone(), out_dims);
                                    }
                                }
                                const_values.insert(out.clone(), shape.clone());
                                let inferred_shape = vec![shape.len() as i64];
                                // Force the correct shape - Shape operation computes exact output shape
                                value_shapes.insert(out.clone(), inferred_shape.clone());
                                value_shapes.insert(sanitize_identifier(&out), inferred_shape);
                                value_types.insert(out, DataType::Int64);
                            }
                        }
                    }
                } else if op_type == "Gather" {
                    if let (Some(data_name), Some(indices_name), Some(out)) = (
                        node.input.as_slice().first(),
                        node.input.as_slice().get(1),
                        node.output.as_slice().first(),
                    ) {
                        if let (Some(data), Some(indices)) =
                            (const_values.get(data_name), const_values.get(indices_name))
                        {
                            let axis = node
                                .attribute
                                .as_slice()
                                .iter()
                                .find(|a| a.name.as_str() == "axis" && a.i != 0)
                                .map(|a| a.i)
                                .unwrap_or(0);

                            if axis == 0 {
                                let mut gathered = Vec::new();
                                let mut gathered_dims = Vec::new();
                                let data_dims = if options.experimental_dynamic_inputs {
                                    value_shape_dims
                                        .get(data_name)
                                        .or_else(|| {
                                            value_shape_dims.get(&sanitize_identifier(data_name))
                                        })
                                        .cloned()
                                } else {
                                    None
                                };
                                for &idx in indices {
                                    let i = if idx < 0 {
                                        (data.len() as i64 + idx) as usize
                                    } else {
                                        idx as usize
                                    };
                                    if let Some(v) = data.get(i) {
                                        gathered.push(*v);
                                        if let Some(ref dd) = data_dims {
                                            if let Some(dim) = dd.get(i) {
                                                gathered_dims.push(dim.clone());
                                            }
                                        }
                                    }
                                }
                                if !gathered.is_empty() {
                                    if options.experimental_dynamic_inputs
                                        && gathered_dims.len() == gathered.len()
                                        && gathered_dims
                                            .iter()
                                            .any(|d| matches!(d, crate::ast::Dimension::Dynamic(_)))
                                    {
                                        value_shape_dims.insert(out.to_string(), gathered_dims);
                                    }
                                    const_values.insert(out.to_string(), gathered.clone());
                                    let out_shape = if gathered.len() == 1 {
                                        Vec::new()
                                    } else {
                                        vec![gathered.len() as i64]
                                    };
                                    // Force the correct shape - Gather operation computes exact output shape
                                    value_shapes.insert(out.to_string(), out_shape.clone());
                                    value_shapes.insert(sanitize_identifier(out), out_shape);
                                    value_types.insert(out.to_string(), DataType::Int64);
                                }
                            }
                        }
                    }
                } else if matches!(op_type, "Add" | "Sub" | "Mul" | "Div") {
                    if node.input.as_slice().len() >= 2 {
                        if let (Some(a_name), Some(b_name), Some(out)) = (
                            node.input.as_slice().first(),
                            node.input.as_slice().get(1),
                            node.output.as_slice().first(),
                        ) {
                            let a = const_values.get(a_name);
                            let b = const_values.get(b_name);
                            if let (Some(a), Some(b)) = (a, b) {
                                let a_shape = const_shape_for_folding(a_name, a, &value_shapes);
                                let b_shape = const_shape_for_folding(b_name, b, &value_shapes);
                                if let Some((result_vals, out_shape)) =
                                    fold_binary_const_i64(op_type, a, b, &a_shape, &b_shape)
                                {
                                    if options.experimental_dynamic_inputs {
                                        let a_dims =
                                            value_shape_dims_for(a_name, &value_shape_dims);
                                        let b_dims =
                                            value_shape_dims_for(b_name, &value_shape_dims);
                                        if let Some(out_dims) = fold_binary_dynamic_dims(
                                            op_type, a, b, &a_shape, &b_shape, a_dims, b_dims,
                                        ) {
                                            value_shape_dims.insert(out.to_string(), out_dims);
                                        }
                                    }
                                    const_values.insert(out.to_string(), result_vals.clone());
                                    // Force the correct shape - Binary operations compute exact output shape
                                    value_shapes.insert(out.to_string(), out_shape.clone());
                                    value_shapes.insert(sanitize_identifier(out), out_shape);
                                    if let Some(dtype) = node
                                        .input
                                        .as_slice()
                                        .iter()
                                        .find_map(|i| value_types.get(i).cloned())
                                    {
                                        value_types.insert(out.to_string(), dtype);
                                    }
                                }
                            }
                        }
                    }
                } else if op_type == "Cast" || op_type == "Unsqueeze" || op_type == "Squeeze" {
                    if let (Some(inp), Some(out)) = (
                        node.input.as_slice().first(),
                        node.output.as_slice().first(),
                    ) {
                        if let Some(vals) = const_values.get(inp).cloned() {
                            // Propagate dynamic dim metadata
                            if options.experimental_dynamic_inputs {
                                if let Some(dims) = value_shape_dims
                                    .get(inp)
                                    .or_else(|| value_shape_dims.get(&sanitize_identifier(inp)))
                                    .cloned()
                                {
                                    value_shape_dims.insert(out.to_string(), dims);
                                }
                            }
                            const_values.insert(out.to_string(), vals.clone());
                            let out_shape = if vals.len() == 1 {
                                Vec::new()
                            } else {
                                vec![vals.len() as i64]
                            };
                            // Force the correct shape - Cast/Unsqueeze/Squeeze compute exact output shape
                            value_shapes.insert(out.to_string(), out_shape);
                            if let Some(dtype) = value_types.get(inp).cloned() {
                                value_types.insert(out.to_string(), dtype);
                            }
                        }
                    }
                } else if op_type == "Range" {
                    if node.input.as_slice().len() == 3 {
                        if let (Some(start_name), Some(limit_name), Some(delta_name)) = (
                            node.input.as_slice().first(),
                            node.input.as_slice().get(1),
                            node.input.as_slice().get(2),
                        ) {
                            if options.experimental_dynamic_inputs {
                                let start_dim = dynamic_scalar_dimension_for_value(
                                    start_name,
                                    &value_shape_dims,
                                );
                                if let Some(limit_dim) = dynamic_scalar_dimension_for_value(
                                    limit_name,
                                    &value_shape_dims,
                                ) {
                                    if let (Some(start_vals), Some(delta_vals), Some(out)) = (
                                        const_values.get(start_name),
                                        const_values.get(delta_name),
                                        node.output.as_slice().first(),
                                    ) {
                                        if !start_vals.is_empty() && !delta_vals.is_empty() {
                                            let start = start_vals[0];
                                            let delta = delta_vals[0];
                                            if let Some(range_dim) = dynamic_range_length_dimension(
                                                start,
                                                delta,
                                                start_dim.as_ref(),
                                                &limit_dim,
                                            ) {
                                                let out_shape = vec![range_dim.max_size as i64];
                                                value_shape_dims.insert(
                                                    out.to_string(),
                                                    vec![Dimension::Dynamic(range_dim.clone())],
                                                );
                                                value_shapes
                                                    .insert(out.to_string(), out_shape.clone());
                                                value_shapes
                                                    .insert(sanitize_identifier(out), out_shape);
                                                value_types
                                                    .insert(out.to_string(), DataType::Int64);
                                            }
                                        }
                                    }
                                    continue;
                                }
                            }

                            // Range(start, limit, delta) -> [start, start+delta, start+2*delta, ...]
                            if let (Some(start_vals), Some(limit_vals), Some(delta_vals)) = (
                                const_values.get(start_name),
                                const_values.get(limit_name),
                                const_values.get(delta_name),
                            ) {
                                if !start_vals.is_empty()
                                    && !limit_vals.is_empty()
                                    && !delta_vals.is_empty()
                                {
                                    let start = start_vals[0];
                                    let limit = limit_vals[0];
                                    let delta = delta_vals[0];

                                    let mut range_vals = Vec::new();
                                    if delta > 0 {
                                        let mut current = start;
                                        while current < limit {
                                            range_vals.push(current);
                                            current += delta;
                                        }
                                    } else if delta < 0 {
                                        let mut current = start;
                                        while current > limit {
                                            range_vals.push(current);
                                            current += delta;
                                        }
                                    }

                                    if let Some(out) = node.output.as_slice().first() {
                                        const_values.insert(out.to_string(), range_vals.clone());
                                        let out_shape = vec![range_vals.len() as i64];
                                        // Force the correct shape - Range computes exact output shape
                                        value_shapes.insert(out.to_string(), out_shape.clone());
                                        value_shapes.insert(sanitize_identifier(out), out_shape);
                                        value_types.insert(out.to_string(), DataType::Int64);
                                    }
                                }
                            }
                        }
                    }
                } else if op_type == "Concat" {
                    // Concatenate constant inputs (often used to build shape tensors)
                    if let Some(out) = node.output.as_slice().first() {
                        let mut concatenated: Vec<i64> = Vec::new();
                        let mut all_const = true;
                        for inp in node.input.as_slice() {
                            if let Some(vals) = const_values.get(inp) {
                                concatenated.extend_from_slice(vals);
                            } else {
                                all_const = false;
                                break;
                            }
                        }

                        // Handle axis=0 or axis=-1 (common for shape building)
                        let axis = node
                            .attribute
                            .as_slice()
                            .iter()
                            .find(|a| a.name.as_str() == "axis" && a.i != 0)
                            .map(|a| a.i)
                            .unwrap_or(0);

                        if all_const && (axis == 0 || axis == -1) {
                            if out.contains("rotary") && out.contains("Where") {
                                crate::debug_println!(
                                    "[CONCAT WRITE] Writing {} = {:?}",
                                    out,
                                    concatenated
                                );
                            }
                            // Propagate dynamic dim metadata through concat
                            if options.experimental_dynamic_inputs {
                                let mut concat_dims: Vec<crate::ast::Dimension> = Vec::new();
                                let mut has_dynamic = false;
                                for inp in node.input.as_slice() {
                                    let inp_s = inp.to_string();
                                    if let Some(dims) = value_shape_dims.get(&inp_s).or_else(|| {
                                        value_shape_dims.get(&sanitize_identifier(&inp_s))
                                    }) {
                                        for d in dims {
                                            if matches!(d, crate::ast::Dimension::Dynamic(_)) {
                                                has_dynamic = true;
                                            }
                                            concat_dims.push(d.clone());
                                        }
                                    } else if let Some(vals) = const_values.get(inp) {
                                        for v in vals {
                                            concat_dims
                                                .push(crate::ast::Dimension::Static(*v as u32));
                                        }
                                    }
                                }
                                if has_dynamic && concat_dims.len() == concatenated.len() {
                                    value_shape_dims.insert(out.to_string(), concat_dims);
                                }
                            }
                            const_values.insert(out.to_string(), concatenated.clone());
                            let out_shape = vec![concatenated.len() as i64];
                            // Force the correct shape - Concat computes exact output shape
                            value_shapes.insert(out.to_string(), out_shape.clone());
                            value_shapes.insert(sanitize_identifier(out), out_shape);
                            value_types.insert(out.to_string(), DataType::Int64);
                        }
                    }
                } else if op_type == "ConstantOfShape" {
                    // ConstantOfShape(shape) -> tensor filled with constant value
                    if let Some(shape_name) = node.input.as_slice().first() {
                        let dynamic_output_dims = if options.experimental_dynamic_inputs {
                            value_shape_dims_for(shape_name, &value_shape_dims)
                                .map(|dims| dims.to_vec())
                                .filter(|dims| dims_contain_dynamic(dims))
                        } else {
                            None
                        };

                        if let (Some(out), Some(dims)) =
                            (node.output.as_slice().first(), dynamic_output_dims.as_ref())
                        {
                            value_shape_dims.insert(out.to_string(), dims.to_vec());
                            const_values.remove(out.as_str());
                        }

                        if let Some(shape_vals) = const_values.get(shape_name).cloned() {
                            // Get the fill value from attributes (default is 0)
                            let mut fill_value = 0i64;
                            for attr in node.attribute.as_slice() {
                                if attr.name.as_str() == "value" {
                                    if let Some(value_tensor) = attr.t.as_ref() {
                                        if value_tensor.data_type
                                            == crate::protos::onnx::TensorProto_DataType::Int64
                                                as i32
                                        {
                                            let raw = value_tensor.raw_data.as_slice();
                                            if !raw.is_empty() && raw.len() >= 8 {
                                                fill_value = i64::from_le_bytes([
                                                    raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
                                                    raw[6], raw[7],
                                                ]);
                                            } else if !value_tensor.int64_data.as_slice().is_empty()
                                            {
                                                fill_value = value_tensor.int64_data.as_slice()[0];
                                            }
                                        }
                                    }
                                }
                            }

                            // Calculate number of elements
                            let numel = if shape_vals.is_empty() {
                                1
                            } else {
                                shape_vals.iter().product::<i64>()
                            };

                            if numel > 0 && numel < 1_000_000 {
                                // Reasonable size limit
                                let filled_tensor = vec![fill_value; numel as usize];
                                if let Some(out) = node.output.as_slice().first() {
                                    let should_keep_const = dynamic_output_dims
                                        .as_ref()
                                        .is_none_or(|dims| !dims_contain_dynamic(dims));
                                    if should_keep_const {
                                        const_values.insert(out.to_string(), filled_tensor);
                                    } else {
                                        const_values.remove(out.as_str());
                                    }
                                    // Force the correct shape - ConstantOfShape creates exact output shape
                                    value_shapes.insert(out.to_string(), shape_vals.clone());
                                    value_shapes
                                        .insert(sanitize_identifier(out), shape_vals.clone());
                                    value_types.insert(out.to_string(), DataType::Int64);
                                }
                            }
                        }
                    }
                } else if op_type == "Equal" {
                    // Equal(a, b) -> boolean tensor (represented as i64: 1 for true, 0 for false)
                    if node.input.as_slice().len() >= 2 {
                        if let (Some(a_name), Some(b_name), Some(out)) = (
                            node.input.as_slice().first(),
                            node.input.as_slice().get(1),
                            node.output.as_slice().first(),
                        ) {
                            let a = const_values.get(a_name);
                            let b = const_values.get(b_name);
                            if let (Some(a), Some(b)) = (a, b) {
                                let a_shape = const_shape_for_folding(a_name, a, &value_shapes);
                                let b_shape = const_shape_for_folding(b_name, b, &value_shapes);
                                if let Some((result_vals, out_shape)) =
                                    fold_binary_const_i64("Equal", a, b, &a_shape, &b_shape)
                                {
                                    const_values.insert(out.to_string(), result_vals.clone());
                                    // Force the correct shape - Equal operation computes exact output shape
                                    value_shapes.insert(out.to_string(), out_shape.clone());
                                    value_shapes.insert(sanitize_identifier(out), out_shape);
                                    value_types.insert(out.to_string(), DataType::Int64);
                                }
                            }
                        }
                    }
                } else if op_type == "Where" {
                    if options.experimental_dynamic_inputs && node.input.as_slice().len() >= 3 {
                        if let Some(out) = node.output.as_slice().first() {
                            let cond = const_values.get(node.input.as_slice()[0].as_str());
                            let a_dims = dimension_vector_for_value(
                                node.input.as_slice()[1].as_str(),
                                &const_values,
                                &value_shape_dims,
                            );
                            let b_dims = dimension_vector_for_value(
                                node.input.as_slice()[2].as_str(),
                                &const_values,
                                &value_shape_dims,
                            );
                            let out_dims = if let (Some(cond), Some(a_dims), Some(b_dims)) =
                                (cond, a_dims.as_ref(), b_dims.as_ref())
                            {
                                if cond.len() == 1 && a_dims.len() == b_dims.len() {
                                    Some(if cond[0] != 0 {
                                        a_dims.clone()
                                    } else {
                                        b_dims.clone()
                                    })
                                } else if cond.len() == a_dims.len() && cond.len() == b_dims.len() {
                                    Some(
                                        cond.iter()
                                            .enumerate()
                                            .map(|(idx, c)| {
                                                if *c != 0 {
                                                    a_dims[idx].clone()
                                                } else {
                                                    b_dims[idx].clone()
                                                }
                                            })
                                            .collect(),
                                    )
                                } else {
                                    None
                                }
                            } else if let (Some(a_dims), Some(b_dims)) =
                                (a_dims.as_ref(), b_dims.as_ref())
                            {
                                let a_has_dynamic =
                                    a_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)));
                                let b_has_dynamic =
                                    b_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)));
                                if a_has_dynamic && !b_has_dynamic {
                                    Some(a_dims.clone())
                                } else if b_has_dynamic && !a_has_dynamic {
                                    Some(b_dims.clone())
                                } else if a_has_dynamic
                                    && b_has_dynamic
                                    && a_dims.len() == b_dims.len()
                                {
                                    Some(
                                        a_dims
                                            .iter()
                                            .zip(b_dims.iter())
                                            .map(|(a_dim, b_dim)| match (a_dim, b_dim) {
                                                (Dimension::Dynamic(dim), _) => {
                                                    Dimension::Dynamic(dim.clone())
                                                }
                                                (_, Dimension::Dynamic(dim)) => {
                                                    Dimension::Dynamic(dim.clone())
                                                }
                                                (Dimension::Static(v), _) => Dimension::Static(*v),
                                            })
                                            .collect(),
                                    )
                                } else {
                                    None
                                }
                            } else if let Some(a_dims) = a_dims.as_ref() {
                                if a_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
                                    && !is_trivial_static_dimension_vector(a_dims)
                                {
                                    Some(a_dims.clone())
                                } else {
                                    None
                                }
                            } else if let Some(b_dims) = b_dims.as_ref() {
                                if b_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
                                    && !is_trivial_static_dimension_vector(b_dims)
                                {
                                    Some(b_dims.clone())
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            if let Some(out_dims) = out_dims {
                                if out_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))) {
                                    value_shape_dims.insert(out.to_string(), out_dims);
                                }
                            }
                        }
                    }
                    // Keep Where dynamic to avoid baking shape-driving expressions
                    // (e.g., past_sequence_length + 1) into fixed constants.
                    continue;
                }
            }

            if const_values.len() == consts_before {
                break;
            }

            // DEBUG: Check value after propagation pass
            if let Some(val) = const_values.get("/model/rotary_emb/Where_output_0") {
                crate::debug_println!("[PROP AFTER] /model/rotary_emb/Where_output_0 = {:?}", val);
            }
        }

        // DEBUG: Check value before node conversion
        if let Some(val) = const_values.get("/model/rotary_emb/Where_output_0") {
            crate::debug_println!("[NODE CONV] /model/rotary_emb/Where_output_0 = {:?}", val);
        }
        for onnx_node in onnx_graph.node.as_slice() {
            // If all outputs are compile-time constants, emit them directly and skip conversion
            let outputs = onnx_node.output.as_slice();
            let has_dynamic_output_metadata = outputs.iter().any(|o| {
                value_shape_dims_for(o.as_str(), &value_shape_dims)
                    .map(|dims| dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))))
                    .unwrap_or(false)
            });
            if !outputs.is_empty()
                && !has_dynamic_output_metadata
                && outputs
                    .iter()
                    .all(|o| const_values.contains_key(o.as_str()))
            {
                // Check if outputs are true scalars (rank 0), not just single-element tensors
                let all_scalar = outputs.iter().all(|o| {
                    value_shapes
                        .get(o.as_str())
                        .map(|s| s.is_empty()) // True scalar has empty shape
                        .unwrap_or_else(|| {
                            // Fallback: check if data length is 1
                            const_values
                                .get(o.as_str())
                                .map(|v| v.len() == 1)
                                .unwrap_or(false)
                        })
                });

                // Handle scalar constants by emitting them inline
                if all_scalar {
                    for out in outputs {
                        if let Some(values) = const_values.get(out) {
                            let const_name = sanitize_identifier(out);
                            // Use the intended shape from value_shapes, not just empty for single-element
                            let shape = value_shapes
                                .get(out.as_str())
                                .map(|s| s.iter().map(|&d| d as u32).collect())
                                .unwrap_or_else(Vec::new);

                            let decl = crate::ast::ConstDecl {
                                data_type: DataType::Int64,
                                shape,
                                init: crate::ast::ConstInit::InlineBytes {
                                    bytes: values[0].to_le_bytes().to_vec(),
                                },
                            };

                            if let Some(existing) = self.graph.consts.get(&const_name) {
                                if existing != &decl {
                                    return Err(OnnxError::InvalidShape(format!(
                                        "Conflicting constant definitions for '{}'",
                                        const_name
                                    )));
                                }
                            } else {
                                self.graph.consts.insert(const_name.clone(), decl);
                            }

                            value_name_map.insert(out.to_string(), const_name.clone());
                            value_name_map.insert(const_name.clone(), const_name.clone());
                            value_types.insert(out.to_string(), DataType::Int64);
                            value_types.insert(const_name, DataType::Int64);
                        }
                    }
                }
                // For non-scalar constants (like Range output), emit inline consts so downstream
                // nodes have a defined producer.
                for out in outputs {
                    if let Some(values) = const_values.get(out) {
                        let const_name = sanitize_identifier(out);
                        let mut shape = value_shapes
                            .get(out.as_str())
                            .cloned()
                            .unwrap_or_else(|| vec![values.len() as i64]);
                        let declared_numel = shape
                            .iter()
                            .try_fold(1usize, |acc, d| usize::try_from(*d).ok().map(|v| acc * v));
                        if declared_numel != Some(values.len()) {
                            // Some folded constants are broadcast candidates where value_shapes
                            // carries the post-broadcast shape but const_values stores the compact payload.
                            // Keep shape/data internally consistent by using the compact shape.
                            shape = vec![values.len() as i64];
                        }
                        let dtype = value_types
                            .get(out.as_str())
                            .cloned()
                            .unwrap_or(DataType::Int64);

                        // Flatten i64 values into little-endian bytes
                        let mut bytes = Vec::with_capacity(values.len() * 8);
                        for v in values {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }

                        let decl = crate::ast::ConstDecl {
                            data_type: dtype.clone(),
                            shape: shape.iter().map(|d| *d as u32).collect(),
                            init: crate::ast::ConstInit::InlineBytes { bytes },
                        };

                        let existing = self.graph.consts.get(&const_name).cloned();
                        if existing.is_none() {
                            self.graph.consts.insert(const_name.clone(), decl);
                        }

                        value_name_map.insert(out.to_string(), const_name.clone());
                        value_name_map.insert(const_name.clone(), const_name.clone());
                        value_types.insert(out.to_string(), dtype.clone());
                        value_types.insert(const_name, dtype);
                    }
                }
                continue;
            }

            let context = crate::onnx::ops::ConversionContext {
                initializers: &initializers_map,
                value_shapes: &value_shapes,
                value_shape_dims: &value_shape_dims,
                const_values: &const_values,
                value_ids: &value_name_map,
                value_types: &value_types,
            };

            let converted = registry.convert_node(onnx_node, &context)?;

            for (name, mut decl) in converted.consts {
                if let crate::ast::ConstInit::InlineBytes { bytes } = &decl.init {
                    let elem_size = match decl.data_type {
                        DataType::Float32 => 4,
                        DataType::Float16 => 2,
                        DataType::Int64 => 8,
                        DataType::Uint64 => 8,
                        DataType::Int32 => 4,
                        DataType::Uint32 => 4,
                        DataType::Int8 => 1,
                        DataType::Uint8 => 1,
                        DataType::Int4 | DataType::Uint4 => 0,
                    };
                    if elem_size > 0 {
                        let declared_numel = decl
                            .shape
                            .iter()
                            .try_fold(1usize, |acc, d| usize::try_from(*d).ok().map(|v| acc * v));
                        let declared_bytes = declared_numel.map(|n| n * elem_size);
                        if declared_bytes != Some(bytes.len()) && bytes.len() % elem_size == 0 {
                            // Keep const metadata internally consistent even when upstream shape
                            // metadata reflects a broadcasted view of compact inline data.
                            decl.shape = vec![(bytes.len() / elem_size) as u32];
                        }
                    }
                }
                let decl_dtype = decl.data_type.clone();
                if let Some(existing) = self.graph.consts.get(&name) {
                    if existing != &decl {
                        return Err(OnnxError::InvalidShape(format!(
                            "Conflicting constant definitions for '{}'",
                            name
                        )));
                    }
                } else {
                    self.graph.consts.insert(name.clone(), decl);
                }
                value_name_map.insert(name.clone(), name.clone());
                value_types.insert(name.clone(), decl_dtype);
            }

            for (onnx_out, webnn_id) in converted.output_mappings {
                value_name_map.insert(onnx_out.clone(), webnn_id.clone());
                value_name_map.insert(sanitize_identifier(&onnx_out), webnn_id.clone());
            }

            for (onnx_out, dtype) in converted.output_types {
                if let Some(webnn_id) = value_name_map.get(&onnx_out).cloned() {
                    value_types.insert(webnn_id, dtype);
                }
            }

            // Track output shapes after conversion to prevent shape inflation
            // Use .insert() to force correct shapes (not .or_insert() which preserves old shapes)
            if let Some(inferred_shape) =
                infer_shape(onnx_node, &value_shapes, &initializers_map, &const_values)
            {
                for output_name in onnx_node.output.as_slice() {
                    // Insert shape for both raw and sanitized names
                    value_shapes.insert(output_name.to_string(), inferred_shape.clone());
                    value_shapes.insert(sanitize_identifier(output_name), inferred_shape.clone());
                }
            }

            self.graph.nodes.extend(converted.nodes);
        }

        // Process outputs
        for output in onnx_graph.output.as_slice() {
            let onnx_name = output.name.as_str();
            if let Some(mapped) = value_name_map.get(onnx_name) {
                self.graph
                    .outputs
                    .insert(sanitize_identifier(onnx_name), mapped.clone());
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "No WebNN value found for ONNX output '{}'",
                    onnx_name
                )));
            }
        }

        let has_dynamic_inputs = self.graph.inputs.values().any(|operand| {
            operand
                .shape
                .iter()
                .any(|dim| matches!(dim, Dimension::Dynamic(_)))
        });
        self.graph.version = if has_dynamic_inputs { 2 } else { 1 };

        Ok(self.graph)
    }
}

/// Convert an ONNX file to WebNN format with optional weight extraction
pub fn convert_onnx<P: AsRef<Path>>(
    onnx_path: P,
    mut options: ConvertOptions,
) -> Result<GraphJson, OnnxError> {
    // Read ONNX file
    let onnx_path_ref = onnx_path.as_ref();
    let onnx_bytes = fs::read(onnx_path_ref)?;

    // Parse protobuf
    let mut model: ModelProto =
        ModelProto::decode(&onnx_bytes[..]).map_err(|e| OnnxError::ProtobufError(e.to_string()))?;

    // Apply constant folding if optimize flag is set
    if options.optimize {
        crate::debug_println!("Running constant folding...");
        let evaluators = crate::onnx::constant_folding::evaluators::get_evaluators();
        let nodes_folded =
            crate::onnx::constant_folding::fold_constants_in_model(&mut model, &evaluators)?;
        crate::debug_println!("Constant folding: {} nodes folded", nodes_folded);
    }

    // Merge overrides from sidecar dims file if provided implicitly and not already set
    if options.free_dim_overrides.is_empty() {
        let mut sidecar = onnx_path_ref.to_path_buf();
        sidecar.set_extension("dims.json");
        if sidecar.exists() {
            let content = fs::read_to_string(&sidecar)?;
            if let Ok(json) = serde_json::from_str::<JsonValue>(&content) {
                if let Some(obj) = json
                    .get("freeDimensionOverrides")
                    .unwrap_or(&json)
                    .as_object()
                {
                    for (name, value) in obj {
                        if let Some(v) = value.as_u64() {
                            options
                                .free_dim_overrides
                                .entry(name.clone())
                                .or_insert(v as u32);
                        }
                    }
                }
            }
        }
    }

    // Create converter
    let converter = OnnxConverter::new(model.clone())?;

    // Extract metadata for debugging
    converter.extract_metadata()?;

    // Convert to GraphJson
    let mut graph = converter.convert(&options)?;

    // Extract weights if requested
    if options.extract_weights {
        if let (Some(weights_path), Some(manifest_path)) =
            (&options.weights_path, &options.manifest_path)
        {
            extract_weights_from_onnx(&model, &mut graph, weights_path, manifest_path)?;
        }
    }

    Ok(graph)
}

/// Extract weights from ONNX model to .weights and .manifest.json files.
/// Also extracts large inline constants from the converted graph into the weights file.
fn extract_weights_from_onnx(
    model: &ModelProto,
    graph: &mut GraphJson,
    weights_path: &str,
    manifest_path: &str,
) -> Result<(), OnnxError> {
    use crate::weights::{TensorEntry, WeightsManifest};

    if model.graph.is_none() {
        return Err(OnnxError::ProtobufError(
            "Missing graph in model".to_string(),
        ));
    }

    let onnx_graph = model.graph.as_ref().unwrap();
    let mut manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: "little".to_string(),
        tensors: BTreeMap::new(),
    };

    let mut weights_data = Vec::new();
    let mut current_offset = 0u64;

    // Process each initializer
    for initializer in onnx_graph.initializer.as_slice() {
        let name = sanitize_identifier(initializer.name.as_str());

        // Convert ONNX data type enum to i32, then to WebNN DataType
        let onnx_type = initializer.data_type;
        let data_type = map_onnx_data_type(onnx_type)?;

        let shape: Vec<u32> = initializer
            .dims
            .as_slice()
            .iter()
            .map(|d| *d as u32)
            .collect();
        let raw_data = initializer.raw_data.as_slice();

        // Convert typed data to bytes if raw_data is empty
        let bytes_to_write: Vec<u8> = if raw_data.is_empty() {
            // Try to extract from typed data fields
            let int64_data = initializer.int64_data.as_slice();
            let float_data = initializer.float_data.as_slice();
            let int32_data = initializer.int32_data.as_slice();
            let double_data = initializer.double_data.as_slice();

            if !int64_data.is_empty() {
                // Convert int64_data to bytes (little-endian)
                int64_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else if !float_data.is_empty() {
                // Convert float_data to bytes (little-endian)
                float_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else if !int32_data.is_empty() {
                // Convert int32_data to bytes (little-endian)
                int32_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else if !double_data.is_empty() {
                // Convert double_data to bytes (little-endian)
                double_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else {
                // No data at all - skip this initializer
                crate::debug_println!("Warning: Skipping initializer '{}' with no data", name);
                continue;
            }
        } else {
            raw_data.to_vec()
        };

        let byte_length = bytes_to_write.len() as u64;

        // Add to manifest
        manifest.tensors.insert(
            name,
            TensorEntry {
                data_type,
                shape,
                byte_offset: current_offset,
                byte_length,
                layout: None,
            },
        );

        // Append to weights data
        weights_data.extend_from_slice(&bytes_to_write);
        current_offset += byte_length;
    }

    // Extract large inline constants from the graph into the weights file.
    // Threshold: constants larger than 1 KiB are moved to external weights.
    const INLINE_THRESHOLD: usize = 1024;
    for (name, decl) in graph.consts.iter_mut() {
        if let crate::ast::ConstInit::InlineBytes { bytes } = &decl.init {
            if bytes.len() > INLINE_THRESHOLD && !manifest.tensors.contains_key(name) {
                let byte_length = bytes.len() as u64;
                manifest.tensors.insert(
                    name.clone(),
                    TensorEntry {
                        data_type: decl.data_type.clone(),
                        shape: decl.shape.clone(),
                        byte_offset: current_offset,
                        byte_length,
                        layout: None,
                    },
                );
                weights_data.extend_from_slice(bytes);
                current_offset += byte_length;
            }
        }
    }
    // Update the graph consts to use weight references instead of inline bytes
    for (name, decl) in graph.consts.iter_mut() {
        if let crate::ast::ConstInit::InlineBytes { bytes } = &decl.init {
            if bytes.len() > INLINE_THRESHOLD {
                decl.init = crate::ast::ConstInit::Weights {
                    r#ref: name.clone(),
                };
            }
        }
    }

    // Write weights file
    fs::write(weights_path, &weights_data)?;

    // Write manifest file
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| OnnxError::ProtobufError(e.to_string()))?;
    fs::write(manifest_path, manifest_json)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_options_default() {
        let options = ConvertOptions::default();
        assert!(options.extract_weights);
        assert_eq!(options.output_path, "output.webnn");
    }

    #[test]
    fn test_sanitize_identifier_replaces_colons() {
        assert_eq!(sanitize_identifier("foo::bar"), "foo__bar");
        assert_eq!(sanitize_identifier("foo:bar"), "foo_bar");
    }

    #[test]
    fn test_sanitize_identifier_replaces_dots() {
        assert_eq!(sanitize_identifier("encoder.block.0"), "encoder_block_0");
        assert_eq!(
            sanitize_identifier("model.layer.weight"),
            "model_layer_weight"
        );
        assert_eq!(sanitize_identifier("a.b.c"), "a_b_c");
    }

    #[test]
    fn test_sanitize_identifier_replaces_combined() {
        // Test combinations of :: : and .
        assert_eq!(
            sanitize_identifier("module::class:method.field"),
            "module__class_method_field"
        );
        assert_eq!(
            sanitize_identifier("encoder.attention::output:dense"),
            "encoder_attention__output_dense"
        );
    }

    #[test]
    fn test_sanitize_identifier_no_change() {
        // Identifiers that don't need sanitization
        assert_eq!(sanitize_identifier("simple_name"), "simple_name");
        assert_eq!(sanitize_identifier("CamelCase"), "CamelCase");
        assert_eq!(sanitize_identifier("name123"), "name123");
    }

    #[test]
    fn test_inline_bytes_encoding_for_i64_values() {
        // Test the inline bytes encoding logic used for non-scalar constants
        // This simulates what happens when Range or similar ops produce constant arrays
        let values: Vec<i64> = vec![0, 1, 2, 3, 4];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Verify byte length
        assert_eq!(bytes.len(), 40); // 5 values * 8 bytes each

        // Verify first value (0)
        let first_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(first_bytes), 0);

        // Verify last value (4)
        let last_bytes: [u8; 8] = bytes[32..40].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(last_bytes), 4);
    }

    #[test]
    fn test_inline_bytes_encoding_single_value() {
        // Test single value encoding
        let values: Vec<i64> = vec![42];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 8);
        let decoded: [u8; 8] = bytes.try_into().unwrap();
        assert_eq!(i64::from_le_bytes(decoded), 42);
    }

    #[test]
    fn test_inline_bytes_encoding_negative_values() {
        // Test with negative values (important for Range with negative delta)
        let values: Vec<i64> = vec![5, 4, 3, 2, 1, 0, -1, -2];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 64); // 8 values * 8 bytes each

        // Verify a negative value
        let neg_bytes: [u8; 8] = bytes[56..64].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(neg_bytes), -2);
    }

    #[test]
    fn test_inline_bytes_encoding_large_values() {
        // Test with large i64 values
        let values: Vec<i64> = vec![i64::MAX, i64::MIN, 0];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 24);

        // Verify MAX value
        let max_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(max_bytes), i64::MAX);

        // Verify MIN value
        let min_bytes: [u8; 8] = bytes[8..16].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(min_bytes), i64::MIN);
    }

    #[test]
    fn test_convert_preserves_dynamic_input_dim_without_override() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{GraphProto, ModelProto, TensorShapeProto, ValueInfoProto};

        let dim_batch = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch_size".to_string(),
            )),
            denotation: String::new(),
        };
        let dim_seq = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto {
            dim: vec![dim_batch, dim_seq],
        };

        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(shape),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let input_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let output_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![input_vi],
                output: vec![output_vi],
                ..Default::default()
            }),
            ..Default::default()
        };

        let converter = OnnxConverter::new(model).expect("converter");
        let graph = converter
            .convert(&ConvertOptions {
                experimental_dynamic_inputs: true,
                ..ConvertOptions::default()
            })
            .expect("convert");

        let input = graph.inputs.get("input_ids").expect("input_ids input");
        assert_eq!(input.shape.len(), 2);
        assert!(matches!(
            &input.shape[0],
            Dimension::Dynamic(d) if d.name == "batch_size"
        ));
        assert!(matches!(&input.shape[1], Dimension::Static(1)));
        assert_eq!(graph.version, 2);
    }

    #[test]
    fn test_convert_rejects_dynamic_input_dim_without_flag() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{GraphProto, ModelProto, TensorShapeProto, ValueInfoProto};

        let dim_batch = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "unknown_dim".to_string(),
            )),
            denotation: String::new(),
        };
        let dim_seq = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto {
            dim: vec![dim_batch, dim_seq],
        };

        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(shape),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let input_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let output_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![input_vi],
                output: vec![output_vi],
                ..Default::default()
            }),
            ..Default::default()
        };

        let converter = OnnxConverter::new(model).expect("converter");
        let err = converter
            .convert(&ConvertOptions::default())
            .expect_err("should require overrides or flag");
        let msg = err.to_string();
        assert!(msg.contains("override-dim"));
        assert!(msg.contains("experimental-dynamic-inputs"));
    }

    #[test]
    fn test_convert_dynamic_shape_concat_reshape_path_with_experimental_flag() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{
            AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
            ValueInfoProto,
        };

        let batch_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let seq_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "sequence_length".to_string(),
            )),
            denotation: String::new(),
        };
        let hidden_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(4)),
            denotation: String::new(),
        };
        let data_shape = TensorShapeProto {
            dim: vec![batch_dim, seq_dim, hidden_dim],
        };

        let data_tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Float.into(),
            shape: Some(data_shape),
        };
        let data_type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(data_tensor_type)),
            denotation: String::new(),
        };

        let data_input = ValueInfoProto {
            name: "data".to_string(),
            r#type: Some(data_type_proto.clone()),
            ..Default::default()
        };
        let data_output = ValueInfoProto {
            name: "out".to_string(),
            r#type: Some(data_type_proto),
            ..Default::default()
        };

        let idx0 = TensorProto {
            name: "idx0".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![0],
            ..Default::default()
        };
        let idx1 = TensorProto {
            name: "idx1".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![1],
            ..Default::default()
        };
        let last_dim = TensorProto {
            name: "last_dim".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![4],
            ..Default::default()
        };

        let shape_node = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["data".to_string()],
            output: vec!["shape_out".to_string()],
            ..Default::default()
        };
        let gather0 = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["shape_out".to_string(), "idx0".to_string()],
            output: vec!["dim0".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let gather1 = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["shape_out".to_string(), "idx1".to_string()],
            output: vec!["dim1".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let concat_shape = NodeProto {
            op_type: "Concat".to_string(),
            input: vec![
                "dim0".to_string(),
                "dim1".to_string(),
                "last_dim".to_string(),
            ],
            output: vec!["shape_for_reshape".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reshape = NodeProto {
            op_type: "Reshape".to_string(),
            input: vec!["data".to_string(), "shape_for_reshape".to_string()],
            output: vec!["out".to_string()],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![data_input],
                output: vec![data_output],
                initializer: vec![idx0, idx1, last_dim],
                node: vec![shape_node, gather0, gather1, concat_shape, reshape],
                ..Default::default()
            }),
            ..Default::default()
        };

        let converter = OnnxConverter::new(model).expect("converter");
        let graph = converter
            .convert(&ConvertOptions {
                optimize: true,
                experimental_dynamic_inputs: true,
                extract_weights: false,
                ..ConvertOptions::default()
            })
            .expect("dynamic reshape path should convert");

        let reshape_node = graph
            .nodes
            .iter()
            .find(|n| n.op == "reshape")
            .expect("reshape node should exist");
        let shape = reshape_node
            .options
            .get("newShape")
            .and_then(|v| v.as_array())
            .expect("newShape should be an array");
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0].as_u64(), Some(1));
        assert_eq!(shape[2].as_u64(), Some(4));
        // The sequence dimension may be a concrete integer (concretized for lowering)
        // or a dynamic dimension object {"name": ..., "maxSize": N} when dynamic
        // dimension metadata is propagated.
        let dim1_ok = shape[1].as_u64().is_some_and(|v| v > 0)
            || shape[1].as_object().is_some_and(|o| {
                o.contains_key("name")
                    && o.get("maxSize")
                        .and_then(|v| v.as_u64())
                        .is_some_and(|v| v > 0)
            });
        assert!(
            dim1_ok,
            "sequence dimension should be concretized or dynamic for lowering, got: {:?}",
            shape[1]
        );
    }

    #[test]
    fn test_convert_reshape_shape_path_survives_add_broadcast() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{
            AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
            ValueInfoProto,
        };

        let batch_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let seq_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(128)),
            denotation: String::new(),
        };
        let hidden_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(4)),
            denotation: String::new(),
        };
        let data_shape = TensorShapeProto {
            dim: vec![batch_dim, seq_dim, hidden_dim],
        };

        let data_tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Float.into(),
            shape: Some(data_shape),
        };
        let data_type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(data_tensor_type)),
            denotation: String::new(),
        };

        let data_input = ValueInfoProto {
            name: "data".to_string(),
            r#type: Some(data_type_proto.clone()),
            ..Default::default()
        };
        let data_output = ValueInfoProto {
            name: "out".to_string(),
            r#type: Some(data_type_proto),
            ..Default::default()
        };

        let bias = TensorProto {
            name: "bias".to_string(),
            data_type: TensorProto_DataType::Float as i32,
            dims: vec![4],
            float_data: vec![0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let idx0 = TensorProto {
            name: "idx0".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![0],
            ..Default::default()
        };
        let idx1 = TensorProto {
            name: "idx1".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![1],
            ..Default::default()
        };
        let last_dim = TensorProto {
            name: "last_dim".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![4],
            ..Default::default()
        };

        let add_node = NodeProto {
            op_type: "Add".to_string(),
            input: vec!["data".to_string(), "bias".to_string()],
            output: vec!["add_out".to_string()],
            ..Default::default()
        };
        let shape_node = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["add_out".to_string()],
            output: vec!["shape_out".to_string()],
            ..Default::default()
        };
        let gather0 = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["shape_out".to_string(), "idx0".to_string()],
            output: vec!["dim0".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let gather1 = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["shape_out".to_string(), "idx1".to_string()],
            output: vec!["dim1".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let concat_shape = NodeProto {
            op_type: "Concat".to_string(),
            input: vec![
                "dim0".to_string(),
                "dim1".to_string(),
                "last_dim".to_string(),
            ],
            output: vec!["shape_for_reshape".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reshape = NodeProto {
            op_type: "Reshape".to_string(),
            input: vec!["add_out".to_string(), "shape_for_reshape".to_string()],
            output: vec!["out".to_string()],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![data_input],
                output: vec![data_output],
                initializer: vec![bias, idx0, idx1, last_dim],
                node: vec![
                    add_node,
                    shape_node,
                    gather0,
                    gather1,
                    concat_shape,
                    reshape,
                ],
                ..Default::default()
            }),
            ..Default::default()
        };

        let converter = OnnxConverter::new(model).expect("converter");
        let graph = converter
            .convert(&ConvertOptions {
                optimize: true,
                extract_weights: false,
                ..ConvertOptions::default()
            })
            .expect("broadcasted shape path should convert");

        let reshape_node = graph
            .nodes
            .iter()
            .find(|n| n.op == "reshape")
            .expect("reshape node should exist");
        assert_eq!(
            reshape_node.options.get("newShape"),
            Some(&serde_json::json!([1, 128, 4]))
        );
    }

    #[test]
    fn test_convert_dynamic_range_lowers_to_slice_and_preserves_dynamic_reshape() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{
            AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
            ValueInfoProto,
        };

        let seq_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "sequence_length".to_string(),
            )),
            denotation: String::new(),
        };
        let data_shape = TensorShapeProto { dim: vec![seq_dim] };

        let data_tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Float.into(),
            shape: Some(data_shape),
        };
        let data_type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(data_tensor_type)),
            denotation: String::new(),
        };

        let data_input = ValueInfoProto {
            name: "data".to_string(),
            r#type: Some(data_type_proto),
            ..Default::default()
        };
        let output_vi = ValueInfoProto {
            name: "out".to_string(),
            ..Default::default()
        };

        let idx0 = TensorProto {
            name: "idx0".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![0],
            ..Default::default()
        };
        let zero = TensorProto {
            name: "zero".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![],
            int64_data: vec![0],
            ..Default::default()
        };
        let one = TensorProto {
            name: "one".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![],
            int64_data: vec![1],
            ..Default::default()
        };

        let shape_node = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["data".to_string()],
            output: vec!["shape_out".to_string()],
            ..Default::default()
        };
        let gather = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["shape_out".to_string(), "idx0".to_string()],
            output: vec!["seq_len".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let add_limit = NodeProto {
            op_type: "Add".to_string(),
            input: vec!["seq_len".to_string(), "one".to_string()],
            output: vec!["range_limit".to_string()],
            ..Default::default()
        };
        let range = NodeProto {
            op_type: "Range".to_string(),
            input: vec![
                "zero".to_string(),
                "range_limit".to_string(),
                "one".to_string(),
            ],
            output: vec!["range_out".to_string()],
            ..Default::default()
        };
        let concat_shape = NodeProto {
            op_type: "Concat".to_string(),
            input: vec!["range_limit".to_string(), "one".to_string()],
            output: vec!["shape_for_reshape".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reshape = NodeProto {
            op_type: "Reshape".to_string(),
            input: vec!["range_out".to_string(), "shape_for_reshape".to_string()],
            output: vec!["out".to_string()],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![data_input],
                output: vec![output_vi],
                initializer: vec![idx0, zero, one],
                node: vec![shape_node, gather, add_limit, range, concat_shape, reshape],
                ..Default::default()
            }),
            ..Default::default()
        };

        let converter = OnnxConverter::new(model).expect("converter");
        let graph = converter
            .convert(&ConvertOptions {
                optimize: true,
                experimental_dynamic_inputs: true,
                extract_weights: false,
                ..ConvertOptions::default()
            })
            .expect("dynamic range path should convert");

        let slice_node = graph
            .nodes
            .iter()
            .find(|n| n.op == "slice")
            .expect("range should lower to slice");
        let slice_sizes = slice_node
            .options
            .get("sizes")
            .and_then(|v| v.as_array())
            .expect("slice sizes should exist");
        assert_eq!(slice_sizes.len(), 1);
        let dynamic_size = slice_sizes[0]
            .as_object()
            .expect("dynamic range size should be a dimension object");
        assert_eq!(
            dynamic_size.get("name").and_then(|v| v.as_str()),
            Some("sequence_length + 1")
        );
        assert_eq!(
            dynamic_size.get("maxSize").and_then(|v| v.as_u64()),
            Some(4097)
        );

        let reshape_node = graph
            .nodes
            .iter()
            .find(|n| n.op == "reshape")
            .expect("reshape node should exist");
        let new_shape = reshape_node
            .options
            .get("newShape")
            .and_then(|v| v.as_array())
            .expect("reshape newShape should exist");
        assert_eq!(new_shape.len(), 2);
        assert_eq!(new_shape[1].as_u64(), Some(1));
        let reshape_dim0 = new_shape[0]
            .as_object()
            .expect("reshape dim 0 should stay dynamic");
        assert_eq!(
            reshape_dim0.get("name").and_then(|v| v.as_str()),
            Some("sequence_length + 1")
        );
        assert_eq!(
            reshape_dim0.get("maxSize").and_then(|v| v.as_u64()),
            Some(4097)
        );
    }

    #[test]
    fn test_convert_dynamic_range_with_dynamic_start_lowers_to_slice_and_add() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{
            AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
            ValueInfoProto,
        };

        let batch_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let seq_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "sequence_length".to_string(),
            )),
            denotation: String::new(),
        };
        let past_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "past_sequence_length".to_string(),
            )),
            denotation: String::new(),
        };
        let heads_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(3)),
            denotation: String::new(),
        };
        let head_dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(4)),
            denotation: String::new(),
        };

        let ids_shape = TensorShapeProto {
            dim: vec![batch_dim.clone(), seq_dim.clone()],
        };
        let past_shape = TensorShapeProto {
            dim: vec![batch_dim, heads_dim, past_dim, head_dim],
        };
        let range_shape = TensorShapeProto {
            dim: vec![seq_dim.clone()],
        };
        let out_shape = TensorShapeProto {
            dim: vec![
                seq_dim,
                tensor_shape_proto::Dimension {
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
                    denotation: String::new(),
                },
            ],
        };

        let ids_tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(ids_shape),
        };
        let past_tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Float.into(),
            shape: Some(past_shape),
        };
        let range_tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(range_shape),
        };
        let out_tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(out_shape),
        };

        let ids_input = ValueInfoProto {
            name: "ids".to_string(),
            r#type: Some(crate::protos::onnx::TypeProto {
                value: Some(type_proto::Value::TensorType(ids_tensor_type)),
                denotation: String::new(),
            }),
            ..Default::default()
        };
        let past_input = ValueInfoProto {
            name: "past".to_string(),
            r#type: Some(crate::protos::onnx::TypeProto {
                value: Some(type_proto::Value::TensorType(past_tensor_type)),
                denotation: String::new(),
            }),
            ..Default::default()
        };
        let range_vi = ValueInfoProto {
            name: "range_out".to_string(),
            r#type: Some(crate::protos::onnx::TypeProto {
                value: Some(type_proto::Value::TensorType(range_tensor_type)),
                denotation: String::new(),
            }),
            ..Default::default()
        };
        let out_vi = ValueInfoProto {
            name: "out".to_string(),
            r#type: Some(crate::protos::onnx::TypeProto {
                value: Some(type_proto::Value::TensorType(out_tensor_type)),
                denotation: String::new(),
            }),
            ..Default::default()
        };

        let idx1 = TensorProto {
            name: "idx1".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![1],
            ..Default::default()
        };
        let idx2 = TensorProto {
            name: "idx2".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![2],
            ..Default::default()
        };
        let one = TensorProto {
            name: "one".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![],
            int64_data: vec![1],
            ..Default::default()
        };
        let reshape_shape = TensorProto {
            name: "reshape_shape".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![2],
            int64_data: vec![4096, 1],
            ..Default::default()
        };

        let shape_past = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["past".to_string()],
            output: vec!["past_shape".to_string()],
            ..Default::default()
        };
        let gather_start = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["past_shape".to_string(), "idx2".to_string()],
            output: vec!["range_start".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let shape_ids = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["ids".to_string()],
            output: vec!["ids_shape".to_string()],
            ..Default::default()
        };
        let gather_seq = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["ids_shape".to_string(), "idx1".to_string()],
            output: vec!["seq_len".to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let add_limit = NodeProto {
            op_type: "Add".to_string(),
            input: vec!["range_start".to_string(), "seq_len".to_string()],
            output: vec!["range_limit".to_string()],
            ..Default::default()
        };
        let range = NodeProto {
            op_type: "Range".to_string(),
            input: vec![
                "range_start".to_string(),
                "range_limit".to_string(),
                "one".to_string(),
            ],
            output: vec!["range_out".to_string()],
            ..Default::default()
        };
        let reshape = NodeProto {
            op_type: "Reshape".to_string(),
            input: vec!["range_out".to_string(), "reshape_shape".to_string()],
            output: vec!["out".to_string()],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![ids_input, past_input],
                output: vec![out_vi.clone()],
                value_info: vec![range_vi, out_vi],
                initializer: vec![idx1, idx2, one, reshape_shape],
                node: vec![
                    shape_past,
                    gather_start,
                    shape_ids,
                    gather_seq,
                    add_limit,
                    range,
                    reshape,
                ],
                ..Default::default()
            }),
            ..Default::default()
        };

        let converter = OnnxConverter::new(model).expect("converter");
        let graph = converter
            .convert(&ConvertOptions {
                optimize: true,
                experimental_dynamic_inputs: true,
                extract_weights: false,
                ..ConvertOptions::default()
            })
            .expect("dynamic range with dynamic start should convert");

        assert!(
            !graph.consts.contains_key("range_out"),
            "range output should stay runtime-computed"
        );

        let slice_node = graph
            .nodes
            .iter()
            .find(|n| n.id == "range_out_slice" && n.op == "slice")
            .expect("range should lower to a slice");
        let slice_sizes = slice_node
            .options
            .get("sizes")
            .and_then(|v| v.as_array())
            .expect("slice sizes should exist");
        let dynamic_size = slice_sizes[0]
            .as_object()
            .expect("slice size should be dynamic");
        assert_eq!(
            dynamic_size.get("name").and_then(|v| v.as_str()),
            Some("sequence_length")
        );
        assert_eq!(
            dynamic_size.get("maxSize").and_then(|v| v.as_u64()),
            Some(4096)
        );

        let add_node = graph
            .nodes
            .iter()
            .find(|n| n.id == "range_out" && n.op == "add")
            .expect("dynamic-start range should add the runtime start offset");
        assert_eq!(add_node.inputs.len(), 2);
        assert_eq!(add_node.inputs[0], "range_out_slice");

        let reshape_node = graph
            .nodes
            .iter()
            .find(|n| n.op == "reshape")
            .expect("reshape node should exist");
        let new_shape = reshape_node
            .options
            .get("newShape")
            .and_then(|v| v.as_array())
            .expect("reshape newShape should exist");
        assert_eq!(new_shape.len(), 2);
        assert_eq!(new_shape[1].as_u64(), Some(1));
        let reshape_dim0 = new_shape[0]
            .as_object()
            .expect("reshape dim 0 should stay dynamic");
        assert_eq!(
            reshape_dim0.get("name").and_then(|v| v.as_str()),
            Some("sequence_length")
        );
        assert_eq!(
            reshape_dim0.get("maxSize").and_then(|v| v.as_u64()),
            Some(4096)
        );
    }

    #[test]
    fn test_binary_const_folding_preserves_broadcast_shape() {
        let a = vec![-1];
        let b = [1, 2, 3, 4].repeat(128);
        let a_shape = Vec::<i64>::new();
        let b_shape = vec![1, 128, 4];
        let (out, out_shape) =
            fold_binary_const_i64("Mul", &a, &b, &a_shape, &b_shape).expect("broadcast fold");
        assert_eq!(out_shape, vec![1, 128, 4]);
        assert_eq!(out.len(), 512);
        assert_eq!(out[0], -1);
        assert_eq!(out[1], -2);
        assert_eq!(out[2], -3);
        assert_eq!(out[3], -4);
    }

    #[test]
    fn test_convert_equal_broadcast_path_does_not_flatten_const_shape() {
        use crate::protos::onnx::{
            type_proto, AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto,
        };

        let a = TensorProto {
            name: "shape_vec".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![4],
            int64_data: vec![1, 128, 4, 8],
            ..Default::default()
        };
        let shape3 = TensorProto {
            name: "shape3".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![3],
            int64_data: vec![1, 128, 4],
            ..Default::default()
        };
        let neg1 = TensorProto {
            name: "neg1".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![],
            int64_data: vec![-1],
            ..Default::default()
        };
        let cos_fill = TensorProto {
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![],
            int64_data: vec![1],
            ..Default::default()
        };

        let cos = NodeProto {
            op_type: "ConstantOfShape".to_string(),
            input: vec!["shape3".to_string()],
            output: vec!["cos_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                t: Some(cos_fill),
                ..Default::default()
            }],
            ..Default::default()
        };
        let mul = NodeProto {
            op_type: "Mul".to_string(),
            input: vec!["cos_out".to_string(), "neg1".to_string()],
            output: vec!["mul_out".to_string()],
            ..Default::default()
        };
        let eq = NodeProto {
            op_type: "Equal".to_string(),
            input: vec!["shape_vec".to_string(), "mul_out".to_string()],
            output: vec!["eq_out".to_string()],
            ..Default::default()
        };

        let output_type = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: TensorProto_DataType::Bool.into(),
                shape: None,
            })),
            denotation: String::new(),
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![a, shape3, neg1],
                node: vec![cos, mul, eq],
                output: vec![crate::protos::onnx::ValueInfoProto {
                    name: "eq_out".to_string(),
                    r#type: Some(output_type),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let converter = OnnxConverter::new(model).expect("converter");
        let graph = converter
            .convert(&ConvertOptions {
                optimize: true,
                extract_weights: false,
                ..ConvertOptions::default()
            })
            .expect("convert");

        let mul_const = graph.consts.get("mul_out").expect("mul_out const");
        assert_eq!(mul_const.shape, vec![1, 128, 4]);
        assert!(
            !graph.consts.contains_key("eq_out")
                || graph
                    .consts
                    .get("eq_out")
                    .is_some_and(|decl| decl.shape == vec![1, 128, 4]),
            "eq_out constant must not be flattened"
        );
    }
}
