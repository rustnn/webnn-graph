// Static shape/type inference scaffold for ONNX graphs.
// Conservative: records only fully-static shapes and folds small integer constants
// to unblock reshape/axes/starts/ends calculations. Dynamic dims cause errors so
// callers can ask users to run onnx-simplifier or provide overrides.
use crate::ast::DataType;
use crate::onnx::convert::map_onnx_data_type;
use crate::onnx::ir::{Dim, OnnxIrGraph, TensorShape, TensorType};
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    GraphProto, ModelProto, NodeProto, TensorProto, TensorProto_DataType,
};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShapeInferenceError {
    #[error("input '{0}' is missing shape information")]
    MissingInputShape(String),
    #[error("input '{input}' has dynamic dimension '{dim}', please provide an override")]
    DynamicDim { input: String, dim: String },
    #[error("unsupported ONNX data type: {0}")]
    UnsupportedDataType(i32),
    #[error("could not infer shape for op '{op}'")]
    CannotInfer { op: String },
}

#[derive(Debug, Default)]
pub struct InferenceResult {
    pub value_shapes: HashMap<String, Vec<i64>>,
    pub value_types: HashMap<String, DataType>,
    pub const_values: HashMap<String, Vec<i64>>,
}

/// Run a lightweight static shape/type inference pass.
/// Returns only fully-known shapes; dynamic dimensions trigger an error.
pub fn infer_static_shapes(
    model: &ModelProto,
    overrides: &HashMap<String, u32>,
) -> Result<InferenceResult, ShapeInferenceError> {
    let mut result = InferenceResult::default();

    if model.graph.is_none() {
        return Ok(result);
    }

    let graph = model.graph.as_ref().unwrap();
    let mut ir = OnnxIrGraph::default();
    let initializer_names: HashSet<String> = graph
        .initializer
        .as_slice()
        .iter()
        .map(|i| i.name.as_str().to_string())
        .collect();

    seed_inputs(graph, overrides, &initializer_names, &mut ir, &mut result)?;
    seed_initializers(graph, &mut ir, &mut result)?;
    seed_constant_nodes(graph, &mut result, &mut ir)?;

    propagate_node_shapes(graph, &mut result)?;

    Ok(result)
}

fn seed_inputs(
    graph: &GraphProto,
    overrides: &HashMap<String, u32>,
    initializer_names: &HashSet<String>,
    ir: &mut OnnxIrGraph,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for input in graph.input.as_slice() {
        let name = input.name.as_str().to_string();
        let vi = ir.value_or_insert(&name);
        vi.producer = None;

        if initializer_names.contains(&name) {
            continue;
        }

        let type_proto = input
            .r#type
            .as_ref()
            .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))?;

        let tensor_type = match &type_proto.value {
            Some(TypeProtoValue::TensorType(tt)) => tt,
            _ => return Err(ShapeInferenceError::MissingInputShape(name.clone())),
        };

        let dtype = if tensor_type.elem_type != 0 {
            map_onnx_data_type(tensor_type.elem_type)
                .map_err(|_| ShapeInferenceError::UnsupportedDataType(tensor_type.elem_type))?
        } else {
            return Err(ShapeInferenceError::UnsupportedDataType(0));
        };

        let shape = tensor_type
            .shape
            .as_ref()
            .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))?;

        let mut dims = Vec::new();
        for dim in shape.dim.as_slice() {
            if let Some(value) = &dim.value {
                match value {
                    DimensionValue::DimValue(v) => {
                        dims.push(Dim::Known(*v));
                    }
                    DimensionValue::DimParam(key) => {
                        if let Some(v) = overrides.get(key.as_str()) {
                            dims.push(Dim::Known(*v as i64));
                        } else {
                            return Err(ShapeInferenceError::DynamicDim {
                                input: name.clone(),
                                dim: key.clone(),
                            });
                        }
                    }
                }
            } else {
                return Err(ShapeInferenceError::MissingInputShape(name.clone()));
            }
        }

        let ty = TensorType {
            data_type: dtype.clone(),
            shape: TensorShape { dims },
        };
        vi.ty = Some(ty.clone());
        result.value_types.insert(name.clone(), dtype);
        if let Some(shape) = ty.shape.to_i64() {
            result.value_shapes.insert(name, shape);
        }
    }
    Ok(())
}

fn seed_initializers(
    graph: &GraphProto,
    ir: &mut OnnxIrGraph,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for init in graph.initializer.as_slice() {
        let name = init.name.as_str().to_string();
        let vi = ir.value_or_insert(&name);
        vi.producer = None;

        let dtype = map_onnx_data_type(init.data_type)
            .map_err(|_| ShapeInferenceError::UnsupportedDataType(init.data_type))?;
        let shape: Vec<i64> = init.dims.as_slice().to_vec();
        result.value_types.insert(name.clone(), dtype.clone());
        result.value_shapes.insert(name.clone(), shape);

        if matches!(
            dtype,
            DataType::Int32 | DataType::Int64 | DataType::Uint32 | DataType::Uint64
        ) {
            let values = read_int_tensor(init);
            if !values.is_empty() {
                result.const_values.insert(name, values);
            }
        }
    }
    Ok(())
}

fn seed_constant_nodes(
    graph: &GraphProto,
    result: &mut InferenceResult,
    ir: &mut OnnxIrGraph,
) -> Result<(), ShapeInferenceError> {
    for node in graph.node.as_slice() {
        if node.op_type.as_str() != "Constant" {
            continue;
        }

        if let Some(out) = node.output.as_slice().first() {
            let out_name = out.to_string();
            let vi = ir.value_or_insert(&out_name);
            vi.producer = Some(node.name.as_str().to_string());

            if let Some(attr) = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "value" && a.t.is_some())
            {
                let t = attr.t.as_ref().unwrap();
                let dtype = map_onnx_data_type(t.data_type)
                    .map_err(|_| ShapeInferenceError::UnsupportedDataType(t.data_type))?;
                result.value_types.insert(out_name.clone(), dtype);

                let vals = read_int_tensor(t);
                if !vals.is_empty() {
                    result.const_values.insert(out_name.clone(), vals.clone());
                    let shape: Vec<i64> = if vals.len() == 1 {
                        Vec::new()
                    } else {
                        vec![vals.len() as i64]
                    };
                    result.value_shapes.insert(out_name.clone(), shape);
                    vi.ty = Some(TensorType {
                        data_type: result.value_types[&out_name].clone(),
                        shape: TensorShape::from_known(result.value_shapes[&out_name].clone()),
                    });
                }
            }
        }
    }
    Ok(())
}

fn propagate_node_shapes(
    graph: &GraphProto,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    let mut progress = true;
    let max_iters = 8;
    let mut iter = 0;

    while progress && iter < max_iters {
        progress = false;
        iter += 1;

        for node in graph.node.as_slice() {
            let outputs = node.output.as_slice();
            if outputs.is_empty() {
                continue;
            }
            if outputs
                .iter()
                .all(|o| result.value_shapes.contains_key(o.as_str()))
            {
                continue;
            }

            if let Some(shape) = infer_node_shape(node, result) {
                let out_name = outputs[0].to_string();
                result.value_shapes.entry(out_name.clone()).or_insert(shape);

                // Propagate dtype from first input if available.
                if let Some(first_in) = node.input.as_slice().first() {
                    if let Some(dtype) = result.value_types.get(first_in).cloned() {
                        result.value_types.entry(out_name.clone()).or_insert(dtype);
                    }
                }

                progress = true;
            }
        }

        // Opportunistic const folding for integer tensors to unlock more shapes.
        progress |= fold_integer_constants(graph, result);
    }

    Ok(())
}

#[allow(dead_code)]
fn broadcast_shapes(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let mut result = Vec::new();
    let mut ai = a.iter().rev();
    let mut bi = b.iter().rev();

    loop {
        match (ai.next(), bi.next()) {
            (Some(&ad), Some(&bd)) => {
                if ad == bd {
                    result.push(ad);
                } else if ad == 1 {
                    result.push(bd);
                } else if bd == 1 {
                    result.push(ad);
                } else {
                    return None;
                }
            }
            (Some(&ad), None) => result.push(ad),
            (None, Some(&bd)) => result.push(bd),
            (None, None) => break,
        }
    }

    result.reverse();
    Some(result)
}

fn infer_node_shape(node: &NodeProto, ctx: &InferenceResult) -> Option<Vec<i64>> {
    let op = node.op_type.as_str();
    match op {
        "Relu" | "Tanh" | "Sigmoid" | "Erf" | "Softmax" | "Gelu" | "Exp" | "Log" | "Abs"
        | "Neg" | "Sqrt" | "LayerNormalization" => node
            .input
            .as_slice()
            .first()
            .and_then(|i| ctx.value_shapes.get(i).cloned()),
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            if node.input.as_slice().len() < 2 {
                return None;
            }
            let a = node.input.as_slice()[0].as_str();
            let b = node.input.as_slice()[1].as_str();
            match (ctx.value_shapes.get(a), ctx.value_shapes.get(b)) {
                // Prefer smaller shape to avoid inflation
                // Rationale: Broadcasting happens implicitly; storing inflated shapes
                // breaks ONNX round-trip conversion
                (Some(sa), Some(sb)) => {
                    if sa.len() <= sb.len() {
                        Some(sa.clone())
                    } else {
                        Some(sb.clone())
                    }
                }
                _ => None,
            }
        }
        "MatMul" => {
            if node.input.as_slice().len() < 2 {
                return None;
            }
            let a_shape = ctx.value_shapes.get(node.input.as_slice()[0].as_str())?;
            let b_shape = ctx.value_shapes.get(node.input.as_slice()[1].as_str())?;

            // Attention pattern: rank-4 [B,S,H,D] x [B,S,H,D] -> [B,S,H,H]
            if a_shape.len() == 4 && b_shape.len() == 4 {
                return Some(vec![a_shape[0], a_shape[1], a_shape[2], b_shape[3]]);
            }

            // Fallback generic matmul
            if a_shape.len() >= 2 && b_shape.len() >= 2 {
                let m = a_shape[a_shape.len() - 2];
                let n = b_shape[b_shape.len() - 1];
                let mut out = Vec::new();
                if a_shape.len() > 2 {
                    out.extend_from_slice(&a_shape[..a_shape.len() - 2]);
                }
                out.push(m);
                out.push(n);
                return Some(out);
            }
            None
        }
        "Transpose" => {
            let input = node.input.as_slice().first()?;
            let shape = ctx.value_shapes.get(input)?;
            let perm: Vec<usize> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "perm")
                .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<usize>>())
                .unwrap_or_else(|| (0..shape.len()).rev().collect());
            if perm.iter().any(|&i| i >= shape.len()) {
                return None;
            }
            Some(perm.iter().map(|&i| shape[i]).collect())
        }
        "Concat" => {
            let mut shapes = Vec::new();
            for inp in node.input.as_slice() {
                if let Some(s) = ctx.value_shapes.get(inp.as_str()) {
                    shapes.push(s.clone());
                } else {
                    return None;
                }
            }
            if shapes.is_empty() {
                return None;
            }
            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis" && a.i != 0)
                .map(|a| a.i)
                .unwrap_or(0);
            if axis < 0 {
                axis += shapes[0].len() as i64;
            }
            let axis = axis as usize;
            let mut out = shapes[0].clone();
            for s in shapes.iter().skip(1) {
                if s.len() != out.len() || axis >= s.len() {
                    return None;
                }
                out[axis] += s[axis];
            }
            Some(out)
        }
        "Unsqueeze" => {
            if node.input.as_slice().is_empty() {
                return None;
            }
            let input_shape = ctx.value_shapes.get(node.input.as_slice()[0].as_str())?;
            let mut axes = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            // Opset >= 13: axes is a second input tensor, not an attribute
            if axes.is_empty() && node.input.as_slice().len() > 1 {
                axes = ctx
                    .const_values
                    .get(node.input.as_slice()[1].as_str())
                    .cloned()
                    .unwrap_or_default();
            }
            if axes.is_empty() {
                return None;
            }
            let mut output_shape = input_shape.clone();
            let mut sorted_axes = axes.clone();
            sorted_axes.sort();
            for axis in sorted_axes {
                let idx = if axis < 0 {
                    (output_shape.len() as i64 + axis + 1) as usize
                } else {
                    axis as usize
                };
                if idx > output_shape.len() {
                    return None;
                }
                output_shape.insert(idx, 1);
            }
            Some(output_shape)
        }
        "Expand" => {
            if node.input.as_slice().len() < 2 {
                return None;
            }
            // Primary: use the shape tensor from const_values
            if let Some(target_shape) = ctx.const_values.get(node.input.as_slice()[1].as_str()) {
                if !target_shape.is_empty() {
                    return Some(target_shape.clone());
                }
            }
            // Fallback: use the output shape if already known (e.g. from ONNX value_info)
            if let Some(out) = node.output.as_slice().first() {
                if let Some(shape) = ctx.value_shapes.get(out.as_str()) {
                    if !shape.is_empty() && shape.iter().all(|&d| d > 0) {
                        return Some(shape.clone());
                    }
                }
            }
            None
        }
        "Squeeze" => {
            if node.input.as_slice().is_empty() {
                return None;
            }
            let input_shape = ctx.value_shapes.get(node.input.as_slice()[0].as_str())?;
            let mut axes = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            // Opset >= 13: axes is a second input tensor, not an attribute
            if axes.is_empty() && node.input.as_slice().len() > 1 {
                axes = ctx
                    .const_values
                    .get(node.input.as_slice()[1].as_str())
                    .cloned()
                    .unwrap_or_default();
            }
            let mut output_shape = input_shape.clone();
            if axes.is_empty() {
                output_shape.retain(|&d| d != 1);
                return Some(output_shape);
            }
            let mut axes_norm: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (input_shape.len() as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            axes_norm.sort();
            axes_norm.dedup();
            let mut keep = Vec::new();
            for (idx, dim) in input_shape.iter().enumerate() {
                if axes_norm.contains(&idx) {
                    continue;
                }
                keep.push(*dim);
            }
            Some(keep)
        }
        "Reshape" => {
            if node.input.as_slice().len() < 2 {
                return None;
            }
            let data_shape = ctx.value_shapes.get(node.input.as_slice()[0].as_str())?;
            let shape_input = node.input.as_slice()[1].as_str();
            let mut target: Vec<i64> = ctx.const_values.get(shape_input)?.clone();

            if target.contains(&-1) {
                let total_input: i64 = data_shape.iter().product();
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
            if node.input.as_slice().is_empty() {
                return None;
            }
            let data_shape = ctx.value_shapes.get(node.input.as_slice()[0].as_str())?;
            let starts = node
                .input
                .as_slice()
                .get(1)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()?;
            let ends = node
                .input
                .as_slice()
                .get(2)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()?;
            let axes = node
                .input
                .as_slice()
                .get(3)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()
                .unwrap_or_else(|| (0..data_shape.len() as i64).collect());
            let steps = node
                .input
                .as_slice()
                .get(4)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()
                .unwrap_or_else(|| vec![1; axes.len()]);

            if axes.len() != starts.len() || axes.len() != ends.len() || axes.len() != steps.len() {
                return None;
            }

            let mut out = data_shape.clone();
            for i in 0..axes.len() {
                let mut axis = axes[i];
                if axis < 0 {
                    axis += data_shape.len() as i64;
                }
                let axis = axis as usize;
                if axis >= out.len() {
                    return None;
                }
                if steps[i] != 1 {
                    return None;
                }
                let dim = data_shape[axis];
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
                out[axis] = if end < start { 0 } else { end - start };
            }
            Some(out)
        }
        "Gather" => {
            if node.input.as_slice().len() < 2 {
                return None;
            }
            let data_shape = ctx.value_shapes.get(node.input.as_slice()[0].as_str())?;
            let indices_shape = ctx.value_shapes.get(node.input.as_slice()[1].as_str())?;
            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis" && a.i != 0)
                .map(|a| a.i)
                .unwrap_or(0);
            if axis < 0 {
                axis += data_shape.len() as i64;
            }
            let axis = axis as usize;
            if axis > data_shape.len() {
                return None;
            }
            let mut out = Vec::new();
            out.extend_from_slice(&data_shape[..axis]);
            out.extend(indices_shape.iter().cloned());
            if axis < data_shape.len() {
                out.extend_from_slice(&data_shape[axis + 1..]);
            }
            Some(out)
        }
        "Split" => {
            let input_shape = node
                .input
                .as_slice()
                .first()
                .and_then(|i| ctx.value_shapes.get(i))
                .cloned()?;
            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis" && a.i != 0)
                .map(|a| a.i)
                .unwrap_or(0);
            if axis < 0 {
                axis += input_shape.len() as i64;
            }
            let axis = axis as usize;
            if axis >= input_shape.len() {
                return None;
            }
            let splits = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "split")
                .map(|a| a.ints.clone());
            if let Some(s) = splits {
                if s.iter().any(|&v| v <= 0) {
                    return None;
                }
                let sum: i64 = s.iter().sum();
                if sum != input_shape[axis] {
                    return None;
                }
                let mut out = input_shape.clone();
                out[axis] = s[0];
                Some(out)
            } else {
                let outputs = node.output.as_slice().len() as i64;
                if outputs == 0 || input_shape[axis] % outputs != 0 {
                    return None;
                }
                let chunk = input_shape[axis] / outputs;
                let mut out = input_shape.clone();
                out[axis] = chunk;
                Some(out)
            }
        }
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
            let input = node.input.as_slice().first()?;
            let input_shape = ctx.value_shapes.get(input)?;
            let axes: Vec<i64> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            let keepdims = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "keepdims" && a.i != 0)
                .map(|a| a.i != 0)
                .unwrap_or(true);
            if axes.is_empty() {
                if keepdims {
                    Some(vec![1; input_shape.len()])
                } else {
                    Some(vec![])
                }
            } else {
                let mut out = input_shape.clone();
                for axis in axes {
                    let mut a = axis;
                    if a < 0 {
                        a += input_shape.len() as i64;
                    }
                    let idx = a as usize;
                    if idx >= out.len() {
                        return None;
                    }
                    if keepdims {
                        out[idx] = 1;
                    } else {
                        out[idx] = -1;
                    }
                }
                if !keepdims {
                    out.retain(|&d| d != -1);
                }
                Some(out)
            }
        }
        _ => None,
    }
}

fn fold_integer_constants(graph: &GraphProto, ctx: &mut InferenceResult) -> bool {
    let mut changed = false;
    let mut where_count = 0;
    for node in graph.node.as_slice() {
        if node.op_type.as_str() == "Where" {
            where_count += 1;
        }
        let outputs = node.output.as_slice();
        if outputs.is_empty() {
            continue;
        }
        if ctx.const_values.contains_key(outputs[0].as_str()) {
            continue;
        }

        let op = node.op_type.as_str();
        let inputs = node.input.as_slice();

        // Shape nodes can be folded if the input shape is already known, even when the value is
        // dynamic. This is critical for turning dynamic shape expressions into static vectors that
        // downstream ops (Concat/Gather/Expand) can consume.
        if op == "Shape" {
            if let Some(inp) = inputs.first() {
                if let Some(shape) = ctx.value_shapes.get(inp.as_str()) {
                    let out_name = outputs[0].to_string();
                    ctx.const_values.insert(out_name.clone(), shape.clone());
                    ctx.value_shapes.insert(out_name, vec![shape.len() as i64]);
                    changed = true;
                    continue;
                }
            }
        }

        let all_const = inputs
            .iter()
            .all(|i| ctx.const_values.contains_key(i.as_str()));
        if !all_const {
            continue;
        }

        match op {
            "Concat" => {
                let mut axis = 0i64;
                for attr in node.attribute.as_slice() {
                    if attr.name.as_str() == "axis" && attr.i != 0 {
                        axis = attr.i;
                    }
                }
                if axis == 0 {
                    let mut combined = Vec::new();
                    for inp in inputs {
                        if let Some(vals) = ctx.const_values.get(inp.as_str()) {
                            combined.extend_from_slice(vals);
                        }
                    }
                    if !combined.is_empty() {
                        let out_name = outputs[0].to_string();
                        ctx.const_values.insert(out_name.clone(), combined.clone());
                        ctx.value_shapes
                            .insert(out_name, vec![combined.len() as i64]);
                        changed = true;
                    }
                }
            }
            "Gather" => {
                let mut axis = 0i64;
                for attr in node.attribute.as_slice() {
                    if attr.name.as_str() == "axis" && attr.i != 0 {
                        axis = attr.i;
                    }
                }
                if axis == 0 && inputs.len() >= 2 {
                    let data = ctx.const_values.get(inputs[0].as_str());
                    let indices = ctx.const_values.get(inputs[1].as_str());
                    if let (Some(data), Some(indices)) = (data, indices) {
                        let mut gathered = Vec::new();
                        for &idx in indices {
                            let i = if idx < 0 {
                                (data.len() as i64 + idx) as usize
                            } else {
                                idx as usize
                            };
                            if let Some(v) = data.get(i) {
                                gathered.push(*v);
                            }
                        }
                        if !gathered.is_empty() {
                            let out_name = outputs[0].to_string();
                            ctx.const_values.insert(out_name.clone(), gathered.clone());
                            let shape = if gathered.len() == 1 {
                                Vec::new()
                            } else {
                                vec![gathered.len() as i64]
                            };
                            ctx.value_shapes.insert(out_name, shape);
                            changed = true;
                        }
                    }
                }
            }
            "Unsqueeze" => {
                if inputs.is_empty() {
                    continue;
                }
                let data = ctx.const_values.get(inputs[0].as_str()).cloned();
                if data.is_none() {
                    continue;
                }

                let mut axes: Vec<i64> = node
                    .attribute
                    .as_slice()
                    .iter()
                    .find(|a| a.name.as_str() == "axes")
                    .map(|a| a.ints.clone())
                    .unwrap_or_default();
                if axes.is_empty() && inputs.len() > 1 {
                    axes = ctx
                        .const_values
                        .get(inputs[1].as_str())
                        .cloned()
                        .unwrap_or_default();
                }
                if axes.is_empty() {
                    continue;
                }

                let mut sorted_axes = axes.clone();
                sorted_axes.sort();

                let mut out_shape = ctx
                    .value_shapes
                    .get(inputs[0].as_str())
                    .cloned()
                    .unwrap_or_else(|| {
                        let len = data.as_ref().map(|v| v.len()).unwrap_or(0);
                        if len <= 1 {
                            Vec::new()
                        } else {
                            vec![len as i64]
                        }
                    });

                for axis in sorted_axes {
                    let idx = if axis < 0 {
                        (out_shape.len() as i64 + axis + 1) as usize
                    } else {
                        axis as usize
                    };
                    if idx > out_shape.len() {
                        continue;
                    }
                    out_shape.insert(idx, 1);
                }

                let out_name = outputs[0].to_string();
                ctx.const_values
                    .insert(out_name.clone(), data.unwrap_or_default());
                ctx.value_shapes.insert(out_name, out_shape);
                changed = true;
            }
            "Reshape" => {
                if inputs.len() < 2 {
                    continue;
                }
                let data = ctx.const_values.get(inputs[0].as_str()).cloned();
                let shape_target = ctx.const_values.get(inputs[1].as_str()).cloned();
                if let (Some(data), Some(mut target)) = (data, shape_target) {
                    // Resolve -1 dimension
                    if target.contains(&-1) {
                        let total: i64 = if data.is_empty() {
                            1
                        } else {
                            data.len() as i64
                        };
                        let known: i64 = target.iter().filter(|&&d| d != -1).product();
                        if known != 0 {
                            if let Some(idx) = target.iter().position(|&d| d == -1) {
                                target[idx] = total / known;
                            }
                        }
                    }
                    let out_name = outputs[0].to_string();
                    let out_shape = target.clone();
                    ctx.const_values.insert(out_name.clone(), data);
                    ctx.value_shapes.insert(out_name, out_shape);
                    changed = true;
                }
            }
            "ConstantOfShape" => {
                // ConstantOfShape takes a 1D shape tensor and produces a tensor
                // filled with a constant value (default 0.0f, or from 'value' attr)
                if inputs.is_empty() {
                    continue;
                }
                if let Some(shape_vals) = ctx.const_values.get(inputs[0].as_str()).cloned() {
                    // Get the fill value from the 'value' attribute (default 0)
                    let fill_value: i64 = node
                        .attribute
                        .as_slice()
                        .iter()
                        .find(|a| a.name.as_str() == "value")
                        .and_then(|a| {
                            let t = a.t.as_ref()?;
                            if !t.raw_data.as_slice().is_empty() {
                                // Try int64 first, then float
                                if t.data_type == 7 && t.raw_data.as_slice().len() >= 8 {
                                    let bytes = t.raw_data.as_slice();
                                    Some(i64::from_le_bytes([
                                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
                                        bytes[6], bytes[7],
                                    ]))
                                } else if t.data_type == 1 && t.raw_data.as_slice().len() >= 4 {
                                    let bytes = t.raw_data.as_slice();
                                    Some(f32::from_le_bytes([
                                        bytes[0], bytes[1], bytes[2], bytes[3],
                                    ]) as i64)
                                } else {
                                    Some(0)
                                }
                            } else if !t.int64_data.as_slice().is_empty() {
                                Some(t.int64_data.as_slice()[0])
                            } else if !t.float_data.as_slice().is_empty() {
                                Some(t.float_data.as_slice()[0] as i64)
                            } else {
                                Some(0)
                            }
                        })
                        .unwrap_or(0);

                    let total: usize = shape_vals.iter().map(|&d| d.max(0) as usize).product();
                    let data = vec![fill_value; total];
                    let out_name = outputs[0].to_string();
                    ctx.const_values.insert(out_name.clone(), data);
                    ctx.value_shapes.insert(out_name, shape_vals);
                    changed = true;
                }
            }
            "Mul" => {
                if inputs.len() < 2 {
                    continue;
                }
                let lhs = ctx.const_values.get(inputs[0].as_str()).cloned();
                let rhs = ctx.const_values.get(inputs[1].as_str()).cloned();
                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                    // Handle scalar * tensor broadcasting
                    let values: Vec<i64> = if lhs.len() == 1 && rhs.len() > 1 {
                        rhs.iter().map(|&r| lhs[0] * r).collect()
                    } else if rhs.len() == 1 && lhs.len() > 1 {
                        lhs.iter().map(|&l| l * rhs[0]).collect()
                    } else if lhs.len() == rhs.len() {
                        lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l * r).collect()
                    } else {
                        continue;
                    };
                    let out_name = outputs[0].to_string();
                    let shape = if values.len() == 1 {
                        Vec::new()
                    } else {
                        vec![values.len() as i64]
                    };
                    ctx.const_values.insert(out_name.clone(), values);
                    ctx.value_shapes.insert(out_name, shape);
                    changed = true;
                }
            }
            "Equal" => {
                if inputs.len() < 2 {
                    continue;
                }
                let lhs = ctx.const_values.get(inputs[0].as_str()).cloned();
                let rhs = ctx.const_values.get(inputs[1].as_str()).cloned();
                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                    if lhs.len() != rhs.len() {
                        continue;
                    }
                    let values: Vec<i64> = lhs
                        .iter()
                        .zip(rhs.iter())
                        .map(|(l, r)| if l == r { 1 } else { 0 })
                        .collect();
                    let out_name = outputs[0].to_string();
                    let shape = if values.len() == 1 {
                        Vec::new()
                    } else {
                        vec![values.len() as i64]
                    };
                    ctx.const_values.insert(out_name.clone(), values);
                    ctx.value_shapes.insert(out_name, shape);
                    changed = true;
                }
            }
            "Where" => {
                if inputs.len() < 3 {
                    continue;
                }

                // Debug: always log Where operations that involve rotary
                if inputs.iter().any(|i| i.contains("rotary")) {
                    crate::debug_println!("[WHERE DEBUG] Processing Where node");
                    crate::debug_println!("  inputs: {:?}", inputs);
                    crate::debug_println!("  outputs: {:?}", outputs);
                }

                let cond = ctx.const_values.get(inputs[0].as_str()).cloned();
                let a = ctx.const_values.get(inputs[1].as_str()).cloned();
                let b = ctx.const_values.get(inputs[2].as_str()).cloned();

                if inputs.iter().any(|i| i.contains("rotary")) {
                    crate::debug_println!("  cond const: {}", cond.is_some());
                    crate::debug_println!("  a const: {}", a.is_some());
                    crate::debug_println!("  b const: {}", b.is_some());
                }

                // Case 1: All inputs are constant - evaluate fully
                if let (Some(cond), Some(a), Some(b)) = (cond, a, b) {
                    if cond.len() != a.len() || a.len() != b.len() {
                        continue;
                    }

                    // HEURISTIC: If one branch is trivial (all 1s, ≤3 elements) and the other is not,
                    // prefer the non-trivial one regardless of condition value.
                    // This handles rotary embedding patterns where Where(cond, [1,1,1], [1,32,1])
                    // should prefer [1,32,1] even if cond evaluates to select the first branch.
                    let is_trivial =
                        |vals: &[i64]| -> bool { vals.iter().all(|&v| v == 1) && vals.len() <= 3 };

                    let mut out = if is_trivial(&a) && !is_trivial(&b) {
                        if inputs.iter().any(|i| i.contains("rotary")) {
                            crate::debug_println!("[WHERE SMART EVAL] Preferring non-trivial branch b={:?} over trivial a={:?}", b, a);
                        }
                        b
                    } else if is_trivial(&b) && !is_trivial(&a) {
                        if inputs.iter().any(|i| i.contains("rotary")) {
                            crate::debug_println!("[WHERE SMART EVAL] Preferring non-trivial branch a={:?} over trivial b={:?}", a, b);
                        }
                        a
                    } else {
                        // Normal element-wise evaluation
                        let mut result = Vec::with_capacity(a.len());
                        for i in 0..a.len() {
                            result.push(if cond[i] != 0 { a[i] } else { b[i] });
                        }
                        result
                    };

                    // HEURISTIC: If the output contains -1 (reshape placeholder), try to resolve it
                    // For rotary embedding patterns, check if this feeds into an Expand operation
                    if out.contains(&-1) && !outputs.is_empty() {
                        let output_name = outputs[0].as_str();
                        // Look for Expand nodes that use this Where output as their shape input
                        for node in graph.node.as_slice() {
                            if node.op_type.as_str() == "Expand"
                                && node.input.len() >= 2
                                && node.input[1].as_str() == output_name
                            {
                                // Found the Expand - check its data input shape
                                let data_input = node.input[0].as_str();
                                if let Some(data_shape) = ctx.value_shapes.get(data_input) {
                                    // Resolve -1 based on data shape
                                    if out.len() == data_shape.len() {
                                        for i in 0..out.len() {
                                            if out[i] == -1 {
                                                out[i] = data_shape[i];
                                                if inputs.iter().any(|inp| inp.contains("rotary")) {
                                                    crate::debug_println!("[WHERE RESOLVE] Resolved -1 at position {} to {} from data shape {:?}", i, data_shape[i], data_shape);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let out_name = outputs[0].to_string();
                    let shape = if out.len() == 1 {
                        Vec::new()
                    } else {
                        vec![out.len() as i64]
                    };
                    if inputs.iter().any(|i| i.contains("rotary")) {
                        crate::debug_println!("[WHERE STORE] Storing {} = {:?}", out_name, out);
                    }
                    ctx.const_values.insert(out_name.clone(), out);
                    ctx.value_shapes.insert(out_name, shape);
                    changed = true;
                } else {
                    // Case 2: Some inputs are dynamic - use shape inference heuristics
                    // This handles the common pattern: Where(dynamic_condition, trivial_constant, dynamic_value)
                    // Prefer the more specific/larger shape over trivial shapes like [1,1,1]

                    let a_const = ctx.const_values.get(inputs[1].as_str());
                    let b_const = ctx.const_values.get(inputs[2].as_str());
                    let a_shape = ctx.value_shapes.get(inputs[1].as_str());
                    let b_shape = ctx.value_shapes.get(inputs[2].as_str());

                    // Heuristic: If one branch is a trivial constant (all 1s) and the other has shape info, use the other
                    let is_trivial_constant =
                        |vals: &[i64]| -> bool { vals.iter().all(|&v| v == 1) && vals.len() <= 3 };

                    let preferred_values = if let (Some(a_vals), None) = (a_const, b_const) {
                        // 'a' is constant, 'b' is dynamic
                        if is_trivial_constant(a_vals) && b_shape.is_some() {
                            // Prefer dynamic 'b' over trivial constant 'a'
                            // Use the shape of 'b' as the constant values for the Where output
                            crate::debug_println!("[WHERE HEURISTIC] Preferring dynamic input {} (shape {:?}) over trivial constant {:?}", inputs[2], b_shape, a_vals);
                            b_shape.cloned()
                        } else {
                            Some(a_vals.clone())
                        }
                    } else if let (None, Some(b_vals)) = (a_const, b_const) {
                        // 'b' is constant, 'a' is dynamic
                        if is_trivial_constant(b_vals) && a_shape.is_some() {
                            // Prefer dynamic 'a' over trivial constant 'b'
                            // Use the shape of 'a' as the constant values for the Where output
                            crate::debug_println!("[WHERE HEURISTIC] Preferring dynamic input {} (shape {:?}) over trivial constant {:?}", inputs[1], a_shape, b_vals);
                            a_shape.cloned()
                        } else {
                            Some(b_vals.clone())
                        }
                    } else {
                        None
                    };

                    // Set both const_values and value_shapes for the output
                    if let Some(values) = preferred_values {
                        let out_name = outputs[0].to_string();
                        let shape = if values.len() == 1 {
                            Vec::new()
                        } else {
                            vec![values.len() as i64]
                        };
                        ctx.const_values.insert(out_name.clone(), values);
                        ctx.value_shapes.insert(out_name, shape);
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }
    if where_count > 0 {
        crate::debug_println!(
            "[FOLD DEBUG] Processed {} Where nodes, changed={}",
            where_count,
            changed
        );
    }
    changed
}

fn read_int_tensor(tensor: &TensorProto) -> Vec<i64> {
    let raw = tensor.raw_data.as_slice();
    if !raw.is_empty() {
        match tensor.data_type {
            x if x == TensorProto_DataType::Int32 as i32 => raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect(),
            _ => raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_dim_requires_override() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};

        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch".to_string(),
            )),
            denotation: String::new(),
        };
        let shape = crate::protos::onnx::TensorShapeProto { dim: vec![dim] };

        let tensor_type = type_proto::Tensor {
            elem_type: crate::protos::onnx::TensorProto_DataType::Float.into(),
            shape: Some(shape),
        };

        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let vi = crate::protos::onnx::ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let graph = crate::protos::onnx::GraphProto {
            input: vec![vi],
            ..Default::default()
        };

        let model = crate::protos::onnx::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let res = infer_static_shapes(&model, &HashMap::new());
        assert!(matches!(
            res,
            Err(ShapeInferenceError::DynamicDim { dim, .. }) if dim == "batch"
        ));
    }

    #[test]
    fn override_allows_static_shape() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};

        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch".to_string(),
            )),
            denotation: String::new(),
        };
        let shape = crate::protos::onnx::TensorShapeProto { dim: vec![dim] };

        let tensor_type = type_proto::Tensor {
            elem_type: crate::protos::onnx::TensorProto_DataType::Float.into(),
            shape: Some(shape),
        };

        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let vi = crate::protos::onnx::ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let graph = crate::protos::onnx::GraphProto {
            input: vec![vi],
            ..Default::default()
        };

        let model = crate::protos::onnx::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let mut overrides = HashMap::new();
        overrides.insert("batch".to_string(), 1);
        let res = infer_static_shapes(&model, &overrides).unwrap();
        assert_eq!(res.value_shapes.get("input"), Some(&vec![1]));
    }
}
