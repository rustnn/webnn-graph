// Utility operators: Shape, Gather, Slice

use crate::ast::Node;
use crate::ast::{ConstDecl, ConstInit, DataType};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{
    normalize_axis_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::NodeProto;
use serde_json::{json, Map};

pub struct UtilityHandler;

impl OpHandler for UtilityHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Shape" | "Gather" | "Slice" | "ConstantOfShape" | "Range" | "Trilu"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Shape" => self.convert_shape(node, &node_name, context),
            "Gather" => self.convert_gather(node, &node_name, context),
            "Slice" => self.convert_slice(node, &node_name, context),
            "ConstantOfShape" => self.convert_constant_of_shape(node, &node_name, context),
            "Range" => self.convert_range(node, &node_name, context),
            "Trilu" => self.convert_trilu(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl UtilityHandler {
    /// Convert ONNX Shape to WebNN shape operation
    /// Returns a 1D tensor containing the dimensions of the input
    fn convert_shape(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Shape expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let options = Map::new();

        // WebNN doesn't have a direct shape operation, but we can use identity
        // and mark it with metadata that this is a shape operation
        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "shape".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    fn read_scalar_i64(&self, name: &str, context: &ConversionContext) -> Option<i64> {
        if let Some(vals) = context.const_values.get(name) {
            return vals.first().copied();
        }
        if let Some(t) = context.initializers.get(name) {
            let raw = t.raw_data.as_slice();
            if !raw.is_empty() {
                if t.data_type == crate::protos::onnx::TensorProto_DataType::Int32 as i32 {
                    return Some(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64);
                }
                if raw.len() >= 8 {
                    return Some(i64::from_le_bytes([
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                    ]));
                }
            } else if !t.int64_data.as_slice().is_empty() {
                return t.int64_data.as_slice().first().copied();
            } else if !t.int32_data.as_slice().is_empty() {
                return t.int32_data.as_slice().first().map(|v| *v as i64);
            }
        }
        None
    }

    fn convert_range(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "Range expects 3 inputs (start, limit, delta), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let start = self.read_scalar_i64(&inputs[0], context);
        let limit = self.read_scalar_i64(&inputs[1], context);
        let delta = self.read_scalar_i64(&inputs[2], context);

        let start_dim = crate::onnx::convert::dynamic_scalar_dimension_for_value(
            &inputs[0],
            context.value_shape_dims,
        );
        if let (Some(start), Some(delta), Some(limit_dim)) = (
            start,
            delta,
            crate::onnx::convert::dynamic_scalar_dimension_for_value(
                &inputs[1],
                context.value_shape_dims,
            ),
        ) {
            let range_dim = crate::onnx::convert::dynamic_range_length_dimension(
                start,
                delta,
                start_dim.as_ref(),
                &limit_dim,
            )
            .ok_or_else(|| {
                OnnxError::InvalidShape(format!(
                    "Range {} requires dynamic range length to be representable as <dim> +/- const with delta=1",
                    node_name,
                ))
            })?;

            let max_len = usize::try_from(range_dim.max_size).map_err(|_| {
                OnnxError::InvalidShape(format!(
                    "Range {} max size {} does not fit in usize",
                    node_name, range_dim.max_size
                ))
            })?;

            let use_runtime_start = start_dim.is_some();
            let mut values = Vec::with_capacity(max_len.max(1));
            let mut current = if use_runtime_start { 0 } else { start };
            for _ in 0..max_len {
                values.push(current);
                current += delta;
            }
            if values.is_empty() {
                values.push(if use_runtime_start { 0 } else { start });
            }

            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| v.to_le_bytes().to_vec())
                .collect();

            let range_const_name = format!("{}_range_const", output_name);
            let range_const = ConstDecl {
                data_type: DataType::Int64,
                shape: vec![values.len() as u32],
                init: ConstInit::InlineBytes { bytes },
            };

            let mut options = Map::new();
            options.insert("starts".to_string(), json!([0]));
            options.insert(
                "sizes".to_string(),
                json!([{
                    "name": range_dim.name,
                    "maxSize": range_dim.max_size
                }]),
            );
            options.insert("strides".to_string(), json!([1]));

            let sliced_name = if use_runtime_start {
                format!("{}_slice", output_name)
            } else {
                output_name.clone()
            };
            let mut nodes = vec![Node {
                id: sliced_name.clone(),
                op: "slice".to_string(),
                inputs: vec![range_const_name.clone()],
                options,
                outputs: None,
            }];
            if use_runtime_start {
                nodes.push(Node {
                    id: output_name.clone(),
                    op: "add".to_string(),
                    inputs: vec![sliced_name, context.resolve_input(&inputs[0])],
                    options: Map::new(),
                    outputs: None,
                });
            }

            let mut result = ConversionResult::new(nodes);
            result.consts.push((range_const_name, range_const));
            if let Some(out) = node.output.as_slice().first() {
                result
                    .output_mappings
                    .insert(out.to_string(), output_name.clone());
                result.output_types.insert(out.to_string(), DataType::Int64);
            }
            return Ok(result);
        }

        let start = start.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {} requires a constant scalar start input",
                node_name
            ))
        })?;
        let limit = limit.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {} requires a constant scalar or supported dynamic limit input",
                node_name
            ))
        })?;
        let delta = delta.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {} requires a constant scalar delta input",
                node_name
            ))
        })?;

        if delta == 0 {
            return Err(OnnxError::InvalidShape(
                "Range delta cannot be zero".to_string(),
            ));
        }

        let mut values = Vec::new();
        let mut v = start;
        if delta > 0 {
            while v < limit {
                values.push(v);
                v += delta;
            }
        } else {
            while v > limit {
                values.push(v);
                v += delta;
            }
        }

        if values.is_empty() {
            values.push(0);
        }

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect();

        let const_decl = ConstDecl {
            data_type: DataType::Int64,
            shape: vec![values.len() as u32],
            init: ConstInit::InlineBytes { bytes },
        };

        let mut result = ConversionResult::new(vec![]);
        result.consts.push((output_name.clone(), const_decl));
        if let Some(out) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(out.to_string(), output_name.clone());
            result.output_types.insert(out.to_string(), DataType::Int64);
        }

        Ok(result)
    }

    fn convert_trilu(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Trilu expects at least 1 input (data)".to_string(),
            ));
        }

        if inputs.len() > 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Trilu expects at most 2 inputs (data, k), got {}",
                inputs.len()
            )));
        }

        let mut upper = true;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "upper" {
                upper = attr.i != 0;
            }
        }

        let mut k: i64 = 0;
        if inputs.len() == 2 {
            let k_input = inputs[1].as_str();
            if let Some(offset) = self.read_scalar_i64(k_input, context) {
                k = offset;
            } else {
                return Err(OnnxError::InvalidShape(
                    "Trilu k input must be a constant scalar for WebNN".to_string(),
                ));
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let mut options = Map::new();
        options.insert("upper".to_string(), json!(upper));
        options.insert("k".to_string(), json!(k));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "triangular".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
    }

    /// Convert ConstantOfShape into an inline constant when the output shape is statically known.
    fn convert_constant_of_shape(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let output_dim_shape = node
            .output
            .as_slice()
            .first()
            .and_then(|out| {
                let out_s = out.to_string();
                context
                    .value_shape_dims
                    .get(&out_s)
                    .or_else(|| context.value_shape_dims.get(&sanitize_identifier(&out_s)))
                    .or_else(|| context.value_shape_dims.get(out_s.trim_start_matches('/')))
            })
            .cloned();

        // Determine the target shape: prefer inferred output shape, otherwise try the shape input const.
        let mut shape: Option<Vec<i64>> = None;
        if let Some(out) = node.output.as_slice().first() {
            if let Some(s) = context.value_shapes.get(out) {
                shape = Some(s.clone());
            } else {
                let sanitized = sanitize_identifier(out);
                if let Some(s) = context.value_shapes.get(&sanitized) {
                    shape = Some(s.clone());
                }
            }
        }
        if shape.is_none() {
            if let Some(shape_input) = node.input.as_slice().first() {
                if let Some(vals) = context.const_values.get(shape_input) {
                    shape = Some(vals.clone());
                } else if let Some(len_shape) = context.value_shapes.get(shape_input) {
                    // If we only know the length of the shape tensor, default the dims to 1s.
                    if len_shape.len() == 1 && len_shape[0] > 0 {
                        shape = Some(vec![1; len_shape[0] as usize]);
                    }
                }
            }
        }

        // Determine fill value and data type (default int64 zero)
        let mut fill_value_i64: i64 = 0;
        let mut dtype = DataType::Int64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "value" {
                if let Some(t) = attr.t.as_ref() {
                    match t.data_type {
                        // FLOAT
                        x if x == crate::protos::onnx::TensorProto_DataType::Float as i32 => {
                            dtype = DataType::Float32;
                            if !t.float_data.as_slice().is_empty() {
                                fill_value_i64 = t.float_data.as_slice()[0].to_bits() as i64;
                            } else if !t.raw_data.as_slice().is_empty()
                                && t.raw_data.as_slice().len() >= 4
                            {
                                let raw = &t.raw_data.as_slice()[..4];
                                let bits = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                                fill_value_i64 = bits as i64;
                            } else {
                                fill_value_i64 = 0f32.to_bits() as i64;
                            }
                        }
                        // INT64
                        x if x == crate::protos::onnx::TensorProto_DataType::Int64 as i32 => {
                            dtype = DataType::Int64;
                            if !t.int64_data.as_slice().is_empty() {
                                fill_value_i64 = t.int64_data.as_slice()[0];
                            } else if !t.raw_data.as_slice().is_empty()
                                && t.raw_data.as_slice().len() >= 8
                            {
                                let raw = &t.raw_data.as_slice()[..8];
                                fill_value_i64 = i64::from_le_bytes([
                                    raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                                ]);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(dims) = output_dim_shape.as_ref().filter(|dims| {
            dims.iter()
                .any(|d| matches!(d, crate::ast::Dimension::Dynamic(_)))
        }) {
            let scalar_name = format!("{}_fill", output_name);
            let scalar_bytes = match dtype {
                DataType::Float32 => {
                    let f = f32::from_bits(fill_value_i64 as u32);
                    f.to_le_bytes().to_vec()
                }
                _ => fill_value_i64.to_le_bytes().to_vec(),
            };
            let scalar_decl = ConstDecl {
                data_type: dtype.clone(),
                shape: vec![1],
                init: ConstInit::InlineBytes {
                    bytes: scalar_bytes,
                },
            };

            let new_shape: Vec<serde_json::Value> = dims
                .iter()
                .map(|d| match d {
                    crate::ast::Dimension::Static(v) => serde_json::json!(v),
                    crate::ast::Dimension::Dynamic(dd) => serde_json::json!({
                        "name": dd.name,
                        "maxSize": dd.max_size
                    }),
                })
                .collect();

            let mut options = Map::new();
            options.insert("newShape".to_string(), serde_json::json!(new_shape));

            let mut result = ConversionResult::new(vec![Node {
                id: output_name.clone(),
                op: "expand".to_string(),
                inputs: vec![scalar_name.clone()],
                options,
                outputs: None,
            }]);
            result.consts.push((scalar_name, scalar_decl));
            if let Some(out) = node.output.as_slice().first() {
                result
                    .output_mappings
                    .insert(out.to_string(), output_name.clone());
                result.output_types.insert(out.to_string(), dtype);
            }
            return Ok(result);
        }

        let shape = shape.unwrap_or_else(|| vec![1]);

        let mut numel: usize = 1;
        for d in &shape {
            if *d <= 0 {
                return Err(OnnxError::InvalidShape(format!(
                    "ConstantOfShape '{}' has non-positive dimension {:?}",
                    node_name, shape
                )));
            }
            numel = numel.saturating_mul(*d as usize);
        }

        let bytes = match dtype {
            DataType::Float32 => {
                let f = f32::from_bits(fill_value_i64 as u32);
                let val = f.to_le_bytes();
                val.repeat(numel)
            }
            _ => {
                let val = fill_value_i64.to_le_bytes();
                val.repeat(numel)
            }
        };

        let const_decl = ConstDecl {
            data_type: dtype.clone(),
            shape: shape.iter().map(|d| *d as u32).collect(),
            init: ConstInit::InlineBytes { bytes },
        };

        let mut result = ConversionResult::new(vec![]);
        result.consts.push((output_name.clone(), const_decl));
        if let Some(out) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(out.to_string(), output_name.clone());
            result.output_types.insert(out.to_string(), dtype);
        }

        Ok(result)
    }

    /// Convert ONNX Gather to WebNN gather
    /// Gathers elements along a specified axis using indices
    fn convert_gather(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gather expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute (default: 0)
        let mut axis = 0i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);
        let input1 = context.resolve_input(&inputs[1]);

        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank)
        } else {
            axis
        };

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));

        // Propagate output shape metadata when available so downstream ops see correct ranks
        if let (Some(data_shape), Some(indices_shape)) = (
            context.value_shapes.get(&inputs[0]),
            context.value_shapes.get(&inputs[1]),
        ) {
            let resolved_axis = axis;
            if resolved_axis >= 0 && (resolved_axis as usize) < data_shape.len() {
                let axis_idx = resolved_axis as usize;
                let mut out_shape = Vec::new();
                out_shape.extend_from_slice(&data_shape[..axis_idx]);
                out_shape.extend(indices_shape.iter().cloned());
                if axis_idx < data_shape.len() {
                    out_shape.extend_from_slice(&data_shape[axis_idx + 1..]);
                }
                options.insert("shape".to_string(), serde_json::json!(out_shape));
            }
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "gather".to_string(),
            inputs: vec![input0, input1],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
    }

    /// Convert ONNX Slice to WebNN slice
    /// Extracts a slice from the input tensor
    fn convert_slice(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Slice expects at least 1 input".to_string(),
            ));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let read_ints = |name: &str, context: &ConversionContext| -> Option<Vec<i64>> {
            if let Some(vals) = context.const_values.get(name) {
                return Some(vals.clone());
            }
            if let Some(t) = context.initializers.get(name) {
                let raw = t.raw_data.as_slice();
                if !raw.is_empty() {
                    if t.data_type == crate::protos::onnx::TensorProto_DataType::Int32 as i32 {
                        return Some(
                            raw.chunks_exact(4)
                                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                .collect(),
                        );
                    }
                    return Some(
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect(),
                    );
                } else if !t.int64_data.as_slice().is_empty() {
                    return Some(t.int64_data.as_slice().to_vec());
                } else if !t.int32_data.as_slice().is_empty() {
                    return Some(t.int32_data.as_slice().iter().map(|&v| v as i64).collect());
                }
            }
            None
        };

        let mut options = Map::new();

        // In opset >= 10, starts/ends/axes/steps are inputs
        // WebNN requires static values, so we enforce const-ness here.
        if inputs.len() >= 3 {
            let starts_name = inputs[1].as_str();
            let ends_name = inputs[2].as_str();
            let mut starts = read_ints(starts_name, context);
            let mut ends = read_ints(ends_name, context);

            if starts.is_none() || ends.is_none() {
                // As a last resort, try to pull starts/ends from sibling consts
                // produced by earlier shape inference passes.
                if let Some(s) = context.const_values.get(starts_name) {
                    starts = Some(s.clone());
                }
                if let Some(e) = context.const_values.get(ends_name) {
                    ends = Some(e.clone());
                }

                let fallback_len = if let Some(axes_name) = inputs.get(3).map(|s| s.as_str()) {
                    read_ints(axes_name, context)
                        .map(|v| v.len())
                        .unwrap_or_else(|| {
                            starts
                                .as_ref()
                                .map(|v| v.len())
                                .or_else(|| {
                                    context
                                        .value_shapes
                                        .get(inputs[0].as_str())
                                        .map(|s| s.len())
                                })
                                .unwrap_or(1)
                        })
                } else {
                    starts
                        .as_ref()
                        .map(|v| v.len())
                        .or_else(|| {
                            context
                                .value_shapes
                                .get(inputs[0].as_str())
                                .map(|s| s.len())
                        })
                        .unwrap_or(1)
                };

                starts.get_or_insert(vec![0; fallback_len]);
                // Keep Slice dynamic when ONNX ends input is non-const.
                ends.get_or_insert(vec![i64::MAX; fallback_len]);

                crate::debug_println!(
                    "[slice] using fallback starts/ends for {}, starts={:?} ends={:?}",
                    node_name,
                    starts,
                    ends
                );
            }

            let starts = starts.ok_or_else(|| {
                OnnxError::InvalidShape("Slice starts must be constant for WebNN".to_string())
            })?;
            let ends = ends.ok_or_else(|| {
                OnnxError::InvalidShape("Slice ends must be constant for WebNN".to_string())
            })?;

            // Normalize lengths: starts/ends must match axes length if provided,
            // otherwise match each other.
            let mut axes_opt: Option<Vec<i64>> = None;
            if inputs.len() >= 4 {
                let axes_name = inputs[3].as_str();
                if let Some(axes) = read_ints(axes_name, context) {
                    axes_opt = Some(axes);
                }
            }

            let desired_len = axes_opt
                .as_ref()
                .map(|a| a.len())
                .unwrap_or_else(|| starts.len().max(ends.len()));
            let mut starts_norm = starts;
            let mut ends_norm = ends;
            if starts_norm.len() > desired_len {
                starts_norm.truncate(desired_len);
            } else {
                starts_norm.resize(desired_len, 0);
            }
            if ends_norm.len() > desired_len {
                ends_norm.truncate(desired_len);
            } else {
                // If we know data shape, use its dims; otherwise use max i64.
                let fill = context
                    .value_shapes
                    .get(inputs[0].as_str())
                    .and_then(|s| s.first())
                    .copied()
                    .unwrap_or(i64::MAX);
                ends_norm.resize(desired_len, fill);
            }

            if let Some(input_shape) = context.resolve_shape(inputs[0].as_str()) {
                let rank = input_shape.len();
                let mut axes = if let Some(a) = axes_opt {
                    if a.is_empty() {
                        (0..desired_len as i64).collect::<Vec<_>>()
                    } else {
                        a
                    }
                } else {
                    (0..desired_len as i64).collect::<Vec<_>>()
                };
                if axes.len() != desired_len {
                    axes.resize(desired_len, 0);
                }
                let axes: Vec<i64> = axes
                    .iter()
                    .map(|&a| normalize_axis_best_effort(a, rank))
                    .collect();

                let mut steps = if inputs.len() >= 5 {
                    let steps_name = inputs[4].as_str();
                    read_ints(steps_name, context).unwrap_or_default()
                } else {
                    Vec::new()
                };
                if steps.len() > desired_len {
                    steps.truncate(desired_len);
                } else {
                    steps.resize(desired_len, 1);
                }

                let mut dense_starts = vec![0i64; rank];
                let mut dense_sizes: Vec<i64> = input_shape.clone();
                let mut dense_strides = vec![1i64; rank];

                // Check if ends input has dynamic dimension metadata
                let ends_dims = context.value_shape_dims.get(ends_name).or_else(|| {
                    context
                        .value_shape_dims
                        .get(&sanitize_identifier(ends_name))
                });

                // Track which dense axes have dynamic sizes
                let mut dynamic_size_info: Vec<Option<crate::ast::DynamicDimension>> =
                    vec![None; rank];

                for i in 0..desired_len {
                    let axis = axes[i] as usize;
                    let dim = input_shape[axis];
                    let step = steps[i];
                    if step <= 0 {
                        return Err(OnnxError::InvalidShape(
                            "Slice currently requires positive step values".to_string(),
                        ));
                    }

                    let mut start = starts_norm[i];
                    let mut end = ends_norm[i];
                    if start < 0 {
                        start += dim;
                    }
                    if end == i64::MAX {
                        end = dim;
                    } else if end < 0 {
                        end += dim;
                    }
                    start = start.clamp(0, dim);
                    end = end.clamp(0, dim);

                    let size = if end <= start {
                        0
                    } else {
                        (end - start + step - 1) / step
                    };

                    // If this end value came from a dynamic dimension, mark the size as dynamic
                    if let Some(dims) = ends_dims {
                        if let Some(crate::ast::Dimension::Dynamic(dd)) = dims.get(i) {
                            dynamic_size_info[axis] = Some(crate::ast::DynamicDimension {
                                name: dd.name.clone(),
                                max_size: size as u32,
                            });
                        }
                    }

                    dense_starts[axis] = start;
                    dense_sizes[axis] = size;
                    dense_strides[axis] = step;
                }

                options.insert("starts".to_string(), serde_json::json!(dense_starts));

                // Emit sizes with dynamic dimension metadata when present
                let has_dynamic = dynamic_size_info.iter().any(|d| d.is_some());
                if has_dynamic {
                    let sizes_json: Vec<serde_json::Value> = dense_sizes
                        .iter()
                        .zip(dynamic_size_info.iter())
                        .map(|(&sz, dyn_info)| match dyn_info {
                            Some(dd) => serde_json::json!({
                                "name": dd.name,
                                "maxSize": dd.max_size
                            }),
                            None => serde_json::json!(sz),
                        })
                        .collect();
                    options.insert("sizes".to_string(), serde_json::json!(sizes_json));
                } else {
                    options.insert("sizes".to_string(), serde_json::json!(dense_sizes));
                }

                options.insert("strides".to_string(), serde_json::json!(dense_strides));
            } else {
                // Fallback for unknown-rank tensors: keep ONNX-style static slice options.
                options.insert("starts".to_string(), serde_json::json!(starts_norm));
                options.insert("ends".to_string(), serde_json::json!(ends_norm));
                if let Some(axes) = axes_opt {
                    options.insert("axes".to_string(), serde_json::json!(axes));
                }
                if inputs.len() >= 5 {
                    let steps_name = inputs[4].as_str();
                    if let Some(steps) = read_ints(steps_name, context) {
                        options.insert("steps".to_string(), serde_json::json!(steps));
                    }
                }
            }
        } else {
            // Extract from attributes (older opset)
            for attr in node.attribute.as_slice() {
                match attr.name.as_str() {
                    "starts" => {
                        options
                            .insert("starts".to_string(), serde_json::json!(&attr.ints.to_vec()));
                    }
                    "ends" => {
                        options.insert("ends".to_string(), serde_json::json!(&attr.ints.to_vec()));
                    }
                    "axes" => {
                        options.insert("axes".to_string(), serde_json::json!(&attr.ints.to_vec()));
                    }
                    "steps" => {
                        options.insert("steps".to_string(), serde_json::json!(&attr.ints.to_vec()));
                    }
                    _ => {}
                }
            }
            if !options.contains_key("starts") || !options.contains_key("ends") {
                return Err(OnnxError::InvalidShape(
                    "Slice requires static starts/ends".to_string(),
                ));
            }

            if let Some(input_shape) = context.resolve_shape(inputs[0].as_str()) {
                let rank = input_shape.len();
                let starts = options
                    .remove("starts")
                    .and_then(|v| serde_json::from_value::<Vec<i64>>(v).ok())
                    .ok_or_else(|| OnnxError::InvalidShape("Slice starts malformed".to_string()))?;
                let ends = options
                    .remove("ends")
                    .and_then(|v| serde_json::from_value::<Vec<i64>>(v).ok())
                    .ok_or_else(|| OnnxError::InvalidShape("Slice ends malformed".to_string()))?;
                let axes = options
                    .remove("axes")
                    .and_then(|v| serde_json::from_value::<Vec<i64>>(v).ok())
                    .unwrap_or_else(|| (0..starts.len() as i64).collect::<Vec<_>>());
                let mut steps = options
                    .remove("steps")
                    .and_then(|v| serde_json::from_value::<Vec<i64>>(v).ok())
                    .unwrap_or_else(|| vec![1; starts.len()]);

                let desired_len = starts.len().max(ends.len()).max(axes.len());
                let mut starts = starts;
                let mut ends = ends;
                let mut axes = axes;
                if starts.len() < desired_len {
                    starts.resize(desired_len, 0);
                }
                if ends.len() < desired_len {
                    ends.resize(desired_len, i64::MAX);
                }
                if axes.len() < desired_len {
                    axes.resize(desired_len, 0);
                }
                if steps.len() < desired_len {
                    steps.resize(desired_len, 1);
                }

                let axes: Vec<i64> = axes
                    .iter()
                    .map(|&a| normalize_axis_best_effort(a, rank))
                    .collect();
                let mut dense_starts = vec![0i64; rank];
                let mut dense_sizes: Vec<i64> = input_shape.clone();
                let mut dense_strides = vec![1i64; rank];

                for i in 0..desired_len {
                    let axis = axes[i] as usize;
                    let dim = input_shape[axis];
                    let step = steps[i];
                    if step <= 0 {
                        return Err(OnnxError::InvalidShape(
                            "Slice currently requires positive step values".to_string(),
                        ));
                    }

                    let mut start = starts[i];
                    let mut end = ends[i];
                    if start < 0 {
                        start += dim;
                    }
                    if end == i64::MAX {
                        end = dim;
                    } else if end < 0 {
                        end += dim;
                    }
                    start = start.clamp(0, dim);
                    end = end.clamp(0, dim);

                    let size = if end <= start {
                        0
                    } else {
                        (end - start + step - 1) / step
                    };

                    dense_starts[axis] = start;
                    dense_sizes[axis] = size;
                    dense_strides[axis] = step;
                }

                options.insert("starts".to_string(), serde_json::json!(dense_starts));
                options.insert("sizes".to_string(), serde_json::json!(dense_sizes));
                options.insert("strides".to_string(), serde_json::json!(dense_strides));
            }
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "slice".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DataType;
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto, TensorProto_DataType};
    use serde_json::json;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        let attr = AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    #[test]
    fn test_utility_handler_supports() {
        let handler = UtilityHandler;
        assert!(handler.supports("Shape"));
        assert!(handler.supports("Gather"));
        assert!(handler.supports("Slice"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_shape() {
        let handler = UtilityHandler;
        let node = create_test_node("Shape", vec!["x"], vec!["shape"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "shape");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_gather() {
        let handler = UtilityHandler;
        let mut node = create_test_node("Gather", vec!["data", "indices"], vec!["output"]);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![2, 3, 4]);
        value_shapes.insert("indices".to_string(), vec![2]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "gather");
        assert_eq!(result.nodes[0].inputs.len(), 2);
        assert!(result.nodes[0].options.contains_key("axis"));
        assert_eq!(
            result.nodes[0].options.get("axis"),
            Some(&serde_json::json!(2))
        );
    }

    #[test]
    fn test_convert_slice() {
        let handler = UtilityHandler;
        let node = create_test_node(
            "Slice",
            vec!["x", "starts", "ends", "axes", "steps"],
            vec!["output"],
        );
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 128]);
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("starts".to_string(), vec![0]);
        const_values.insert("ends".to_string(), vec![128]);
        const_values.insert("axes".to_string(), vec![1]);
        const_values.insert("steps".to_string(), vec![1]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "slice");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("starts"));
        assert_eq!(
            result.nodes[0].options.get("starts"),
            Some(&serde_json::json!([0, 0]))
        );
        assert_eq!(
            result.nodes[0].options.get("sizes"),
            Some(&serde_json::json!([1, 128]))
        );
        assert_eq!(
            result.nodes[0].options.get("strides"),
            Some(&serde_json::json!([1, 1]))
        );
        assert!(!result.nodes[0].options.contains_key("ends"));
        assert!(!result.nodes[0].options.contains_key("axes"));
        assert!(!result.nodes[0].options.contains_key("steps"));
    }

    #[test]
    fn test_convert_constant_of_shape_prefers_dynamic_output_dims() {
        let handler = UtilityHandler;
        let mut node = create_test_node("ConstantOfShape", vec!["shape"], vec!["output"]);
        node.attribute.push(AttributeProto {
            name: "value".to_string(),
            t: Some(TensorProto {
                data_type: TensorProto_DataType::Float as i32,
                dims: vec![],
                raw_data: 0f32.to_le_bytes().to_vec(),
                ..Default::default()
            }),
            ..Default::default()
        });

        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("output".to_string(), vec![4096, 4096]);
        let mut value_shape_dims = std::collections::HashMap::new();
        value_shape_dims.insert(
            "output".to_string(),
            vec![
                crate::ast::Dimension::Dynamic(crate::ast::DynamicDimension {
                    name: "sequence_length".to_string(),
                    max_size: 4096,
                }),
                crate::ast::Dimension::Dynamic(crate::ast::DynamicDimension {
                    name: "past_sequence_length + 1".to_string(),
                    max_size: 4096,
                }),
            ],
        );
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("shape".to_string(), vec![4096, 4096]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: &value_shape_dims,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "expand");
        assert_eq!(result.nodes[0].inputs.len(), 1);
        assert_eq!(result.consts.len(), 1);
        assert_eq!(result.consts[0].1.shape, vec![1]);
        assert_eq!(
            result.nodes[0].options.get("newShape"),
            Some(&json!([
                {"name": "sequence_length", "maxSize": 4096},
                {"name": "past_sequence_length + 1", "maxSize": 4096}
            ]))
        );
        assert_eq!(result.output_types.get("output"), Some(&DataType::Float32));
    }

    #[test]
    fn test_convert_trilu_defaults() {
        let handler = UtilityHandler;
        let node = create_test_node("Trilu", vec!["x"], vec!["y"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let mut value_types = std::collections::HashMap::new();
        value_types.insert("x".to_string(), DataType::Float32);
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "triangular");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert_eq!(result.nodes[0].options.get("upper"), Some(&json!(true)));
        assert_eq!(result.nodes[0].options.get("k"), Some(&json!(0)));
        assert_eq!(result.output_mappings.get("y"), Some(&"y".to_string()));
        assert_eq!(result.output_types.get("y"), Some(&DataType::Float32));
    }

    #[test]
    fn test_convert_trilu_with_k_and_lower() {
        let handler = UtilityHandler;
        let mut node = create_test_node("Trilu", vec!["x", "k"], vec!["y"]);
        add_int_attribute(&mut node, "upper", 0);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("k".to_string(), vec![2]);
        let value_ids = std::collections::HashMap::new();
        let mut value_types = std::collections::HashMap::new();
        value_types.insert("x".to_string(), DataType::Float16);
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "triangular");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert_eq!(result.nodes[0].options.get("upper"), Some(&json!(false)));
        assert_eq!(result.nodes[0].options.get("k"), Some(&json!(2)));
        assert_eq!(result.output_mappings.get("y"), Some(&"y".to_string()));
        assert_eq!(result.output_types.get("y"), Some(&DataType::Float16));
    }
}
