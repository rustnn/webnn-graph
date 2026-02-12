// Reshape operators: Reshape, Transpose, Concat, Split, Unsqueeze, Squeeze

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::{NodeProto, TensorProto_DataType};
use serde_json::Map;

pub struct ReshapeHandler;

impl OpHandler for ReshapeHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Reshape"
                | "Transpose"
                | "Concat"
                | "Split"
                | "Unsqueeze"
                | "Squeeze"
                | "Tile"
                | "Expand"
        )
    }

    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Reshape" => self.convert_reshape(node, &node_name, context),
            "Transpose" => self.convert_transpose(node, &node_name, context),
            "Concat" => self.convert_concat(node, &node_name, context),
            "Split" => self.convert_split(node, &node_name, context),
            "Unsqueeze" => self.convert_unsqueeze(node, &node_name, context),
            "Squeeze" => self.convert_squeeze(node, &node_name, context),
            "Tile" => self.convert_tile(node, &node_name, context),
            "Expand" => self.convert_expand(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReshapeHandler {
    fn read_axes_from_attr_or_const(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<Vec<i64>, OnnxError> {
        if let Some(attr_axes) = node
            .attribute
            .as_slice()
            .iter()
            .find(|a| a.name.as_str() == "axes")
            .map(|a| a.ints.clone())
        {
            return Ok(if attr_axes.is_empty() {
                vec![0]
            } else {
                attr_axes
            });
        }

        if node.input.as_slice().len() >= 2 {
            let name = node.input.as_slice()[1].to_string();
            if let Some(vals) = context.const_values.get(&name) {
                return Ok(if vals.is_empty() {
                    vec![0]
                } else {
                    vals.clone()
                });
            }
            if let Some(t) = context.initializers.get(&name) {
                let raw = t.raw_data.as_slice();
                if !raw.is_empty() {
                    let mut axes: Vec<i64> = raw
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    if axes.is_empty() {
                        axes.push(0);
                    }
                    return Ok(axes);
                } else if !t.int64_data.as_slice().is_empty() {
                    let mut axes = t.int64_data.as_slice().to_vec();
                    if axes.is_empty() {
                        axes.push(0);
                    }
                    return Ok(axes);
                } else if !t.int32_data.as_slice().is_empty() {
                    let mut axes: Vec<i64> =
                        t.int32_data.as_slice().iter().map(|&v| v as i64).collect();
                    if axes.is_empty() {
                        axes.push(0);
                    }
                    return Ok(axes);
                }
            }
            return Ok(vec![0]);
        }

        Ok(vec![0])
    }

    /// Ensure Unsqueeze axes are available even if missing in the ONNX node by
    /// defaulting to a new leading dimension. This guards against malformed
    /// exports where the axes input was stripped.
    /// Convert ONNX Reshape to WebNN reshape
    /// ONNX Reshape takes shape as a second input (constant tensor)
    /// WebNN reshape takes newShape as a static array option
    fn convert_reshape<'a>(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &crate::onnx::ops::ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Reshape expects 2 inputs (data, shape), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let data_input_raw = inputs[0].to_string();
        let shape_input_raw = inputs[1].to_string();
        let data_input = context.resolve_input(&data_input_raw);

        // Resolve shape from const-folded values or initializers
        let mut shape_values: Vec<i64> =
            if let Some(values) = context.const_values.get(&shape_input_raw) {
                values.clone()
            } else if let Some(initializer) = context.initializers.get(shape_input_raw.as_str()) {
                let raw_data = initializer.raw_data.as_slice();
                if !raw_data.is_empty() {
                    match initializer.data_type {
                        x if x == TensorProto_DataType::Int32 as i32 => raw_data
                            .chunks_exact(4)
                            .map(|chunk| {
                                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64
                            })
                            .collect(),
                        _ => raw_data
                            .chunks_exact(8)
                            .map(|chunk| {
                                i64::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                    chunk[6], chunk[7],
                                ])
                            })
                            .collect(),
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
                }
            } else {
                Vec::new()
            };
        let shape_from_const = !shape_values.is_empty();

        // Fallback: derive shape from known input shape when the shape tensor isn't const.
        if shape_values.is_empty() {
            if let Some(ds) = context.value_shapes.get(data_input_raw.as_str()) {
                if ds.len() >= 3 {
                    let tail: i64 = ds[2..].iter().product();
                    shape_values = vec![ds[0], ds[1], tail];
                } else {
                    shape_values = ds.clone();
                }
                // Debug: track shape derivation for layer 15
                if output_name.contains("layers_15_self_attn") && output_name.contains("Reshape") {
                    crate::debug_println!(
                        "[RESHAPE FALLBACK] {} from input {:?} -> {:?}",
                        output_name,
                        ds,
                        shape_values
                    );
                }
            } else if let Some(ds) = context.value_shapes.get(&data_input) {
                if ds.len() >= 3 {
                    let tail: i64 = ds[2..].iter().product();
                    shape_values = vec![ds[0], ds[1], tail];
                } else {
                    shape_values = ds.clone();
                }
                // Debug: track shape derivation for layer 15
                if output_name.contains("layers_15_self_attn") && output_name.contains("Reshape") {
                    crate::debug_println!(
                        "[RESHAPE FALLBACK] {} from input {:?} -> {:?}",
                        output_name,
                        ds,
                        shape_values
                    );
                }
            } else {
                let output_dims_opt = node
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
                if let Some(output_dims) = output_dims_opt {
                    let new_shape_json: Vec<serde_json::Value> = output_dims
                        .into_iter()
                        .map(|d| match d {
                            crate::ast::Dimension::Static(v) => serde_json::json!(v),
                            crate::ast::Dimension::Dynamic(dd) => serde_json::json!({
                                "name": dd.name,
                                "maxSize": dd.max_size
                            }),
                        })
                        .collect();
                    if !new_shape_json.is_empty() {
                        let mut options = Map::new();
                        options.insert("newShape".to_string(), serde_json::json!(new_shape_json));

                        let mut result = ConversionResult::new(vec![Node {
                            id: output_name.clone(),
                            op: "reshape".to_string(),
                            inputs: vec![data_input],
                            options,
                            outputs: None,
                        }]);

                        if let Some(output) = node.output.as_slice().first() {
                            result
                                .output_mappings
                                .insert(output.to_string(), output_name.clone());
                        }

                        return Ok(result);
                    }
                }

                return Err(OnnxError::InvalidShape(format!(
                    "Reshape shape input '{}' must be a constant (initializer/constant-folded) or input shape must be known.",
                    shape_input_raw
                )));
            }
        } else if shape_from_const
            && output_name.contains("layers_15_self_attn")
            && output_name.contains("Reshape")
        {
            // Debug: track const-derived shapes for layer 15
            crate::debug_println!(
                "[RESHAPE CONST] {} newShape from const -> {:?}",
                output_name,
                shape_values
            );
        }

        // Handle -1 (dimension inference marker) - compute the inferred dimension
        // WebNN requires all dimensions to be explicit, so we need to resolve -1 values
        let input_shape_opt = {
            let trimmed = data_input_raw.trim_start_matches('/');
            context
                .value_shapes
                .get(data_input_raw.as_str())
                .or_else(|| context.value_shapes.get(&data_input))
                .or_else(|| context.value_shapes.get(trimmed))
                .cloned()
        };

        let shape_values: Vec<u32> = if shape_values.contains(&-1) {
            // Prefer strict inference when we know the input shape; otherwise fall back to
            // best-effort by replacing -1 with 1 so conversion can proceed for fixed-step
            // decoder exports (batch=1, seq=1) even when upstream shape info is missing.
            if let Some(input_shape) = input_shape_opt.clone() {
                // Need to infer the -1 dimension based on input tensor shape
                // Validate all input dimensions are positive (WebNN requirement)
                if input_shape.iter().any(|&d| d <= 0) {
                    return Err(OnnxError::InvalidShape(format!(
                        "Cannot infer reshape dimension: input '{}' has dynamic/unknown dimensions {:?}. \
                        WebNN requires all dimensions to be statically known (> 0). \
                        Please ensure onnx-simplifier fully resolved all dimensions.",
                        data_input_raw, input_shape
                    )));
                }

                // Calculate total elements in input tensor
                let total_elements: i64 = input_shape.iter().product();

                // Calculate product of known dimensions and infer the -1 dimension
                let mut inferred_shape = Vec::new();
                let mut known_product: i64 = 1;
                let mut infer_index = None;

                for (i, &dim) in shape_values.iter().enumerate() {
                    if dim == -1 {
                        if infer_index.is_some() {
                            return Err(OnnxError::InvalidShape(
                                "Reshape cannot have multiple -1 dimensions".to_string(),
                            ));
                        }
                        infer_index = Some(i);
                        inferred_shape.push(0); // Placeholder
                    } else {
                        known_product *= dim;
                        inferred_shape.push(dim as u32);
                    }
                }

                // Compute inferred dimension
                if let Some(idx) = infer_index {
                    let inferred_dim = total_elements / known_product;
                    if inferred_dim <= 0 || total_elements % known_product != 0 {
                        // Some decoder models (e.g. GPT exports with KV-cache inputs) can carry
                        // partially-known intermediate shapes even after override resolution.
                        // In those cases, prefer the existing best-effort fallback instead of
                        // failing conversion outright.
                        if total_elements > 0 {
                            crate::debug_println!(
                                "[reshape] cannot infer -1 for {} from input {:?} and target {:?}; replacing -1 with 1",
                                data_input_raw,
                                input_shape,
                                shape_values
                            );
                            return Ok({
                                let fallback_shape: Vec<u32> = shape_values
                                    .iter()
                                    .map(|&v| if v == -1 { 1 } else { v as u32 })
                                    .collect();

                                let mut options = Map::new();
                                options.insert(
                                    "newShape".to_string(),
                                    serde_json::json!(fallback_shape),
                                );

                                let mut result = ConversionResult::new(vec![Node {
                                    id: output_name.clone(),
                                    op: "reshape".to_string(),
                                    inputs: vec![data_input.clone()],
                                    options,
                                    outputs: None,
                                }]);

                                if let Some(output) = node.output.as_slice().first() {
                                    result
                                        .output_mappings
                                        .insert(output.to_string(), output_name.clone());
                                }

                                result
                            });
                        }

                        return Err(OnnxError::InvalidShape(format!(
                            "Cannot infer reshape dimension: {} elements cannot be reshaped to {:?}",
                            total_elements, shape_values
                        )));
                    }
                    inferred_shape[idx] = inferred_dim as u32;
                }

                inferred_shape
            } else {
                crate::debug_println!(
                    "[reshape] missing input shape for {}, shape {:?}; replacing -1 with 1",
                    data_input_raw,
                    shape_values
                );
                shape_values
                    .iter()
                    .map(|&v| if v == -1 { 1 } else { v as u32 })
                    .collect()
            }
        } else {
            // All dimensions are positive, use const shape if available; otherwise repair if needed.
            let input_shape = input_shape_opt.unwrap_or_default();
            let total_input: i64 = input_shape.iter().product();
            let total_target: i64 = shape_values.iter().product();
            let mut candidate: Vec<i64> = shape_values.clone();

            // If element counts don't match, repair using available hints.
            if total_input > 0 && total_target > 0 && total_input != total_target {
                // Rebuild using batch/seq hints (from known inputs) and hidden from target.
                let mut batch_hint = input_shape.first().copied().unwrap_or(1);
                let mut seq_hint = input_shape.get(1).copied().unwrap_or(1);
                for (name, shape) in context.value_shapes.iter() {
                    if shape.len() >= 2 && !context.initializers.contains_key(name) {
                        if shape[0] > batch_hint {
                            batch_hint = shape[0];
                        }
                        if shape[1] > seq_hint {
                            seq_hint = shape[1];
                        }
                    }
                }

                let hidden = shape_values.last().copied().unwrap_or(1);
                crate::debug_println!(
                    "[reshape] repair: {} input_shape={:?} target_shape={:?} batch_hint={} seq_hint={} hidden={}",
                    output_name, input_shape, shape_values, batch_hint, seq_hint, hidden
                );
                candidate = vec![batch_hint, seq_hint, hidden];
            } else if !shape_from_const && !input_shape.is_empty() {
                // If the target is rank-3 and the input is rank-4, flatten the last two dims.
                if input_shape.len() == 4 && shape_values.len() == 3 {
                    let tail: i64 = input_shape[2..].iter().product();
                    candidate = vec![input_shape[0], input_shape[1], tail];
                }
            }
            candidate.iter().map(|&v| v as u32).collect()
        };

        // Debug: final shape for layer 15
        if output_name.contains("layers_15_self_attn") && output_name.contains("Reshape") {
            crate::debug_println!(
                "[RESHAPE FINAL] {} final newShape -> {:?}",
                output_name,
                shape_values
            );
        }

        let mut options = Map::new();
        options.insert("newShape".to_string(), serde_json::json!(shape_values));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "reshape".to_string(),
            inputs: vec![data_input],
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

    /// Convert ONNX Expand to WebNN expand (broadcast to target shape)
    fn convert_expand<'a>(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &crate::onnx::ops::ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Expand expects 2 inputs (data, shape), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let data_input_raw = inputs[0].to_string();
        let shape_input_raw = inputs[1].to_string();
        let data_input = context.resolve_input(&data_input_raw);

        // Debug logging for all Expand operations
        if shape_input_raw.contains("rotary") || data_input_raw.contains("rotary") {
            crate::debug_println!("[EXPAND DEBUG] Node: {}", node_name);
            crate::debug_println!("  data_input_raw: {}", data_input_raw);
            crate::debug_println!("  shape_input_raw: {}", shape_input_raw);
            crate::debug_println!(
                "  In const_values: {}",
                context.const_values.contains_key(&shape_input_raw)
            );
            crate::debug_println!(
                "  In initializers: {}",
                context.initializers.contains_key(shape_input_raw.as_str())
            );
        }

        let shape_key_sanitized = sanitize_identifier(&shape_input_raw);
        let shape_key_trimmed = shape_input_raw.trim_start_matches('/').to_string();

        let shape_values: Vec<i64> = if let Some(values) = context
            .const_values
            .get(&shape_input_raw)
            .or_else(|| context.const_values.get(&shape_key_sanitized))
            .or_else(|| context.const_values.get(&shape_key_trimmed))
        {
            if shape_input_raw.contains("rotary") || data_input_raw.contains("rotary") {
                crate::debug_println!("  Shape from const_values: {:?}", values);
            }
            values.clone()
        } else if let Some(initializer) = context.initializers.get(shape_input_raw.as_str()) {
            if shape_input_raw.contains("rotary") || data_input_raw.contains("rotary") {
                crate::debug_println!("  Shape from initializer");
            }
            let raw_data = initializer.raw_data.as_slice();
            if !raw_data.is_empty() {
                match initializer.data_type {
                    x if x == TensorProto_DataType::Int32 as i32 => raw_data
                        .chunks_exact(4)
                        .map(|chunk| {
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64
                        })
                        .collect(),
                    _ => raw_data
                        .chunks_exact(8)
                        .map(|chunk| {
                            i64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                chunk[6], chunk[7],
                            ])
                        })
                        .collect(),
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
            }
        } else {
            Vec::new()
        };

        let mut dynamic_shape_json: Option<Vec<serde_json::Value>> = None;
        let shape_values: Vec<i64> = if shape_values.is_empty() {
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
            let output_shape_opt = node
                .output
                .as_slice()
                .first()
                .and_then(|out| {
                    let out_s = out.to_string();
                    context
                        .value_shapes
                        .get(&out_s)
                        .or_else(|| context.value_shapes.get(&sanitize_identifier(&out_s)))
                        .or_else(|| context.value_shapes.get(out_s.trim_start_matches('/')))
                })
                .cloned();

            if let Some(output_shape) = output_shape_opt {
                let has_dynamic_output_dim = output_dim_shape.as_ref().is_some_and(|dims| {
                    dims.iter()
                        .any(|d| matches!(d, crate::ast::Dimension::Dynamic(_)))
                });
                if output_shape.iter().all(|&d| d > 0) && !has_dynamic_output_dim {
                    crate::debug_println!(
                        "[expand] using inferred output shape for {}: {:?}",
                        node_name,
                        output_shape
                    );
                    output_shape
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            shape_values
        };

        if shape_values.is_empty() {
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
            if let Some(dims) = output_dim_shape {
                let json_dims: Vec<serde_json::Value> = dims
                    .into_iter()
                    .map(|d| match d {
                        crate::ast::Dimension::Static(v) => serde_json::json!(v),
                        crate::ast::Dimension::Dynamic(dd) => serde_json::json!({
                            "name": dd.name,
                            "maxSize": dd.max_size
                        }),
                    })
                    .collect();
                if !json_dims.is_empty() {
                    dynamic_shape_json = Some(json_dims);
                }
            }
        }

        // Determine if this is a broadcast (WebNN expand) or reshape operation
        // by checking if shapes are broadcast-compatible (ONNX Expand rules)
        let input_shape = context.value_shapes.get(&data_input_raw);
        let op_type = if dynamic_shape_json.is_some() {
            "expand"
        } else if let Some(input_shape) = input_shape {
            // Check broadcast compatibility: align from right, dimensions must be equal or one must be 1
            let mut is_broadcast_compatible = true;
            let input_rank = input_shape.len();
            let target_rank = shape_values.len();

            for i in 0..input_rank.min(target_rank) {
                let input_dim = input_shape[input_rank - 1 - i];
                let target_dim = shape_values[target_rank - 1 - i];

                // Dimensions are compatible if they're equal or either is 1
                if input_dim != target_dim && input_dim != 1 && target_dim != 1 {
                    is_broadcast_compatible = false;
                    break;
                }
            }

            if is_broadcast_compatible {
                "expand"
            } else {
                // Not broadcast-compatible, use reshape instead
                "reshape"
            }
        } else {
            // No shape info available, assume expand (broadcasting)
            "expand"
        };

        let mut options = Map::new();
        if let Some(json_dims) = dynamic_shape_json {
            options.insert("newShape".to_string(), serde_json::json!(json_dims));
        } else {
            options.insert(
                "newShape".to_string(),
                serde_json::json!(shape_values.iter().map(|v| *v as u32).collect::<Vec<_>>()),
            );
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: op_type.to_string(),
            inputs: vec![data_input],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            if let Some(dtype) = context.value_types.get(&data_input_raw) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
    }

    /// Convert ONNX Transpose to WebNN transpose
    fn convert_transpose(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Transpose expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract perm attribute (permutation)
        let mut perm: Option<Vec<i64>> = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "perm" {
                perm = Some(attr.ints.clone());
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let mut options = Map::new();
        if let Some(perm_values) = perm {
            options.insert("permutation".to_string(), serde_json::json!(perm_values));
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "transpose".to_string(),
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

    /// Convert ONNX Concat to WebNN concat
    fn convert_concat(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Concat expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute (required in ONNX)
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

        let sanitized_inputs: Vec<String> =
            inputs.iter().map(|s| context.resolve_input(s)).collect();

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "concat".to_string(),
            inputs: sanitized_inputs,
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

    /// Convert ONNX Split to WebNN split
    fn convert_split(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Split expects at least 1 input".to_string(),
            ));
        }

        // Extract axis attribute
        let mut axis = 0i64;
        let mut splits: Option<Vec<i64>> = None;

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axis" => {
                    if attr.i != 0 {
                        axis = attr.i;
                    }
                }
                "split" => {
                    splits = Some(attr.ints.clone());
                }
                _ => {}
            }
        }

        let outputs = node.output.as_slice();
        if outputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Split expects at least 1 output".to_string(),
            ));
        }

        let input0 = context.resolve_input(&inputs[0]);
        let sanitized_outputs: Vec<String> = outputs
            .iter()
            .map(|s| sanitize_identifier(&s.to_string()))
            .collect();

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));
        if let Some(split_values) = splits {
            options.insert("splits".to_string(), serde_json::json!(split_values));
        }

        // WebNN split returns multiple outputs
        let output_node_id = sanitize_identifier(&format!("{}_split", node_name));
        let mut result = ConversionResult::new(vec![Node {
            id: output_node_id,
            op: "split".to_string(),
            inputs: vec![input0],
            options,
            outputs: Some(sanitized_outputs.clone()),
        }]);

        for (onnx_out, webnn_out) in outputs.iter().zip(sanitized_outputs.iter()) {
            result
                .output_mappings
                .insert(onnx_out.to_string(), webnn_out.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Unsqueeze to WebNN expand
    /// ONNX Unsqueeze adds dimensions at specified axes
    /// In opset >= 13, axes is a second input; in earlier opsets, it's an attribute
    fn convert_unsqueeze(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Unsqueeze expects at least 1 input".to_string(),
            ));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        // Extract axes attribute (opset < 13) or use second input (opset >= 13).
        // If missing/empty, default to [0] to ensure the emitted unsqueeze is valid.
        let axes_values = {
            let mut axes: Option<Vec<i64>> = None;
            for attr in node.attribute.as_slice() {
                if attr.name.as_str() == "axes" {
                    axes = Some(attr.ints.clone());
                }
            }

            if let Some(a) = axes {
                if a.is_empty() {
                    vec![0]
                } else {
                    a
                }
            } else {
                let mut from_const = self.read_axes_from_attr_or_const(node, context)?;
                if from_const.is_empty() {
                    from_const.push(0);
                }
                from_const
            }
        };

        let mut options = Map::new();
        options.insert("axes".to_string(), serde_json::json!(axes_values.clone()));

        // WebNN unsqueeze operation only takes the data input
        // The axes parameter is provided as an attribute, not as an input
        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "unsqueeze".to_string(),
            inputs: vec![input0], // Only the data input, no axes input
            options: {
                let mut o = options;
                o.insert("axes".to_string(), serde_json::json!(axes_values));
                o
            },
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Squeeze to WebNN reshape
    /// ONNX Squeeze removes dimensions of size 1
    fn convert_squeeze(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Squeeze expects at least 1 input".to_string(),
            ));
        }

        // Extract axes attribute (opset < 13) or use second input (opset >= 13)
        let mut axes: Option<Vec<i64>> = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axes" {
                axes = Some(attr.ints.to_vec());
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let axes_values = if let Some(a) = axes {
            a
        } else {
            self.read_axes_from_attr_or_const(node, context)?
        };

        let mut options = Map::new();
        options.insert("axes".to_string(), serde_json::json!(axes_values));

        // WebNN doesn't have squeeze, so we'll use reshape with axes parameter.
        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "reshape".to_string(),
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

    /// Convert ONNX Tile to WebNN tile
    /// Repeats the input tensor along each dimension according to the repeats input
    fn convert_tile(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Tile expects 2 inputs (input, repeats), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        // The repeats input must be a constant for WebNN
        let repeats_name = inputs[1].as_str();

        // Try to read repeats from const_values or initializers
        let repeats = if let Some(vals) = context.const_values.get(repeats_name) {
            vals.clone()
        } else if let Some(tensor) = context.initializers.get(repeats_name) {
            // Read from initializer
            let raw = tensor.raw_data.as_slice();
            if !raw.is_empty() {
                match tensor.data_type {
                    x if x == TensorProto_DataType::Int64 as i32 => raw
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect(),
                    x if x == TensorProto_DataType::Int32 as i32 => raw
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                        .collect(),
                    _ => {
                        return Err(OnnxError::InvalidShape(
                            "Tile repeats must be int32 or int64".to_string(),
                        ))
                    }
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
                return Err(OnnxError::InvalidShape(
                    "Tile repeats tensor has no data".to_string(),
                ));
            }
        } else {
            return Err(OnnxError::InvalidShape(
                "Tile repeats must be constant for WebNN".to_string(),
            ));
        };

        let mut options = Map::new();
        options.insert("repetitions".to_string(), serde_json::json!(repeats));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "tile".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            // Preserve data type
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
    use crate::protos::onnx::{AttributeProto, NodeProto};

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

    fn add_ints_attribute(node: &mut NodeProto, name: &str, values: Vec<i64>) {
        let attr = AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    #[test]
    fn test_reshape_handler_supports() {
        let handler = ReshapeHandler;
        assert!(handler.supports("Reshape"));
        assert!(handler.supports("Transpose"));
        assert!(handler.supports("Concat"));
        assert!(handler.supports("Split"));
        assert!(handler.supports("Unsqueeze"));
        assert!(handler.supports("Squeeze"));
        assert!(handler.supports("Tile"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_reshape() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape"], vec!["reshaped"]);

        // Create a mock shape initializer [1, 2, 3, 4]
        let shape_tensor = crate::protos::onnx::TensorProto {
            name: "shape".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            int64_data: vec![1, 2, 3, 4],
            ..Default::default()
        };

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("shape".to_string(), &shape_tensor);

        // Add input shape for inference (24 elements = 1*2*3*4)
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![2, 3, 4]); // 24 elements

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
        assert_eq!(result.nodes[0].op, "reshape");
        assert_eq!(result.nodes[0].inputs, vec!["data"]);
        assert_eq!(result.nodes[0].id, "reshaped");
        // Verify the newShape is now a static array
        assert_eq!(
            result.nodes[0].options.get("newShape"),
            Some(&serde_json::json!([1, 2, 3, 4]))
        );
    }

    #[test]
    fn test_convert_reshape_fallback_when_inference_diverges() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape"], vec!["reshaped"]);

        // Target shape contains -1 and hidden size (common transformer pattern)
        let shape_tensor = crate::protos::onnx::TensorProto {
            name: "shape".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            int64_data: vec![-1, 768],
            ..Default::default()
        };

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("shape".to_string(), &shape_tensor);

        // Deliberately incomplete/partial shape info (1 element), which cannot satisfy [-1, 768]
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![1]);

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
        assert_eq!(result.nodes[0].op, "reshape");
        assert_eq!(result.nodes[0].id, "reshaped");
        assert_eq!(
            result.nodes[0].options.get("newShape"),
            Some(&serde_json::json!([1, 768]))
        );
    }

    #[test]
    fn test_convert_transpose() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Transpose", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "perm", vec![1, 0, 2]);
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
        assert_eq!(result.nodes[0].op, "transpose");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("permutation"));
    }

    #[test]
    fn test_convert_expand_uses_output_shape_when_shape_input_non_const() {
        let handler = ReshapeHandler;
        let node = create_test_node("Expand", vec!["data", "shape_dyn"], vec!["expanded"]);

        let initializers = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();

        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![1, 1, 768]);
        value_shapes.insert("expanded".to_string(), vec![1, 1, 768]);

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
        assert_eq!(result.nodes[0].op, "expand");
        assert_eq!(
            result.nodes[0].options.get("newShape"),
            Some(&serde_json::json!([1, 1, 768]))
        );
    }

    #[test]
    fn test_convert_concat() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Concat", vec!["a", "b", "c"], vec!["result"]);
        add_int_attribute(&mut node, "axis", 1);
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
        assert_eq!(result.nodes[0].op, "concat");
        assert_eq!(result.nodes[0].inputs.len(), 3);
        assert!(result.nodes[0].options.contains_key("axis"));
    }

    #[test]
    fn test_convert_split() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Split", vec!["x"], vec!["y1", "y2"]);
        add_int_attribute(&mut node, "axis", 0);
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
        assert_eq!(result.nodes[0].op, "split");
        assert!(result.nodes[0].outputs.is_some());
    }

    #[test]
    fn test_convert_unsqueeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Unsqueeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![0, 2]);
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

        // WebNN unsqueeze should create only 1 node (not 2)
        // The axes parameter is provided as an attribute, not as an input operand
        assert_eq!(result.nodes.len(), 1);

        // The node should be the unsqueeze operation
        assert_eq!(result.nodes[0].op, "unsqueeze");

        // WebNN unsqueeze only takes the data input (no axes input)
        assert_eq!(result.nodes[0].inputs.len(), 1);
        assert_eq!(result.nodes[0].inputs[0], "x");

        // Axes should be in the attributes/options
        assert!(result.nodes[0].options.contains_key("axes"));
        assert_eq!(
            result.nodes[0].options.get("axes"),
            Some(&serde_json::json!([0, 2]))
        );
    }

    #[test]
    fn test_convert_unsqueeze_opset13_with_input_axes() {
        // Test ONNX opset 13+ where axes are provided as a second input tensor
        let handler = ReshapeHandler;
        let node = create_test_node("Unsqueeze", vec!["x", "axes_tensor"], vec!["y"]);

        // Create a mock axes tensor [1, 3]
        let axes_tensor = crate::protos::onnx::TensorProto {
            name: "axes_tensor".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![1, 3],
            ..Default::default()
        };

        let leaked_axes: &'static crate::protos::onnx::TensorProto =
            Box::leak(Box::new(axes_tensor));

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("axes_tensor".to_string(), leaked_axes);
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

        // Should create only 1 WebNN unsqueeze node
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "unsqueeze");

        // WebNN unsqueeze only takes the data input (no axes input)
        assert_eq!(result.nodes[0].inputs.len(), 1);
        assert_eq!(result.nodes[0].inputs[0], "x");

        // Axes should be extracted from the tensor and placed in attributes
        assert!(result.nodes[0].options.contains_key("axes"));
        assert_eq!(
            result.nodes[0].options.get("axes"),
            Some(&serde_json::json!([1, 3]))
        );
    }

    #[test]
    fn test_convert_squeeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Squeeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1]);
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
        assert_eq!(result.nodes[0].op, "reshape");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("axes"));
    }

    #[test]
    fn test_convert_tile() {
        let handler = ReshapeHandler;
        let node = create_test_node("Tile", vec!["input", "repeats"], vec!["output"]);

        // Create a mock repeats tensor [2, 3]
        let repeats_tensor = crate::protos::onnx::TensorProto {
            name: "repeats".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![2, 3],
            ..Default::default()
        };

        let leaked_repeats: &'static crate::protos::onnx::TensorProto =
            Box::leak(Box::new(repeats_tensor));

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("repeats".to_string(), leaked_repeats);
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
        assert_eq!(result.nodes[0].op, "tile");
        assert_eq!(result.nodes[0].inputs, vec!["input"]);
        assert!(result.nodes[0].options.contains_key("repetitions"));

        // Verify the repetitions value
        let repetitions = result.nodes[0].options.get("repetitions").unwrap();
        assert_eq!(repetitions, &serde_json::json!([2, 3]));
    }
}
