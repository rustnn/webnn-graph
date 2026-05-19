// Pooling operators: MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool
//
// Maps ONNX pooling ops to WebNN maxPool2d / averagePool2d (NCHW layout).
//
// ONNX MaxPool / AveragePool attributes (spatial-rank-aware):
//   * kernel_shape:    required, length = spatial_rank
//   * strides:         default = [1; spatial_rank]
//   * dilations:       default = [1; spatial_rank]  (MaxPool only)
//   * pads:            default = [0; 2*spatial_rank], layout [b1, b2, ..., e1, e2, ...]
//   * auto_pad:        NOTSET | SAME_UPPER | SAME_LOWER | VALID
//   * ceil_mode:       0 (floor) | 1 (ceil)
//   * count_include_pad (AveragePool): 0 (default) | 1
//   * storage_order (MaxPool): 0 (row major, default) | 1 (column major) — not exposed in WebNN
//
// Global pooling variants take no attributes and pool over the entire spatial volume.

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::{json, Map, Value};

pub struct PoolHandler;

impl OpHandler for PoolHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "MaxPool" | "AveragePool" | "GlobalMaxPool" | "GlobalAveragePool"
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
            "MaxPool" => self.convert_pool(node, &node_name, context, PoolKind::Max),
            "AveragePool" => self.convert_pool(node, &node_name, context, PoolKind::Average),
            "GlobalMaxPool" => self.convert_global_pool(node, &node_name, context, PoolKind::Max),
            "GlobalAveragePool" => {
                self.convert_global_pool(node, &node_name, context, PoolKind::Average)
            }
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PoolKind {
    Max,
    Average,
}

impl PoolKind {
    fn webnn_op(self) -> &'static str {
        match self {
            PoolKind::Max => "maxPool2d",
            PoolKind::Average => "averagePool2d",
        }
    }
}

#[derive(Debug, Clone)]
struct PoolAttrs {
    kernel_shape: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    dilations: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    auto_pad: String,
    ceil_mode: bool,
    count_include_pad: bool,
}

fn parse_pool_attrs(node: &NodeProto) -> PoolAttrs {
    let mut attrs = PoolAttrs {
        kernel_shape: None,
        strides: None,
        dilations: None,
        pads: None,
        auto_pad: "NOTSET".to_string(),
        ceil_mode: false,
        count_include_pad: false,
    };

    for attr in node.attribute.as_slice() {
        match attr.name.as_str() {
            "auto_pad" => {
                if let Ok(s) = String::from_utf8(attr.s.clone()) {
                    if !s.is_empty() {
                        attrs.auto_pad = s;
                    }
                }
            }
            "kernel_shape" if !attr.ints.is_empty() => {
                attrs.kernel_shape = Some(attr.ints.clone());
            }
            "strides" if !attr.ints.is_empty() => {
                attrs.strides = Some(attr.ints.clone());
            }
            "dilations" if !attr.ints.is_empty() => {
                attrs.dilations = Some(attr.ints.clone());
            }
            "pads" if !attr.ints.is_empty() => {
                attrs.pads = Some(attr.ints.clone());
            }
            "ceil_mode" => {
                attrs.ceil_mode = attr.i != 0;
            }
            "count_include_pad" => {
                attrs.count_include_pad = attr.i != 0;
            }
            _ => {}
        }
    }

    attrs
}

fn lookup_shape(name: &str, context: &ConversionContext) -> Option<Vec<i64>> {
    if let Some(s) = context.value_shapes.get(name) {
        return Some(s.clone());
    }
    let sanitized = sanitize_identifier(name);
    if let Some(s) = context.value_shapes.get(&sanitized) {
        return Some(s.clone());
    }
    None
}

fn map_auto_pad(auto_pad: &str) -> &'static str {
    match auto_pad {
        "SAME_UPPER" => "same-upper",
        "SAME_LOWER" => "same-lower",
        _ => "explicit",
    }
}

fn onnx_pads_to_webnn(pads: &[i64], spatial_rank: usize) -> Vec<i64> {
    if pads.len() != 2 * spatial_rank {
        return pads.to_vec();
    }
    let mut out = Vec::with_capacity(2 * spatial_rank);
    for i in 0..spatial_rank {
        out.push(pads[i]);
        out.push(pads[i + spatial_rank]);
    }
    out
}

impl PoolHandler {
    fn convert_pool(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        kind: PoolKind,
    ) -> Result<ConversionResult, OnnxError> {
        let op_label = match kind {
            PoolKind::Max => "MaxPool",
            PoolKind::Average => "AveragePool",
        };

        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                op_label,
                inputs.len()
            )));
        }
        // Reject the optional second MaxPool output (indices) — WebNN has no equivalent.
        if matches!(kind, PoolKind::Max) && node.output.as_slice().len() > 1 {
            return Err(OnnxError::UnsupportedOp {
                op: "MaxPool(with indices output)".to_string(),
                node: node_name.to_string(),
            });
        }

        let input_raw = inputs[0].to_string();
        let input_id = context.resolve_input(&input_raw);
        let input_shape = lookup_shape(&input_raw, context);

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let attrs = parse_pool_attrs(node);

        let kernel = attrs
            .kernel_shape
            .clone()
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "kernel_shape".to_string(),
                op: op_label.to_string(),
            })?;
        let spatial_rank = kernel.len();

        match spatial_rank {
            2 => self.emit_pool_2d(
                node,
                node_name,
                &output_name,
                &input_id,
                &attrs,
                &kernel,
                kind,
            ),
            1 => self.emit_pool_1d_via_2d(
                node,
                node_name,
                &output_name,
                &input_id,
                &attrs,
                &kernel,
                kind,
                input_shape.as_deref(),
            ),
            _ => Err(OnnxError::UnsupportedOp {
                op: format!("{}{}D", op_label, spatial_rank),
                node: node_name.to_string(),
            }),
        }
    }

    fn build_pool_2d_options(
        &self,
        attrs: &PoolAttrs,
        kernel: &[i64],
        kind: PoolKind,
    ) -> Result<Map<String, Value>, OnnxError> {
        let mut options = Map::new();
        let strides = attrs.strides.clone().unwrap_or_else(|| vec![1, 1]);
        let dilations = attrs.dilations.clone().unwrap_or_else(|| vec![1, 1]);
        let pads = attrs.pads.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);
        if strides.len() != 2 || dilations.len() != 2 || kernel.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "pool2d: expected length-2 kernel/strides/dilations, got kernel={:?} strides={:?} dilations={:?}",
                kernel, strides, dilations
            )));
        }

        options.insert("windowDimensions".to_string(), json!(kernel));
        options.insert("strides".to_string(), json!(strides));
        // AveragePool in ONNX has no dilations; only emit dilations when non-default
        // to keep generated calls minimal for the average case.
        if matches!(kind, PoolKind::Max) || dilations.iter().any(|&d| d != 1) {
            options.insert("dilations".to_string(), json!(dilations));
        }

        let mapped_auto_pad = map_auto_pad(&attrs.auto_pad);
        if mapped_auto_pad == "explicit" {
            let effective_pads = if attrs.auto_pad == "VALID" {
                vec![0, 0, 0, 0]
            } else {
                onnx_pads_to_webnn(&pads, 2)
            };
            if effective_pads.len() != 4 {
                return Err(OnnxError::InvalidShape(format!(
                    "pool2d: padding must yield 4 values for 2D, got {:?}",
                    effective_pads
                )));
            }
            options.insert("padding".to_string(), json!(effective_pads));
        } else {
            options.insert("autoPad".to_string(), json!(mapped_auto_pad));
        }

        if attrs.ceil_mode {
            options.insert("roundingType".to_string(), json!("ceil"));
        }

        if matches!(kind, PoolKind::Average) && attrs.count_include_pad {
            // The WebNN spec does not currently expose a count_include_pad knob; surface
            // a clear error rather than silently producing incorrect results.
            return Err(OnnxError::UnsupportedOp {
                op: "AveragePool(count_include_pad=1)".to_string(),
                node: "".to_string(),
            });
        }

        Ok(options)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_pool_2d(
        &self,
        node: &NodeProto,
        _node_name: &str,
        output_name: &str,
        input_id: &str,
        attrs: &PoolAttrs,
        kernel: &[i64],
        kind: PoolKind,
    ) -> Result<ConversionResult, OnnxError> {
        let options = self.build_pool_2d_options(attrs, kernel, kind)?;
        let mut result = ConversionResult::new(vec![Node {
            id: output_name.to_string(),
            op: kind.webnn_op().to_string(),
            inputs: vec![input_id.to_string()],
            options,
            outputs: None,
        }]);

        if let Some(onnx_out) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(onnx_out.to_string(), output_name.to_string());
        }
        Ok(result)
    }

    /// Emulate a 1D pool by reshaping the input to 4-D (trailing W=1), pooling, then reshaping
    /// the output back to 3-D.
    #[allow(clippy::too_many_arguments)]
    fn emit_pool_1d_via_2d(
        &self,
        node: &NodeProto,
        node_name: &str,
        output_name: &str,
        input_id: &str,
        attrs: &PoolAttrs,
        kernel: &[i64],
        kind: PoolKind,
        input_shape: Option<&[i64]>,
    ) -> Result<ConversionResult, OnnxError> {
        let input_shape = input_shape.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "1D pool emulation requires known shape for input of node {}",
                node_name
            ))
        })?;
        if input_shape.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "1D pool emulation expects rank-3 input, got {:?}",
                input_shape
            )));
        }

        let mut attrs_2d = attrs.clone();
        attrs_2d.strides = Some(extend_with(attrs.strides.as_deref(), 1, 2));
        attrs_2d.dilations = Some(extend_with(attrs.dilations.as_deref(), 1, 2));
        attrs_2d.pads = Some(extend_pads_to_2d(attrs.pads.as_deref()));
        let kernel_2d: Vec<i64> = {
            let mut k = kernel.to_vec();
            if k.len() == 1 {
                k.push(1);
            }
            k
        };
        attrs_2d.kernel_shape = Some(kernel_2d.clone());

        let options = self.build_pool_2d_options(&attrs_2d, &kernel_2d, kind)?;

        let reshape_in_id = sanitize_identifier(&format!("{}_x4d", node_name));
        let pool_id = sanitize_identifier(&format!("{}_pool2d", node_name));

        let in_4d: Vec<i64> = vec![input_shape[0], input_shape[1], input_shape[2], 1];
        let mut reshape_in_opts = Map::new();
        reshape_in_opts.insert("newShape".to_string(), json!(in_4d));

        let nodes = vec![
            Node {
                id: reshape_in_id.clone(),
                op: "reshape".to_string(),
                inputs: vec![input_id.to_string()],
                options: reshape_in_opts,
                outputs: None,
            },
            Node {
                id: pool_id.clone(),
                op: kind.webnn_op().to_string(),
                inputs: vec![reshape_in_id],
                options,
                outputs: None,
            },
            Node {
                id: output_name.to_string(),
                op: "reshape".to_string(),
                inputs: vec![pool_id],
                options: {
                    let mut m = Map::new();
                    m.insert(
                        "newShape".to_string(),
                        json!([input_shape[0], input_shape[1], -1i64]),
                    );
                    m
                },
                outputs: None,
            },
        ];

        let mut result = ConversionResult::new(nodes);
        if let Some(onnx_out) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(onnx_out.to_string(), output_name.to_string());
        }
        Ok(result)
    }

    fn convert_global_pool(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        kind: PoolKind,
    ) -> Result<ConversionResult, OnnxError> {
        let op_label = match kind {
            PoolKind::Max => "GlobalMaxPool",
            PoolKind::Average => "GlobalAveragePool",
        };
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                op_label,
                inputs.len()
            )));
        }

        let input_raw = inputs[0].to_string();
        let input_id = context.resolve_input(&input_raw);
        let input_shape = lookup_shape(&input_raw, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "{}: input '{}' shape is unknown — required to determine spatial window size",
                op_label, input_raw
            ))
        })?;
        if input_shape.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "{}: input must be at least rank-3 (N, C, spatial...), got {:?}",
                op_label, input_shape
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let spatial = &input_shape[2..];
        match spatial.len() {
            2 => {
                let mut options = Map::new();
                options.insert("windowDimensions".to_string(), json!(spatial.to_vec()));
                // Strides default to 1 — fine since the window covers the whole spatial.
                let mut result = ConversionResult::new(vec![Node {
                    id: output_name.clone(),
                    op: kind.webnn_op().to_string(),
                    inputs: vec![input_id],
                    options,
                    outputs: None,
                }]);
                if let Some(onnx_out) = node.output.as_slice().first() {
                    result
                        .output_mappings
                        .insert(onnx_out.to_string(), output_name);
                }
                Ok(result)
            }
            1 => {
                // Reshape to 4-D (trailing 1), pool with windowDimensions=[L, 1], reshape back.
                let reshape_in_id = sanitize_identifier(&format!("{}_x4d", node_name));
                let pool_id = sanitize_identifier(&format!("{}_pool2d", node_name));
                let in_4d: Vec<i64> = vec![input_shape[0], input_shape[1], spatial[0], 1];
                let mut reshape_in_opts = Map::new();
                reshape_in_opts.insert("newShape".to_string(), json!(in_4d));

                let mut pool_opts = Map::new();
                pool_opts.insert("windowDimensions".to_string(), json!([spatial[0], 1]));

                let nodes = vec![
                    Node {
                        id: reshape_in_id.clone(),
                        op: "reshape".to_string(),
                        inputs: vec![input_id],
                        options: reshape_in_opts,
                        outputs: None,
                    },
                    Node {
                        id: pool_id.clone(),
                        op: kind.webnn_op().to_string(),
                        inputs: vec![reshape_in_id],
                        options: pool_opts,
                        outputs: None,
                    },
                    Node {
                        id: output_name.clone(),
                        op: "reshape".to_string(),
                        inputs: vec![pool_id],
                        options: {
                            let mut m = Map::new();
                            m.insert(
                                "newShape".to_string(),
                                json!([input_shape[0], input_shape[1], 1i64]),
                            );
                            m
                        },
                        outputs: None,
                    },
                ];

                let mut result = ConversionResult::new(nodes);
                if let Some(onnx_out) = node.output.as_slice().first() {
                    result
                        .output_mappings
                        .insert(onnx_out.to_string(), output_name);
                }
                Ok(result)
            }
            _ => Err(OnnxError::UnsupportedOp {
                op: format!("{}{}D", op_label, spatial.len()),
                node: node_name.to_string(),
            }),
        }
    }
}

fn extend_with(src: Option<&[i64]>, fill: i64, target_len: usize) -> Vec<i64> {
    let mut out = src.map(|v| v.to_vec()).unwrap_or_default();
    while out.len() < target_len {
        out.push(fill);
    }
    out
}

fn extend_pads_to_2d(pads: Option<&[i64]>) -> Vec<i64> {
    match pads {
        Some(p) if p.len() == 2 => vec![p[0], 0, p[1], 0],
        Some(p) if p.len() == 4 => p.to_vec(),
        _ => vec![0, 0, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto};
    use std::collections::HashMap;

    fn make_node(
        op_type: &str,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        attrs: Vec<AttributeProto>,
    ) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            attribute: attrs,
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        }
    }

    fn ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        }
    }

    fn string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    fn make_context<'a>(
        initializers: &'a HashMap<String, &'a crate::protos::onnx::TensorProto>,
        value_shapes: &'a HashMap<String, Vec<i64>>,
        const_values: &'a HashMap<String, Vec<i64>>,
        value_ids: &'a HashMap<String, String>,
        value_types: &'a HashMap<String, crate::ast::DataType>,
    ) -> ConversionContext<'a> {
        ConversionContext {
            initializers,
            value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values,
            value_ids,
            value_types,
        }
    }

    #[test]
    fn supports_pool_ops() {
        let h = PoolHandler;
        assert!(h.supports("MaxPool"));
        assert!(h.supports("AveragePool"));
        assert!(h.supports("GlobalMaxPool"));
        assert!(h.supports("GlobalAveragePool"));
        assert!(!h.supports("Conv"));
    }

    #[test]
    fn maxpool2d_basic() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("strides", vec![2, 2]),
                ints_attr("pads", vec![1, 1, 1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 64, 112, 112]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        assert_eq!(result.nodes.len(), 1);
        let n = &result.nodes[0];
        assert_eq!(n.op, "maxPool2d");
        assert_eq!(n.inputs, vec!["x"]);
        assert_eq!(n.options.get("windowDimensions"), Some(&json!([3, 3])));
        assert_eq!(n.options.get("strides"), Some(&json!([2, 2])));
        assert_eq!(n.options.get("padding"), Some(&json!([1, 1, 1, 1])));
        assert_eq!(n.options.get("dilations"), Some(&json!([1, 1])));
    }

    #[test]
    fn maxpool2d_pads_layout_reordered() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("pads", vec![1, 2, 3, 4]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 64, 32, 32]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        // ONNX [top, left, bottom, right] -> WebNN [top, bottom, left, right] = [1, 3, 2, 4]
        assert_eq!(
            result.nodes[0].options.get("padding"),
            Some(&json!([1, 3, 2, 4]))
        );
    }

    #[test]
    fn maxpool2d_with_ceil_mode() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![2, 2]),
                int_attr("ceil_mode", 1),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 7, 7]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        assert_eq!(
            result.nodes[0].options.get("roundingType"),
            Some(&json!("ceil"))
        );
    }

    #[test]
    fn maxpool2d_auto_pad_same_upper() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                string_attr("auto_pad", "SAME_UPPER"),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 32, 32]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        let n = &result.nodes[0];
        assert_eq!(n.options.get("autoPad"), Some(&json!("same-upper")));
        assert!(n.options.get("padding").is_none());
    }

    #[test]
    fn averagepool2d_basic() {
        let h = PoolHandler;
        let node = make_node(
            "AveragePool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![2, 2]),
                ints_attr("strides", vec![2, 2]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 14, 14]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        let n = &result.nodes[0];
        assert_eq!(n.op, "averagePool2d");
        assert_eq!(n.options.get("windowDimensions"), Some(&json!([2, 2])));
        // AveragePool: dilations not emitted unless non-default
        assert!(n.options.get("dilations").is_none());
    }

    #[test]
    fn averagepool_count_include_pad_rejected() {
        let h = PoolHandler;
        let node = make_node(
            "AveragePool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![2, 2]),
                int_attr("count_include_pad", 1),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 14, 14]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let err = h.convert(&node, &ctx).unwrap_err();
        match err {
            OnnxError::UnsupportedOp { op, .. } => {
                assert!(op.contains("count_include_pad"));
            }
            other => panic!("expected UnsupportedOp, got {:?}", other),
        }
    }

    #[test]
    fn global_average_pool_2d() {
        let h = PoolHandler;
        let node = make_node("GlobalAveragePool", vec!["x"], vec!["y"], vec![]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2048, 7, 7]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        assert_eq!(result.nodes.len(), 1);
        let n = &result.nodes[0];
        assert_eq!(n.op, "averagePool2d");
        assert_eq!(n.options.get("windowDimensions"), Some(&json!([7, 7])));
    }

    #[test]
    fn global_max_pool_2d() {
        let h = PoolHandler;
        let node = make_node("GlobalMaxPool", vec!["x"], vec!["y"], vec![]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 14, 14]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        let n = &result.nodes[0];
        assert_eq!(n.op, "maxPool2d");
        assert_eq!(n.options.get("windowDimensions"), Some(&json!([14, 14])));
    }

    #[test]
    fn maxpool_missing_kernel_shape_errors() {
        let h = PoolHandler;
        let node = make_node("MaxPool", vec!["x"], vec!["y"], vec![]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 14, 14]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let err = h.convert(&node, &ctx).unwrap_err();
        match err {
            OnnxError::MissingAttribute { attr, .. } => {
                assert_eq!(attr, "kernel_shape");
            }
            other => panic!("expected MissingAttribute, got {:?}", other),
        }
    }

    #[test]
    fn maxpool_rejects_indices_output() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y", "indices"],
            vec![ints_attr("kernel_shape", vec![2, 2])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 14, 14]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let err = h.convert(&node, &ctx).unwrap_err();
        match err {
            OnnxError::UnsupportedOp { op, .. } => {
                assert!(op.contains("indices"));
            }
            other => panic!("expected UnsupportedOp, got {:?}", other),
        }
    }

    #[test]
    fn maxpool1d_emulated_via_2d() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3]),
                ints_attr("strides", vec![2]),
                ints_attr("pads", vec![1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 64]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        assert_eq!(result.nodes.len(), 3); // reshape -> pool -> reshape
        assert_eq!(result.nodes[0].op, "reshape");
        assert_eq!(result.nodes[1].op, "maxPool2d");
        assert_eq!(result.nodes[2].op, "reshape");
        let pool = &result.nodes[1];
        assert_eq!(pool.options.get("windowDimensions"), Some(&json!([3, 1])));
        assert_eq!(pool.options.get("strides"), Some(&json!([2, 1])));
        // ONNX 1D pads [1, 1] -> 2D [1, 0, 1, 0] -> WebNN [1, 1, 0, 0]
        assert_eq!(pool.options.get("padding"), Some(&json!([1, 1, 0, 0])));
    }
}
