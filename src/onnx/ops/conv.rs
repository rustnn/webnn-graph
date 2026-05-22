// Convolution operators: Conv, ConvTranspose
//
// Maps ONNX Conv/ConvTranspose to WebNN conv2d/convTranspose2d (NCHW layout).
//
// ONNX layout assumptions (the spec defaults):
//   * input X : (N, C_in, ...spatial)
//   * filter W: Conv          -> (M, C_in / group, kH, kW, ...)
//                ConvTranspose -> (C_in, M / group, kH, kW, ...)
//   * bias B  : (M,)  (optional)
//
// WebNN defaults match ONNX:
//   * inputLayout  = "nchw"
//   * filterLayout = "oihw" for conv2d
//   * filterLayout = "iohw" for convTranspose2d
//
// Spatial dimensionality:
//   * 2D (4-D input)              -> emitted as conv2d / convTranspose2d directly
//   * 1D (3-D input)              -> emulated as reshape -> conv2d -> reshape
//   * Anything else (1D w/o shape info, 3D, etc.) -> UnsupportedOp error.

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::{json, Map, Value};

pub struct ConvHandler;

impl OpHandler for ConvHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Conv" | "ConvTranspose")
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
            "Conv" => self.convert_conv(node, &node_name, context, false),
            "ConvTranspose" => self.convert_conv(node, &node_name, context, true),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct ConvAttrs {
    auto_pad: String,
    dilations: Option<Vec<i64>>,
    group: i64,
    kernel_shape: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    output_padding: Option<Vec<i64>>,
    output_shape: Option<Vec<i64>>,
}

fn parse_conv_attrs(node: &NodeProto) -> ConvAttrs {
    let mut attrs = ConvAttrs {
        auto_pad: "NOTSET".to_string(),
        dilations: None,
        group: 1,
        kernel_shape: None,
        pads: None,
        strides: None,
        output_padding: None,
        output_shape: None,
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
            "dilations" if !attr.ints.is_empty() => {
                attrs.dilations = Some(attr.ints.clone());
            }
            "group" if attr.i > 0 => {
                attrs.group = attr.i;
            }
            "kernel_shape" if !attr.ints.is_empty() => {
                attrs.kernel_shape = Some(attr.ints.clone());
            }
            "pads" if !attr.ints.is_empty() => {
                attrs.pads = Some(attr.ints.clone());
            }
            "strides" if !attr.ints.is_empty() => {
                attrs.strides = Some(attr.ints.clone());
            }
            "output_padding" if !attr.ints.is_empty() => {
                attrs.output_padding = Some(attr.ints.clone());
            }
            "output_shape" if !attr.ints.is_empty() => {
                attrs.output_shape = Some(attr.ints.clone());
            }
            _ => {}
        }
    }

    attrs
}

/// Look up a tensor's shape from value_shapes / initializers.
fn lookup_shape(name: &str, context: &ConversionContext) -> Option<Vec<i64>> {
    if let Some(s) = context.value_shapes.get(name) {
        return Some(s.clone());
    }
    let sanitized = sanitize_identifier(name);
    if let Some(s) = context.value_shapes.get(&sanitized) {
        return Some(s.clone());
    }
    if let Some(init) = context.initializers.get(name) {
        return Some(init.dims.as_slice().to_vec());
    }
    None
}

/// Map ONNX auto_pad string to WebNN autoPad option string.
fn map_auto_pad(auto_pad: &str) -> &'static str {
    match auto_pad {
        "SAME_UPPER" => "same-upper",
        "SAME_LOWER" => "same-lower",
        // VALID and NOTSET both map to explicit padding; for VALID we'll also zero pads out.
        _ => "explicit",
    }
}

/// Convert ONNX pads layout [b1, b2, ..., bk, e1, e2, ..., ek]
/// to WebNN padding layout [b1, e1, b2, e2, ..., bk, ek].
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

impl ConvHandler {
    fn convert_conv(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        transpose: bool,
    ) -> Result<ConversionResult, OnnxError> {
        let op_label = if transpose { "ConvTranspose" } else { "Conv" };
        let inputs = node.input.as_slice();
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 2 or 3 inputs (X, W[, B]), got {}",
                op_label,
                inputs.len()
            )));
        }

        let input_raw = inputs[0].to_string();
        let filter_raw = inputs[1].to_string();
        let bias_raw = inputs.get(2).map(|s| s.to_string());

        let input_id = context.resolve_input(&input_raw);
        let filter_id = context.resolve_input(&filter_raw);
        let bias_id = bias_raw.as_ref().map(|n| context.resolve_input(n));

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let attrs = parse_conv_attrs(node);

        // Determine spatial rank.  Prefer the explicit kernel_shape attribute, then the
        // filter's declared shape, then the input's spatial rank.
        let filter_shape = lookup_shape(&filter_raw, context);
        let input_shape = lookup_shape(&input_raw, context);
        let spatial_rank = if let Some(ks) = attrs.kernel_shape.as_ref() {
            ks.len()
        } else if let Some(fs) = filter_shape.as_ref() {
            if fs.len() >= 2 {
                fs.len() - 2
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "{}: filter '{}' has rank {} (need >= 2)",
                    op_label,
                    filter_raw,
                    fs.len()
                )));
            }
        } else if let Some(is) = input_shape.as_ref() {
            if is.len() >= 2 {
                is.len() - 2
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "{}: cannot determine spatial rank from input '{}' of rank {}",
                    op_label,
                    input_raw,
                    is.len()
                )));
            }
        } else {
            return Err(OnnxError::InvalidShape(format!(
                "{}: cannot determine spatial rank — provide kernel_shape attribute or filter/input shape info",
                op_label,
            )));
        };

        match spatial_rank {
            2 => self.emit_conv_2d(
                node_name,
                &output_name,
                &input_id,
                &filter_id,
                bias_id.as_deref(),
                &attrs,
                transpose,
                node,
            ),
            1 => self.emit_conv_1d_via_2d(
                node_name,
                &output_name,
                &input_id,
                &filter_id,
                bias_id.as_deref(),
                &attrs,
                transpose,
                node,
                input_shape.as_deref(),
                filter_shape.as_deref(),
            ),
            _ => Err(OnnxError::UnsupportedOp {
                op: format!("{}{}D", op_label, spatial_rank),
                node: node_name.to_string(),
            }),
        }
    }

    /// Build the options map shared by conv2d / convTranspose2d.
    fn build_conv2d_options(
        &self,
        attrs: &ConvAttrs,
        transpose: bool,
    ) -> Result<Map<String, Value>, OnnxError> {
        let mut options = Map::new();

        let strides = attrs.strides.clone().unwrap_or_else(|| vec![1, 1]);
        let dilations = attrs.dilations.clone().unwrap_or_else(|| vec![1, 1]);
        let pads = attrs.pads.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);

        if strides.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "conv2d: strides must have length 2, got {:?}",
                strides
            )));
        }
        if dilations.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "conv2d: dilations must have length 2, got {:?}",
                dilations
            )));
        }

        options.insert("strides".to_string(), json!(strides));
        options.insert("dilations".to_string(), json!(dilations));

        let mapped_auto_pad = map_auto_pad(&attrs.auto_pad);
        // Always emit padding for explicit case (or VALID, which uses zero pads).
        if mapped_auto_pad == "explicit" {
            // VALID means no padding regardless of `pads` attribute.
            let effective_pads = if attrs.auto_pad == "VALID" {
                vec![0, 0, 0, 0]
            } else {
                onnx_pads_to_webnn(&pads, 2)
            };
            if effective_pads.len() != 4 {
                return Err(OnnxError::InvalidShape(format!(
                    "conv2d: pads must yield 4 values for 2D, got {:?}",
                    effective_pads
                )));
            }
            options.insert("padding".to_string(), json!(effective_pads));
        } else {
            options.insert("autoPad".to_string(), json!(mapped_auto_pad));
        }

        if attrs.group != 1 {
            options.insert("groups".to_string(), json!(attrs.group));
        }

        if transpose {
            if let Some(op) = attrs.output_padding.as_ref() {
                if op.len() == 2 {
                    options.insert("outputPadding".to_string(), json!(op));
                }
            }
            if let Some(os) = attrs.output_shape.as_ref() {
                // ONNX output_shape is the full N×C×H×W; WebNN outputSizes is the spatial part
                // [H, W].  Accept either form for robustness.
                let sizes: Vec<i64> = if os.len() == 2 {
                    os.clone()
                } else if os.len() >= 2 {
                    os[os.len() - 2..].to_vec()
                } else {
                    Vec::new()
                };
                if sizes.len() == 2 {
                    options.insert("outputSizes".to_string(), json!(sizes));
                }
            }
        }

        Ok(options)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_conv_2d(
        &self,
        _node_name: &str,
        output_name: &str,
        input_id: &str,
        filter_id: &str,
        bias_id: Option<&str>,
        attrs: &ConvAttrs,
        transpose: bool,
        node: &NodeProto,
    ) -> Result<ConversionResult, OnnxError> {
        let webnn_op = if transpose {
            "convTranspose2d"
        } else {
            "conv2d"
        };
        let options = self.build_conv2d_options(attrs, transpose)?;

        let mut inputs_vec = vec![input_id.to_string(), filter_id.to_string()];
        if let Some(b) = bias_id {
            inputs_vec.push(b.to_string());
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.to_string(),
            op: webnn_op.to_string(),
            inputs: inputs_vec,
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

    /// Emulate a 1D convolution by reshaping the input/filter to 4-D (W=1),
    /// running conv2d, then reshaping the output back to 3-D.
    #[allow(clippy::too_many_arguments)]
    fn emit_conv_1d_via_2d(
        &self,
        node_name: &str,
        output_name: &str,
        input_id: &str,
        filter_id: &str,
        bias_id: Option<&str>,
        attrs: &ConvAttrs,
        transpose: bool,
        node: &NodeProto,
        input_shape: Option<&[i64]>,
        filter_shape: Option<&[i64]>,
    ) -> Result<ConversionResult, OnnxError> {
        let input_shape = input_shape.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "1D Conv emulation requires known shape for input of node {}",
                node_name
            ))
        })?;
        let filter_shape = filter_shape.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "1D Conv emulation requires known shape for filter of node {}",
                node_name
            ))
        })?;
        if input_shape.len() != 3 || filter_shape.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "1D Conv emulation expects rank-3 input/filter, got input {:?} filter {:?}",
                input_shape, filter_shape
            )));
        }

        // Extend 1D attrs to 2D by appending a trailing dim of "1" (a no-op extra dim).
        let mut attrs_2d = attrs.clone();
        attrs_2d.strides = Some(extend_with_one(attrs.strides.as_deref(), 1, 2));
        attrs_2d.dilations = Some(extend_with_one(attrs.dilations.as_deref(), 1, 2));
        attrs_2d.pads = Some(extend_pads_to_2d(attrs.pads.as_deref()));
        attrs_2d.kernel_shape = attrs
            .kernel_shape
            .as_ref()
            .map(|ks| extend_with_one(Some(ks.as_slice()), 1, 2));
        if transpose {
            attrs_2d.output_padding = Some(extend_with_one(attrs.output_padding.as_deref(), 0, 2));
            // output_shape only makes sense for the full spatial range; we drop it for the
            // 1D-via-2D rewrite to avoid mis-encoding [H] vs [H, W]=1.
            attrs_2d.output_shape = None;
        }

        let options = self.build_conv2d_options(&attrs_2d, transpose)?;

        let reshape_in_id = sanitize_identifier(&format!("{}_x4d", node_name));
        let reshape_w_id = sanitize_identifier(&format!("{}_w4d", node_name));
        let conv_id = sanitize_identifier(&format!("{}_conv2d", node_name));

        let in_4d_shape: Vec<i64> = vec![input_shape[0], input_shape[1], input_shape[2], 1];
        let w_4d_shape: Vec<i64> = vec![filter_shape[0], filter_shape[1], filter_shape[2], 1];

        let mut reshape_in_opts = Map::new();
        reshape_in_opts.insert("newShape".to_string(), json!(in_4d_shape));
        let mut reshape_w_opts = Map::new();
        reshape_w_opts.insert("newShape".to_string(), json!(w_4d_shape));

        let mut nodes = vec![
            Node {
                id: reshape_in_id.clone(),
                op: "reshape".to_string(),
                inputs: vec![input_id.to_string()],
                options: reshape_in_opts,
                outputs: None,
            },
            Node {
                id: reshape_w_id.clone(),
                op: "reshape".to_string(),
                inputs: vec![filter_id.to_string()],
                options: reshape_w_opts,
                outputs: None,
            },
        ];

        let webnn_op = if transpose {
            "convTranspose2d"
        } else {
            "conv2d"
        };
        let mut conv_inputs = vec![reshape_in_id.clone(), reshape_w_id.clone()];
        if let Some(b) = bias_id {
            conv_inputs.push(b.to_string());
        }
        nodes.push(Node {
            id: conv_id.clone(),
            op: webnn_op.to_string(),
            inputs: conv_inputs,
            options,
            outputs: None,
        });

        // Reshape back to 3D.  We can compute the spatial output dim with the standard formula
        // but we conservatively rely on shape inference downstream by using -1.
        let out_shape: Vec<Value> =
            vec![json!(input_shape[0]), json!(filter_shape[0]), json!(-1i64)];
        let mut final_reshape_opts = Map::new();
        final_reshape_opts.insert("newShape".to_string(), json!(out_shape));
        nodes.push(Node {
            id: output_name.to_string(),
            op: "reshape".to_string(),
            inputs: vec![conv_id.clone()],
            options: final_reshape_opts,
            outputs: None,
        });

        let mut result = ConversionResult::new(nodes);
        if let Some(onnx_out) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(onnx_out.to_string(), output_name.to_string());
        }
        Ok(result)
    }
}

fn extend_with_one(src: Option<&[i64]>, fill: i64, target_len: usize) -> Vec<i64> {
    let mut out = src.map(|v| v.to_vec()).unwrap_or_default();
    while out.len() < target_len {
        out.push(fill);
    }
    out
}

/// Extend an ONNX-style pads list (1D = [begin, end]) to 2D ([begin_h, begin_w, end_h, end_w])
/// by appending zero padding on the trailing dimension.
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
    fn supports_conv_ops() {
        let h = ConvHandler;
        assert!(h.supports("Conv"));
        assert!(h.supports("ConvTranspose"));
        assert!(!h.supports("MatMul"));
        assert!(!h.supports("Pool"));
    }

    #[test]
    fn conv2d_basic_defaults() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![ints_attr("kernel_shape", vec![3, 3])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 224, 224]);
        value_shapes.insert("w".to_string(), vec![64, 3, 3, 3]);
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
        assert_eq!(n.op, "conv2d");
        assert_eq!(n.id, "y");
        assert_eq!(n.inputs, vec!["x", "w"]);
        assert_eq!(n.options.get("strides"), Some(&json!([1, 1])));
        assert_eq!(n.options.get("dilations"), Some(&json!([1, 1])));
        assert_eq!(n.options.get("padding"), Some(&json!([0, 0, 0, 0])));
        // group=1 should not be emitted.
        assert!(n.options.get("groups").is_none());
        // No autoPad option for explicit (default).
        assert!(n.options.get("autoPad").is_none());
    }

    #[test]
    fn conv2d_with_strides_pads_dilations_groups() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w", "b"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("strides", vec![2, 2]),
                ints_attr("pads", vec![1, 1, 1, 1]),
                ints_attr("dilations", vec![1, 1]),
                int_attr("group", 4),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 4, 112, 112]);
        value_shapes.insert("w".to_string(), vec![8, 1, 3, 3]);
        value_shapes.insert("b".to_string(), vec![8]);
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
        assert_eq!(n.op, "conv2d");
        assert_eq!(n.inputs, vec!["x", "w", "b"]);
        assert_eq!(n.options.get("strides"), Some(&json!([2, 2])));
        assert_eq!(n.options.get("dilations"), Some(&json!([1, 1])));
        // ONNX pads [b1, b2, e1, e2] -> WebNN padding [b1, e1, b2, e2]
        assert_eq!(n.options.get("padding"), Some(&json!([1, 1, 1, 1])));
        assert_eq!(n.options.get("groups"), Some(&json!(4)));
    }

    #[test]
    fn conv2d_pads_layout_reordered() {
        // Asymmetric pads to verify the ONNX->WebNN reordering.
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                // ONNX layout: [top, left, bottom, right]
                ints_attr("pads", vec![1, 2, 3, 4]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3]);
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
        // WebNN layout: [top, bottom, left, right] = [1, 3, 2, 4]
        assert_eq!(n.options.get("padding"), Some(&json!([1, 3, 2, 4])));
    }

    #[test]
    fn conv2d_auto_pad_same_upper() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                string_attr("auto_pad", "SAME_UPPER"),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3]);
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
    fn conv2d_auto_pad_valid_zeroes_pads() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                string_attr("auto_pad", "VALID"),
                // even if pads attribute is set, VALID forces zero padding
                ints_attr("pads", vec![1, 1, 1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3]);
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
        assert_eq!(n.options.get("padding"), Some(&json!([0, 0, 0, 0])));
        assert!(n.options.get("autoPad").is_none());
    }

    #[test]
    fn conv_transpose_basic() {
        let h = ConvHandler;
        let node = make_node(
            "ConvTranspose",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("strides", vec![2, 2]),
                ints_attr("output_padding", vec![1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 32, 32]);
        value_shapes.insert("w".to_string(), vec![16, 8, 3, 3]);
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
        assert_eq!(n.op, "convTranspose2d");
        assert_eq!(n.options.get("strides"), Some(&json!([2, 2])));
        assert_eq!(n.options.get("outputPadding"), Some(&json!([1, 1])));
    }

    #[test]
    fn conv_transpose_output_shape_full_form() {
        let h = ConvHandler;
        // ONNX output_shape is typically N×C×H×W; we should pick H×W for outputSizes.
        let node = make_node(
            "ConvTranspose",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("output_shape", vec![1, 8, 64, 64]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 32, 32]);
        value_shapes.insert("w".to_string(), vec![16, 8, 3, 3]);
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
        assert_eq!(n.options.get("outputSizes"), Some(&json!([64, 64])));
    }

    #[test]
    fn conv1d_emulated_via_2d() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
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
        value_shapes.insert("w".to_string(), vec![8, 16, 3]);
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
        // reshape input -> reshape filter -> conv2d -> reshape output
        assert_eq!(result.nodes.len(), 4);
        assert_eq!(result.nodes[0].op, "reshape");
        assert_eq!(result.nodes[1].op, "reshape");
        assert_eq!(result.nodes[2].op, "conv2d");
        assert_eq!(result.nodes[3].op, "reshape");
        // Strides/dilations/pads extended to 2D
        let conv = &result.nodes[2];
        assert_eq!(conv.options.get("strides"), Some(&json!([2, 1])));
        assert_eq!(conv.options.get("dilations"), Some(&json!([1, 1])));
        // ONNX 1D pads [1, 1] -> 2D [1, 0, 1, 0] -> WebNN [1, 1, 0, 0]
        assert_eq!(conv.options.get("padding"), Some(&json!([1, 1, 0, 0])));
    }

    #[test]
    fn conv_3d_unsupported() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![ints_attr("kernel_shape", vec![3, 3, 3])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 16, 16, 16]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3, 3]);
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
                assert!(op.contains("3D"), "expected 3D in op label, got {}", op);
            }
            other => panic!("expected UnsupportedOp, got {:?}", other),
        }
    }

    #[test]
    fn conv_resolves_input_aliases() {
        // Ensure input IDs go through ConversionContext::resolve_input.
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["onnx_x", "onnx_w"],
            vec!["y"],
            vec![ints_attr("kernel_shape", vec![3, 3])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("onnx_x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("onnx_w".to_string(), vec![8, 3, 3, 3]);
        let const_values = HashMap::new();
        let mut value_ids = HashMap::new();
        value_ids.insert("onnx_x".to_string(), "x_id".to_string());
        value_ids.insert("onnx_w".to_string(), "w_id".to_string());
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let result = h.convert(&node, &ctx).unwrap();
        assert_eq!(result.nodes[0].inputs, vec!["x_id", "w_id"]);
    }

    #[test]
    fn onnx_pads_to_webnn_reorders() {
        // ONNX 2D: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
        assert_eq!(onnx_pads_to_webnn(&[1, 2, 3, 4], 2), vec![1, 3, 2, 4]);
        // ONNX 1D: [begin, end] -> WebNN: [begin, end]
        assert_eq!(onnx_pads_to_webnn(&[5, 6], 1), vec![5, 6]);
    }
}
