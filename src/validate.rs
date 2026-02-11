use std::collections::BTreeSet;

use thiserror::Error;

use crate::ast::{ConstInit, GraphJson};
use crate::weights::{dtype_size, numel, WeightsManifest};

#[derive(Debug, Error)]
pub enum ValidateError {
    #[error("unsupported format/version: {0} v{1}")]
    BadFormat(String, u32),

    #[error("outputs must be non-empty")]
    EmptyOutputs,

    #[error("duplicate node id: {0}")]
    DuplicateNodeId(String),

    #[error("unknown reference '{ref_}' used in node '{node}'")]
    UnknownRef { node: String, ref_: String },

    #[error("output '{out}' references unknown value '{ref_}'")]
    BadOutputRef { out: String, ref_: String },

    #[error("missing weights manifest entry for const ref '{0}'")]
    MissingWeight(String),

    #[error("weights type/shape mismatch for '{ref_}'")]
    WeightMismatch { ref_: String },

    #[error("weights byteLength mismatch for '{ref_}': expected {expected} got {got}")]
    WeightByteLengthMismatch {
        ref_: String,
        expected: u64,
        got: u64,
    },
}

pub fn validate_graph(g: &GraphJson) -> Result<(), ValidateError> {
    if g.format != "webnn-graph-json" || (g.version != 1 && g.version != 2) {
        return Err(ValidateError::BadFormat(g.format.clone(), g.version));
    }
    if g.outputs.is_empty() {
        return Err(ValidateError::EmptyOutputs);
    }

    let mut known: BTreeSet<String> = g.inputs.keys().cloned().collect();
    known.extend(g.consts.keys().cloned());

    let mut ids = BTreeSet::new();

    for n in &g.nodes {
        if !ids.insert(n.id.clone()) {
            return Err(ValidateError::DuplicateNodeId(n.id.clone()));
        }
        for r in &n.inputs {
            if !known.contains(r) {
                return Err(ValidateError::UnknownRef {
                    node: n.id.clone(),
                    ref_: r.clone(),
                });
            }
        }
        known.insert(n.id.clone());
        if let Some(outs) = &n.outputs {
            for o in outs {
                known.insert(o.clone());
            }
        }
    }

    for (out, r) in &g.outputs {
        if !known.contains(r) {
            return Err(ValidateError::BadOutputRef {
                out: out.clone(),
                ref_: r.clone(),
            });
        }
    }
    Ok(())
}

pub fn validate_weights(g: &GraphJson, m: &WeightsManifest) -> Result<(), ValidateError> {
    for c in g.consts.values() {
        if let ConstInit::Weights { r#ref } = &c.init {
            let entry = m
                .tensors
                .get(r#ref)
                .ok_or_else(|| ValidateError::MissingWeight(r#ref.clone()))?;

            if entry.data_type != c.data_type || entry.shape != c.shape {
                return Err(ValidateError::WeightMismatch {
                    ref_: r#ref.clone(),
                });
            }

            let expected = dtype_size(&c.data_type) * numel(&c.shape);
            if entry.byte_length != expected {
                return Err(ValidateError::WeightByteLengthMismatch {
                    ref_: r#ref.clone(),
                    expected,
                    got: entry.byte_length,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{new_graph_json, to_dimension_vector, ConstDecl, DataType, Node, OperandDesc};
    use crate::weights::TensorEntry;
    use std::collections::BTreeMap;

    #[test]
    fn test_validate_graph_success() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1, 10]),
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        assert!(validate_graph(&g).is_ok());
    }

    #[test]
    fn test_validate_graph_bad_format() {
        let mut g = new_graph_json();
        g.format = "invalid".to_string();
        g.outputs.insert("x".to_string(), "x".to_string());

        let result = validate_graph(&g);
        assert!(matches!(result, Err(ValidateError::BadFormat(_, _))));
    }

    #[test]
    fn test_validate_graph_empty_outputs() {
        let g = new_graph_json();
        let result = validate_graph(&g);
        assert!(matches!(result, Err(ValidateError::EmptyOutputs)));
    }

    #[test]
    fn test_validate_graph_duplicate_node_id() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "sigmoid".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let result = validate_graph(&g);
        assert!(matches!(result, Err(ValidateError::DuplicateNodeId(_))));
    }

    #[test]
    fn test_validate_graph_unknown_ref() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "add".to_string(),
            inputs: vec!["x".to_string(), "unknown".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let result = validate_graph(&g);
        assert!(matches!(result, Err(ValidateError::UnknownRef { .. })));
    }

    #[test]
    fn test_validate_graph_bad_output_ref() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
            },
        );
        g.outputs
            .insert("out".to_string(), "nonexistent".to_string());

        let result = validate_graph(&g);
        assert!(matches!(result, Err(ValidateError::BadOutputRef { .. })));
    }

    #[test]
    fn test_validate_weights_success() {
        let mut g = new_graph_json();
        g.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );
        g.outputs.insert("x".to_string(), "x".to_string());

        let mut manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };
        manifest.tensors.insert(
            "W".to_string(),
            TensorEntry {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                byte_offset: 0,
                byte_length: 200, // 10 * 5 * 4 bytes
                layout: None,
            },
        );

        assert!(validate_weights(&g, &manifest).is_ok());
    }

    #[test]
    fn test_validate_weights_missing_weight() {
        let mut g = new_graph_json();
        g.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );
        g.outputs.insert("x".to_string(), "x".to_string());

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };

        let result = validate_weights(&g, &manifest);
        assert!(matches!(result, Err(ValidateError::MissingWeight(_))));
    }

    #[test]
    fn test_validate_weights_type_mismatch() {
        let mut g = new_graph_json();
        g.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );
        g.outputs.insert("x".to_string(), "x".to_string());

        let mut manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };
        manifest.tensors.insert(
            "W".to_string(),
            TensorEntry {
                data_type: DataType::Float16, // Mismatched type
                shape: vec![10, 5],
                byte_offset: 0,
                byte_length: 100,
                layout: None,
            },
        );

        let result = validate_weights(&g, &manifest);
        assert!(matches!(result, Err(ValidateError::WeightMismatch { .. })));
    }

    #[test]
    fn test_validate_weights_byte_length_mismatch() {
        let mut g = new_graph_json();
        g.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );
        g.outputs.insert("x".to_string(), "x".to_string());

        let mut manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };
        manifest.tensors.insert(
            "W".to_string(),
            TensorEntry {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                byte_offset: 0,
                byte_length: 100, // Wrong: should be 200
                layout: None,
            },
        );

        let result = validate_weights(&g, &manifest);
        assert!(matches!(
            result,
            Err(ValidateError::WeightByteLengthMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_weights_scalar_init_skipped() {
        let mut g = new_graph_json();
        g.consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![1],
                init: ConstInit::Scalar {
                    value: serde_json::json!(1.0),
                },
            },
        );
        g.outputs.insert("x".to_string(), "x".to_string());

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };

        // Should succeed because scalar init doesn't require weights manifest entry
        assert!(validate_weights(&g, &manifest).is_ok());
    }
}
