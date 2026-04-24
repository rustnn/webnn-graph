//! Resolve `@weights` / [`ConstInit::Weights`] using sidecar files
//! next to a graph path (SafeTensors or manifest + raw weights blob).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use half::bf16;
use safetensors::tensor::Dtype as StDtype;
use safetensors::SafeTensors;
use serde::Deserialize;
use thiserror::Error;

use crate::ast::{ConstInit, DataType as AstDataType, GraphJson};

/// Default graph JSON basename (typical sidecar layout next to weights / manifest).
pub const DEFAULT_PATH_JSON: &str = "model.json";
/// Default raw weights blob basename when not using a stem-prefixed `*.weights` file.
pub const DEFAULT_PATH_WEIGHTS: &str = "model.weights";
/// Default SafeTensors archive basename when not using a stem-prefixed `*.safetensors` file.
pub const DEFAULT_PATH_SAFETENSORS: &str = "model.safetensors";
/// Default weights manifest basename when not using a stem-prefixed `*.manifest.json` file.
pub const DEFAULT_PATH_MANIFEST: &str = "manifest.json";

/// Failure while resolving external weights for a [`GraphJson`].
#[derive(Debug, Error)]
pub enum WeightResolveError {
    /// Could not read a required file from disk.
    #[error("failed to read `{path}`: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Manifest JSON is invalid.
    #[error("failed to parse manifest JSON at `{path}`: {source}")]
    ManifestJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    /// SafeTensors–specific validation or parse error.
    #[error("[safetensors] {0}")]
    Safetensors(String),
    /// Manifest + weights blob resolution error.
    #[error("[manifest-weights] {0}")]
    Manifest(String),
    /// No usable weight source was found next to the graph.
    #[error("[weights] {0}")]
    Missing(String),
}

fn graph_has_external_weight_refs(graph_json: &GraphJson) -> bool {
    graph_json
        .consts
        .values()
        .any(|c| matches!(c.init, ConstInit::Weights { .. }))
}

/// Normalizes tensor / manifest key strings for lookup when graphs use sanitized weight refs.
#[inline]
fn sanitize_weight_key(name: &str) -> String {
    name.replace("::", "__").replace('.', "_")
}

fn safetensors_st_dtype_matches_ast(st: StDtype, ast: &AstDataType) -> bool {
    matches!(
        (ast, st),
        (AstDataType::Float32, StDtype::F32)
            | (AstDataType::Float16, StDtype::F16)
            | (AstDataType::Int32, StDtype::I32)
            | (AstDataType::Uint32, StDtype::U32)
            | (AstDataType::Int64, StDtype::I64)
            | (AstDataType::Uint64, StDtype::U64)
            | (AstDataType::Int8, StDtype::I8)
            | (AstDataType::Uint8, StDtype::U8)
    )
}

fn st_shape_matches_const(st_shape: &[usize], const_shape: &[u32]) -> bool {
    if st_shape.len() != const_shape.len() {
        return false;
    }
    st_shape
        .iter()
        .zip(const_shape.iter())
        .all(|(&s, &c)| s as u32 == c)
}

/// Convert little-endian BF16 payload to little-endian F32 (WebNN float32 constants).
fn bf16_bytes_to_f32_le_bytes(data: &[u8]) -> Result<Vec<u8>, WeightResolveError> {
    if !data.len().is_multiple_of(2) {
        return Err(WeightResolveError::Safetensors(format!(
            "BF16 data length {} is not a multiple of 2",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() * 2);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let v = bf16::from_bits(bits).to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

fn safetensors_sanitized_name_map(
    st: &SafeTensors<'_>,
) -> Result<HashMap<String, String>, WeightResolveError> {
    let mut out: HashMap<String, String> = HashMap::new();
    for name in st.names() {
        let sanitized = sanitize_weight_key(name);
        if let Some(prev) = out.insert(sanitized.clone(), name.to_string()) {
            if prev.as_str() != name {
                return Err(WeightResolveError::Safetensors(format!(
                    "ambiguous sanitized tensor name `{sanitized}` (both `{prev}` and `{name}`)"
                )));
            }
        }
    }
    Ok(out)
}

fn resolve_tensor_view<'a>(
    st: &'a SafeTensors<'a>,
    sanitized_map: &HashMap<String, String>,
    r#ref: &str,
) -> Result<safetensors::tensor::TensorView<'a>, WeightResolveError> {
    if let Ok(v) = st.tensor(r#ref) {
        return Ok(v);
    }
    let orig = sanitized_map.get(r#ref).ok_or_else(|| {
        WeightResolveError::Safetensors(format!("tensor `{ref}` not found in safetensors archive"))
    })?;
    st.tensor(orig.as_str())
        .map_err(|e| WeightResolveError::Safetensors(format!("tensor `{ref}` (via `{orig}`): {e}")))
}

fn inline_weights_from_safetensors(
    graph_json: &mut GraphJson,
    safetensors_path: &Path,
) -> Result<(), WeightResolveError> {
    let weight_ref_count = graph_json
        .consts
        .values()
        .filter(|c| matches!(c.init, ConstInit::Weights { .. }))
        .count();
    eprintln!(
        "[webnn-graph] resolve safetensors: path=`{}` weight_ref_count={}",
        safetensors_path.display(),
        weight_ref_count
    );

    let bytes = fs::read(safetensors_path).map_err(|source| WeightResolveError::ReadFile {
        path: safetensors_path.to_path_buf(),
        source,
    })?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| {
        WeightResolveError::Safetensors(format!("`{}`: {e}", safetensors_path.display()))
    })?;
    let sanitized_map = safetensors_sanitized_name_map(&st)?;

    for (const_name, const_decl) in graph_json.consts.iter_mut() {
        let ConstInit::Weights { r#ref } = &const_decl.init else {
            continue;
        };
        let view = match resolve_tensor_view(&st, &sanitized_map, r#ref) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "[webnn-graph] warning: safetensors could not resolve weight ref `{ref}` \
                     (constant `{const_name}`) from `{}`: {e}",
                    safetensors_path.display()
                );
                return Err(e);
            }
        };
        if !st_shape_matches_const(view.shape(), &const_decl.shape) {
            let msg = format!(
                "shape mismatch for weight `{ref}` (constant `{const_name}`): graph {:?} vs safetensors {:?}",
                const_decl.shape,
                view.shape()
            );
            eprintln!(
                "[webnn-graph] warning: safetensors could not resolve weight `{ref}` \
                 (constant `{const_name}`) from `{}`: {msg}",
                safetensors_path.display()
            );
            return Err(WeightResolveError::Safetensors(msg));
        }

        let st_dtype = view.dtype();
        let raw = view.data();
        let bytes = if safetensors_st_dtype_matches_ast(st_dtype, &const_decl.data_type) {
            raw.to_vec()
        } else if matches!(
            (&const_decl.data_type, st_dtype),
            (AstDataType::Float32, StDtype::BF16)
        ) {
            let elem_count: usize = const_decl.shape.iter().map(|&x| x as usize).product();
            let expected = elem_count.checked_mul(2).ok_or_else(|| {
                WeightResolveError::Safetensors(format!(
                    "element count overflow for weight `{ref}` (constant `{const_name}`)"
                ))
            })?;
            if raw.len() != expected {
                return Err(WeightResolveError::Safetensors(format!(
                    "BF16 tensor `{ref}` (constant `{const_name}`): byte length {} != expected {} ({} BF16 elements)",
                    raw.len(),
                    expected,
                    elem_count
                )));
            }
            eprintln!(
                "[webnn-graph] safetensors: converting BF16 → float32 for weight `{ref}` (constant `{const_name}`)"
            );
            bf16_bytes_to_f32_le_bytes(raw)?
        } else {
            let msg = format!(
                "dtype mismatch for weight `{ref}` (constant `{const_name}`): graph declares {:?} but safetensors has {:?}",
                const_decl.data_type,
                st_dtype
            );
            eprintln!(
                "[webnn-graph] warning: safetensors could not resolve weight `{ref}` \
                 (constant `{const_name}`) from `{}`: {msg}",
                safetensors_path.display()
            );
            return Err(WeightResolveError::Safetensors(msg));
        };

        const_decl.init = ConstInit::InlineBytes { bytes };
    }

    let still_count = graph_json
        .consts
        .values()
        .filter(|c| matches!(c.init, ConstInit::Weights { .. }))
        .count();
    if still_count > 0 {
        return Err(WeightResolveError::Safetensors(format!(
            "safetensors `{}` did not provide all tensors referenced by the graph ({still_count} still missing)",
            safetensors_path.display()
        )));
    }

    Ok(())
}

/// Weight manifest JSON next to a graph (supports `webnn-weights-manifest` and related layouts).
#[derive(Debug, Deserialize)]
struct FlexibleManifest {
    #[serde(default)]
    tensors: HashMap<String, FlexibleTensorEntry>,
}

#[derive(Debug, Deserialize, Clone)]
struct FlexibleTensorEntry {
    #[serde(rename = "byteOffset")]
    byte_offset: u64,
    #[serde(rename = "byteLength")]
    byte_length: u64,
}

fn inline_weights_from_manifest(
    graph_json: &mut GraphJson,
    manifest_path: &Path,
    weights_path: &Path,
) -> Result<(), WeightResolveError> {
    let manifest_text =
        fs::read_to_string(manifest_path).map_err(|source| WeightResolveError::ReadFile {
            path: manifest_path.to_path_buf(),
            source,
        })?;
    let weights_bytes = fs::read(weights_path).map_err(|source| WeightResolveError::ReadFile {
        path: weights_path.to_path_buf(),
        source,
    })?;

    let manifest: FlexibleManifest = serde_json::from_str(&manifest_text).map_err(|source| {
        WeightResolveError::ManifestJson {
            path: manifest_path.to_path_buf(),
            source,
        }
    })?;

    let mut manifest_by_sanitized: HashMap<String, Vec<FlexibleTensorEntry>> = HashMap::new();
    for (name, entry) in &manifest.tensors {
        let sanitized = sanitize_weight_key(name);
        manifest_by_sanitized
            .entry(sanitized)
            .or_default()
            .push(entry.clone());
    }

    for (const_name, const_decl) in graph_json.consts.iter_mut() {
        let ConstInit::Weights { r#ref } = &const_decl.init else {
            continue;
        };
        let entry = manifest
            .tensors
            .get(r#ref)
            .cloned()
            .or_else(|| {
                manifest_by_sanitized.get(r#ref).and_then(|entries| {
                    if entries.len() == 1 {
                        Some(entries[0].clone())
                    } else {
                        None
                    }
                })
            })
            .ok_or_else(|| {
                WeightResolveError::Manifest(format!(
                    "no manifest tensor entry for weight ref `{ref}` (constant `{const_name}`)"
                ))
            })?;

        let start = usize::try_from(entry.byte_offset).map_err(|_| {
            WeightResolveError::Manifest(format!(
                "byteOffset {} for `{ref}` does not fit in usize",
                entry.byte_offset
            ))
        })?;
        let len = usize::try_from(entry.byte_length).map_err(|_| {
            WeightResolveError::Manifest(format!(
                "byteLength {} for `{ref}` does not fit in usize",
                entry.byte_length
            ))
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            WeightResolveError::Manifest(format!("byte range overflow for `{ref}`"))
        })?;
        if end > weights_bytes.len() {
            return Err(WeightResolveError::Manifest(format!(
                "byte range [{start}, {end}) for `{ref}` exceeds weights file length {} (`{}`)",
                weights_bytes.len(),
                weights_path.display()
            )));
        }
        const_decl.init = ConstInit::InlineBytes {
            bytes: weights_bytes[start..end].to_vec(),
        };
    }
    Ok(())
}

/// Resolves `path_str` relative to the parent directory of `graph_path`, or as an absolute path
/// when `path_str` is absolute.
fn resolve_path_relative_to_graph(graph_path: &Path, path_str: &str) -> PathBuf {
    let p = Path::new(path_str);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        graph_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path_str)
    }
}

fn discover_sidecar_manifest(graph_path: &Path) -> Option<PathBuf> {
    let stem = graph_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    [
        graph_path.with_file_name(format!("{stem}.manifest.json")),
        graph_path.with_file_name(DEFAULT_PATH_MANIFEST),
    ]
    .into_iter()
    .find(|p| p.exists())
}

/// Discovers a single weights file next to `graph_path`, in order: `{stem}.safetensors`,
/// `{stem}.weights`, [`DEFAULT_PATH_SAFETENSORS`], [`DEFAULT_PATH_WEIGHTS`].
fn discover_weights_file(graph_path: &Path) -> Option<PathBuf> {
    let stem = graph_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    [
        graph_path.with_file_name(format!("{stem}.safetensors")),
        graph_path.with_file_name(format!("{stem}.weights")),
        graph_path.with_file_name(DEFAULT_PATH_SAFETENSORS),
        graph_path.with_file_name(DEFAULT_PATH_WEIGHTS),
    ]
    .into_iter()
    .find(|p| p.exists())
}

/// Whether `path` refers to a SafeTensors archive (by extension).
fn path_looks_like_safetensors(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()).is_some_and(|e| {
        e.eq_ignore_ascii_case("safetensors") || e.eq_ignore_ascii_case("safetensor")
    })
}

/// If `graph_json` contains any `ConstInit::Weights` references, load tensors from disk next to
/// `graph_path` and replace them with [`ConstInit::InlineBytes`].
///
/// ## Resolution
///
/// 1. **No-op.** If the graph has no [`ConstInit::Weights`] initializers, return `Ok(())` without
///    reading the filesystem.
///
/// 2. **Resolve weights path** (discovery is separate from loading):
///    - If `weights_path` is set: resolve relative to the graph’s directory (or absolute as-is); the file
///      must exist or return [`WeightResolveError::Missing`].
///    - Else: [`discover_weights_file`] searches next to the graph in order: `{stem}.safetensors`,
///      `{stem}.weights`, [`DEFAULT_PATH_SAFETENSORS`], [`DEFAULT_PATH_WEIGHTS`]. If none exist, return
///      [`WeightResolveError::Missing`].
///
/// 3. **Load by kind:**
///    - If the weights path is SafeTensors → [`inline_weights_from_safetensors`] and return (any
///      `manifest_path` is ignored).
///    - Otherwise it is a binary blob → resolve manifest: explicit `manifest_path` must exist, or
///      [`discover_sidecar_manifest`] must find `{stem}.manifest.json` / [`DEFAULT_PATH_MANIFEST`], else
///      [`WeightResolveError::Missing`]. Then [`inline_weights_from_manifest`].
///
/// Incomplete SafeTensors resolution returns [`WeightResolveError::Safetensors`]; manifest errors use
/// [`WeightResolveError::Manifest`] / [`WeightResolveError::ManifestJson`].
pub fn resolve_external_weights(
    graph_json: &mut GraphJson,
    graph_path: &Path,
    weights_path: Option<&str>,
    manifest_path: Option<&str>,
) -> Result<(), WeightResolveError> {
    eprintln!(
        "[webnn graph] resolve external weights: graph={}, weights_path={}, manifest_path={}",
        graph_path.display(),
        weights_path.unwrap_or("<discover next to graph>"),
        manifest_path.unwrap_or("<discover next to graph>"),
    );

    if !graph_has_external_weight_refs(graph_json) {
        return Ok(());
    }

    let stem = graph_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();

    let wp = if let Some(s) = weights_path {
        let p = resolve_path_relative_to_graph(graph_path, s);
        if !p.exists() {
            return Err(WeightResolveError::Missing(format!(
                "weights path `{}` does not exist",
                p.display()
            )));
        }
        p
    } else {
        discover_weights_file(graph_path).ok_or_else(|| {
            WeightResolveError::Missing(format!(
                "no weights file found next to `{0}`; expected `{1}.safetensors`, `{1}.weights`, \
                 `{DEFAULT_PATH_SAFETENSORS}`, or `{DEFAULT_PATH_WEIGHTS}`, or pass `weights_path`",
                graph_path.display(),
                stem,
            ))
        })?
    };

    if path_looks_like_safetensors(&wp) {
        return inline_weights_from_safetensors(graph_json, &wp);
    }

    let mp = if let Some(s) = manifest_path {
        let p = resolve_path_relative_to_graph(graph_path, s);
        if !p.exists() {
            return Err(WeightResolveError::Missing(format!(
                "manifest path `{}` does not exist",
                p.display()
            )));
        }
        p
    } else {
        discover_sidecar_manifest(graph_path).ok_or_else(|| {
            WeightResolveError::Missing(format!(
                "weights blob `{0}` requires a manifest; pass `manifest_path` or place `{1}.manifest.json` / \
                 `{DEFAULT_PATH_MANIFEST}` next to `{2}`",
                wp.display(),
                stem,
                graph_path.display()
            ))
        })?
    };

    inline_weights_from_manifest(graph_json, &mp, &wp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::{serialize, Dtype};
    use tempfile::TempDir;

    fn write_safetensors_f32(path: &Path, tensor_name: &str, shape: Vec<usize>, data: &[u8]) {
        let view = TensorView::new(Dtype::F32, shape, data).unwrap();
        let bytes = serialize(vec![(tensor_name.to_string(), view)], None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn write_safetensors_bf16(path: &Path, tensor_name: &str, shape: Vec<usize>, data: &[u8]) {
        let view = TensorView::new(Dtype::BF16, shape, data).unwrap();
        let bytes = serialize(vec![(tensor_name.to_string(), view)], None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn manifest_and_weights_inline() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let manifest_path = temp_dir.path().join("model.manifest.json");
        let weights_path = temp_dir.path().join(DEFAULT_PATH_WEIGHTS);

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        let manifest_content = r#"{
            "format": "webnn-weights-manifest",
            "version": 1,
            "endianness": "little",
            "tensors": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "byteOffset": 0,
                    "byteLength": 8
                }
            }
        }"#;

        let weights_data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40];
        std::fs::write(&graph_path, graph_content).unwrap();
        std::fs::write(&manifest_path, manifest_content).unwrap();
        std::fs::write(&weights_path, &weights_data).unwrap();

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        resolve_external_weights(&mut graph, &graph_path, None, None).unwrap();
        match &graph.consts["weight"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes.len(), 8),
            other => panic!("expected inline bytes, got {:?}", other),
        }
    }

    #[test]
    fn explicit_manifest_and_weights_paths() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let manifest_path = temp_dir.path().join("custom.manifest.json");
        let weights_path = temp_dir.path().join("blob.weights");

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        let manifest_content = r#"{
            "format": "webnn-weights-manifest",
            "version": 1,
            "endianness": "little",
            "tensors": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "byteOffset": 0,
                    "byteLength": 8
                }
            }
        }"#;

        let weights_data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40];
        std::fs::write(&graph_path, graph_content).unwrap();
        std::fs::write(&manifest_path, manifest_content).unwrap();
        std::fs::write(&weights_path, &weights_data).unwrap();

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        resolve_external_weights(
            &mut graph,
            &graph_path,
            Some("blob.weights"),
            Some("custom.manifest.json"),
        )
        .unwrap();
        match &graph.consts["weight"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes.len(), 8),
            other => panic!("expected inline bytes, got {:?}", other),
        }
    }

    #[test]
    fn explicit_safetensors_weights_path() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let st_path = temp_dir.path().join("custom.safetensors");

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        let tensor_bytes: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40];
        std::fs::write(&graph_path, graph_content).unwrap();
        write_safetensors_f32(&st_path, "weight", vec![2], &tensor_bytes);

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        resolve_external_weights(&mut graph, &graph_path, Some("custom.safetensors"), None)
            .unwrap();
        match &graph.consts["weight"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes, &tensor_bytes),
            other => panic!("expected inline bytes, got {:?}", other),
        }
    }

    #[test]
    fn manifest_arg_ignored_when_weights_path_is_safetensors() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let st_path = temp_dir.path().join("weights.safetensors");

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        let tensor_bytes: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40];
        std::fs::write(&graph_path, graph_content).unwrap();
        write_safetensors_f32(&st_path, "weight", vec![2], &tensor_bytes);

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        resolve_external_weights(
            &mut graph,
            &graph_path,
            Some("weights.safetensors"),
            Some("this_manifest_is_not_read.json"),
        )
        .unwrap();
        match &graph.consts["weight"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes, &tensor_bytes),
            other => panic!("expected inline bytes, got {:?}", other),
        }
    }

    #[test]
    fn safetensors_inline() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let st_path = temp_dir.path().join(DEFAULT_PATH_SAFETENSORS);

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        let tensor_bytes: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40];
        std::fs::write(&graph_path, graph_content).unwrap();
        write_safetensors_f32(&st_path, "weight", vec![2], &tensor_bytes);

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        resolve_external_weights(&mut graph, &graph_path, None, None).unwrap();
        match &graph.consts["weight"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes, &tensor_bytes),
            other => panic!("expected inline bytes, got {:?}", other),
        }
    }

    #[test]
    fn out_of_bounds_manifest_errors() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let manifest_path = temp_dir.path().join(DEFAULT_PATH_MANIFEST);
        let weights_path = temp_dir.path().join(DEFAULT_PATH_WEIGHTS);

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        let manifest_content = r#"{
            "format": "webnn-weights-manifest",
            "version": 1,
            "tensors": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "byteOffset": 0,
                    "byteLength": 100
                }
            }
        }"#;

        std::fs::write(&graph_path, graph_content).unwrap();
        std::fs::write(&manifest_path, manifest_content).unwrap();
        std::fs::write(&weights_path, vec![0u8; 8]).unwrap();

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        let err = resolve_external_weights(&mut graph, &graph_path, None, None).unwrap_err();
        assert!(matches!(err, WeightResolveError::Manifest(_)));
    }

    #[test]
    fn safetensors_preferred_over_invalid_manifest() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let manifest_path = temp_dir.path().join(DEFAULT_PATH_MANIFEST);
        let weights_path = temp_dir.path().join(DEFAULT_PATH_WEIGHTS);
        let st_path = temp_dir.path().join(DEFAULT_PATH_SAFETENSORS);

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        std::fs::write(&graph_path, graph_content).unwrap();
        std::fs::write(&manifest_path, "{ not valid manifest json").unwrap();
        std::fs::write(&weights_path, [0u8; 8]).unwrap();
        write_safetensors_f32(
            &st_path,
            "weight",
            vec![2],
            &[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40],
        );

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        resolve_external_weights(&mut graph, &graph_path, None, None).unwrap();
    }

    #[test]
    fn safetensors_bf16_converts_to_float32_for_graph_constants() {
        use half::bf16;

        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join(DEFAULT_PATH_JSON);
        let st_path = temp_dir.path().join(DEFAULT_PATH_SAFETENSORS);

        let graph_content = r#"{
            "format": "webnn-graph-json",
            "version": 1,
            "inputs": { "x": { "dataType": "float32", "shape": [2] } },
            "consts": {
                "weight": {
                    "dataType": "float32",
                    "shape": [2],
                    "init": { "kind": "weights", "ref": "weight" }
                }
            },
            "nodes": [],
            "outputs": { "y": "x" }
        }"#;

        let mut bf16_bytes = Vec::new();
        bf16_bytes.extend_from_slice(&bf16::from_f32(1.0f32).to_bits().to_le_bytes());
        bf16_bytes.extend_from_slice(&bf16::from_f32(2.0f32).to_bits().to_le_bytes());

        std::fs::write(&graph_path, graph_content).unwrap();
        write_safetensors_bf16(&st_path, "weight", vec![2], &bf16_bytes);

        let mut graph: GraphJson = serde_json::from_str(graph_content).unwrap();
        resolve_external_weights(&mut graph, &graph_path, None, None).unwrap();

        let expected: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        match &graph.consts["weight"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes, &expected),
            other => panic!("expected inline bytes, got {:?}", other),
        }
    }
}
