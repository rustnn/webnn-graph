//! Resolve `@weights` / [`ConstInit::Weights`] using sidecar files
//! next to a graph path (SafeTensors or manifest + raw weights blob).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::bf16;
use safetensors::tensor::{Dtype as StDtype, TensorView};
#[cfg(test)]
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

/// SafeTensors archive metadata key for packed logical Int4/Uint4 tensors.
pub const PACKED_4BIT_METADATA_KEY: &str = "rustnn.webnn.packed4";
/// Current version of the packed logical Int4/Uint4 storage extension.
pub const PACKED_4BIT_METADATA_VERSION: &str = "1";

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

/// Failure while writing a SafeTensors archive for a [`GraphJson`].
#[derive(Debug, Error)]
pub enum WeightWriteError {
    /// The graph declarations or supplied tensor bytes are inconsistent.
    #[error("[safetensors] {0}")]
    Validation(String),
    /// SafeTensors could not serialize the archive.
    #[error("failed to write SafeTensors archive `{path}`: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },
    /// The completed temporary archive could not be installed at its destination.
    #[error("failed to install SafeTensors archive `{path}`: {source}")]
    Install {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

fn ast_dtype_to_safetensors(ast: &AstDataType) -> Option<StDtype> {
    Some(match ast {
        AstDataType::Float32 => StDtype::F32,
        AstDataType::Float16 => StDtype::F16,
        AstDataType::Int32 => StDtype::I32,
        AstDataType::Uint32 => StDtype::U32,
        AstDataType::Int64 => StDtype::I64,
        AstDataType::Uint64 => StDtype::U64,
        AstDataType::Int8 => StDtype::I8,
        AstDataType::Uint8 => StDtype::U8,
        AstDataType::Int4 | AstDataType::Uint4 => return None,
    })
}

fn logical_element_count(name: &str, shape: &[u32]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension as usize)
            .ok_or_else(|| format!("constant `{name}` element count overflows usize"))
    })
}

fn packed_byte_length(name: &str, shape: &[u32]) -> Result<usize, String> {
    Ok(logical_element_count(name, shape)?.div_ceil(2))
}

fn validate_packed4_marker(
    metadata: &safetensors::tensor::Metadata,
    path: &Path,
) -> Result<(), WeightResolveError> {
    let marker = metadata
        .metadata()
        .as_ref()
        .and_then(|values| values.get(PACKED_4BIT_METADATA_KEY));
    match marker.map(String::as_str) {
        Some(PACKED_4BIT_METADATA_VERSION) => Ok(()),
        Some(version) => Err(WeightResolveError::Safetensors(format!(
            "`{}` uses unsupported packed 4-bit format version `{version}`",
            path.display()
        ))),
        None => Err(WeightResolveError::Safetensors(format!(
            "`{}` contains external int4/uint4 weights but lacks the `{PACKED_4BIT_METADATA_KEY}` metadata marker",
            path.display()
        ))),
    }
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
    for chunk in data.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let v = bf16::from_bits(bits).to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

fn safetensors_sanitized_name_map(
    st: &safetensors::tensor::Metadata,
) -> Result<HashMap<String, String>, WeightResolveError> {
    let mut out: HashMap<String, String> = HashMap::new();
    for name in st.tensors().keys() {
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

fn resolve_tensor_info<'a>(
    st: &'a safetensors::tensor::Metadata,
    sanitized_map: &HashMap<String, String>,
    name: &str,
) -> Result<&'a safetensors::tensor::TensorInfo, WeightResolveError> {
    st.info(name)
        .or_else(|| sanitized_map.get(name).and_then(|n| st.info(n)))
        .ok_or_else(|| {
            WeightResolveError::Safetensors(format!(
                "tensor `{name}` not found in safetensors archive"
            ))
        })
}
/// Read only the bounded header. Metadata's Deserialize validates shape sizes,
/// contiguous offsets and overflow; the final extent must match the actual file.
fn read_archive_header(
    path: &Path,
) -> Result<(fs::File, Vec<u8>, safetensors::tensor::Metadata), WeightResolveError> {
    use std::io::Read;
    let io = |source| WeightResolveError::ReadFile {
        path: path.to_path_buf(),
        source,
    };
    let bad = |message: String| {
        WeightResolveError::Safetensors(format!("`{}`: {message}", path.display()))
    };
    let mut file = fs::File::open(path).map_err(io)?;
    let len = file.metadata().map_err(io)?.len();
    let mut size = [0u8; 8];
    file.read_exact(&mut size).map_err(io)?;
    let n = u64::from_le_bytes(size);
    if n > 100_000_000 || n > len.saturating_sub(8) {
        return Err(bad("invalid header size".into()));
    }
    let mut bytes = vec![0; n as usize + 8];
    bytes[..8].copy_from_slice(&size);
    file.read_exact(&mut bytes[8..]).map_err(io)?;
    let metadata: safetensors::tensor::Metadata =
        serde_json::from_slice(&bytes[8..]).map_err(|e| bad(e.to_string()))?;
    let end = metadata
        .tensors()
        .values()
        .map(|t| t.data_offsets.1 as u64)
        .max()
        .unwrap_or(0);
    if end.checked_add(n).and_then(|v| v.checked_add(8)) != Some(len) {
        return Err(bad("archive payload length differs from header".into()));
    }
    Ok((file, bytes, metadata))
}

#[cfg(test)]
fn read_file(path: &Path) -> Result<Vec<u8>, WeightResolveError> {
    fs::read(path).map_err(|source| WeightResolveError::ReadFile {
        path: path.to_path_buf(),
        source,
    })
}

fn inline_weights_from_safetensors(
    graph_json: &mut GraphJson,
    safetensors_path: &Path,
    mut consumer: Option<&mut WeightConsumer<'_>>,
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

    let (mut archive, archive_bytes, st) = read_archive_header(safetensors_path)?;
    let sanitized_map = safetensors_sanitized_name_map(&st)?;
    let has_packed_refs = graph_json.consts.values().any(|decl| {
        matches!(decl.data_type, AstDataType::Int4 | AstDataType::Uint4)
            && matches!(decl.init, ConstInit::Weights { .. })
    });
    if has_packed_refs {
        validate_packed4_marker(&st, safetensors_path)?;
    }

    for (const_name, const_decl) in graph_json.consts.iter_mut() {
        let ConstInit::Weights { r#ref } = &const_decl.init else {
            continue;
        };
        let info = match resolve_tensor_info(&st, &sanitized_map, r#ref) {
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

        use std::io::{Read, Seek, SeekFrom};
        let mut payload = vec![0; info.data_offsets.1 - info.data_offsets.0];
        let io = |source| WeightResolveError::ReadFile {
            path: safetensors_path.to_path_buf(),
            source,
        };
        archive
            .seek(SeekFrom::Start(
                archive_bytes.len() as u64 + info.data_offsets.0 as u64,
            ))
            .map_err(io)?;
        archive.read_exact(&mut payload).map_err(io)?;
        let view = safetensors::tensor::TensorView::new(info.dtype, info.shape.clone(), &payload)
            .map_err(|e| WeightResolveError::Safetensors(e.to_string()))?;

        if matches!(const_decl.data_type, AstDataType::Int4 | AstDataType::Uint4) {
            let expected = packed_byte_length(const_name, &const_decl.shape)
                .map_err(WeightResolveError::Safetensors)?;
            if view.dtype() != StDtype::U8 {
                return Err(WeightResolveError::Safetensors(format!(
                    "packed 4-bit weight `{ref}` (constant `{const_name}`) must use SafeTensors U8 storage, found {:?}",
                    view.dtype()
                )));
            }
            if view.shape() != [expected] {
                return Err(WeightResolveError::Safetensors(format!(
                    "packed 4-bit weight `{ref}` (constant `{const_name}`) storage shape {:?} does not match expected [{expected}] for logical shape {:?}",
                    view.shape(), const_decl.shape
                )));
            }
            if view.data().len() != expected {
                return Err(WeightResolveError::Safetensors(format!(
                    "packed 4-bit weight `{ref}` (constant `{const_name}`) has {} bytes, expected {expected}",
                    view.data().len()
                )));
            }
            let bytes = if let Some(sink) = consumer.as_deref_mut() {
                sink(const_name, const_decl, payload)?
            } else {
                payload
            };
            const_decl.init = ConstInit::InlineBytes { bytes };
            continue;
        }
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
            payload
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

        let bytes = if let Some(sink) = consumer.as_deref_mut() {
            sink(const_name, const_decl, bytes)?
        } else {
            bytes
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
    mut consumer: Option<&mut WeightConsumer<'_>>,
) -> Result<(), WeightResolveError> {
    let manifest_text =
        fs::read_to_string(manifest_path).map_err(|source| WeightResolveError::ReadFile {
            path: manifest_path.to_path_buf(),
            source,
        })?;
    use std::io::{Read, Seek, SeekFrom};
    let io = |source| WeightResolveError::ReadFile {
        path: weights_path.to_path_buf(),
        source,
    };
    let mut weights = fs::File::open(weights_path).map_err(io)?;
    let weights_len = weights.metadata().map_err(io)?.len();

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
        if end as u64 > weights_len {
            return Err(WeightResolveError::Manifest(format!(
                "byte range [{start}, {end}) for `{ref}` exceeds weights file length {} (`{}`)",
                weights_len,
                weights_path.display()
            )));
        }
        let mut bytes = vec![0; len];
        weights
            .seek(SeekFrom::Start(entry.byte_offset))
            .map_err(io)?;
        weights.read_exact(&mut bytes).map_err(io)?;
        let bytes = if let Some(sink) = consumer.as_deref_mut() {
            sink(const_name, const_decl, bytes)?
        } else {
            bytes
        };
        const_decl.init = ConstInit::InlineBytes { bytes };
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
type WeightConsumer<'a> =
    dyn FnMut(&str, &crate::ast::ConstDecl, Vec<u8>) -> Result<Vec<u8>, WeightResolveError> + 'a;

pub fn resolve_external_weights(
    graph_json: &mut GraphJson,
    graph_path: &Path,
    weights_path: Option<&str>,
    manifest_path: Option<&str>,
) -> Result<(), WeightResolveError> {
    resolve_external_weights_impl(graph_json, graph_path, weights_path, manifest_path, None)
}

/// Resolve one external tensor at a time through a consuming sink. The returned
/// replacement bytes become the declaration's inline payload. A sink may return
/// an empty vector only when it retains equivalent external storage separately;
/// that graph is then metadata for that consumer, not a standalone resolved AST.
pub fn resolve_external_weights_with_consumer(
    graph_json: &mut GraphJson,
    graph_path: &Path,
    weights_path: Option<&str>,
    manifest_path: Option<&str>,
    consumer: &mut WeightConsumer<'_>,
) -> Result<(), WeightResolveError> {
    resolve_external_weights_impl(
        graph_json,
        graph_path,
        weights_path,
        manifest_path,
        Some(consumer),
    )
}

fn resolve_external_weights_impl(
    graph_json: &mut GraphJson,
    graph_path: &Path,
    weights_path: Option<&str>,
    manifest_path: Option<&str>,
    consumer: Option<&mut WeightConsumer<'_>>,
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
        return inline_weights_from_safetensors(graph_json, &wp, consumer);
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

    inline_weights_from_manifest(graph_json, &mp, &wp, consumer)
}

fn temporary_archive_path(path: &Path) -> PathBuf {
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("weights.safetensors");
    path.with_file_name(format!(".{name}.{}.{}.tmp", std::process::id(), sequence))
}

/// Writes all external constants declared by `graph_json` to one SafeTensors archive.
///
/// `external_weights` is keyed by each [`ConstInit::Weights`] reference. Logical Int4 and Uint4
/// payloads remain packed low-nibble-first and are stored as one-dimensional U8 tensors.
pub fn write_external_weights_safetensors(
    graph_json: &GraphJson,
    external_weights: &HashMap<String, &[u8]>,
    destination: &Path,
) -> Result<(), WeightWriteError> {
    let mut declarations: BTreeMap<&str, (&AstDataType, &[u32])> = BTreeMap::new();
    for (constant_name, declaration) in &graph_json.consts {
        let ConstInit::Weights { r#ref } = &declaration.init else {
            continue;
        };
        if let Some((data_type, shape)) = declarations.get(r#ref.as_str()) {
            if *data_type != &declaration.data_type || *shape != declaration.shape.as_slice() {
                return Err(WeightWriteError::Validation(format!(
                    "weight reference `{ref}` has conflicting declarations (constant `{constant_name}`)"
                )));
            }
        } else {
            declarations.insert(r#ref, (&declaration.data_type, &declaration.shape));
        }
    }

    let mut views = Vec::with_capacity(declarations.len());
    let mut has_packed4 = false;
    for (weight_ref, (data_type, logical_shape)) in declarations {
        let bytes = external_weights.get(weight_ref).copied().ok_or_else(|| {
            WeightWriteError::Validation(format!(
                "weight reference `{weight_ref}` has no supplied tensor bytes"
            ))
        })?;
        let (storage_dtype, storage_shape) = if matches!(
            data_type,
            AstDataType::Int4 | AstDataType::Uint4
        ) {
            let packed_len = packed_byte_length(weight_ref, logical_shape)
                .map_err(WeightWriteError::Validation)?;
            if bytes.len() != packed_len {
                return Err(WeightWriteError::Validation(format!(
                        "packed 4-bit weight `{weight_ref}` has {} bytes, expected {packed_len} for logical shape {logical_shape:?}",
                        bytes.len()
                    )));
            }
            has_packed4 = true;
            (StDtype::U8, vec![packed_len])
        } else {
            let dtype = ast_dtype_to_safetensors(data_type).ok_or_else(|| {
                WeightWriteError::Validation(format!(
                    "weight reference `{weight_ref}` has unsupported data type {data_type:?}"
                ))
            })?;
            (
                dtype,
                logical_shape
                    .iter()
                    .map(|&dimension| dimension as usize)
                    .collect(),
            )
        };
        let view = TensorView::new(storage_dtype, storage_shape, bytes).map_err(|error| {
            WeightWriteError::Validation(format!("weight reference `{weight_ref}`: {error}"))
        })?;
        views.push((weight_ref.to_string(), view));
    }

    let metadata = has_packed4.then(|| {
        HashMap::from([(
            PACKED_4BIT_METADATA_KEY.to_string(),
            PACKED_4BIT_METADATA_VERSION.to_string(),
        )])
    });
    let temporary = temporary_archive_path(destination);
    if let Err(source) = safetensors::serialize_to_file(views, metadata, &temporary) {
        let _ = fs::remove_file(&temporary);
        return Err(WeightWriteError::Serialize {
            path: destination.to_path_buf(),
            source,
        });
    }

    let install_result = match fs::rename(&temporary, destination) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(destination).and_then(|()| fs::rename(&temporary, destination))
        }
        Err(source) => Err(source),
    };
    if let Err(source) = install_result {
        let _ = fs::remove_file(&temporary);
        return Err(WeightWriteError::Install {
            path: destination.to_path_buf(),
            source,
        });
    }
    Ok(())
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

    fn external_graph(constants: Vec<(&str, AstDataType, Vec<u32>, &str)>) -> GraphJson {
        let mut graph = crate::ast::new_graph_json();
        for (name, data_type, shape, weight_ref) in constants {
            graph.consts.insert(
                name.to_string(),
                crate::ast::ConstDecl {
                    data_type,
                    shape,
                    init: ConstInit::Weights {
                        r#ref: weight_ref.to_string(),
                    },
                },
            );
        }
        graph
    }

    fn write_test_archive(
        path: &Path,
        tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
        marker: Option<&str>,
    ) {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let metadata = marker.map(|version| {
            HashMap::from([(PACKED_4BIT_METADATA_KEY.to_string(), version.to_string())])
        });
        safetensors::serialize_to_file(views, metadata, path).unwrap();
    }

    #[test]
    fn mixed_ordinary_and_packed_weights_roundtrip_in_one_archive() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join("mixed.webnn");
        let archive_path = temp_dir.path().join("mixed.safetensors");
        let mut graph = external_graph(vec![
            ("even", AstDataType::Int4, vec![4], "even"),
            ("odd", AstDataType::Uint4, vec![3], "odd"),
            ("float", AstDataType::Float32, vec![2], "float"),
        ]);
        let even = [0x21, 0x43];
        let odd = [0x65, 0x07];
        let float = [0_u8, 0, 128, 63, 0, 0, 0, 64];
        let bytes = HashMap::from([
            ("even".to_string(), even.as_slice()),
            ("odd".to_string(), odd.as_slice()),
            ("float".to_string(), float.as_slice()),
        ]);

        write_external_weights_safetensors(&graph, &bytes, &archive_path).unwrap();
        let archive_bytes = read_file(&archive_path).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(&archive_bytes).unwrap();
        assert_eq!(
            metadata.metadata().as_ref().unwrap()[PACKED_4BIT_METADATA_KEY],
            PACKED_4BIT_METADATA_VERSION
        );
        let archive = SafeTensors::deserialize(&archive_bytes).unwrap();
        assert_eq!(archive.tensor("even").unwrap().shape(), [2]);
        assert_eq!(archive.tensor("odd").unwrap().shape(), [2]);
        assert_eq!(archive.tensor("float").unwrap().shape(), [2]);

        resolve_external_weights(&mut graph, &graph_path, None, None).unwrap();
        assert_eq!(
            graph.consts["even"].init,
            ConstInit::InlineBytes {
                bytes: even.to_vec()
            }
        );
        assert_eq!(
            graph.consts["odd"].init,
            ConstInit::InlineBytes {
                bytes: odd.to_vec()
            }
        );
        assert_eq!(
            graph.consts["float"].init,
            ConstInit::InlineBytes {
                bytes: float.to_vec()
            }
        );
    }

    #[test]
    fn packed_weights_require_supported_marker() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join("packed.webnn");
        let archive_path = temp_dir.path().join("packed.safetensors");
        let graph = external_graph(vec![("q", AstDataType::Uint4, vec![3], "q")]);

        for marker in [None, Some("999")] {
            write_test_archive(
                &archive_path,
                vec![("q".to_string(), Dtype::U8, vec![2], vec![0x21, 0x03])],
                marker,
            );
            let mut candidate = graph.clone();
            let error = resolve_external_weights(&mut candidate, &graph_path, None, None)
                .unwrap_err()
                .to_string();
            assert!(error.contains(if marker.is_some() {
                "unsupported"
            } else {
                "lacks"
            }));
        }
    }

    #[test]
    fn packed_weights_validate_storage_dtype_and_shape() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join("packed.webnn");
        let archive_path = temp_dir.path().join("packed.safetensors");
        let graph = external_graph(vec![("q", AstDataType::Int4, vec![3], "q")]);

        write_test_archive(
            &archive_path,
            vec![("q".to_string(), Dtype::I8, vec![2], vec![0x21, 0x03])],
            Some(PACKED_4BIT_METADATA_VERSION),
        );
        let error = resolve_external_weights(&mut graph.clone(), &graph_path, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must use SafeTensors U8"));

        write_test_archive(
            &archive_path,
            vec![("q".to_string(), Dtype::U8, vec![1, 2], vec![0x21, 0x03])],
            Some(PACKED_4BIT_METADATA_VERSION),
        );
        let error = resolve_external_weights(&mut graph.clone(), &graph_path, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("storage shape"));
    }

    #[test]
    fn writer_rejects_missing_malformed_and_overflowing_weights_without_partials() {
        let temp_dir = TempDir::new().unwrap();
        let archive_path = temp_dir.path().join("packed.safetensors");
        let graph = external_graph(vec![("q", AstDataType::Uint4, vec![3], "q")]);

        let error = write_external_weights_safetensors(&graph, &HashMap::new(), &archive_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no supplied tensor bytes"));

        let malformed = [0x21];
        let error = write_external_weights_safetensors(
            &graph,
            &HashMap::from([("q".to_string(), malformed.as_slice())]),
            &archive_path,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("expected 2"));

        let overflow = external_graph(vec![(
            "q",
            AstDataType::Int4,
            vec![u32::MAX, u32::MAX, u32::MAX],
            "q",
        )]);
        let empty = [];
        let error = write_external_weights_safetensors(
            &overflow,
            &HashMap::from([("q".to_string(), empty.as_slice())]),
            &archive_path,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("element count overflows"));
        assert!(!archive_path.exists());
        assert!(temp_dir.path().read_dir().unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")));
    }

    #[test]
    fn packed_weights_share_sanitized_lookup_and_ambiguity_rules() {
        let temp_dir = TempDir::new().unwrap();
        let graph_path = temp_dir.path().join("packed.webnn");
        let archive_path = temp_dir.path().join("packed.safetensors");
        let graph = external_graph(vec![("q", AstDataType::Uint4, vec![3], "a__b")]);

        write_test_archive(
            &archive_path,
            vec![("a::b".to_string(), Dtype::U8, vec![2], vec![0x21, 0x03])],
            Some(PACKED_4BIT_METADATA_VERSION),
        );
        let mut resolved = graph.clone();
        resolve_external_weights(&mut resolved, &graph_path, None, None).unwrap();
        assert_eq!(
            resolved.consts["q"].init,
            ConstInit::InlineBytes {
                bytes: vec![0x21, 0x03]
            }
        );

        write_test_archive(
            &archive_path,
            vec![
                ("a::b".to_string(), Dtype::U8, vec![2], vec![0x21, 0x03]),
                ("a..b".to_string(), Dtype::U8, vec![2], vec![0x21, 0x03]),
            ],
            Some(PACKED_4BIT_METADATA_VERSION),
        );
        let error = resolve_external_weights(&mut graph.clone(), &graph_path, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous sanitized tensor name"));

        write_test_archive(
            &archive_path,
            vec![("other".to_string(), Dtype::U8, vec![2], vec![0x21, 0x03])],
            Some(PACKED_4BIT_METADATA_VERSION),
        );
        let missing = external_graph(vec![("q", AstDataType::Uint4, vec![3], "missing")]);
        let error = resolve_external_weights(&mut missing.clone(), &graph_path, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not found"));
    }
    #[test]
    fn consuming_sink_owns_payload_and_replaces_inline_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.webnn");
        let archive = path.with_extension("safetensors");
        write_test_archive(
            &archive,
            vec![
                ("a".into(), Dtype::I8, vec![2], vec![0x80, 0x7f]),
                (
                    "b".into(),
                    Dtype::F32,
                    vec![1],
                    1.25f32.to_le_bytes().to_vec(),
                ),
            ],
            None,
        );
        let original = external_graph(vec![
            ("a", AstDataType::Int8, vec![2], "a"),
            ("b", AstDataType::Float32, vec![1], "b"),
        ]);
        let mut inline = original.clone();
        resolve_external_weights(&mut inline, &path, None, None).unwrap();
        let mut graph = original.clone();
        let mut retained = BTreeMap::new();
        resolve_external_weights_with_consumer(
            &mut graph,
            &path,
            None,
            None,
            &mut |name, decl, bytes| {
                assert!(matches!(decl.init, ConstInit::Weights { .. }));
                retained.insert(name.to_string(), bytes);
                Ok(vec![])
            },
        )
        .unwrap();
        assert_eq!(retained.len(), 2);
        for (name, decl) in &inline.consts {
            let ConstInit::InlineBytes { bytes } = &decl.init else {
                panic!()
            };
            assert_eq!(retained[name], *bytes);
            assert!(
                matches!(&graph.consts[name].init, ConstInit::InlineBytes { bytes } if bytes.is_empty())
            );
        }
        let mut graph = original;
        resolve_external_weights_with_consumer(
            &mut graph,
            &path,
            None,
            None,
            &mut |_, _, bytes| Ok(bytes),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(graph).unwrap(),
            serde_json::to_value(inline).unwrap()
        );
    }

    #[test]
    fn consuming_sink_error_is_propagated_without_replacing_the_payload() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.webnn");
        write_safetensors_f32(
            &path.with_extension("safetensors"),
            "w",
            vec![1],
            &1f32.to_le_bytes(),
        );
        let mut graph = external_graph(vec![("w", AstDataType::Float32, vec![1], "w")]);
        let err = resolve_external_weights_with_consumer(
            &mut graph,
            &path,
            None,
            None,
            &mut |_, _, _| Err(WeightResolveError::Manifest("sink failed".into())),
        )
        .unwrap_err();
        assert!(err.to_string().contains("sink failed"));
        assert!(matches!(graph.consts["w"].init, ConstInit::Weights { .. }));
    }

    #[test]
    fn streaming_header_rejects_truncated_extended_and_oversized_archives_before_sink() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.webnn");
        let archive = path.with_extension("safetensors");
        write_safetensors_f32(&archive, "w", vec![1], &1f32.to_le_bytes());
        let original = fs::read(&archive).unwrap();
        for bytes in [
            original[..original.len() - 1].to_vec(),
            [original.clone(), vec![0]].concat(),
            u64::MAX.to_le_bytes().to_vec(),
        ] {
            fs::write(&archive, bytes).unwrap();
            let mut graph = external_graph(vec![("w", AstDataType::Float32, vec![1], "w")]);
            let mut called = false;
            assert!(resolve_external_weights_with_consumer(
                &mut graph,
                &path,
                None,
                None,
                &mut |_, _, bytes| {
                    called = true;
                    Ok(bytes)
                }
            )
            .is_err());
            assert!(!called);
        }
    }

    #[test]
    fn consuming_manifest_reads_only_requested_span_and_preserves_error_contract() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.webnn");
        fs::write(path.with_extension("weights"), [9, 0x80, 0x7f, 8]).unwrap();
        let manifest = path.with_extension("manifest.json");
        fs::write(
            &manifest,
            r#"{"tensors":{"w":{"byteOffset":1,"byteLength":2}}}"#,
        )
        .unwrap();
        let mut graph = external_graph(vec![("w", AstDataType::Int8, vec![2], "w")]);
        resolve_external_weights_with_consumer(
            &mut graph,
            &path,
            None,
            None,
            &mut |name, _, bytes| {
                assert_eq!(name, "w");
                assert_eq!(bytes, vec![0x80, 0x7f]);
                Ok(bytes)
            },
        )
        .unwrap();
        fs::write(
            &manifest,
            r#"{"tensors":{"w":{"byteOffset":18446744073709551615,"byteLength":2}}}"#,
        )
        .unwrap();
        let mut graph = external_graph(vec![("w", AstDataType::Int8, vec![2], "w")]);
        assert!(resolve_external_weights_with_consumer(
            &mut graph,
            &path,
            None,
            None,
            &mut |_, _, _| panic!("invalid span reached sink")
        )
        .is_err());
    }
}
