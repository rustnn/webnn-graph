//! End-to-end conversion test for ResNet-50 (ONNX Model Zoo).
//!
//! The downloaded model is cached at `target/test-data/resnet50_Opset16.onnx`.
//! If a copy already exists at the repository root (the same file used during
//! manual testing) the test uses that and skips the download.

use std::fs;
use std::path::{Path, PathBuf};

use webnn_graph::ast::GraphJson;
use webnn_graph::emit_js::{emit_builder_js, emit_weights_loader_js};
use webnn_graph::onnx::convert::{convert_onnx, ConvertOptions};
use webnn_graph::serialize::{serialize_graph_to_wg_text, SerializeOptions};
use webnn_graph::validate::{validate_graph, validate_weights};
use webnn_graph::weights::WeightsManifest;

const MODEL_URL: &str = "https://media.githubusercontent.com/media/onnx/models/refs/heads/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx";
const MODEL_FILENAME: &str = "resnet50_Opset16.onnx";
// Exact size of the LFS-resolved blob. Catches partial downloads, LFS pointer
// files (~130 B), and upstream model swaps. Bump this if MODEL_URL ever points
// to a different file.
const EXPECTED_MODEL_SIZE_BYTES: u64 = 102_146_206;

fn locate_or_download_model() -> PathBuf {
    // Download and cache the model under target/test-data/.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cache_dir = repo_root.join("target").join("test-data");
    fs::create_dir_all(&cache_dir).expect("create test-data cache dir");
    let cached = cache_dir.join(MODEL_FILENAME);
    if cached.exists() && file_size(&cached) == EXPECTED_MODEL_SIZE_BYTES {
        return cached;
    }

    eprintln!("Downloading {} -> {}", MODEL_URL, cached.display());
    let mut builder = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(600));
    // Honor the conventional proxy env vars so the test works behind a corporate
    // proxy without code changes. ureq does not read these automatically.
    if let Some(proxy_url) = proxy_from_env() {
        match ureq::Proxy::new(&proxy_url) {
            Ok(proxy) => {
                eprintln!("Using proxy from environment: {}", proxy_url);
                builder = builder.proxy(proxy);
            }
            Err(e) => eprintln!("Ignoring malformed proxy env var '{}': {}", proxy_url, e),
        }
    }
    let agent = builder.build();
    let response = agent
        .get(MODEL_URL)
        .call()
        .expect("download ResNet-50 model");
    let mut out = fs::File::create(&cached).expect("create cache file");
    std::io::copy(&mut response.into_reader(), &mut out).expect("stream model to disk");

    let downloaded_size = file_size(&cached);
    assert_eq!(
        downloaded_size, EXPECTED_MODEL_SIZE_BYTES,
        "downloaded model size mismatch (got {} bytes, expected {}). \
         Likely a Git LFS pointer file, a truncated download, or the upstream \
         model was swapped — bump EXPECTED_MODEL_SIZE_BYTES if intentional.",
        downloaded_size, EXPECTED_MODEL_SIZE_BYTES,
    );

    cached
}

fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Read the conventional HTTPS/HTTP proxy environment variables. The HTTPS variant
/// is preferred since the model URL is HTTPS, but we fall back to plain HTTP_PROXY
/// for environments that only set one.
fn proxy_from_env() -> Option<String> {
    for var in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

fn count_op(graph: &GraphJson, op: &str) -> usize {
    graph.nodes.iter().filter(|n| n.op == op).count()
}

#[test]
fn resnet50_converts_validates_and_emits_js() {
    let model_path = locate_or_download_model();
    let tmp = tempfile::tempdir().expect("create temp dir");

    let weights_path = tmp.path().join("resnet50.weights");
    let manifest_path = tmp.path().join("resnet50.manifest.json");
    let webnn_path = tmp.path().join("resnet50.webnn");

    let options = ConvertOptions {
        extract_weights: true,
        output_path: webnn_path.to_string_lossy().into_owned(),
        weights_path: Some(weights_path.to_string_lossy().into_owned()),
        manifest_path: Some(manifest_path.to_string_lossy().into_owned()),
        free_dim_overrides: Default::default(),
        optimize: true,
        experimental_dynamic_inputs: false,
    };

    let graph = convert_onnx(&model_path, options).expect("convert ResNet-50");

    // ResNet-50 stem + 4 stages of bottlenecks: expect a large but predictable op mix.
    let conv2d = count_op(&graph, "conv2d");
    let max_pool = count_op(&graph, "maxPool2d");
    let avg_pool = count_op(&graph, "averagePool2d");
    let relu = count_op(&graph, "relu");
    let add = count_op(&graph, "add");
    let matmul = count_op(&graph, "matmul");
    let reshape = count_op(&graph, "reshape");

    eprintln!(
        "ResNet-50 conversion produced: conv2d={conv2d}, maxPool2d={max_pool}, \
         averagePool2d={avg_pool}, relu={relu}, add={add}, matmul={matmul}, reshape={reshape}, \
         total_nodes={total}",
        total = graph.nodes.len()
    );

    // Exact counts: this is a fixed, versioned ONNX file so any drift here means
    // either the converter changed semantics or the upstream model was swapped.
    assert_eq!(conv2d, 53, "expected exactly 53 conv2d nodes");
    assert_eq!(max_pool, 1, "expected exactly 1 maxPool2d node (stem)");
    assert_eq!(
        avg_pool, 1,
        "expected exactly 1 averagePool2d node (global pool head)"
    );
    assert_eq!(relu, 49, "expected exactly 49 relu nodes");
    assert_eq!(
        add, 17,
        "expected exactly 17 add nodes (16 residual + FC bias)"
    );
    assert_eq!(matmul, 1, "expected exactly 1 matmul (final FC layer)");
    assert_eq!(reshape, 1, "expected exactly 1 reshape (from Flatten)");
    assert_eq!(graph.nodes.len(), 124, "expected exactly 124 total nodes");

    // The graph has a single image input and a single classification output.
    assert_eq!(graph.inputs.len(), 1, "expected exactly 1 input");
    assert_eq!(graph.outputs.len(), 1, "expected exactly 1 output");

    // Stem conv: 7×7 kernel, stride 2, padding 3 on every side. Locating it via
    // the windowDimensions option keeps the assertion robust to op ordering.
    let stem_conv = graph
        .nodes
        .iter()
        .find(|n| n.op == "conv2d" && n.options.get("strides") == Some(&serde_json::json!([2, 2])))
        .expect("stem conv2d (stride 2) not found");
    assert_eq!(
        stem_conv.options.get("padding"),
        Some(&serde_json::json!([3, 3, 3, 3])),
        "stem conv2d should have padding [3,3,3,3]"
    );

    // MaxPool stem: 3×3 window, stride 2, padding 1 on every side.
    let max_pool_node = graph
        .nodes
        .iter()
        .find(|n| n.op == "maxPool2d")
        .expect("maxPool2d not found");
    assert_eq!(
        max_pool_node.options.get("windowDimensions"),
        Some(&serde_json::json!([3, 3])),
    );
    assert_eq!(
        max_pool_node.options.get("strides"),
        Some(&serde_json::json!([2, 2])),
    );
    assert_eq!(
        max_pool_node.options.get("padding"),
        Some(&serde_json::json!([1, 1, 1, 1])),
    );

    // GlobalAveragePool over the final 7×7 feature map.
    let avg_pool_node = graph
        .nodes
        .iter()
        .find(|n| n.op == "averagePool2d")
        .expect("averagePool2d not found");
    assert_eq!(
        avg_pool_node.options.get("windowDimensions"),
        Some(&serde_json::json!([7, 7])),
    );

    // Flatten lowered to reshape [1, 2048] (batch=1, FC input dim).
    let flatten_reshape = graph
        .nodes
        .iter()
        .find(|n| {
            n.op == "reshape" && n.options.get("newShape") == Some(&serde_json::json!([1, 2048]))
        })
        .expect("flatten-as-reshape [1, 2048] not found");
    let _ = flatten_reshape; // assertion above is enough; kept binding for clarity.

    // Validate against the extracted manifest.
    let manifest_text = fs::read_to_string(&manifest_path).expect("read manifest");
    let manifest: WeightsManifest = serde_json::from_str(&manifest_text).expect("parse manifest");
    validate_graph(&graph).expect("graph passes structural validation");
    validate_weights(&graph, &manifest).expect("manifest matches graph constants");

    // Round-trip through the .webnn text format (this used to fail for ResNet-50
    // because some intermediate ONNX values are pure digits, e.g. "495", which the
    // grammar rejects unless sanitize_identifier prefixes them).
    let serialized = serialize_graph_to_wg_text(&graph, SerializeOptions::default())
        .expect("serialize to .webnn text");
    let reparsed =
        webnn_graph::parser::parse_wg_text(&serialized).expect("re-parse serialized .webnn");
    assert_eq!(reparsed.nodes.len(), graph.nodes.len());
    assert_eq!(reparsed.inputs.len(), graph.inputs.len());
    assert_eq!(reparsed.outputs.len(), graph.outputs.len());

    // Emit JS and spot-check that all four new operators reached the output.
    let js = format!("{}\n{}", emit_weights_loader_js(), emit_builder_js(&graph));
    for needle in [
        "MLGraphBuilder",
        "builder[\"conv2d\"]",
        "builder[\"maxPool2d\"]",
        "builder[\"averagePool2d\"]",
        "builder[\"reshape\"]",
        "builder[\"add\"]",
        "builder[\"relu\"]",
        "builder.build(outputs)",
    ] {
        assert!(
            js.contains(needle),
            "emitted JS missing expected token: {needle}",
        );
    }
}
