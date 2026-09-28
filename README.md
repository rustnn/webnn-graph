# webnn-graph

`webnn-graph` is a Rust library and command-line tool for a WebNN-oriented graph DSL. It parses and
serializes `.webnn` files, validates graph structure, manages external weights, emits JavaScript and
interactive HTML, and optionally converts ONNX models into the DSL.

The browser-based graph visualizer is published at
[rustnn.github.io/webnn-graph](https://rustnn.github.io/webnn-graph/).

## File model

A model consists of a graph and, when constants are external, a weight sidecar:

- `.webnn` is the compact, human-readable graph representation.
- `GraphJson` is the equivalent JSON AST used by the Rust API and tooling.
- `.safetensors` is a self-describing external-weight archive.
- `.weights` plus `.manifest.json` is the raw binary alternative used by the weight utilities and
  ONNX converter.

The canonical contracts are documented in:

- [WebNN graph format](docs/webnn-format.md)
- [External weight format](docs/external-weights.md)

## Install and build

```bash
cargo build
cargo test
```

ONNX conversion is enabled by the default `onnx` feature. Build only the parser, serializer, validators,
emitters, and weight utilities with:

```bash
cargo build --no-default-features
```

## CLI

The CLI accepts `.webnn` or `GraphJson` where indicated. Run `webnn-graph <command> --help` for the complete
option list.

| Command | Purpose |
| --- | --- |
| `parse` | Parse `.webnn` and print `GraphJson`. |
| `serialize` | Serialize `GraphJson` as `.webnn`. |
| `validate` | Validate a graph and, optionally, a raw-weight manifest. |
| `emit-js` | Emit WebNN builder JavaScript and the raw `.weights` loader. |
| `emit-html` | Emit a standalone interactive graph visualizer. |
| `pack-weights` | Pack tensor files into a `WGWT` `.weights` archive. |
| `unpack-weights` | Extract tensors from a `WGWT` `.weights` archive. |
| `create-manifest` | Create a raw-weight manifest from tensor files. |
| `extract-weights` | Move inline graph constants into a raw-weight archive. |
| `inline-weights` | Copy raw external weights into `GraphJson`. |
| `convert-onnx` | Convert ONNX to `.webnn` or `GraphJson` when the `onnx` feature is enabled. |

Examples:

```bash
# Parse and validate a graph.
cargo run -- parse examples/resnet_head.webnn > /tmp/resnet_head.json
cargo run -- validate /tmp/resnet_head.json

# Serialize GraphJson back to the text format.
cargo run -- serialize /tmp/resnet_head.json > /tmp/resnet_head.webnn

# Generate JavaScript or a standalone visualizer.
cargo run -- emit-js examples/resnet_head.webnn > /tmp/build_graph.js
cargo run -- emit-html examples/resnet_head.webnn > /tmp/graph.html
```

See [examples/README.md](examples/README.md) for the raw-weight workflow.

## ONNX conversion

The converter accepts `ai.onnx` opsets 11 through 20. Static dimension overrides and optional constant
folding can resolve shape-critical ONNX expressions. Experimental bounded dynamic input metadata is available,
but operations whose arguments must be static still require concrete values.

```bash
cargo run -- convert-onnx \
  --input model.onnx \
  --output model.webnn \
  --weights model.weights \
  --manifest model.manifest.json \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128 \
  --optimize
```

Without `--inline-weights`, conversion produces `.webnn`, `.weights`, and `.manifest.json` artifacts. Use
`--experimental-dynamic-inputs` to preserve unresolved input dimensions as bounded `dyn(...)` metadata where
the lowering can otherwise proceed.

See:

- [ONNX to WebNN lowering](docs/onnx-lowering.md)
- [Dynamic dimensions](docs/dynamic-dimensions-guide.md)

## Rust library

The public library exposes the format AST, parser, serializer, validation helpers, external-weight resolver and
SafeTensors writer, JavaScript/HTML emitters, and the optional ONNX converter.

```rust
use std::error::Error;

use webnn_graph::parser::parse_wg_text;
use webnn_graph::serialize::{serialize_graph_to_wg_text, SerializeOptions};

fn main() -> Result<(), Box<dyn Error>> {
let graph = parse_wg_text(r#"
webnn_graph "identity" v1 {
  inputs { x: f32[1]; }
  nodes { y = identity(x); }
  outputs { y; }
}
"#)?;

let text = serialize_graph_to_wg_text(&graph, SerializeOptions::default())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Development

```bash
make fmt-check
make lint
make test
cargo test --all-features
cargo test --no-default-features
```

The repository keeps prose lines at or below 120 characters.
