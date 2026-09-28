# ONNX to WebNN lowering

The optional `onnx` feature converts ONNX models into the `GraphJson` AST and serializes them as `.webnn` or
JSON. The converter accepts `ai.onnx` opsets 11 through 20. Other domains are retained for operator-specific
handling rather than being checked by the `ai.onnx` opset guard.

## Conversion flow

1. Read and decode the ONNX protobuf.
2. Optionally run registered constant-folding evaluators with `--optimize`.
3. Collect explicit, sidecar, and model-metadata dimension overrides.
4. Convert graph inputs and initializers into WebNN inputs and constants.
5. Infer intermediate types and shapes required by each supported lowering.
6. Lower each ONNX node through the operator registry.
7. Optionally extract constants to `.weights` plus `.manifest.json`.
8. Serialize the resulting `GraphJson` as `.webnn` or JSON.

Unsupported opsets, operators, dtypes, attributes, or unresolved shape-critical values fail conversion with an
error. The converter does not silently omit a node.

## Dimensions and shape expressions

Static overrides specialize symbolic ONNX inputs. Experimental bounded dynamic input metadata can preserve an
unresolved input dimension when downstream lowering can still determine every required operation argument.

See [Dynamic dimensions](dynamic-dimensions-guide.md) for override precedence, `.dims.json`, metadata, inferred
batch defaults, and the `dyn(...)` representation.

ONNX graphs often calculate operation arguments through small tensor subgraphs. With `--optimize`, registered
evaluators fold expressions whose inputs are known constants before lowering. Without a foldable value, an
operation that requires a static reshape target, axis, permutation, slice bound, or similar parameter is rejected.

The exact supported behavior is operator- and opset-specific. Source tests are authoritative; this page does not
claim that every variant of a named ONNX operator is supported.

Standard-domain `Gelu` is supported from opset 20. Exact GELU (an absent `approximate` attribute or
`"none"`) maps to WebNN `gelu`; `approximate="tanh"` lowers to the ONNX polynomial with
`mul`/`add`/`tanh`, not to exact GELU. FP16 inputs are promoted to FP32 for the intermediate
polynomial and rounded back to FP16 at the output. This avoids half-precision intermediate overflow
and excessive negative-tail cancellation. Scalar inputs remain rank zero. Invalid approximation
attributes fail conversion; the older `com.microsoft::Gelu` accepts no approximation attribute.

The opset-19/20 audit preserves `AveragePool` dilation, and reductions resolve constant axes inputs,
`keepdims=0`, and `noop_with_empty_axes`. Dynamic reduction axes still fail conversion. Added float8,
bfloat16, string, sequence, and optional types remain unsupported rather than being reinterpreted as
FP32. `Cast`'s float8-only `saturate` option has no effect on supported destination types. New operators
without handlers (including `CastLike`, `Resize`, `QuantizeLinear`, `DequantizeLinear`, `GridSample`,
`AffineGrid`, and `DFT`) remain explicit errors; accepting an opset does not claim every operator in it.

`tests/onnx_gelu.rs` covers optimized/unoptimized full import, reference numerical results, typed
constants, scalar inputs, generated-name collisions, and emitted JavaScript signatures. WebNN WPT
does not import ONNX, so passing exact-GELU WPT is independent of support for ONNX's tanh variant.

## Constants and output artifacts

By default, `convert-onnx` extracts initializers and large inline constants into a headerless `.weights` blob and
a manifest whose offsets start at zero. Constants larger than 1 KiB are moved out of the graph when extraction is
enabled. Smaller scalar and byte constants may remain inline.

```bash
webnn-graph convert-onnx \
  --input model.onnx \
  --output model.webnn \
  --weights model.weights \
  --manifest model.manifest.json \
  --override-dim batch_size=1 \
  --optimize
```

Omitting explicit output paths derives all three names from the ONNX filename. `--inline-weights` suppresses the
raw sidecars and retains constants in the graph representation.

The ONNX converter currently emits manifest-backed raw weights, not SafeTensors. Consumers may subsequently save
the graph through a writer that produces `.webnn` plus `.safetensors`.

See [External weight format](external-weights.md) for both raw and SafeTensors contracts.

## Graph versions

The converter emits graph version 2 when at least one input retains a bounded dynamic dimension; otherwise it
emits version 1. Both versions use the same nodes, constants, and output structures.

The `@quantized` graph flag is metadata carried by `GraphJson` and serialization. It does not select a different
ONNX lowering pipeline by itself.

## Diagnostics

Use the global `--debug` flag before the subcommand to enable converter diagnostics:

```bash
webnn-graph --debug convert-onnx --input model.onnx --optimize
```

When conversion fails, first check:

- whether the model uses `ai.onnx` opset 11–20;
- whether every required symbolic dimension has an override or a usable bounded representation;
- whether `--optimize` can fold the shape-producing expression;
- whether the specific operator form and attributes have a registered lowering.

After conversion, parse and validate the artifact independently:

```bash
webnn-graph parse model.webnn > model.json
webnn-graph validate model.webnn --weights-manifest model.manifest.json
```
