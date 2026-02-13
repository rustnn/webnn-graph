# Proposal Draft: W3C WebNN DSL and Graph File Format Specification

Suggested target repo: `w3c/webmachinelearning`

Suggested issue title:

`Proposal: Standardize a WebNN Graph DSL and Portable File Format (.webnn + JSON + weights manifest)`

## Proposal name

WebNN Graph DSL and Portable File Format

## Short description

Today, developers can author and execute WebNN graphs through JavaScript APIs, but there is no interoperable,
text-based graph format that can be exchanged across tools, validated offline, versioned in source control, and
compiled into runtime code. This creates friction for model conversion pipelines, tooling interoperability,
long-term archival, and debugging.

This proposal suggests starting a W3C specification for a WebNN-oriented graph DSL and companion file format,
including: (1) a human-readable text representation (`.webnn`), (2) a canonical JSON representation for tooling,
and (3) an optional external weights manifest + binary container. The target audience is ML tooling authors,
framework and converter maintainers, browser implementers, and application teams shipping WebNN models at scale.

## Example use cases

### 1) Toolchain interoperability: ONNX -> WebNN deployment

End user goal: convert an ONNX model once and deploy consistently across browsers and runtimes.

Step-by-step:
1. A model engineer converts ONNX into a standardized `.webnn` graph + external weights files.
2. CI validates the graph and manifest with a standards-compliant validator.
3. Build tooling emits WebNN JavaScript builder code from the standardized format.
4. Runtime loads the same portable artifacts in browser or server environments.

Result: one portable representation across converter, validator, codegen, and runtime boundaries.

### 2) Team collaboration and code review

End user goal: review model-graph changes like source code.

Step-by-step:
1. A developer modifies graph structure in a text `.webnn` file.
2. Teammates review diffs in pull requests without opening binary assets.
3. Automated checks parse and validate graph references, node wiring, and type/shape declarations.
4. Approved changes are serialized for downstream tools without semantic drift.

Result: graph evolution becomes auditable, reviewable, and easy to reason about.

### 3) Long-term reproducibility and governance

End user goal: preserve model topology and weights mapping for compliance and future replay.

Step-by-step:
1. A release pipeline stores `.webnn`, JSON, manifest, and weights as versioned artifacts.
2. Audit tooling verifies that graph declarations match manifest metadata and tensor layout.
3. Future teams can reconstruct the execution graph and regenerate builders from archived artifacts.

Result: stable, vendor-neutral artifacts for reproducibility, governance, and lifecycle management.

## A rough idea or two about implementation

A practical path is to scope the first version around graph structure, typing, and serialization invariants,
not full operator semantics (which remain defined by the WebNN API specification). The spec can define:
- Concrete grammar for a text DSL (`.webnn`) with versioning and forward-compatible extensions.
- Canonical JSON mapping rules and round-trip requirements (`.webnn` <-> JSON).
- Optional weights manifest schema (dtype, shape, offsets, lengths, endianness) and binary container framing.
- Validation and conformance requirements (references, uniqueness, type/shape consistency).

For standardization process, start as an incubated Community Group / Working Group deliverable with a small,
interoperable core and a conformance test suite. Existing open-source implementations can provide seed grammar,
fixtures, and round-trip tests, while WPT-style tests verify cross-implementation parsing, validation behavior,
and serialization determinism. This keeps the initial scope tight and measurable while leaving room for future
operator-coverage and extension profiles.

## Optional starter scope for v1

- Graph header/versioning and quantization annotation.
- Inputs / const declarations / nodes / outputs blocks.
- Referencing model for external weights.
- JSON canonical form and lossless round-trip criteria.
- Validator rules and machine-readable error categories.

## Why now

WebNN adoption is growing, model conversion pipelines are already producing WebNN-oriented artifacts, and the
ecosystem would benefit from a portable format that reduces bespoke tooling and improves interoperability.
Standardizing a minimal DSL + file format now can prevent fragmentation before incompatible conventions harden.
