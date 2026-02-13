# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [0.3.0] - to be released

### Added
- Dynamic dimension representation in AST and WG grammar via `dyn("name", maxSize)` input dims.
- ONNX conversion support for preserving unresolved dynamic input dimensions in graph metadata.
- New `convert-onnx` CLI flag: `--experimental-dynamic-inputs` (opt-in dynamic input preservation).
- ONNX converter/operator support improvements:
  - `ScatterND`
  - `Where`, `Equal`, comparison operators, `Cos`, `Sin`, `TriLu`, `Tile`
  - `ConstantOfShape`, `Range`, and additional constant-folding evaluators
- Built-in constant folding in `webnn-graph` (`--optimize`) to reduce dynamic-shape plumbing.
- Global debug switch for converter diagnostics (`--debug`).
- Pre-commit hook setup script and make-based local checks.

### Changed
- ONNX conversion now supports static-lowering + dynamic metadata workflows in one pipeline.
- Graph parser/serializer now support richer values (including object literals in options).
- JS/HTML emitters and visualizer now render mixed static/dynamic shapes.
- Docs expanded and corrected:
  - ONNX lowering behavior
  - Dynamic dimension guidance
  - SmolLM-135M conversion example from Hugging Face

### Fixed
- Multiple ONNX conversion correctness fixes, including:
  - dynamic reshape/expand conversion edge cases
  - shape inflation prevention and post-conversion shape tracking
  - `Unsqueeze` v14 handling
  - identifier sanitization robustness (including `$` prefixes)
  - clippy/robustness cleanup across converter and shape inference

### Compatibility
- Existing static graphs remain supported.
- Validator/serializer support both graph versions `v1` and `v2`.
- Dynamic input metadata is experimental and must be enabled with
  `--experimental-dynamic-inputs`.

## [0.2.1] - 2025-12-28

### Added
- ONNX shape inference and `Expand` conversion support.
- Initial ONNX lowering documentation.

### Fixed
- BERT conversion fixes.
- Identifier sanitization updates.

## [0.2.0] - 2025-12-24

### Added
- Interactive HTML visualizer and `emit-html` command.
- Drag-and-drop `.webnn` loading and parser improvements.
- Graph/weights split workflow improvements and docs refinements.

## [0.1.0] - 2025-12-24

### Added
- Initial release with core DSL parsing/serialization/validation scaffold.
- Binary weights support and foundational CLI commands.
