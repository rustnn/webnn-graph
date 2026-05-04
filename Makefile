.PHONY: all build test fmt clean run check lint help parse validate emit-js emit-html coverage minimal

CARGO_FLAGS ?=

# Default target
all: build

# Build the project
build:
	cargo build $(CARGO_FLAGS)

# Build with --no-default-features
minimal:
	cargo build --no-default-features

# Build with release optimizations
release:
	cargo build --release $(CARGO_FLAGS)

# Run all tests
test:
	cargo test $(CARGO_FLAGS)

# Run tests with output
test-verbose:
	cargo test $(CARGO_FLAGS) -- --nocapture

# Format code
fmt:
	cargo fmt

# Check formatting
fmt-check:
	cargo fmt --check

# Run clippy linter
lint:
	cargo clippy $(CARGO_FLAGS) -- -D warnings

# Quick compile check without building
check:
	cargo check $(CARGO_FLAGS)

# Clean build artifacts
clean:
	cargo clean

# Run the CLI tool (parse example)
run:
	cargo run $(CARGO_FLAGS) --bin webnn-graph -- parse examples/resnet_head.webnn

# Parse example graph
parse:
	cargo run $(CARGO_FLAGS) --bin webnn-graph -- parse examples/resnet_head.webnn

# Validate example graph
validate:
	cargo run $(CARGO_FLAGS) --bin webnn-graph -- parse examples/resnet_head.webnn > /tmp/graph.json && \
	cargo run $(CARGO_FLAGS) --bin webnn-graph -- validate /tmp/graph.json --weights-manifest examples/weights.manifest.json

# Emit JavaScript for example graph
emit-js:
	cargo run $(CARGO_FLAGS) --bin webnn-graph -- parse examples/resnet_head.webnn > /tmp/graph.json && \
	cargo run $(CARGO_FLAGS) --bin webnn-graph -- emit-js /tmp/graph.json

# Generate HTML visualizer for example graph
emit-html:
	cargo run $(CARGO_FLAGS) --bin webnn-graph -- emit-html examples/resnet_head.webnn > /tmp/webnn_viz.html
	@echo "Visualizer generated: /tmp/webnn_viz.html"
	@echo "Open it with: open /tmp/webnn_viz.html"

# Run tests with coverage (requires cargo-tarpaulin)
coverage:
	@if command -v cargo-tarpaulin >/dev/null 2>&1; then \
		cargo tarpaulin --out Html --output-dir coverage; \
		echo "Coverage report generated in coverage/"; \
	else \
		echo "cargo-tarpaulin not installed. Install with: cargo install cargo-tarpaulin"; \
		exit 1; \
	fi

# Install development dependencies
dev-deps:
	@echo "Installing development dependencies..."
	@command -v cargo-tarpaulin >/dev/null 2>&1 || cargo install cargo-tarpaulin
	@echo "Development dependencies installed"

# Help target
help:
	@echo "WebNN Graph - Available Make Targets:"
	@echo ""
	@echo "  make build         - Build the project"
	@echo "  make release       - Build with release optimizations"
	@echo "  make test          - Run all tests"
	@echo "  make test-verbose  - Run tests with output"
	@echo "  make fmt           - Format code"
	@echo "  make fmt-check     - Check code formatting"
	@echo "  make lint          - Run clippy linter"
	@echo "  make check         - Quick compile check"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make run           - Run CLI with example"
	@echo "  make parse         - Parse example graph"
	@echo "  make validate      - Validate example graph"
	@echo "  make emit-js       - Emit JavaScript for example"
	@echo "  make emit-html     - Generate HTML visualizer for example"
	@echo "  make coverage      - Generate test coverage report"
	@echo "  make dev-deps      - Install development dependencies"
	@echo "  make help          - Show this help message"
