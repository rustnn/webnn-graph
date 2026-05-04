use clap::{Parser, Subcommand};
use std::fs;
#[cfg(feature = "onnx")]
use std::path::Path;

use webnn_graph::ast::GraphJson;
use webnn_graph::emit_html::emit_html;
use webnn_graph::emit_js::{emit_builder_js, emit_weights_loader_js};
#[cfg(feature = "onnx")]
use webnn_graph::onnx::convert::{convert_onnx, ConvertOptions};
use webnn_graph::parser::parse_wg_text;
use webnn_graph::serialize::serialize_graph_to_wg_text;
use webnn_graph::validate::{validate_graph, validate_weights};
use webnn_graph::weights::WeightsManifest;
use webnn_graph::weights_io::{
    create_manifest, extract_weights, inline_weights, pack_weights, unpack_weights,
};

#[derive(Parser)]
#[command(name = "webnn-graph")]
#[command(about = "WebNN Graph DSL tools", long_about = None)]
struct Cli {
    /// Enable debug output
    #[arg(long, global = true)]
    debug: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    Parse {
        path: String,
    },
    Validate {
        path: String,
        #[arg(long)]
        weights_manifest: Option<String>,
    },
    EmitJs {
        path: String,
    },
    EmitHtml {
        path: String,
    },
    Serialize {
        path: String,
        /// Mark output as quantized (adds @quantized flag to header/options)
        #[arg(long)]
        quantized: bool,
    },
    PackWeights {
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        input_dir: String,
        #[arg(long)]
        output: String,
    },
    UnpackWeights {
        #[arg(long)]
        weights: String,
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        output_dir: String,
    },
    CreateManifest {
        #[arg(long)]
        input_dir: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "little")]
        endianness: String,
    },
    ExtractWeights {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output_dir: String,
        #[arg(long)]
        weights: String,
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        output_graph: String,
    },
    InlineWeights {
        #[arg(long)]
        input: String,
        #[arg(long)]
        weights: String,
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        output: String,
    },
    #[cfg(feature = "onnx")]
    ConvertOnnx {
        /// Input ONNX model file (.onnx)
        #[arg(long)]
        input: String,

        /// Output graph file (.webnn or .json)
        #[arg(long)]
        output: Option<String>,

        /// Inline weights instead of extracting to external file
        #[arg(long)]
        inline_weights: bool,

        /// Weights output file (.weights)
        #[arg(long)]
        weights: Option<String>,

        /// Weights manifest file (.manifest.json)
        #[arg(long)]
        manifest: Option<String>,

        /// Override dynamic dimension values (e.g., batch_size=1)
        /// Can be specified multiple times for different dimensions
        #[arg(long = "override-dim", value_name = "NAME=VALUE")]
        override_dims: Vec<String>,

        /// Read dynamic dimension overrides from a JSON file (supports { "freeDimensionOverrides": { ... } } or a flat object)
        #[arg(long = "override-dims-file")]
        override_dims_file: Option<String>,

        /// Enable constant folding and shape propagation optimizations
        #[arg(long)]
        optimize: bool,

        /// Experimental: preserve unresolved dynamic input dimensions in v2 graph metadata
        #[arg(long)]
        experimental_dynamic_inputs: bool,
    },
}

/// Load a graph from either .webnn text format or JSON format (auto-detect)
fn load_graph(path: &str) -> anyhow::Result<GraphJson> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim_start();

    // Auto-detect format: .webnn starts with "webnn_graph", JSON starts with "{"
    if trimmed.starts_with("webnn_graph") {
        Ok(parse_wg_text(&content)?)
    } else if trimmed.starts_with('{') {
        Ok(serde_json::from_str(&content)?)
    } else {
        Err(anyhow::anyhow!(
            "Unknown format: file must be .webnn text (starts with 'webnn_graph') or JSON (starts with '{{')"
        ))
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Enable debug mode if --debug flag is set
    if cli.debug {
        webnn_graph::debug::enable();
    }

    match cli.cmd {
        Command::Parse { path } => {
            let txt = fs::read_to_string(path)?;
            let g = parse_wg_text(&txt)?;
            println!("{}", serde_json::to_string_pretty(&g)?);
        }
        Command::Validate {
            path,
            weights_manifest,
        } => {
            let g = load_graph(&path)?;
            validate_graph(&g)?;

            if let Some(mpath) = weights_manifest {
                let mtxt = fs::read_to_string(mpath)?;
                let m: WeightsManifest = serde_json::from_str(&mtxt)?;
                validate_weights(&g, &m)?;
            }
            eprintln!("OK");
        }
        Command::EmitJs { path } => {
            let g = load_graph(&path)?;
            validate_graph(&g)?;

            // Emit WeightsFile helper class
            print!("{}", emit_weights_loader_js());
            println!();

            // Emit buildGraph function
            let js = emit_builder_js(&g);
            print!("{js}");
        }
        Command::EmitHtml { path } => {
            let g = load_graph(&path)?;
            validate_graph(&g)?;
            let html = emit_html(&g);
            print!("{html}");
        }
        Command::Serialize { path, quantized } => {
            let txt = fs::read_to_string(path)?;
            let g: GraphJson = serde_json::from_str(&txt)?;
            let mut g = g;
            if quantized {
                g.quantized = true;
            }
            let wg_text = serialize_graph_to_wg_text(
                &g,
                webnn_graph::serialize::SerializeOptions { quantized },
            )?;
            print!("{}", wg_text);
        }
        Command::PackWeights {
            manifest,
            input_dir,
            output,
        } => {
            pack_weights(&manifest, &input_dir, &output)?;
        }
        Command::UnpackWeights {
            weights,
            manifest,
            output_dir,
        } => {
            unpack_weights(&weights, &manifest, &output_dir)?;
        }
        Command::CreateManifest {
            input_dir,
            output,
            endianness,
        } => {
            create_manifest(&input_dir, &output, &endianness)?;
        }
        Command::ExtractWeights {
            input,
            output_dir,
            weights,
            manifest,
            output_graph,
        } => {
            let g = load_graph(&input)?;
            let new_graph = extract_weights(&g, &output_dir, &weights, &manifest)?;
            let json = serde_json::to_string_pretty(&new_graph)?;
            fs::write(&output_graph, json)?;
            eprintln!("Wrote graph with weight references to: {}", output_graph);
        }
        Command::InlineWeights {
            input,
            weights,
            manifest,
            output,
        } => {
            let g = load_graph(&input)?;
            let new_graph = inline_weights(&g, &weights, &manifest)?;
            let json = serde_json::to_string_pretty(&new_graph)?;
            fs::write(&output, json)?;
            eprintln!("Wrote graph with inline weights to: {}", output);
        }
        #[cfg(feature = "onnx")]
        Command::ConvertOnnx {
            input,
            output,
            inline_weights,
            weights,
            manifest,
            override_dims,
            override_dims_file,
            optimize,
            experimental_dynamic_inputs,
        } => {
            // Parse dimension overrides
            let mut free_dim_overrides = if let Some(path) = override_dims_file {
                let content = fs::read_to_string(&path)?;
                let json: serde_json::Value = serde_json::from_str(&content)?;
                let overrides = json
                    .get("freeDimensionOverrides")
                    .unwrap_or(&json)
                    .as_object()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "override-dims-file must be a JSON object (optionally nested under freeDimensionOverrides)"
                        )
                    })?;

                let mut map = std::collections::HashMap::new();
                for (name, value) in overrides {
                    let parsed = value.as_u64().ok_or_else(|| {
                        anyhow::anyhow!(
                            "override value for '{}' must be an integer, got {}",
                            name,
                            value
                        )
                    })?;
                    map.insert(name.to_string(), parsed as u32);
                }
                map
            } else {
                std::collections::HashMap::new()
            };
            for override_dim in override_dims {
                let parts: Vec<&str> = override_dim.split('=').collect();
                if parts.len() != 2 {
                    return Err(anyhow::anyhow!(
                        "Invalid override-dim format: '{}'. Expected NAME=VALUE",
                        override_dim
                    ));
                }
                let name = parts[0].trim().to_string();
                let value: u32 = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("Invalid dimension value: '{}'", parts[1]))?;
                free_dim_overrides.insert(name, value);
            }

            // Determine output paths
            let input_path = Path::new(&input);
            let base_name = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");

            let output_path = output.unwrap_or_else(|| format!("{}.webnn", base_name));
            let weights_path = if inline_weights {
                None
            } else {
                Some(weights.unwrap_or_else(|| format!("{}.weights", base_name)))
            };
            let manifest_path = if inline_weights {
                None
            } else {
                Some(manifest.unwrap_or_else(|| format!("{}.manifest.json", base_name)))
            };

            // Convert ONNX to GraphJson
            let options = ConvertOptions {
                extract_weights: !inline_weights,
                output_path: output_path.clone(),
                weights_path: weights_path.clone(),
                free_dim_overrides,
                optimize,
                manifest_path: manifest_path.clone(),
                experimental_dynamic_inputs,
            };

            let graph = convert_onnx(&input, options)?;

            // Serialize to .webnn text format (default) or JSON
            let output_content = if output_path.ends_with(".json") {
                serde_json::to_string_pretty(&graph)?
            } else {
                serialize_graph_to_wg_text(
                    &graph,
                    webnn_graph::serialize::SerializeOptions::default(),
                )?
            };

            fs::write(&output_path, output_content)?;
            eprintln!("✓ Converted ONNX model to: {}", output_path);

            if !inline_weights {
                if let Some(w_path) = &weights_path {
                    eprintln!("✓ Extracted weights to: {}", w_path);
                }
                if let Some(m_path) = &manifest_path {
                    eprintln!("✓ Generated manifest: {}", m_path);
                }
            }
        }
    }

    Ok(())
}
