use crate::ast::GraphJson;

const HTML_TEMPLATE: &str = include_str!("html_template.html");

/// Emit a standalone HTML visualizer for the graph
pub fn emit_html(graph: &GraphJson) -> String {
    let graph_json = serde_json::to_string(graph).unwrap();

    // Escape closing script tags to prevent injection
    let escaped = graph_json.replace("</script>", "<\\/script>");

    HTML_TEMPLATE.replace("{{GRAPH_DATA}}", &escaped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{new_graph_json, to_dimension_vector, DataType, OperandDesc};

    #[test]
    fn test_emit_html_basic_graph() {
        let mut g = new_graph_json();
        g.name = Some("test".to_string());
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1, 10]),
            },
        );

        let html = emit_html(&g);

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("WebNN Graph Visualizer"));
        assert!(html.contains("const GRAPH_DATA = "));
        assert!(html.contains("\"name\":\"test\""));
    }

    #[test]
    fn test_html_contains_dagre() {
        let g = new_graph_json();
        let html = emit_html(&g);
        assert!(html.contains("dagre"));
    }

    #[test]
    fn test_json_escaping() {
        let mut g = new_graph_json();
        g.name = Some("test</script><script>alert('xss')".to_string());
        let html = emit_html(&g);

        // Should not contain unescaped closing script tag
        assert!(!html.contains("</script><script>alert"));
        // Should contain escaped version
        assert!(html.contains("<\\/script>"));
    }
}
