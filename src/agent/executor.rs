use serde_json::Value;

/// Result placeholder for step execution (kept for compatibility with older MVP flows).
#[derive(Debug, Clone)]
pub struct StepResult {
    pub output: Value,
    pub sources_found: Vec<String>,
}
