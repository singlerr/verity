use std::collections::HashSet;

use serde_json::{json, Value};

use crate::app::{PlanStep, Source, Tool as PlanTool};
use crate::agent::tools::ToolRegistry;
use crate::agent::orchestrator::AgentEvent;

/// Result placeholder for step execution (kept for compatibility with older MVP flows).
#[derive(Debug, Clone)]
pub struct StepResult {
    pub output: Value,
    pub sources_found: Vec<String>,
}

/// Iterative search loop: deduplicates by URL, max 3 iterations, early stop.
pub struct SearchLoop { pub max_iterations: u8 }
impl Default for SearchLoop { fn default() -> Self { Self::new() } }
impl SearchLoop {
    pub fn new() -> Self { Self { max_iterations: 3 } }
    pub async fn run(&self, queries: &[String], tools: &ToolRegistry, sources: &mut Vec<Source>, tx: &std::sync::mpsc::Sender<AgentEvent>) -> Option<String> {
        let mut seen = HashSet::new();
        let mut out = String::new();
        let mut qs = queries.to_vec();
        for cur in 1..=self.max_iterations {
            if qs.is_empty() { break; }
            let mut n = 0usize;
            if let Some(svc) = tools.get("search") {
                for q in &qs {
                    let _ = tx.send(AgentEvent::SearchingIteration { current: cur, max: self.max_iterations, query: q.clone() });
                    if let Ok(r) = svc.execute(&json!({"query": q})).await {
                        if let Some(arr) = r.get("results").and_then(|v| v.as_array()) {
                            for item in arr {
                                if let Some(url) = item.get("url").and_then(|v| v.as_str()) {
                                    if seen.insert(url.to_string()) {
                                        let src = Source { num: sources.len() + 1, domain: extract_domain(url), title: item.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(), url: url.to_string(), snippet: item.get("snippet").and_then(|v| v.as_str()).unwrap_or("").to_string(), quote: String::new() };
                                        sources.push(src.clone());
                                        let _ = tx.send(AgentEvent::SourceFound(src));
                                        n += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if let Some(rd) = tools.get("read_url") {
                for url in sources.iter().rev().take(3).map(|s| s.url.clone()) {
                    if let Ok(v) = rd.execute(&json!({"url": url})).await {
                        if let Some(c) = v.get("content").and_then(|s| s.as_str()) {
                            out.push_str(&format!("Page {}: {}\n", url, c.chars().take(3000).collect::<String>()));
                        }
                    }
                }
            }
            if n == 0 { break; }
            qs.clear();
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

async fn call_tool(name: &str, input: Value, tools: &ToolRegistry) -> Result<Value, String> {
    match tools.get(name) {
        Some(t) => t.execute(&input).await.map_err(|e| e.to_string()),
        None => Err(format!("{} tool not found", name)),
    }
}

/// Execute a single plan step using the tool registry.
/// Returns Some(output_string) on success, or None if the step could not be executed.
pub async fn execute_step(
    step: &PlanStep,
    idx: usize,
    tools: &ToolRegistry,
    all_sources: &mut Vec<Source>,
    tx: &std::sync::mpsc::Sender<crate::agent::orchestrator::AgentEvent>,
) -> Option<String> {
    // Start reporting this step to UI
    let _ = tx.send(AgentEvent::StepStarted(idx));

    let mut step_output: String = String::new();

    match step.tool {
        PlanTool::Search => {
            if let Some(tool) = tools.get("search") {
                let input = json!({"query": step.title});
                match tool.execute(&input).await {
                    Ok(val) => {
                        if let Some(results) = val.get("results").and_then(|v| v.as_array()) {
                            let mut snippet_buf = String::new();
                            for (i, item) in results.iter().enumerate() {
                                let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
                                let url = item.get("url").and_then(|v| v.as_str()).unwrap_or("");
                                let snippet = item.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                                let domain = extract_domain(url);
                                let src = Source {
                                    num: all_sources.len() + i + 1,
                                    domain,
                                    title: title.to_string(),
                                    url: url.to_string(),
                                    snippet: snippet.to_string(),
                                    quote: String::new(),
                                };
                                snippet_buf.push_str(&format!("- {} {}: {}\n", title, url, snippet));
                                all_sources.push(src.clone());
                                let _ = tx.send(AgentEvent::SourceFound(src));
                            }
                            step_output.push_str(&format!("Search results for \"{}\":\n{}", step.title, snippet_buf));
                        } else {
                            let _ = tx.send(AgentEvent::StepFailed(idx, "Search: unexpected response".to_string()));
                        }
                        let _ = tx.send(AgentEvent::StepDone(idx));
                    }
                    Err(e) => {
                        let _ = tx.send(AgentEvent::StepFailed(idx, format!("Search tool failed: {}", e)));
                    }
                }
            } else {
                let _ = tx.send(AgentEvent::StepFailed(idx, "Search tool not found".to_string()));
            }
        }
        PlanTool::Read => {
            if let Some(tool) = tools.get("read_url") {
                let urls_to_read: Vec<_> = all_sources.iter()
                    .take(5)
                    .map(|s| s.url.clone())
                    .collect();
                if urls_to_read.is_empty() {
                    let _ = tx.send(AgentEvent::StepProgress(idx, "No sources to read".to_string()));
                    let _ = tx.send(AgentEvent::StepDone(idx));
                } else {
                    for url in urls_to_read {
                        let input = json!({"url": url});
                        match tool.execute(&input).await {
                            Ok(v) => {
                                if let Some(content) = v.get("content").and_then(|s| s.as_str()) {
                                    let truncated: String = content.chars().take(3000).collect();
                                    step_output.push_str(&format!("Page content from {}:\n{}\n\n", url, truncated));
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(AgentEvent::StepProgress(idx, format!("Failed to read {}: {}", url, e)));
                            }
                        }
                    }
                    let _ = tx.send(AgentEvent::StepDone(idx));
                }
            } else {
                let _ = tx.send(AgentEvent::StepFailed(idx, "Read tool not found".to_string()));
            }
        }
        PlanTool::Think => {
            let _ = tx.send(AgentEvent::StepProgress(idx, "Thinking...".to_string()));
            let _ = tx.send(AgentEvent::StepDone(idx));
        }
        PlanTool::Edit => match call_tool("write_file", json!({"path": step.title}), tools).await {
            Ok(_) => { let _ = tx.send(AgentEvent::StepDone(idx)); }
            Err(e) => { let _ = tx.send(AgentEvent::StepFailed(idx, format!("Write: {}", e))); }
        },
        PlanTool::Shell => match call_tool("shell", json!({"command": step.title}), tools).await {
            Ok(v) => { if let Some(o) = v.get("stdout").and_then(|s| s.as_str()) { step_output.push_str(&format!("Shell `{}`:\n{}", step.title, o)); let _ = tx.send(AgentEvent::StepProgress(idx, o.to_string())); } let _ = tx.send(AgentEvent::StepDone(idx)); }
            Err(e) => { let _ = tx.send(AgentEvent::StepFailed(idx, format!("Shell: {}", e))); }
        },
        PlanTool::ReadFile => match call_tool("read_file", json!({"path": step.title}), tools).await {
            Ok(v) => { if let Some(c) = v.get("content").and_then(|s| s.as_str()) { step_output.push_str(&format!("File `{}`:\n{}", step.title, c)); let _ = tx.send(AgentEvent::StepProgress(idx, format!("Read {} bytes", c.len()))); } let _ = tx.send(AgentEvent::StepDone(idx)); }
            Err(e) => { let _ = tx.send(AgentEvent::StepFailed(idx, format!("read_file: {}", e))); }
        },
        PlanTool::ListDir => match call_tool("list_dir", json!({"path": step.title}), tools).await {
            Ok(v) => { if let Some(t) = v.get("tree").and_then(|s| s.as_str()) { step_output.push_str(&format!("Dir `{}`:\n{}", step.title, t)); let _ = tx.send(AgentEvent::StepProgress(idx, format!("Listed {}", step.title))); } let _ = tx.send(AgentEvent::StepDone(idx)); }
            Err(e) => { let _ = tx.send(AgentEvent::StepFailed(idx, format!("list_dir: {}", e))); }
        },
    }

    (!step_output.is_empty()).then_some(step_output)
}

fn extract_domain(url: &str) -> String {
    if let Some(pos) = url.find("://") {
        let rest = &url[(pos + 3)..];
        rest.split('/')
            .next()
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    }
}

#[cfg(test)] mod tests { use super::*; #[test] fn domain() { assert_eq!(extract_domain("https://a.b/c"), "a.b"); assert_eq!(extract_domain("no_url"), ""); } #[test] fn max_iter() { assert_eq!(SearchLoop::new().max_iterations, 3); } }
