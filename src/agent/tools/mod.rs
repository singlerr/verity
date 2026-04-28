//! Agent tool implementations for search, URL reading, and file reading.

pub mod registry;
pub mod local;
pub use registry::{build_tool_registry, tool_manifest};
pub use local::{WriteFileTool, ListDirTool, ShellTool};

use crate::fs::read::read_file;
use crate::search::{SearchEngine, SearchResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Trait for agent tools.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, input: &Value) -> Result<Value>;
}

/// Search tool using a SearchEngine implementation.
pub struct SearchTool {
    search_engine: Arc<dyn SearchEngine>,
}

impl SearchTool {
    pub fn new(search_engine: Arc<dyn SearchEngine>) -> Self {
        Self { search_engine }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .context("Missing 'query' field in input")?;

        let results = self.search_engine.search(query).await?;
        let formatted: Vec<Value> = results
            .into_iter()
            .take(5)
            .map(|r: SearchResult| {
                json!({
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                })
            })
            .collect();

        Ok(json!({"results": formatted}))
    }
}

/// Tool for reading URL content.
pub struct ReadUrlTool {
    client: reqwest::Client,
}

impl ReadUrlTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    fn extract_title(&self, html: &str) -> Option<String> {
        html.find("<title>").and_then(|start| {
            html.find("</title>")
                .map(|end| html[start + 7..end].trim().to_string())
        })
    }
}

#[async_trait]
impl Tool for ReadUrlTool {
    fn name(&self) -> &str {
        "read_url"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .context("Missing 'url' field in input")?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("Failed to fetch URL")?;

        let html = response.text().await.context("Failed to read response body")?;
        let title = self.extract_title(&html).unwrap_or_else(|| url.to_string());
        let text = html2text::from_read(html.as_bytes(), 80);
        let truncated: String = text.chars().take(5000).collect();

        Ok(json!({"content": truncated, "title": title}))
    }
}

/// Tool for reading local files.
pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .context("Missing 'path' field in input")?;

        let path = Path::new(path_str);
        let range = input.get("range").and_then(|v| {
            v.as_array().and_then(|arr| {
                if arr.len() == 2 {
                    let start = arr[0].as_u64()? as usize;
                    let end = arr[1].as_u64()? as usize;
                    Some((start, end))
                } else {
                    None
                }
            })
        });

        let content = read_file(path, range)?;
        Ok(json!({"content": content}))
    }
}

/// Registry for managing and dispatching tools.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}


