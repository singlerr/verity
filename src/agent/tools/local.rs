//! Local file and shell tool implementations.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;

use super::Tool;
use crate::agent::tools::registry::format_dir_tree;

/// Tool for writing content to a local file.
pub struct WriteFileTool;

impl WriteFileTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .context("Missing 'path' field")?;
        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .context("Missing 'content' field")?;
        crate::fs::write_file(Path::new(path_str), content)?;
        Ok(json!({"success": true, "path": path_str}))
    }
}

/// Tool for listing directory contents as a tree.
pub struct ListDirTool;

impl ListDirTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let path_str = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let tree = crate::fs::walk_dir(Path::new(path_str))?;
        Ok(json!({"tree": format_dir_tree(&tree, 0)}))
    }
}

/// Tool for executing shell commands in the current working directory.
pub struct ShellTool;

impl ShellTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let cmd = input
            .get("command")
            .and_then(|v| v.as_str())
            .context("Missing 'command' field")?;
        let cwd = std::env::current_dir().context("Failed to get current directory")?;
        let out = crate::shell::execute(cmd, &cwd).await?;
        Ok(json!({
            "stdout": out.stdout,
            "stderr": out.stderr,
            "exit_code": out.exit_code,
        }))
    }
}
