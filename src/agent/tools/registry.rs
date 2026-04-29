//! Tool registry and manifest utilities.

use std::sync::Arc;

use crate::search::{SearXngClient, SearchEngine};

use super::{
    ListDirTool, ReadFileTool, ReadUrlTool, SearchTool, ShellTool, ToolRegistry, WriteFileTool,
};
use crate::fs::FileTree;

/// Build a ToolRegistry with default tools configured.
pub fn build_tool_registry(searxng_url: &str) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    let search_engine: Arc<dyn SearchEngine> = Arc::new(SearXngClient::new(searxng_url));
    registry.register(SearchTool::new(search_engine));
    registry.register(ReadUrlTool::new());
    registry.register(ReadFileTool::new());
    registry.register(WriteFileTool::new());
    registry.register(ListDirTool::new());
    registry.register(ShellTool::new());
    registry
}

/// Returns a human-readable tool manifest for use in system prompts.
pub fn tool_manifest() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    format!(
        "Current working directory: {cwd}\n\n\
         Available tools:\n\
         - search(query) — search the web\n\
         - read_url(url) — fetch and read a web page\n\
         - read_file(path, range?) — read a local file; range is optional [start_line, end_line]\n\
         - write_file(path, content) — write or overwrite a local file\n\
         - list_dir(path?) — list directory tree; defaults to current directory\n\
         - shell(command) — run a shell command in the current directory",
        cwd = cwd
    )
}

// Default impls for tool types

impl Default for ReadUrlTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ListDirTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}

// Helper for ListDirTool

fn format_tree(tree: &FileTree, depth: usize) -> String {
    let indent = "  ".repeat(depth);
    let marker = if tree.is_dir { "[dir] " } else { "" };
    let mut out = format!("{}{}{}\n", indent, marker, tree.name);
    for child in &tree.children {
        out.push_str(&format_tree(child, depth + 1));
    }
    out
}

pub(crate) fn format_dir_tree(tree: &FileTree, depth: usize) -> String {
    format_tree(tree, depth)
}
