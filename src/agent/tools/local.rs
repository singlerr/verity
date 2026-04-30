//! Local file and shell tool implementations.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use super::Tool;
use crate::agent::tools::registry::format_dir_tree;

pub fn sandbox_path(path: &Path) -> Result<PathBuf> {
    let cwd = std::fs::canonicalize(std::env::current_dir()?)
        .context("Failed to canonicalize current directory")?;

    let full_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };

    let canonical = if full_path.exists() {
        std::fs::canonicalize(&full_path)
            .context(format!("Failed to canonicalize path: {}", full_path.display()))?
    } else {
        let parent = full_path
            .parent()
            .context(format!("Path has no parent directory: {}", full_path.display()))?;
        let file_name = full_path
            .file_name()
            .context(format!("Path has no file name: {}", full_path.display()))?;
        let canonical_parent = std::fs::canonicalize(parent)
            .context(format!("Failed to canonicalize parent: {}", parent.display()))?;
        canonical_parent.join(file_name)
    };

    if !canonical.starts_with(&cwd) {
        anyhow::bail!("Path outside working directory: {}", path.display());
    }

    Ok(canonical)
}

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
        let safe_path = sandbox_path(Path::new(path_str))?;
        crate::fs::write_file(&safe_path, content)?;
        Ok(json!({"success": true, "path": path_str}))
    }
}

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
        let safe_path = sandbox_path(Path::new(path_str))?;
        let tree = crate::fs::walk_dir(&safe_path)?;
        Ok(json!({"tree": format_dir_tree(&tree, 0, 3)}))
    }
}

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

pub struct GrepTool;

impl GrepTool {
    pub fn new() -> Self {
        Self
    }

    fn is_binary(data: &[u8]) -> bool {
        const BINARY_CHECK_SIZE: usize = 8192;
        let check_len = data.len().min(BINARY_CHECK_SIZE);
        data[..check_len].contains(&0)
    }

    fn should_skip_dir(name: &str) -> bool {
        matches!(name, "target" | "node_modules" | ".git")
    }

    fn grep_file(&self, path: &Path, pattern: &str) -> Result<Vec<(usize, String)>> {
        let content = std::fs::read(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        if Self::is_binary(&content) {
            return Ok(Vec::new());
        }

        let text = String::from_utf8_lossy(&content);
        let mut matches = Vec::new();

        for (line_num, line) in text.lines().enumerate() {
            if line.contains(pattern) {
                matches.push((line_num + 1, line.to_string()));
            }
        }

        Ok(matches)
    }

    fn grep_recursive(
        &self,
        dir: &Path,
        pattern: &str,
        results: &mut Vec<String>,
        total_count: &mut usize,
    ) -> Result<()> {
        const MAX_MATCHES: usize = 50;

        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry
                .file_name()
                .into_string()
                .unwrap_or_else(|_| String::new());

            if path.is_dir() {
                if !Self::should_skip_dir(&file_name) {
                    self.grep_recursive(&path, pattern, results, total_count)?;
                }
            } else if path.is_file() {
                match self.grep_file(&path, pattern) {
                    Ok(file_matches) => {
                        for (line_num, line) in file_matches {
                            *total_count += 1;
                            if results.len() < MAX_MATCHES {
                                let relative_path = path.strip_prefix(&std::env::current_dir()?).unwrap_or(&path);
                                results.push(format!("{}:{}: {}", relative_path.display(), line_num, line));
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let pattern = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .context("Missing 'pattern' field")?;

        let path_str = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let safe_path = sandbox_path(Path::new(path_str))?;

        let mut results = Vec::new();
        let mut total_count = 0usize;
        const MAX_MATCHES: usize = 50;

        if safe_path.is_file() {
            match self.grep_file(&safe_path, pattern) {
                Ok(file_matches) => {
                    for (line_num, line) in file_matches {
                        total_count += 1;
                        if results.len() < MAX_MATCHES {
                            let relative_path = safe_path.strip_prefix(&std::env::current_dir()?).unwrap_or(&safe_path);
                            results.push(format!("{}:{}: {}", relative_path.display(), line_num, line));
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        } else if safe_path.is_dir() {
            self.grep_recursive(&safe_path, pattern, &mut results, &mut total_count)?;
        } else {
            anyhow::bail!("Path does not exist: {}", safe_path.display());
        }

        const MAX_CHARS: usize = 10_000;

        let mut output = results.join("\n");

        if output.len() > MAX_CHARS {
            let truncated: String = output.chars().take(MAX_CHARS).collect();
            output = format!("{}\n\n[output truncated]", truncated);
        }

        if total_count >= MAX_MATCHES {
            output.push_str(&format!("\n\n[{} more matches not shown]", total_count - MAX_MATCHES + 1));
        }

        Ok(json!({
            "matches": output,
            "count": total_count
        }))
    }
}

/// Tool for editing files by replacing exact string matches.
pub struct EditFileTool;

impl EditFileTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .context("Missing 'path' field")?;
        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .context("Missing 'old_string' field")?;
        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .context("Missing 'new_string' field")?;
        let replace_all = input
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let safe_path = sandbox_path(Path::new(path_str))?;

        let content = std::fs::read_to_string(&safe_path)
            .context(format!("Failed to read file: {}", safe_path.display()))?;

        let count = content.matches(old_string).count();

        if count == 0 {
            anyhow::bail!("old_string not found in file");
        }

        if count > 1 && !replace_all {
            anyhow::bail!(
                "Found {} occurrences of old_string. Set replace_all to true to replace all, or provide a more specific old_string.",
                count
            );
        }

        let new_content = content.replace(old_string, new_string);
        std::fs::write(&safe_path, new_content)
            .context(format!("Failed to write file: {}", safe_path.display()))?;

        Ok(json!({"success": true, "path": path_str, "replacements": count}))
    }
}

const GLOB_SKIP_DIRS: &[&str] = &["target", "node_modules", ".git"];

pub struct GlobTool;

impl GlobTool {
    pub fn new() -> Self {
        Self
    }

    fn matches_pattern(path: &Path, pattern: &str) -> bool {
        let pattern = pattern.trim();

        if let Some(remainder) = pattern.strip_prefix("**/") {
            if remainder.starts_with('*') {
                let ext = &remainder[1..];
                if !ext.contains('*') {
                    return path
                        .file_name()
                        .map(|n| n.to_string_lossy().ends_with(ext))
                        .unwrap_or(false);
                }
            } else if remainder.ends_with('*') {
                let prefix = &remainder[..remainder.len() - 1];
                return path
                    .file_name()
                    .map(|n| n.to_string_lossy().starts_with(prefix))
                    .unwrap_or(false);
            }
            return false;
        }

        if !pattern.contains("**/") {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let ext = parts[1];
                if parts[0].is_empty() && !ext.is_empty() {
                    return path
                        .file_name()
                        .map(|n| n.to_string_lossy().ends_with(ext))
                        .unwrap_or(false);
                }
            }
            if parts.len() == 3 && parts[0].is_empty() && parts[2].is_empty() {
                let middle = parts[1];
                if !middle.is_empty() {
                    return path
                        .file_name()
                        .map(|n| n.to_string_lossy().contains(middle))
                        .unwrap_or(false);
                }
            }
        }

        false
    }

    fn walk_and_glob(dir: &Path, base_dir: &Path, pattern: &str, results: &mut Vec<String>, limit: usize) {
        if results.len() >= limit {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.filter_map(|e| e.ok()) {
            if results.len() >= limit {
                return;
            }

            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();

            if file_name.starts_with('.') {
                continue;
            }

            if GLOB_SKIP_DIRS.contains(&file_name.as_str()) {
                continue;
            }

            if path.is_dir() {
                Self::walk_and_glob(&path, base_dir, pattern, results, limit);
            } else if path.is_file() {
                if Self::matches_pattern(&path, pattern) {
                    if let Ok(relative) = path.strip_prefix(base_dir) {
                        let normalized = relative.to_string_lossy().replace('\\', "/");
                        results.push(normalized);
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let pattern = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .context("Missing 'pattern' field")?;

        let path_str = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let safe_path = sandbox_path(Path::new(path_str))?;

        let mut results: Vec<String> = Vec::new();
        const LIMIT: usize = 100;
        Self::walk_and_glob(&safe_path, &safe_path, pattern, &mut results, LIMIT);

        let count = results.len();
        let files = results;

        Ok(json!({
            "files": files,
            "count": count,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::fmt::Write;

    fn create_dir_tree(tmpdir: &std::path::Path, levels: usize) {
        if levels == 0 {
            return;
        }
        let subdir = tmpdir.join(format!("level{}", levels));
        fs::create_dir(&subdir).unwrap();
        for i in 0..3 {
            fs::write(subdir.join(format!("file{}.txt", i)), "content").unwrap();
        }
        create_dir_tree(&subdir, levels - 1);
    }

    #[tokio::test]
    async fn test_list_dir_respects_depth_limit() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        create_dir_tree(tmpdir.path(), 5);

        let tool = ListDirTool::new();
        let input = json!({"path": tmpdir.path().to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let tree = result.get("tree").unwrap().as_str().unwrap();
        assert!(tree.contains("["));
        assert!(tree.contains("items not shown]"));

        drop(tmpdir);
    }

    #[test]
    fn sandbox_allows_cwd_files() {
        let cwd = std::env::current_dir().unwrap();
        let test_file = cwd.join("Cargo.toml");
        let result = sandbox_path(Path::new("Cargo.toml"));
        assert!(result.is_ok(), "Relative path within CWD should be allowed");
        assert!(result.unwrap().starts_with(&cwd));

        let result = sandbox_path(&test_file);
        assert!(result.is_ok(), "Absolute path within CWD should be allowed");
    }

    #[test]
    fn sandbox_rejects_parent_traversal() {
        let result = sandbox_path(Path::new("../../../etc/passwd"));
        assert!(result.is_err(), "Parent traversal should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Path outside working directory"),
            "Error should mention path escape: {err_msg}"
        );
    }

    #[test]
    fn sandbox_rejects_absolute_outside() {
        let result = sandbox_path(Path::new("/etc/passwd"));
        assert!(result.is_err(), "Absolute path outside CWD should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Path outside working directory"),
            "Error should mention path escape: {err_msg}"
        );
    }

    #[test]
    fn sandbox_rejects_symlink_escape() {
        let cwd = std::env::current_dir().unwrap();
        let inner_link = cwd.join("sandbox_test_escape_link");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc", &inner_link).unwrap();
            let result = sandbox_path(&inner_link);
            assert!(
                result.is_err(),
                "Symlink inside CWD pointing outside should be rejected"
            );
            let _ = fs::remove_file(&inner_link);
        }
    }

    #[tokio::test]
    async fn grep_finds_pattern_in_file() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("test.txt");
        fs::write(&test_file, "line 1\nhello world\nline 3\nhello rust\n").unwrap();

        let tool = GrepTool::new();
        let input = json!({"pattern": "hello", "path": tmpdir.path().to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let matches = result.get("matches").unwrap().as_str().unwrap();
        assert!(matches.contains("hello world"));
        assert!(matches.contains("hello rust"));
        assert!(matches.contains(":2:"));
        assert!(matches.contains(":4:"));

        let count = result.get("count").unwrap().as_u64().unwrap();
        assert_eq!(count, 2);

        drop(tmpdir);
    }

    #[tokio::test]
    async fn grep_returns_limited_matches() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("test.txt");
        let mut content = String::new();
        for i in 0..60 {
            let _ = writeln!(content, "match line {}", i);
        }
        fs::write(&test_file, &content).unwrap();

        let tool = GrepTool::new();
        let input = json!({"pattern": "match", "path": tmpdir.path().to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let matches = result.get("matches").unwrap().as_str().unwrap();
        assert!(matches.contains("[11 more matches not shown]"));

        let count = result.get("count").unwrap().as_u64().unwrap();
        assert_eq!(count, 60);

        drop(tmpdir);
    }

    #[tokio::test]
    async fn grep_skips_binary_files() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        
        let binary_file = tmpdir.path().join("binary.bin");
        let mut binary_content = vec![0u8; 100];
        binary_content[50] = b'X';
        fs::write(&binary_file, &binary_content).unwrap();

        let text_file = tmpdir.path().join("text.txt");
        fs::write(&text_file, "hello world\n").unwrap();

        let tool = GrepTool::new();
        let input = json!({"pattern": "hello", "path": tmpdir.path().to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let matches = result.get("matches").unwrap().as_str().unwrap();
        assert!(!matches.contains("binary.bin"));
        assert!(matches.contains("text.txt"));

        drop(tmpdir);
    }

    #[tokio::test]
    async fn grep_rejects_outside_cwd() {
        let tool = GrepTool::new();
        let input = json!({"pattern": "test", "path": "/etc/passwd"});
        let result = tool.execute(&input).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Path outside working directory"));
    }

    #[tokio::test]
    async fn edit_file_replaces_single() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("test.txt");
        fs::write(&test_file, "Hello World").unwrap();

        let tool = EditFileTool::new();
        let input = json!({
            "path": test_file.to_str().unwrap(),
            "old_string": "Hello",
            "new_string": "Hi"
        });
        let result = tool.execute(&input).await.unwrap();

        assert_eq!(result.get("success").unwrap(), true);
        assert_eq!(result.get("path").unwrap().as_str().unwrap(), test_file.to_str().unwrap());
        assert_eq!(result.get("replacements").unwrap(), 1);

        let content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "Hi World");
    }

    #[tokio::test]
    async fn edit_file_rejects_ambiguous() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("test.txt");
        fs::write(&test_file, "foo bar foo").unwrap();

        let tool = EditFileTool::new();
        let input = json!({
            "path": test_file.to_str().unwrap(),
            "old_string": "foo",
            "new_string": "baz"
        });
        let result = tool.execute(&input).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Found 2 occurrences of old_string"));
        assert!(err_msg.contains("Set replace_all to true"));
    }

    #[tokio::test]
    async fn edit_file_replace_all() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("test.txt");
        fs::write(&test_file, "foo bar foo").unwrap();

        let tool = EditFileTool::new();
        let input = json!({
            "path": test_file.to_str().unwrap(),
            "old_string": "foo",
            "new_string": "baz",
            "replace_all": true
        });
        let result = tool.execute(&input).await.unwrap();

        assert_eq!(result.get("success").unwrap(), true);
        assert_eq!(result.get("replacements").unwrap(), 2);

        let content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "baz bar baz");
    }

    #[tokio::test]
    async fn edit_file_rejects_outside_cwd() {
        let tool = EditFileTool::new();
        let input = json!({
            "path": "../../../etc/passwd",
            "old_string": "root",
            "new_string": "admin"
        });
        let result = tool.execute(&input).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Path outside working directory"));
    }

    #[tokio::test]
    async fn edit_file_rejects_missing_string() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("test.txt");
        fs::write(&test_file, "Hello World").unwrap();

        let tool = EditFileTool::new();
        let input = json!({
            "path": test_file.to_str().unwrap(),
            "old_string": "Goodbye",
            "new_string": "Hi"
        });
        let result = tool.execute(&input).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("old_string not found in file"));
    }

    #[tokio::test]
    async fn glob_finds_rs_files() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let subdir = tmpdir.path().join("src");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("main.rs"), "fn main() {}").unwrap();
        fs::write(subdir.join("lib.rs"), "pub fn foo() {}").unwrap();
        fs::write(tmpdir.path().join("Cargo.toml"), "[package]").unwrap();

        let tool = GlobTool::new();
        let input = json!({"pattern": "**/*.rs", "path": tmpdir.path().to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let files = result.get("files").unwrap().as_array().unwrap();
        assert_eq!(files.len(), 2);
        let counts = result.get("count").unwrap().as_i64().unwrap();
        assert_eq!(counts, 2);

        drop(tmpdir);
    }

    #[tokio::test]
    async fn glob_finds_toml_files() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        fs::write(tmpdir.path().join("Cargo.toml"), "[package]").unwrap();
        fs::write(tmpdir.path().join("workspace.toml"), "[workspace]").unwrap();
        let subdir = tmpdir.path().join("nested");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("config.toml"), "[config]").unwrap();

        let tool = GlobTool::new();
        let input = json!({"pattern": "**/*.toml", "path": tmpdir.path().to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let files = result.get("files").unwrap().as_array().unwrap();
        assert_eq!(files.len(), 3);

        drop(tmpdir);
    }

    #[tokio::test]
    async fn glob_returns_limited_results() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        for i in 0..150 {
            fs::write(tmpdir.path().join(format!("file{}.txt", i)), "content").unwrap();
        }

        let tool = GlobTool::new();
        let input = json!({"pattern": "*.txt", "path": tmpdir.path().to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let files = result.get("files").unwrap().as_array().unwrap();
        assert_eq!(files.len(), 100);

        drop(tmpdir);
    }

    #[tokio::test]
    async fn glob_rejects_outside_cwd() {
        let tool = GlobTool::new();
        let input = json!({"pattern": "**/*.rs", "path": "/etc"});
        let result = tool.execute(&input).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("outside working directory") || err_msg.contains("outside"));

        drop(tool);
    }
}
