use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    ".vscode",
    "__pycache__",
    ".idea",
    "dist",
    "build",
    ".cache",
    ".temp",
    ".tmp",
];

#[derive(Debug, Clone)]
pub struct FileTree {
    pub name: String,
    pub children: Vec<FileTree>,
    pub is_dir: bool,
    pub size: u64,
}

impl FileTree {
    fn new(name: String, is_dir: bool) -> Self {
        Self {
            name,
            children: Vec::new(),
            is_dir,
            size: 0,
        }
    }

    fn with_children(name: String, is_dir: bool, children: Vec<FileTree>, size: u64) -> Self {
        Self {
            name,
            children,
            is_dir,
            size,
        }
    }
}

fn should_skip(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

pub fn walk_dir(path: &Path) -> Result<FileTree> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());

    let metadata =
        fs::metadata(path).context(format!("Failed to read metadata: {}", path.display()))?;

    if metadata.is_file() {
        return Ok(FileTree::with_children(
            name,
            false,
            Vec::new(),
            metadata.len(),
        ));
    }

    let mut children = Vec::new();
    let mut total_size = 0u64;

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            tracing::warn!("Permission denied: {}", path.display());
            return Ok(FileTree::new(name, true));
        }
        Err(e) => return Err(e).context(format!("Failed to read directory: {}", path.display())),
    };

    for entry in entries.filter_map(|e| e.ok()) {
        let entry_path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();

        if file_name.starts_with('.') && file_name != "." {
            continue;
        }
        if should_skip(&file_name) {
            continue;
        }

        match walk_dir(&entry_path) {
            Ok(child) => {
                total_size += child.size;
                children.push(child);
            }
            Err(e) => {
                tracing::warn!("Skipping {}: {}", entry_path.display(), e);
            }
        }
    }

    children.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    Ok(FileTree::with_children(name, true, children, total_size))
}
