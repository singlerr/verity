use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

pub fn write_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context(format!("Failed to create directory: {}", parent.display()))?;
    }
    fs::write(path, content).context(format!("Failed to write file: {}", path.display()))
}

pub fn create_backup(path: &Path) -> Result<PathBuf> {
    let backup_path = PathBuf::from(format!("{}.verity.bak", path.display()));
    fs::copy(path, &backup_path).context(format!("Failed to create backup: {}", path.display()))?;
    Ok(backup_path)
}