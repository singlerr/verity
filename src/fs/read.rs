use std::fs;
use std::path::Path;
use anyhow::{Context, Result};

const BINARY_CHECK_SIZE: usize = 8192;

pub fn read_file(path: &Path, range: Option<(usize, usize)>) -> Result<String> {
    let bytes = fs::read(path).context(format!("Failed to read file: {}", path.display()))?;

    if bytes.len() > BINARY_CHECK_SIZE {
        let check_len = BINARY_CHECK_SIZE;
        if bytes[..check_len].contains(&0) {
            anyhow::bail!("Binary file");
        }
    } else if bytes.contains(&0) {
        anyhow::bail!("Binary file");
    }

    let content = String::from_utf8_lossy(&bytes);

    match range {
        Some((start, end)) => {
            let lines: Vec<&str> = content.lines().collect();
            let start_idx = start.min(lines.len());
            let end_idx = end.min(lines.len()).max(start_idx);
            Ok(lines[start_idx..end_idx].join("\n"))
        }
        None => Ok(content.into_owned()),
    }
}