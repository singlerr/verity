use std::path::Path;
use std::time::Instant;
use anyhow::{Context, Result};
use tokio::process::Command;

const _DEFAULT_TIMEOUT_MS: u64 = 30_000;

#[derive(Debug)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub duration_ms: u64,
}

impl CommandOutput {
    pub fn is_success(&self) -> bool {
        self.exit_code == 0
    }
}

pub async fn execute(cmd: &str, cwd: &Path) -> Result<CommandOutput> {
    let start = Instant::now();

    let mut command = Command::new("sh");
    command.args(["-c", cmd]);
    command.current_dir(cwd);
    command.kill_on_drop(true);

    let output = command
        .output()
        .await
        .context("Failed to execute command")?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    Ok(CommandOutput {
        stdout,
        stderr,
        exit_code,
        duration_ms,
    })
}