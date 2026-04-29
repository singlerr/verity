//! Verity TUI — Entry point.

use anyhow::Result;
use clap::Parser;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio_util::sync::CancellationToken;

use verity::agent::orchestrator::AgentEvent;
use verity::app::App;
use verity::event_loop::run_event_loop;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = verity::cli::Cli::parse();
    if let Some(query) = cli.ask {
        eprintln!("[verity] ask mode: {}", query);
        return Ok(());
    }
    if let Some(cmd) = cli.command {
        verity::cli::handle_command(cmd).await?;
        return Ok(());
    }
    let mut terminal = verity::cli::setup_terminal()?;
    let (std_tx, std_rx) = std::sync::mpsc::channel::<AgentEvent>();
    let (tokio_tx, tokio_rx) = tokio::sync::mpsc::channel::<AgentEvent>(100);
    thread::spawn(move || {
        while let Ok(event) = std_rx.recv() {
            if tokio_tx.blocking_send(event).is_err() {
                break;
            }
        }
    });
    let app = App::new();
    let cancel_token: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
    if let Err(e) = run_event_loop(&mut terminal, app, tokio_rx, std_tx, cancel_token).await {
        verity::cli::restore_terminal();
        return Err(anyhow::anyhow!("Event loop error: {}", e));
    }
    verity::cli::restore_terminal();
    Ok(())
}
