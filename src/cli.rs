use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::time::Duration;

use crossterm::{
    cursor::Show,
    event::{DisableBracketedPaste, EnableBracketedPaste, Event, KeyEventKind},
    terminal::{disable_raw_mode, is_raw_mode_enabled, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::backend::CrosstermBackend;
use std::io::stdout;

use crate::agent::orchestrator::AgentEvent;
use crate::auth::login::{AuthAction, AuthLoginScreen};
use crate::auth::store::{AuthStatus, CredentialStore};
use crate::config::Config;
use crate::llm::provider::ModelEntry;
use crate::llm::ProviderRegistry;
use tokio::time::timeout;

/// CLI entrypoint shared with the previous main.rs
#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
    #[arg(short, long)]
    pub ask: Option<String>,
}

#[derive(Subcommand)]
pub enum Commands {
    Auth {
        #[arg(long)]
        list: bool,
    },
    /// Manage configuration
    Config {
        #[command(subcommand)]
        cmd: ConfigCmd,
    },
}

#[derive(Subcommand)]
pub enum ConfigCmd {
    /// Set the SearXNG instance URL
    Searxng {
        /// SearXNG base URL (e.g. http://localhost:8080)
        url: String,
    },
    /// Show current configuration
    Show,
}

/// Guard terminal and restore on drop
struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if is_raw_mode_enabled().unwrap_or(false) {
            stdout().execute(LeaveAlternateScreen).ok();
            disable_raw_mode().ok();
            stdout().execute(Show).ok();
        }
    }
}

pub fn setup_terminal() -> Result<ratatui::Terminal<CrosstermBackend<std::io::Stdout>>> {
    stdout().execute(EnterAlternateScreen)?;
    stdout().execute(crossterm::cursor::Hide)?;
    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(std::io::stdout());
    Ok(ratatui::Terminal::new(backend)?)
}

pub fn restore_terminal() {
    stdout().execute(DisableBracketedPaste).ok();
    stdout().execute(LeaveAlternateScreen).ok();
    disable_raw_mode().ok();
    stdout().execute(Show).ok();
}

/// Fetch model list from provider registry and emit to the given channel.
pub async fn fetch_model_list(pr: Arc<ProviderRegistry>, tx: std::sync::mpsc::Sender<AgentEvent>) {
    let cred_store = CredentialStore::load().unwrap_or_default();
    let mut all_entries: Vec<ModelEntry> = Vec::new();

    for provider_name in pr.provider_names() {
        let meta = match pr.get_metadata(&provider_name) {
            Some(m) => m,
            None => continue,
        };

        if let Some(handle) = pr.get(&provider_name) {
            if meta.requires_api_key {
                if let Some(creds) = cred_store.get(&provider_name) {
                    let mut w = handle.write().await;
                    let _ = w.authenticate(&creds.api_key).await;
                }
            } else {
                // Providers like Ollama don't need API keys — just connectivity check
                let mut w = handle.write().await;
                let _ = w.authenticate("").await;
            }

            let lock = handle.read().await;
            let models = match timeout(Duration::from_secs(5), lock.list_models()).await {
                Ok(Ok(list)) => list,
                _ => meta.fallback_models.clone(),
            };
            let entries: Vec<ModelEntry> = models
                .into_iter()
                .map(|name| ModelEntry {
                    name,
                    provider: provider_name.clone(),
                })
                .collect();
            all_entries.extend(entries);
        }
    }
    let _ = tx.send(AgentEvent::ModelListReady(all_entries));
}

pub async fn handle_command(cmd: Commands) -> Result<()> {
    match cmd {
        Commands::Auth { list } => {
            if list {
                let store = CredentialStore::load().unwrap_or_default();
                let pr = crate::llm::build_registry();
                for provider_name in pr.provider_names() {
                    let display_name = pr.get_metadata(&provider_name)
                        .map(|m| m.display_name.as_str())
                        .unwrap_or(&provider_name);
                    let status = match store.status(&provider_name) {
                        AuthStatus::Authenticated => "authenticated",
                        AuthStatus::NotAuthenticated => "not authenticated",
                        AuthStatus::Expired => "expired",
                    };
                    println!("{}: {}", display_name, status);
                }
            } else {
                // Run interactive auth screen
                crate::cli::run_auth_screen()?;
            }
        }
        Commands::Config { cmd } => match cmd {
            ConfigCmd::Searxng { url } => {
                let mut config = Config::load().unwrap_or_default();
                config.searxng_url = url.clone();
                config.save()?;
                println!("searxng_url set to {}", url);
            }
            ConfigCmd::Show => {
                let config = Config::load().unwrap_or_default();
                println!("searxng_url: {}", config.searxng_url);
                println!("active_model: {}", config.active_model);
            }
        },
    }
    Ok(())
}

pub fn run_auth_screen() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let _guard = TerminalGuard;
    // Drain buffered events (e.g. the Enter that launched `verity auth`)
    while crossterm::event::poll(std::time::Duration::ZERO)? {
        crossterm::event::read()?;
    }
    stdout().execute(EnableBracketedPaste)?;
    let mut screen = AuthLoginScreen::new()?;
    loop {
        terminal.draw(|frame| screen.render(frame, frame.area()))?;
        match crossterm::event::read() {
            Ok(Event::Key(key_event)) if key_event.kind == KeyEventKind::Press => {
                match screen.handle_key(key_event) {
                    AuthAction::Quit | AuthAction::Done => break,
                    AuthAction::Continue => {}
                }
            }
            Ok(Event::Paste(text)) => {
                screen.handle_paste(text);
            }
            _ => {}
        }
    }
    restore_terminal();
    Ok(())
}
