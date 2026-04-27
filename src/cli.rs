use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::time::Duration;

use crossterm::{cursor::Show, event::{Event, KeyEventKind}, terminal::{disable_raw_mode, is_raw_mode_enabled, EnterAlternateScreen, LeaveAlternateScreen}, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use std::io::stdout;

use crate::agent::orchestrator::AgentEvent;
use crate::auth::login::{AuthAction, AuthLoginScreen};
use crate::auth::store::{AuthStatus, CredentialStore};
use crate::config::Config;
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
    Auth { #[arg(long)] list: bool },
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
    stdout().execute(LeaveAlternateScreen).ok();
    disable_raw_mode().ok();
    stdout().execute(Show).ok();
}

pub fn fallback_models(provider: &str) -> Vec<String> {
    match provider {
        "openai" => vec!["gpt-4o".into(), "gpt-4o-mini".into(), "gpt-4-turbo".into(), "gpt-3.5-turbo".into()],
        "anthropic" => vec![
            "claude-opus-4-5".into(),
            "claude-sonnet-4-5".into(),
            "claude-haiku-4-5".into(),
            "claude-3-5-sonnet-latest".into(),
            "claude-3-5-haiku-latest".into(),
            "claude-3-opus-latest".into(),
        ],
        "google" => vec![
            "gemini-2.0-flash".into(),
            "gemini-1.5-pro".into(),
            "gemini-1.5-flash".into(),
        ],
        _ => vec![],
    }
}

/// Fetch model list from provider registry and emit to the given channel
pub async fn fetch_model_list(pr: Arc<ProviderRegistry>, tx: std::sync::mpsc::Sender<AgentEvent>) {
    let cred_store = CredentialStore::load().unwrap_or_default();
    let mut all_models: Vec<String> = Vec::new();
    for provider_name in &["anthropic", "openai", "google", "ollama"] {
        if let Some(handle) = pr.get(provider_name) {
            if let Some(creds) = cred_store.get(provider_name) {
                let mut w = handle.write().await;
                let _ = w.authenticate(&creds.api_key).await;
            }
            let lock = handle.read().await;
            let models = match timeout(Duration::from_secs(5), lock.list_models()).await {
                Ok(Ok(list)) => list,
                _ => fallback_models(provider_name),
            };
            all_models.extend(models);
        }
    }
    let _ = tx.send(AgentEvent::ModelListReady(all_models));
}

pub async fn handle_command(cmd: Commands) -> Result<()> {
    match cmd {
        Commands::Auth { list } => {
            if list {
                let store = CredentialStore::load().unwrap_or_default();
                for provider in ["openai", "anthropic", "gemini", "ollama"] {
                    let status = match store.status(provider) {
                        AuthStatus::Authenticated => "authenticated",
                        AuthStatus::NotAuthenticated => "not authenticated",
                        AuthStatus::Expired => "expired",
                    };
                    println!("{}: {}", provider, status);
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
    let mut screen = AuthLoginScreen::new()?;
    loop {
        terminal.draw(|frame| screen.render(frame, frame.area()))?;
        if let Ok(Event::Key(key_event)) = crossterm::event::read() {
            if key_event.kind == KeyEventKind::Press {
                match screen.handle_key(key_event) {
                    AuthAction::Quit | AuthAction::Done => break,
                    AuthAction::Continue => {}
                }
            }
        }
    }
    restore_terminal();
    Ok(())
}
