//! Verity TUI — Entry point with terminal setup and async event loop.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossterm::{
    cursor::Show,
    event::{Event, KeyCode, KeyEvent, KeyEventKind},
    terminal::{disable_raw_mode, is_raw_mode_enabled, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::backend::CrosstermBackend;
use std::io::stdout;

use verity::agent::orchestrator::{AgentEvent, AgentOrchestrator};
use verity::agent::planner::AgentPlanner;
use verity::agent::build_tool_registry;
use verity::app::{App, AppState, Focus};
use verity::auth::login::{AuthAction, AuthLoginScreen};
use verity::auth::store::{AuthStatus, CredentialStore};
use verity::config::Config;
use verity::ui::{render_error_overlay, render_layout, render_model_select_popup};

#[derive(Parser)]
#[command(name = "verity", version = "0.1.0", about = "AI Harness TUI — Research & Code")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    #[arg(short, long)]
    ask: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    Auth { #[arg(long)] list: bool },
}

struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if is_raw_mode_enabled().unwrap_or(false) {
            std::io::stdout().execute(LeaveAlternateScreen).ok();
            disable_raw_mode().ok();
            std::io::stdout().execute(Show).ok();
        }
    }
}

fn setup_terminal() -> Result<ratatui::Terminal<CrosstermBackend<std::io::Stdout>>> {
    stdout().execute(EnterAlternateScreen)?;
    stdout().execute(crossterm::cursor::Hide)?;
    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(std::io::stdout());
    Ok(ratatui::Terminal::new(backend)?)
}

fn restore_terminal() {
    stdout().execute(LeaveAlternateScreen).ok();
    disable_raw_mode().ok();
    stdout().execute(Show).ok();
}

async fn run_event_loop(
    terminal: &mut ratatui::Terminal<CrosstermBackend<std::io::Stdout>>,
    mut app: App,
    mut agent_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    agent_tx: std::sync::mpsc::Sender<AgentEvent>,
) -> Result<()> {
    loop {
        terminal.draw(|frame| {
            render_layout(frame, &app);
            if let AppState::Error(ref msg) = app.state {
                render_error_overlay(frame, frame.area(), "Error", msg);
            }
            if app.model_select_open {
                render_model_select_popup(frame, frame.area(), &app.model_list, app.selected_model_idx);
            }
        })?;
        tokio::select! {
            poll_res = tokio::task::spawn_blocking(|| crossterm::event::poll(Duration::from_millis(50))) => {
                if let Ok(Ok(true)) = poll_res {
                    if let Ok(event) = crossterm::event::read() {
                        // Handle resize — next draw uses new dimensions automatically
                        if let Event::Resize(_, _) = event { continue; }
                        if let Event::Key(KeyEvent { kind: KeyEventKind::Press, code, modifiers, .. }) = event {
                            // Dismiss error overlay on any key
                            if let AppState::Error(_) = app.state {
                                app.state = AppState::Idle;
                                continue;
                            }
                            if app.model_select_open {
                                match code {
                                    KeyCode::Char('j') => {
                                        if !app.model_list.is_empty() {
                                            app.selected_model_idx = (app.selected_model_idx + 1) % app.model_list.len();
                                        }
                                    }
                                    KeyCode::Char('k') => {
                                        app.selected_model_idx = app.selected_model_idx.saturating_sub(1);
                                    }
                                    KeyCode::Enter => {
                                        if let Some(model) = app.model_list.get(app.selected_model_idx) {
                                            let mut config = Config::load().unwrap_or_default();
                                            config.active_model = model.clone();
                                            let _ = config.save();
                                            app.active_model = model.clone();
                                        }
                                        app.model_select_open = false;
                                    }
                                    KeyCode::Esc => app.model_select_open = false,
                                    _ => {}
                                }
                                continue;
                            }
                            match (code, modifiers) {
                                (KeyCode::Char('d'), m) if m.contains(crossterm::event::KeyModifiers::CONTROL) => return Ok(()),
                                (KeyCode::Char('c'), m) if m.contains(crossterm::event::KeyModifiers::CONTROL) => return Ok(()),
                                (KeyCode::Char('q'), _) if app.focus != Focus::Command => return Ok(()),
                                (KeyCode::Char('/'), _) if app.focus != Focus::Command => app.focus = Focus::Command,
                                (KeyCode::Char('m'), _) if app.focus != Focus::Command => {
                                    app.model_select_open = true;
                                    app.model_list.clear();
                                    app.selected_model_idx = 0;
                                    let pr = Arc::new(verity::llm::build_registry());
                                    let model = Config::load().unwrap_or_default().active_model.clone();
                                    let tx = agent_tx.clone();
                                    tokio::spawn(async move {
                                        if let Some(handle) = pr.resolve(&model) {
                                            let lock = handle.read().await;
                                            if let Ok(models) = lock.list_models().await {
                                                let _ = tx.send(AgentEvent::ModelListReady(models));
                                            }
                                        }
                                    });
                                }
                                (KeyCode::Esc, _) => { if app.focus == Focus::Command { app.focus = Focus::Left; } }
                                (KeyCode::Enter, _) if app.focus == Focus::Command && !app.query.is_empty() => {
                                    let query = app.query.clone();
                                    app.submit_query();
                                    let tx = agent_tx.clone();
                                    let config = Config::load().unwrap_or_default();
                                    let pr = Arc::new(verity::llm::build_registry());
                                    let planner = AgentPlanner::new(pr.clone(), config.active_model.clone());
                                    let tools = build_tool_registry(&config.searxng_url);
                                    let model = config.active_model.clone();
                                    tokio::spawn(async move {
                                        AgentOrchestrator::new(planner, tools, pr, model).run(&query, tx).await;
                                    });
                                }
                                (KeyCode::Tab, _) if app.focus == Focus::Command => {
                                    let matches = verity::ui::autocomplete::matching(&app.query);
                                    if !matches.is_empty() {
                                        let idx = app.autocomplete_idx % matches.len();
                                        app.query = verity::ui::autocomplete::COMMANDS[matches[idx]].0.to_string();
                                        app.autocomplete_idx = (app.autocomplete_idx + 1) % matches.len();
                                    }
                                }
                                (KeyCode::Char(c), _) if app.focus == Focus::Command => {
                                    app.query.push(c);
                                    app.autocomplete_idx = 0;
                                }
                                (KeyCode::Backspace, _) if app.focus == Focus::Command => {
                                    app.query.pop();
                                    app.autocomplete_idx = 0;
                                }
                                (KeyCode::Char('j'), _) if app.focus != Focus::Command => {
                                    if app.selected_source.is_none() { app.selected_source = Some(0); }
                                    else if let Some(s) = app.selected_source {
                                        if s + 1 < app.sources.len() { app.selected_source = Some(s + 1); }
                                    }
                                }
                                (KeyCode::Char('k'), _) if app.focus != Focus::Command => {
                                    if app.selected_source == Some(0) { app.selected_source = None; }
                                    else if let Some(s) = app.selected_source {
                                        app.selected_source = Some(s.saturating_sub(1));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            maybe_event = agent_rx.recv() => {
                if let Some(event) = maybe_event { app.handle_event(event); }
            }
        }
    }
}

fn run_auth_screen() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let _guard = TerminalGuard;
    let mut screen = AuthLoginScreen::new()?;
    loop {
        terminal.draw(|frame| screen.render(frame, frame.area()))?;
        if let Ok(Event::Key(key_event)) = crossterm::event::read() {
            match screen.handle_key(key_event) {
                AuthAction::Quit | AuthAction::Done => break,
                AuthAction::Continue => {}
            }
        }
    }
    restore_terminal();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(query) = cli.ask {
        eprintln!("[verity] ask mode: {query}");
        return Ok(());
    }
    if let Some(cmd) = cli.command {
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
                    run_auth_screen()?;
                }
            }
        }
        return Ok(());
    }
    let _guard = TerminalGuard;
    let mut terminal = setup_terminal()?;
    let (std_tx, std_rx) = std::sync::mpsc::channel::<AgentEvent>();
    let (tokio_tx, tokio_rx) = tokio::sync::mpsc::channel::<AgentEvent>(100);
    thread::spawn(move || {
        while let Ok(event) = std_rx.recv() { if tokio_tx.blocking_send(event).is_err() { break; } }
    });
    let app = App::new();
    if let Err(e) = run_event_loop(&mut terminal, app, tokio_rx, std_tx).await {
        restore_terminal();
        return Err(anyhow::anyhow!("Event loop error: {}", e));
    }
    restore_terminal();
    Ok(())
}
