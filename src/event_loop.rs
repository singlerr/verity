//! Async event loop for the Verity TUI.

use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::backend::CrosstermBackend;

use crate::agent::build_tool_registry;
use crate::agent::orchestrator::{AgentEvent, AgentOrchestrator};
use crate::agent::planner::AgentPlanner;
use crate::app::{App, AppState, Focus, Mode};
use crate::config::Config;
use crate::ui::{render_error_overlay, render_layout, render_model_select_popup};

pub async fn run_event_loop(
    terminal: &mut ratatui::Terminal<CrosstermBackend<std::io::Stdout>>,
    mut app: App,
    mut agent_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    agent_tx: std::sync::mpsc::Sender<AgentEvent>,
    _cancel_token: Arc<Mutex<Option<CancellationToken>>>,
) -> Result<()> {
    loop {
        terminal.draw(|frame| {
            render_layout(frame, &app);
            if let AppState::Error(ref msg) = app.state {
                render_error_overlay(frame, frame.area(), "Error", msg);
            }
            if app.model_select_open {
                render_model_select_popup(
                    frame,
                    frame.area(),
                    &app.model_list,
                    app.selected_model_idx,
                );
            }
        })?;
        tokio::select! {
            poll_res = tokio::task::spawn_blocking(|| crossterm::event::poll(Duration::from_millis(50))) => {
                if let Ok(Ok(true)) = poll_res {
                    if let Ok(event) = crossterm::event::read() {
                        if let Event::Resize(_, _) = event { continue; }
                        if let Event::Key(KeyEvent { kind: KeyEventKind::Press, code, modifiers, .. }) = event {
                            if let AppState::Error(_) = app.state { app.state = AppState::Idle; continue; }
                            if app.model_select_open { handle_model_select(&mut app, code); continue; }
                            if !handle_global_key(code, modifiers, &mut app, &agent_tx) {
                                handle_focus_key(code, &mut app);
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

fn open_model_select(app: &mut App, agent_tx: &std::sync::mpsc::Sender<AgentEvent>) {
    app.model_select_open = true;
    app.model_list.clear();
    app.selected_model_idx = 0;
    let pr = Arc::new(crate::llm::build_registry());
    let tx = agent_tx.clone();
    tokio::spawn(crate::cli::fetch_model_list(pr, tx));
}

fn set_model(app: &mut App, model: String) {
    let mut config = Config::load().unwrap_or_default();
    config.active_model = model.clone();
    let _ = config.save();
    app.active_model = model;
}

fn handle_model_select(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Char('j') | KeyCode::Down if !app.model_list.is_empty() => {
            app.selected_model_idx = (app.selected_model_idx + 1) % app.model_list.len()
        }
        KeyCode::Char('k') | KeyCode::Up => {
            app.selected_model_idx = app.selected_model_idx.saturating_sub(1)
        }
        KeyCode::Enter => {
            if let Some(entry) = app.model_list.get(app.selected_model_idx) {
                set_model(app, entry.config_id());
            }
            app.model_select_open = false;
        }
        KeyCode::Esc => app.model_select_open = false,
        _ => {}
    }
}

fn autocomplete_next(app: &mut App, forward: bool) {
    let matches = crate::ui::autocomplete::matching(&app.query);
    if matches.is_empty() {
        return;
    }
    app.autocomplete_idx = if forward {
        (app.autocomplete_idx + 1) % matches.len()
    } else {
        app.autocomplete_idx.wrapping_sub(1).min(matches.len() - 1)
    };
    app.query = crate::ui::autocomplete::COMMANDS[matches[app.autocomplete_idx]]
        .0
        .to_string();
}

fn handle_global_key(
    code: KeyCode,
    mods: crossterm::event::KeyModifiers,
    app: &mut App,
    tx: &std::sync::mpsc::Sender<AgentEvent>,
) -> bool {
    let ctrl = mods.contains(crossterm::event::KeyModifiers::CONTROL);
    let in_cmd = app.focus == Focus::Command;
    match code {
        _ if ctrl && matches!(code, KeyCode::Char('d' | 'c')) => std::process::exit(0),
        KeyCode::Char('q') if !in_cmd => std::process::exit(0),
        KeyCode::Char('/') if !in_cmd => {
            app.focus = Focus::Command;
            app.query.push('/');
            true
        }
        KeyCode::Char('m') if !in_cmd => {
            open_model_select(app, tx);
            true
        }
        KeyCode::Esc if in_cmd => {
            app.focus = Focus::Left;
            true
        }
        KeyCode::Enter if in_cmd && !app.query.is_empty() => {
            handle_command(app, tx);
            true
        }
        KeyCode::Tab | KeyCode::Down if in_cmd => {
            autocomplete_next(app, true);
            true
        }
        KeyCode::Up if in_cmd => {
            autocomplete_next(app, false);
            true
        }
        KeyCode::Char(c) if in_cmd => {
            app.query.push(c);
            app.autocomplete_idx = 0;
            true
        }
        KeyCode::Backspace if in_cmd => {
            app.query.pop();
            app.autocomplete_idx = 0;
            true
        }
        _ => false,
    }
}

fn handle_focus_key(code: KeyCode, app: &mut App) {
    let f = app.focus.clone();
    match code {
        KeyCode::Tab if f != Focus::Command => {
            app.focus = if f == Focus::Left {
                Focus::Right
            } else {
                Focus::Left
            }
        }
        KeyCode::Char('j') if f == Focus::Left => {
            if app.selected_source.is_none() {
                app.selected_source = Some(0);
            } else if let Some(s) = app.selected_source {
                if s + 1 < app.sources.len() {
                    app.selected_source = Some(s + 1);
                }
            }
        }
        KeyCode::Char('k') if f == Focus::Left => {
            if app.selected_source == Some(0) {
                app.selected_source = None;
            } else if let Some(s) = app.selected_source {
                app.selected_source = Some(s.saturating_sub(1));
            }
        }
        KeyCode::Char('j') | KeyCode::Down if f == Focus::Right => {
            app.answer_scroll = app.answer_scroll.saturating_add(1)
        }
        KeyCode::Char('k') | KeyCode::Up if f == Focus::Right => {
            app.answer_scroll = app.answer_scroll.saturating_sub(1)
        }
        _ => {}
    }
}

fn handle_command(app: &mut App, agent_tx: &std::sync::mpsc::Sender<AgentEvent>) {
    let trimmed = app.query.trim().to_string();
    let (cmd, args) = trimmed
        .split_once(char::is_whitespace)
        .unwrap_or((&trimmed, ""));
    let args = args.trim();
    match cmd {
        "/model" => {
            app.query.clear();
            app.focus = Focus::Left;
            if args.is_empty() {
                open_model_select(app, agent_tx);
            } else {
                set_model(app, args.to_string());
            }
        }
        "/mode" => {
            app.mode = match args {
                "research" => Mode::Research,
                "code" => Mode::Code,
                _ => app.mode.clone(),
            };
            app.query.clear();
            app.focus = Focus::Left;
        }
        _ => {
            let query = app.query.clone();
            app.submit_query();
            let tx = agent_tx.clone();
            let config = Config::load().unwrap_or_default();
            let pr = Arc::new(crate::llm::build_registry());
            let planner = AgentPlanner::new(pr.clone(), config.active_model.clone());
            let tools = build_tool_registry(&config.searxng_url);
            let model = config.active_model.clone();
            let cancel_token = tokio_util::sync::CancellationToken::new();
            tokio::spawn(async move {
                AgentOrchestrator::new(planner, tools, pr, model)
                    .run(&query, tx, cancel_token)
                    .await;
            });
        }
    }
}
