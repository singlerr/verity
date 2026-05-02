//! TUI authentication login screen for provider selection and API key entry.
use crate::auth::store::{CredentialStore, Credentials};
use crate::ui::layout::ColorScheme;
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthAction {
    Continue,
    Done,
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InputState {
    SelectingProvider,
    EnteringKey { provider: String },
    EnteringUrl { provider: String },
}

struct ProviderInfo {
    display_name: String,
    key: String,
    requires_api_key: bool,
}

pub struct AuthLoginScreen {
    providers: Vec<ProviderInfo>,
    selection: usize,
    state: InputState,
    key_input: String,
    status: String,
    store: CredentialStore,
}

impl AuthLoginScreen {
    pub fn new() -> Result<Self> {
        let store = CredentialStore::load()?;
        let pr = crate::llm::build_registry();
        let providers: Vec<ProviderInfo> = pr.provider_names().into_iter()
            .filter_map(|name| {
                pr.get_metadata(&name).map(|meta| ProviderInfo {
                    display_name: meta.display_name.clone(),
                    key: name.clone(),
                    requires_api_key: meta.requires_api_key,
                })
            })
            .collect();
        Ok(Self {
            providers,
            selection: 0,
            state: InputState::SelectingProvider,
            key_input: String::new(),
            status: String::new(),
            store,
        })
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let c = ColorScheme::default();
        match &self.state {
            InputState::SelectingProvider => self.render_select(frame, area, &c),
            InputState::EnteringKey { provider } => self.render_input(frame, area, provider, &c, false),
            InputState::EnteringUrl { provider } => self.render_input(frame, area, provider, &c, true),
        }
    }

    fn render_select(&self, frame: &mut Frame, area: Rect, c: &ColorScheme) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .margin(2)
            .split(area);
        frame.render_widget(
            Paragraph::new("verity auth")
                .style(Style::default().fg(c.ink).add_modifier(Modifier::BOLD)),
            chunks[0],
        );
        let mut lines: Vec<Line> = vec![
            Line::from("Select a provider to authenticate:").style(Style::default().fg(c.dim)),
            Line::from(""),
        ];
        for (idx, provider) in self.providers.iter().enumerate() {
            let sel = idx == self.selection;
            let style = if sel {
                Style::default().fg(c.accent).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(c.ink)
            };
            lines.push(Line::from(vec![
                Span::styled(if sel { "❯ " } else { "  " }, style),
                Span::styled(provider.display_name.clone(), style),
            ]));
        }
        if !self.status.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(self.status.clone()).style(Style::default().fg(c.success)));
        }
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::NONE)
                    .style(Style::default().bg(c.bg)),
            ),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new("[↑/↓] navigate  [Enter] select  [q] quit")
                .style(Style::default().fg(c.dim))
                .alignment(Alignment::Center),
            chunks[2],
        );
    }

    fn render_input(&self, frame: &mut Frame, area: Rect, provider: &str, c: &ColorScheme, is_url: bool) {
        let label = if is_url { format!("Enter base URL for {}", provider) } else { format!("Enter API key for {}", provider) };
        let hint = "[Enter] submit  [Esc] back  [q] quit";
        let display_input = if is_url { self.key_input.clone() } else { self.key_input.chars().map(|_| '•').collect::<String>() };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(4),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .margin(2)
            .split(area);
        frame.render_widget(
            Paragraph::new(label)
                .style(Style::default().fg(c.ink).add_modifier(Modifier::BOLD)),
            chunks[0],
        );
        let input = Paragraph::new(display_input)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(c.accent))
                    .style(Style::default().bg(c.header_bg)),
            )
            .style(Style::default().fg(c.ink));
        frame.render_widget(input, chunks[1]);
        if !self.status.is_empty() {
            let s = if self.status.starts_with("Error") || self.status.starts_with("Invalid") {
                Style::default().fg(c.accent)
            } else {
                Style::default().fg(c.success)
            };
            frame.render_widget(Paragraph::new(self.status.clone()).style(s), chunks[2]);
        }
        frame.render_widget(
            Paragraph::new(hint)
                .style(Style::default().fg(c.dim))
                .alignment(Alignment::Center),
            chunks[3],
        );
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> AuthAction {
        match &self.state {
            InputState::SelectingProvider => self.handle_select_keys(key),
            InputState::EnteringKey { .. } | InputState::EnteringUrl { .. } => self.handle_input_keys(key),
        }
    }

    fn handle_select_keys(&mut self, key: KeyEvent) -> AuthAction {
        match key.code {
            KeyCode::Char('q') => AuthAction::Quit,
            KeyCode::Up => {
                if self.selection > 0 {
                    self.selection -= 1;
                }
                AuthAction::Continue
            }
            KeyCode::Down => {
                if self.selection < self.providers.len() - 1 {
                    self.selection += 1;
                }
                AuthAction::Continue
            }
            KeyCode::Enter => {
                let info = &self.providers[self.selection];
                if info.requires_api_key {
                    self.state = InputState::EnteringKey {
                        provider: info.key.clone(),
                    };
                } else {
                    self.state = InputState::EnteringUrl {
                        provider: info.key.clone(),
                    };
                }
                self.key_input.clear();
                self.status.clear();
                AuthAction::Continue
            }
            _ => AuthAction::Continue,
        }
    }

    fn handle_input_keys(&mut self, key: KeyEvent) -> AuthAction {
        match key.code {
            KeyCode::Char('q') => AuthAction::Quit,
            KeyCode::Esc => {
                self.state = InputState::SelectingProvider;
                self.key_input.clear();
                self.status.clear();
                AuthAction::Continue
            }
            KeyCode::Enter => {
                match &self.state {
                    InputState::EnteringKey { provider } => {
                        let p = provider.clone();
                        let k = self.key_input.clone();
                        match self.save_api_key(&p, &k) {
                            Ok(()) => {
                                self.status = format!("✓ Authenticated with {}", p);
                                self.state = InputState::SelectingProvider;
                                AuthAction::Done
                            }
                            Err(e) => {
                                self.status = format!("Error: {}", e);
                                AuthAction::Continue
                            }
                        }
                    }
                    InputState::EnteringUrl { provider } => {
                        let p = provider.clone();
                        let url = self.key_input.clone();
                        match self.save_base_url(&p, &url) {
                            Ok(()) => {
                                self.status = format!("✓ Configured {} with base URL", p);
                                self.state = InputState::SelectingProvider;
                                AuthAction::Done
                            }
                            Err(e) => {
                                self.status = format!("Error: {}", e);
                                AuthAction::Continue
                            }
                        }
                    }
                    InputState::SelectingProvider => AuthAction::Continue,
                }
            }
            KeyCode::Char(c) => {
                self.key_input.push(c);
                AuthAction::Continue
            }
            KeyCode::Backspace => {
                self.key_input.pop();
                AuthAction::Continue
            }
            _ => AuthAction::Continue,
        }
    }

    pub fn handle_paste(&mut self, text: String) {
        if matches!(self.state, InputState::EnteringKey { .. } | InputState::EnteringUrl { .. }) {
            let clean: String = text.chars().filter(|&c| c != '\n' && c != '\r').collect();
            self.key_input.push_str(&clean);
        }
    }

    fn save_api_key(&mut self, provider: &str, api_key: &str) -> Result<()> {
        let key = api_key.trim();
        if key.is_empty() {
            return Err(anyhow::anyhow!("API key cannot be empty"));
        }
        let existing = self.store.get(provider).cloned().unwrap_or_default();
        self.store.set(
            provider.to_string(),
            Credentials {
                api_key: key.to_string(),
                base_url: existing.base_url,
            },
        );
        self.store.save()
    }

    fn save_base_url(&mut self, provider: &str, url: &str) -> Result<()> {
        let trimmed = url.trim();
        if trimmed.is_empty() {
            return Err(anyhow::anyhow!("Base URL cannot be empty"));
        }
        if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
            return Err(anyhow::anyhow!("Base URL must start with http:// or https://"));
        }
        let existing = self.store.get(provider).cloned().unwrap_or_default();
        self.store.set(
            provider.to_string(),
            Credentials {
                api_key: existing.api_key,
                base_url: Some(trimmed.to_string()),
            },
        );
        self.store.save()
    }
}

impl Default for AuthLoginScreen {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            providers: Vec::new(),
            selection: 0,
            state: InputState::SelectingProvider,
            key_input: String::new(),
            status: String::from("Warning: Failed to load credential store"),
            store: CredentialStore::default(),
        })
    }
}
