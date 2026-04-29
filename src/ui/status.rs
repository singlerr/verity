//! Status bar — one borderless line showing state, tokens, model, and keybinds.
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::ColorScheme;
use crate::app::{App, AppState};

pub fn render_status_bar(f: &mut Frame, area: Rect, app: &App) {
    let colors = ColorScheme::default();

    let (state_str, spinner_char, state_style) = match &app.state {
        AppState::Idle | AppState::AnswerReady => (
            "ready",
            "\u{25cf}", // ●
            Style::default()
                .fg(colors.success)
                .add_modifier(Modifier::BOLD),
        ),
        AppState::Classifying => (
            "analyzing intent\u{2026}",
            app.spinner.frame(),
            Style::default()
                .fg(colors.accent)
                .add_modifier(Modifier::BOLD),
        ),
        AppState::Planning => (
            "planning\u{2026}",
            app.spinner.frame(),
            Style::default()
                .fg(colors.accent)
                .add_modifier(Modifier::BOLD),
        ),
        AppState::Researching => (
            "searching\u{2026}",
            app.spinner.frame(),
            Style::default()
                .fg(colors.accent)
                .add_modifier(Modifier::BOLD),
        ),
        AppState::Error(_) => (
            "error",
            "\u{25cf}",
            Style::default()
                .fg(colors.accent)
                .add_modifier(Modifier::BOLD),
        ),
    };

    // Elapsed time: frozen final value after done, live counter while running
    let elapsed = app
        .elapsed
        .map(|d| format!("{}s", d.as_secs()))
        .or_else(|| {
            app.start_time
                .map(|t| format!("{}s", t.elapsed().as_secs()))
        })
        .unwrap_or_else(|| "\u{2014}".to_string()); // —

    let sep = Span::styled(" \u{b7} ", Style::default().fg(colors.dim)); // ·

    // Left side spans
    let mut left_spans: Vec<Span> = vec![
        Span::raw(" "),
        Span::styled(spinner_char.to_string(), state_style),
        Span::raw(" "),
        Span::styled(
            state_str.to_string(),
            Style::default().fg(colors.ink).add_modifier(Modifier::BOLD),
        ),
        sep.clone(),
        Span::styled(
            format!("tokens \u{2014} \u{b7} {}", elapsed),
            Style::default().fg(colors.dim),
        ),
        sep.clone(),
        Span::styled(
            format!("model: {}", app.active_model),
            Style::default().fg(colors.dim),
        ),
    ];

    // Right side hints — change based on current focus
    let hints = match app.focus {
        crate::app::Focus::Command => "\u{23ce} submit \u{b7} Esc cancel",
        crate::app::Focus::Left => "j/k sources \u{b7} Tab answer \u{b7} / cmd \u{b7} m model",
        crate::app::Focus::Right => "j/k scroll \u{b7} Tab sources \u{b7} / cmd \u{b7} m model",
    };
    let right_span = Span::styled(hints, Style::default().fg(colors.dim));

    // Compute spacer
    let left_len: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
    let right_len = hints.chars().count() + 1; // +1 for trailing space
    let spacer_len = (area.width as usize).saturating_sub(left_len + right_len);

    left_spans.push(Span::raw(" ".repeat(spacer_len)));
    left_spans.push(right_span);
    left_spans.push(Span::raw(" "));

    f.render_widget(
        Paragraph::new(Line::from(left_spans)).style(Style::default().bg(colors.status_bg)),
        area,
    );
}
