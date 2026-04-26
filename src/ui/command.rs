//! Bottom command input bar (height 1 line).
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::layout::ColorScheme;

/// Render the command input bar. Visual style changes when the bar is focused.
pub fn render_command(frame: &mut Frame, area: Rect, app: &crate::app::App) {
    use crate::app::Focus;
    let colors = ColorScheme::default();
    let focused = app.focus == Focus::Command;

    let prompt_style = if focused {
        Style::default().fg(colors.accent).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(colors.dim)
    };

    let query_span = if focused && app.query.is_empty() {
        Span::styled("type to search...", Style::default().fg(colors.dim))
    } else {
        Span::styled(app.query.as_str(), Style::default().fg(colors.ink))
    };

    let ghost = if focused {
        crate::ui::autocomplete::completion_suffix(&app.query, app.autocomplete_idx)
    } else {
        None
    };

    let suffix = if focused {
        Span::styled("  ⏎ send", Style::default().fg(colors.accent))
    } else {
        Span::styled("  / to search", Style::default().fg(colors.dim))
    };

    let mut spans = vec![Span::styled("❯ ", prompt_style), query_span];
    if let Some(ref g) = ghost {
        spans.push(Span::styled(g.as_str(), Style::default().fg(colors.dim)));
    }
    spans.push(suffix);
    let line = Line::from(spans);

    let bg = if focused { colors.header_bg } else { colors.bg };
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(bg)),
        area,
    );
}
