//! Answer panel renderer.
use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use crate::app::{App, AppState};
use crate::ui::layout::ColorScheme;
use crate::ui::markdown::render_markdown;
use crate::ui::pane_title;

pub fn render_answer(frame: &mut Frame, area: Rect, app: &App) {
    let colors = ColorScheme::default();

    let status_str = match app.state {
        AppState::Idle | AppState::Planning | AppState::Researching => "pending",
        AppState::AnswerReady => "done",
        AppState::Error(_) => "error",
    };

    let mut lines: Vec<Line> = Vec::new();
    lines.push(pane_title("ANSWER", status_str, area.width, &colors));
    lines.push(Line::raw(""));

    if app.answer_chunks.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "waiting for plan to complete\u{2026}",
            Style::default().fg(colors.dim),
        )]));
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(colors.bg)),
            area,
        );
        return;
    }

    // Concatenate answer chunk text and render as markdown
    let content: String = app.answer_chunks.iter().map(|ch| ch.text.as_str()).collect();
    let inner_width = area.width.saturating_sub(1).max(1);
    let md_lines = render_markdown(&content, inner_width);

    // Prepend pane title + blank, then markdown lines
    let all_lines: Vec<Line> = lines.into_iter().chain(md_lines).collect();

    frame.render_widget(
        Paragraph::new(all_lines).style(Style::default().bg(colors.bg)),
        area,
    );
}
