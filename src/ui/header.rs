//! Top header bar: shows version, workspace, and key hints.
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::layout::ColorScheme;

pub fn render_header(frame: &mut Frame, area: Rect, _app: &crate::app::App) {
    let colors = ColorScheme::default();

    // Left side spans
    let left_spans: Vec<Span> = vec![
        Span::raw(" "),
        Span::styled("\u{276f} ", Style::default().fg(colors.accent).add_modifier(Modifier::BOLD)), // ❯
        Span::styled("verity", Style::default().fg(colors.ink).add_modifier(Modifier::BOLD)),
        Span::styled(" v0.1.0", Style::default().fg(colors.dim)),
        Span::styled(" \u{b7} workspace: ", Style::default().fg(colors.dim)), // ·
        Span::styled("default", Style::default().fg(colors.ink)),
    ];

    // Right side key-hint spans
    let right_spans: Vec<Span> = vec![
        Span::styled("[?]", Style::default().fg(colors.ink)),
        Span::styled("help", Style::default().fg(colors.dim)),
        Span::raw("  "),
        Span::styled("[\u{2303}K]", Style::default().fg(colors.ink)), // ⌃K
        Span::styled("new", Style::default().fg(colors.dim)),
        Span::raw("  "),
        Span::styled("[\u{2303}D]", Style::default().fg(colors.ink)), // ⌃D
        Span::styled("quit", Style::default().fg(colors.dim)),
        Span::raw(" "),
    ];

    let left_len: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
    let right_len: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
    let spacer_len = (area.width as usize).saturating_sub(left_len + right_len);

    let mut all_spans = left_spans;
    all_spans.push(Span::raw(" ".repeat(spacer_len)));
    all_spans.extend(right_spans);

    frame.render_widget(
        Paragraph::new(Line::from(all_spans))
            .style(Style::default().bg(colors.header_bg)),
        area,
    );
}
