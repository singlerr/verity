//! Trace log panel — renders the command execution transcript.
use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::app::TerminalLine;
use super::layout::ColorScheme;

/// Render the trace log section: "trace.log" label + bordered log box.
pub fn render_trace(frame: &mut Frame, area: Rect, app: &crate::app::App) {
    let colors = ColorScheme::default();

    let label_line = Line::from(vec![Span::styled(
        "trace.log",
        Style::default().fg(colors.dim),
    )]);

    // Build log entry lines (last N entries to fill available rows)
    // area.height - 1 for the label row, -2 for block borders
    let content_rows = area.height.saturating_sub(3) as usize;
    let entries = build_log_lines(&app.trace_lines, &colors);
    let start = entries.len().saturating_sub(content_rows);
    let visible: Vec<Line> = entries.into_iter().skip(start).collect();

    let inner_lines = if visible.is_empty() {
        vec![Line::from(vec![Span::styled(
            "waiting for input\u{2026}",
            Style::default().fg(colors.dim),
        )])]
    } else {
        visible
    };

    // Render label first (first row of the area)
    let label_area = Rect { x: area.x, y: area.y, width: area.width, height: 1 };
    let log_area = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: area.height.saturating_sub(1),
    };

    frame.render_widget(
        Paragraph::new(label_line.clone()).style(Style::default().bg(colors.bg)),
        label_area,
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(colors.dim))
        .style(Style::default().bg(colors.status_bg));

    frame.render_widget(Paragraph::new(inner_lines).block(block), log_area);
}

fn build_log_lines<'a>(trace_lines: &'a [TerminalLine], colors: &'a ColorScheme) -> Vec<Line<'a>> {
    trace_lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let elapsed = 2 + i as u32 * 3;
            let ts = format!("{:02}:{:02}", elapsed / 60, elapsed % 60);

            let (tag_str, tag_style) = match line.kind {
                crate::app::LineKind::Cmd => ("PLAN", Style::default().fg(colors.accent)),
                crate::app::LineKind::Ok  => (" OK ", Style::default().fg(colors.success)),
                crate::app::LineKind::Dim => ("NET ", Style::default().fg(colors.net)),
                crate::app::LineKind::Out => ("READ", Style::default().fg(colors.read)),
            };

            Line::from(vec![
                Span::styled(ts, Style::default().fg(colors.dim)),
                Span::raw(" "),
                Span::styled(format!("[{}]", tag_str), tag_style),
                Span::raw(" "),
                Span::styled(line.text.clone(), Style::default().fg(colors.ink)),
            ])
        })
        .collect()
}
