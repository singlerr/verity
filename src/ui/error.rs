//! Error overlay for displaying errors over the main layout.
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::ui::layout::ColorScheme;

/// Render a full-screen error overlay with dimmed background.
pub fn render_error_overlay(frame: &mut Frame, area: Rect, title: &str, message: &str) {
    let colors = ColorScheme::default();

    // Clear the entire area first
    frame.render_widget(Clear, area);

    // Centered box: 80% width, min 5 rows for content
    let overlay = centered_rect(area, 80, 40);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(colors.accent))
            .style(Style::default().bg(colors.bg)),
        overlay,
    );

    // Split overlay into: title / message / suggestion
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // title
            Constraint::Min(1),    // message
            Constraint::Length(2), // footer
        ])
        .margin(1)
        .split(overlay);

    // Title
    frame.render_widget(
        Paragraph::new(Line::from(vec![ratatui::text::Span::styled(
            format!("⚠ {}", title),
            Style::default()
                .fg(colors.accent)
                .add_modifier(Modifier::BOLD),
        )]))
        .style(Style::default().bg(colors.bg)),
        chunks[0],
    );

    // Message
    frame.render_widget(
        Paragraph::new(message).style(Style::default().fg(colors.ink).bg(colors.bg)),
        chunks[1],
    );

    // Footer — dismiss hint
    frame.render_widget(
        Paragraph::new("Press any key to dismiss")
            .style(Style::default().fg(colors.dim).bg(colors.bg))
            .alignment(Alignment::Center),
        chunks[2],
    );
}

/// Compute a centered rect with given width/height percentages.
fn centered_rect(area: Rect, width_pct: u16, height_pct: u16) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_pct) / 2),
            Constraint::Percentage(height_pct),
            Constraint::Percentage((100 - height_pct) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_pct) / 2),
            Constraint::Percentage(width_pct),
            Constraint::Percentage((100 - width_pct) / 2),
        ])
        .split(popup_layout[1])[1]
}
