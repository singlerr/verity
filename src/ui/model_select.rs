//! Model selection popup UI.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::layout::ColorScheme;

/// Render a centered model-selection popup.
pub fn render_model_select_popup(
    frame: &mut Frame,
    area: Rect,
    models: &[String],
    selected: usize,
) {
    if models.is_empty() {
        return;
    }
    let colors = ColorScheme::default();
    let n = models.len().min(10) as u16;
    let popup_height = n + 2; // borders
    let popup_width = 44u16.min(area.width.saturating_sub(4));
    let popup_area = Rect {
        x: area.x + (area.width.saturating_sub(popup_width)) / 2,
        y: area.y + (area.height.saturating_sub(popup_height)) / 2,
        width: popup_width,
        height: popup_height,
    };

    let block = Block::default()
        .title(" Select Model ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(colors.accent))
        .style(Style::default().bg(colors.header_bg));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let sel = selected % models.len();
    for (row, model) in models.iter().take(10).enumerate() {
        let is_sel = row == sel;
        let marker = if is_sel { "▶ " } else { "  " };
        let style = if is_sel {
            Style::default().fg(colors.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(colors.ink)
        };
        let bg = if is_sel { colors.status_bg } else { colors.header_bg };

        let line = Line::from(vec![
            Span::styled(marker, Style::default().fg(colors.accent)),
            Span::styled(model.clone(), style),
        ]);
        let row_area = Rect {
            x: inner.x,
            y: inner.y + row as u16,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(line).style(Style::default().bg(bg)),
            row_area,
        );
    }
}
