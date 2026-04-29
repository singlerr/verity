//! Slash-command autocomplete: suggestions and popup rendering.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::layout::ColorScheme;

/// Known slash commands: (full command text, short description).
pub const COMMANDS: &[(&str, &str)] = &[
    ("/mode research", "research mode"),
    ("/mode code", "code mode"),
    ("/model ", "set model  e.g. gpt-4o"),
    ("/help", "show keybindings"),
];

/// Returns indices into COMMANDS whose text starts with `query`.
/// Empty if `query` does not start with `/`.
pub fn matching(query: &str) -> Vec<usize> {
    if !query.starts_with('/') {
        return vec![];
    }
    COMMANDS
        .iter()
        .enumerate()
        .filter(|(_, (cmd, _))| cmd.starts_with(query))
        .map(|(i, _)| i)
        .collect()
}

/// Returns the trailing suffix that Tab would insert for the currently selected suggestion.
pub fn completion_suffix(query: &str, selected: usize) -> Option<String> {
    let m = matching(query);
    if m.is_empty() {
        return None;
    }
    let cmd = COMMANDS[m[selected % m.len()]].0;
    if cmd.len() > query.len() {
        Some(cmd[query.len()..].to_string())
    } else {
        None
    }
}

/// Render a suggestion popup just above `command_area`.
///
/// `selected` wraps modulo the number of matches so callers can freely increment it.
pub fn render_popup(frame: &mut Frame, command_area: Rect, query: &str, selected: usize) {
    let matches = matching(query);
    if matches.is_empty() {
        return;
    }

    let colors = ColorScheme::default();
    let n = matches.len().min(4) as u16;
    let popup_height = n + 2; // border rows

    if command_area.y < popup_height {
        return;
    }

    let popup_width = 54u16.min(command_area.width.saturating_sub(2));
    let area = Rect {
        x: command_area.x + 1,
        y: command_area.y.saturating_sub(popup_height),
        width: popup_width,
        height: popup_height,
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(colors.dim))
        .style(Style::default().bg(colors.header_bg));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let sel = selected % matches.len();
    for (row, &idx) in matches.iter().take(4).enumerate() {
        let (cmd, desc) = COMMANDS[idx];
        let is_sel = row == sel;

        let marker_style = Style::default().fg(colors.accent);
        let cmd_style = if is_sel {
            Style::default()
                .fg(colors.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(colors.ink)
        };
        let bg = if is_sel {
            colors.status_bg
        } else {
            colors.header_bg
        };

        let line = Line::from(vec![
            Span::styled(if is_sel { "▶ " } else { "  " }, marker_style),
            Span::styled(cmd, cmd_style),
            Span::styled("  ", Style::default()),
            Span::styled(desc, Style::default().fg(colors.dim)),
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
