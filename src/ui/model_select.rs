//! Model selection popup UI — grouped by provider.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::layout::ColorScheme;
use crate::llm::provider::ModelEntry;

/// Group models by provider, preserving insertion order.
fn group_by_provider<'a>(
    models: &'a [ModelEntry],
    display_names: &std::collections::HashMap<String, String>,
) -> Vec<(String, Vec<&'a ModelEntry>)> {
    let mut groups: Vec<(String, Vec<&'a ModelEntry>)> = Vec::new();
    for entry in models {
        let prov = display_names.get(&entry.provider)
            .cloned()
            .unwrap_or_else(|| entry.provider.clone());
        if let Some(g) = groups.iter_mut().find(|(p, _)| *p == prov) {
            g.1.push(entry);
        } else {
            groups.push((prov, vec![entry]));
        }
    }
    groups
}

/// Returns the visual row (0-based, within inner area) of the selected flat index.
fn selected_visual_row(groups: &[(String, Vec<&ModelEntry>)], selected: usize) -> usize {
    let mut flat_idx = 0usize;
    let mut visual = 0usize;
    for (g_idx, (_, ms)) in groups.iter().enumerate() {
        if g_idx > 0 {
            visual += 1;
        } // blank line between groups
        visual += 1; // provider header
        for _ in ms.iter() {
            if flat_idx == selected {
                return visual;
            }
            flat_idx += 1;
            visual += 1;
        }
    }
    0
}

/// Render a centered model-selection popup, grouped by provider.
pub fn render_model_select_popup(
    frame: &mut Frame,
    area: Rect,
    models: &[ModelEntry],
    selected: usize,
    display_names: &std::collections::HashMap<String, String>,
) {
    let colors = ColorScheme::default();

    if models.is_empty() {
        let popup_height = 3u16;
        let popup_width = 48u16.min(area.width.saturating_sub(4));
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
        frame.render_widget(
            Paragraph::new("  Loading models...").style(Style::default().fg(colors.dim)),
            inner,
        );
        return;
    }

    let groups = group_by_provider(models, display_names);

    // Total visual rows: per group = 1 header + models; blank line between groups
    let content_rows: u16 = groups
        .iter()
        .enumerate()
        .map(|(i, (_, ms))| {
            let blank: u16 = if i > 0 { 1 } else { 0 };
            blank + 1 + ms.len() as u16
        })
        .sum();

    let max_inner = area.height.saturating_sub(6).max(4);
    let inner_h = content_rows.min(max_inner);
    let popup_height = inner_h + 2;
    let popup_width = 50u16.min(area.width.saturating_sub(4));
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

    // Scroll offset: keep selected item visible
    let sel_row = selected_visual_row(&groups, selected);
    let scroll: usize = if sel_row >= inner.height as usize {
        sel_row + 1 - inner.height as usize
    } else {
        0
    };

    let mut flat_idx = 0usize;
    let mut visual_row: i32 = 0;

    for (g_idx, (provider, provider_models)) in groups.iter().enumerate() {
        if g_idx > 0 {
            visual_row += 1; // blank separator between groups
        }

        // Provider header
        let draw = visual_row - scroll as i32;
        if draw >= 0 && (draw as u16) < inner.height {
            let hdr = Line::from(vec![Span::styled(
                format!(" {} ", provider),
                Style::default().fg(colors.dim).add_modifier(Modifier::BOLD),
            )]);
            frame.render_widget(
                Paragraph::new(hdr).style(Style::default().bg(colors.header_bg)),
                Rect {
                    x: inner.x,
                    y: inner.y + draw as u16,
                    width: inner.width,
                    height: 1,
                },
            );
        }
        visual_row += 1;

        for model in provider_models.iter() {
            let draw = visual_row - scroll as i32;
            if draw >= 0 && (draw as u16) < inner.height {
                let is_sel = flat_idx == selected;
                let marker = if is_sel { "▶ " } else { "  " };
                let style = if is_sel {
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
                    Span::styled(marker, Style::default().fg(colors.accent)),
                    Span::styled(model.display_name(), style),
                ]);
                frame.render_widget(
                    Paragraph::new(line).style(Style::default().bg(bg)),
                    Rect {
                        x: inner.x,
                        y: inner.y + draw as u16,
                        width: inner.width,
                        height: 1,
                    },
                );
            }
            flat_idx += 1;
            visual_row += 1;
        }
    }
}
