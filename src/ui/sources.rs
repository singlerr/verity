//! Source list renderer for the right pane upper section.
use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use crate::app::{App, Source};
use crate::ui::layout::ColorScheme;
use crate::ui::pane_title;

pub fn render_sources(frame: &mut Frame, area: Rect, app: &App) {
    let colors = ColorScheme::default();
    let count = app.sources.len();

    let count_str = if count > 0 {
        format!("{} found", count)
    } else {
        "\u{2014}".to_string() // em dash
    };

    let mut lines: Vec<Line> = Vec::new();
    lines.push(pane_title("SOURCES", &count_str, area.width, &colors));

    if app.sources.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("\u{2026} ", Style::default().fg(colors.accent)),
            Span::styled("gathering\u{2026}", Style::default().fg(colors.dim)),
        ]));
    } else {
        for (idx, source) in app.sources.iter().enumerate() {
            lines.push(build_source_line(
                source,
                idx,
                app.selected_source == Some(idx),
                &colors,
            ));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(colors.bg)),
        area,
    );
}

fn build_source_line<'a>(
    source: &'a Source,
    idx: usize,
    is_selected: bool,
    colors: &'a ColorScheme,
) -> Line<'a> {
    let (indicator, row_style) = if is_selected {
        (
            Span::styled("\u{276f} ", Style::default().fg(colors.accent)), // ❯
            Style::default().bg(colors.accent).fg(colors.bg),
        )
    } else {
        (
            Span::styled("  ", Style::default().fg(colors.dim)),
            Style::default().fg(colors.ink),
        )
    };

    // [N] in accent
    let num_span = Span::styled(
        format!("[{}] ", source.num),
        if is_selected {
            Style::default().bg(colors.accent).fg(colors.bg)
        } else {
            Style::default().fg(colors.accent)
        },
    );

    // Favicon badge — first char of domain
    let domain_char = source
        .domain
        .chars()
        .next()
        .unwrap_or('?')
        .to_uppercase()
        .to_string();
    let badge_bg = cycle_color(idx, colors);
    let favicon = Span::styled(
        format!(" {} ", domain_char),
        Style::default().fg(colors.bg).bg(badge_bg),
    );

    // Domain (truncated)
    let domain = truncate(&source.domain, 15);
    let domain_span = Span::styled(
        format!(" {} ", domain),
        if is_selected {
            Style::default().bg(colors.accent).fg(colors.bg)
        } else {
            Style::default().fg(colors.dim)
        },
    );

    // Title (fills remaining width)
    let title = truncate(&source.title, 30);
    let title_span = Span::styled(title, row_style);

    Line::from(vec![indicator, num_span, favicon, domain_span, title_span])
}

fn truncate(s: &str, max_width: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_width {
        s.to_string()
    } else {
        chars
            .into_iter()
            .take(max_width.saturating_sub(1))
            .collect::<String>()
            + "\u{2026}"
    }
}

fn cycle_color(idx: usize, colors: &ColorScheme) -> Color {
    match idx % 3 {
        0 => colors.accent,
        1 => colors.net,
        _ => colors.read,
    }
}
