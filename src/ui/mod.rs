//! UI module: layout and rendering.
pub mod answer;
pub mod autocomplete;
pub mod command;
pub mod error;
pub mod header;
pub mod layout;
pub mod markdown;
pub mod model_select;
pub mod spinner;
pub mod status;

// Re-exports for convenience
pub use answer::render_answer;
pub use markdown::render_markdown;
pub mod plan;
pub mod sources;
pub mod trace;

pub use command::render_command;
pub use error::render_error_overlay;
pub use header::render_header;
pub use layout::{compute_layout, render_layout, AppLayout, ColorScheme};
pub use model_select::render_model_select_popup;
pub use plan::render_plan;
pub use sources::render_sources;
pub use spinner::Spinner;
pub use status::render_status_bar;
pub use trace::render_trace;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

/// Render a pane title bar: `▎ TITLE ─────── right`
pub fn pane_title(
    title: &str,
    right: &str,
    width: u16,
    colors: &layout::ColorScheme,
) -> Line<'static> {
    let prefix_len = 2usize; // "▎ "
    let title_len = title.len();
    let space_after_title = 1usize;
    let left_used = prefix_len + title_len + space_after_title;
    let right_used = if right.is_empty() { 0 } else { 1 + right.len() }; // " " + right
    let dash_count = (width as usize).saturating_sub(left_used + right_used);

    let mut spans: Vec<Span<'static>> = vec![
        Span::styled(
            "▎ ",
            Style::default()
                .fg(colors.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            title.to_string(),
            Style::default().fg(colors.ink).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled("─".repeat(dash_count), Style::default().fg(colors.dim)),
    ];
    if !right.is_empty() {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            right.to_string(),
            Style::default().fg(colors.dim),
        ));
    }
    Line::from(spans)
}
