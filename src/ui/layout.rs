//! Two-pane split layout with warm color scheme.
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Color,
    widgets::{Block, Borders},
};

/// Warm color scheme matching the design spec.
pub struct ColorScheme {
    pub bg: Color,        // #f4eee3 (warm off-white)
    pub ink: Color,       // #2a241c (dark brown)
    pub dim: Color,       // #8a7f6f (muted brown)
    pub accent: Color,    // #c96442 (burnt orange)
    pub success: Color,   // #5b8a5b (green)
    pub net: Color,       // #3a74a8 (blue)
    pub read: Color,      // #8a6c2a (yellow-brown)
    pub header_bg: Color, // #ebe4d5 (slightly darker than bg)
    pub status_bg: Color, // #e8dfce (slightly darker)
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            bg: Color::Rgb(244, 238, 227),
            ink: Color::Rgb(42, 36, 28),
            dim: Color::Rgb(138, 127, 111),
            accent: Color::Rgb(201, 100, 66),
            success: Color::Rgb(91, 138, 91),
            net: Color::Rgb(58, 116, 168),
            read: Color::Rgb(138, 108, 42),
            header_bg: Color::Rgb(235, 228, 213),
            status_bg: Color::Rgb(232, 223, 206),
        }
    }
}

/// Layout regions for the application.
pub struct AppLayout {
    pub header: Rect,
    pub left: Rect,  // agent plan + trace
    pub right: Rect, // sources + answer
    pub status: Rect,
    pub command: Rect,
}

/// Compute the layout rectangles from the terminal area.
pub fn compute_layout(area: Rect) -> AppLayout {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(0),    // middle (splits horizontally)
            Constraint::Length(1), // status
            Constraint::Length(1), // command
        ])
        .split(area);

    let middle = vertical[1];
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(middle);

    AppLayout {
        header: vertical[0],
        left: horizontal[0],
        right: horizontal[1],
        status: vertical[2],
        command: vertical[3],
    }
}

/// Render the full application layout by calling each component renderer.
pub fn render_layout(frame: &mut ratatui::Frame, app: &crate::app::App) {
    use ratatui::style::Style;

    let area = frame.area();
    let colors = ColorScheme::default();

    // Fill background
    frame.render_widget(Block::default().style(Style::default().bg(colors.bg)), area);

    let layout = compute_layout(area);

    // Header (1 line)
    crate::ui::header::render_header(frame, layout.header, app);

    // Left pane: draw right-border separator then split into plan + trace
    frame.render_widget(
        Block::default()
            .borders(Borders::RIGHT)
            .border_style(Style::default().fg(colors.dim))
            .style(Style::default().bg(colors.bg)),
        layout.left,
    );
    let left_inner = Rect {
        x: layout.left.x,
        y: layout.left.y,
        width: layout.left.width.saturating_sub(1),
        height: layout.left.height,
    };
    let trace_height = 9u16; // "trace.log" label + 8 log rows
    let left_split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(trace_height)])
        .split(left_inner);

    crate::ui::plan::render_plan(frame, left_split[0], app, &app.spinner);
    crate::ui::trace::render_trace(frame, left_split[1], app);

    // Right pane: split into sources (fixed) + answer (flexible)
    let source_rows = (app.sources.len().min(6) + 3) as u16; // pane_title + entries + spacing
    let right_split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(source_rows.max(4)), Constraint::Min(0)])
        .split(layout.right);

    crate::ui::sources::render_sources(frame, right_split[0], app);
    crate::ui::answer::render_answer(frame, right_split[1], app);

    // Status bar (borderless)
    crate::ui::status::render_status_bar(frame, layout.status, app);

    // Command bar (borderless)
    crate::ui::command::render_command(frame, layout.command, app);

    // Autocomplete popup — rendered above the command bar, overlays content
    if app.focus == crate::app::Focus::Command {
        crate::ui::autocomplete::render_popup(
            frame,
            layout.command,
            &app.query,
            app.autocomplete_idx,
        );
    }

    // Show cursor at end of query input when focused on command bar
    if app.focus == crate::app::Focus::Command {
        let cx = layout.command.x + 2 + app.query.chars().count() as u16;
        let cy = layout.command.y;
        frame.set_cursor_position((cx, cy));
    }
}
