//! Plan tree renderer for the left pane.
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use crate::app::{App, AppState, PlanStep, StepStatus, Tool};
use crate::ui::layout::ColorScheme;
use crate::ui::Spinner;
use crate::ui::pane_title;

pub fn render_plan(frame: &mut Frame, area: Rect, app: &App, spinner: &Spinner) {
    let colors = ColorScheme::default();
    let steps = &app.plan_steps;
    let mut lines: Vec<Line> = Vec::new();

    // PaneTitle: ▎ AGENT ─────── running/idle
    let status_str = match app.state {
        AppState::Idle | AppState::AnswerReady => "idle",
        AppState::Planning | AppState::Researching => "running",
        AppState::Error(_) => "error",
    };
    lines.push(pane_title("AGENT", status_str, area.width, &colors));
    lines.push(Line::raw(""));

    // Query echo block with orange left-border approximation
    if !app.submitted_query.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("▏ ", Style::default().fg(colors.accent)),
            Span::styled("$ verity ask", Style::default().fg(colors.dim)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("▏ ", Style::default().fg(colors.accent)),
            Span::styled(app.submitted_query.clone(), Style::default().fg(colors.ink)),
        ]));
        lines.push(Line::raw(""));
    } else {
        lines.push(Line::from(vec![Span::styled(
            "waiting for input\u{2026}",
            Style::default().fg(colors.dim),
        )]));
    }

    if !steps.is_empty() {
        // Plan header: "plan (N/M)"
        let done = steps.iter().filter(|s| s.status == StepStatus::Done).count();
        lines.push(Line::from(vec![
            Span::styled("plan ", Style::default().fg(colors.dim)),
            Span::styled(
                format!("({}/{})", done, steps.len()),
                Style::default().fg(colors.ink),
            ),
        ]));

        lines.extend(render_steps(steps, &colors, spinner));
    }

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(colors.bg).fg(colors.ink)),
        area,
    );
}

fn render_steps<'a>(steps: &'a [PlanStep], colors: &'a ColorScheme, spinner: &Spinner) -> Vec<Line<'a>> {
    let mut lines = Vec::new();
    let last_idx = steps.len().saturating_sub(1);

    for (i, step) in steps.iter().enumerate() {
        let is_last = i == last_idx;
        let prefix = if is_last { "└─ " } else { "├─ " };

        let (marker, marker_color) = match step.status {
            StepStatus::Done    => ("●", colors.ink),
            StepStatus::Running => (spinner.frame(), colors.accent),
            StepStatus::Queued  => ("○", colors.dim),
        };

        let title_style = match step.status {
            StepStatus::Running => Style::default().fg(colors.ink).add_modifier(Modifier::BOLD),
            StepStatus::Queued  => Style::default().fg(colors.dim),
            StepStatus::Done    => Style::default().fg(colors.ink),
        };

        let tool_str = format!("[{:6}]", tool_label(&step.tool));

        let mut spans: Vec<Span> = vec![
            Span::styled(prefix.to_string(), Style::default().fg(colors.dim)),
            Span::styled(marker.to_string(), Style::default().fg(marker_color)),
            Span::raw(" "),
            Span::styled(tool_str, Style::default().fg(colors.accent)),
            Span::raw(" "),
            Span::styled(step.title.clone(), title_style),
        ];

        if let Some(d) = step.duration {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(
                format!("{:.1}s", d),
                Style::default().fg(colors.dim),
            ));
        }

        lines.push(Line::from(spans));

        // Thoughts: "│  // reason" — // in accent, text in dim
        for thought in &step.thoughts {
            let vert = if is_last { "     " } else { "│    " };
            lines.push(Line::from(vec![
                Span::styled(vert.to_string(), Style::default().fg(colors.dim)),
                Span::styled("// ", Style::default().fg(colors.accent)),
                Span::styled(thought.clone(), Style::default().fg(colors.dim)),
            ]));
        }
    }

    lines
}

fn tool_label(tool: &Tool) -> &'static str {
    match tool {
        Tool::Search => "search",
        Tool::Read   => "read",
        Tool::Think  => "think",
        Tool::Edit   => "edit",
        Tool::Shell  => "shell",
    }
}
