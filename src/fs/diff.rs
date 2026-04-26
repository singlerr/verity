use ratatui::prelude::Stylize;
use ratatui::text::Line;
use similar::{ChangeTag, TextDiff};

#[derive(Debug, Clone)]
pub enum DiffLine {
    Context(String),
    Add(String),
    Remove(String),
}

pub fn compute_diff(old: &str, new: &str) -> Vec<DiffLine> {
    let diff = TextDiff::from_lines(old, new);
    diff.iter_all_changes()
        .map(|change| match change.tag() {
            ChangeTag::Delete => DiffLine::Remove(change.value().to_string()),
            ChangeTag::Insert => DiffLine::Add(change.value().to_string()),
            ChangeTag::Equal => DiffLine::Context(change.value().to_string()),
            // Replace is not a ChangeTag variant; Replace ops emit Delete+Insert pairs
        })
        .collect()
}

pub fn render_diff(diff_lines: &[DiffLine]) -> Vec<Line<'static>> {
    diff_lines
        .iter()
        .map(|line| match line {
            DiffLine::Add(text) => Line::from(format!("+ {}", text)).green(),
            DiffLine::Remove(text) => Line::from(format!("- {}", text)).red(),
            DiffLine::Context(text) => Line::from(format!("  {}", text)).dim(),
        })
        .collect()
}
