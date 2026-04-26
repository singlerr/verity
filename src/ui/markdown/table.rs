use comrak::nodes::{AstNode, NodeValue};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;
use super::leak;

/// Build a border line (top/sep/bottom) from column widths.
fn make_border(left: &str, mid: &str, right: &str, widths: &[usize]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = vec![Span::raw(leak(left.to_string()))];
    for (i, &w) in widths.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw(leak(mid.to_string())));
        }
        spans.push(Span::raw(leak("─".repeat(w + 2))));
    }
    spans.push(Span::raw(leak(right.to_string())));
    Line::from(spans)
}

/// Render a comrak Table node into box-drawing bordered lines.
pub fn render_table<'a>(table_node: &'a AstNode<'a>, _width: u16) -> Vec<Line<'static>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut is_header: Vec<bool> = Vec::new();

    for row_node in table_node.children() {
        if let NodeValue::TableRow(hdr) = &row_node.data.borrow().value {
            is_header.push(*hdr);
            let mut cells: Vec<String> = Vec::new();
            for cell_node in row_node.children() {
                if let NodeValue::TableCell = &cell_node.data.borrow().value {
                    cells.push(super::block::collect_inline_text(cell_node));
                }
            }
            rows.push(cells);
        }
    }

    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if ncols == 0 {
        return Vec::new();
    }

    // Compute CJK-aware column widths
    let mut widths = vec![0usize; ncols];
    for r in &rows {
        for (i, c) in r.iter().enumerate() {
            let w = UnicodeWidthStr::width(c.as_str());
            if w > widths[i] { widths[i] = w; }
        }
    }

    let mut out: Vec<Line<'static>> = Vec::new();

    // Top border: ┌─────┬─────┐
    out.push(make_border("┌", "┬", "┐", &widths));

    for (ri, r) in rows.iter().enumerate() {
        // Cell row: │ text │ text │
        let mut line = String::from("│");
        for (i, cell) in r.iter().enumerate() {
            let cw = UnicodeWidthStr::width(cell.as_str());
            let pad = widths.get(i).copied().unwrap_or(0).saturating_sub(cw);
            line.push(' ');
            line.push_str(cell);
            line.push_str(&" ".repeat(pad + 1));
            line.push('│');
        }
        out.push(Line::from(vec![Span::raw(leak(line))]));

        // Header separator: ├───┼───┤
        if is_header.get(ri) == Some(&true) {
            out.push(make_border("├", "┼", "┤", &widths));
        }
    }

    // Bottom border: └─────┴─────┘
    out.push(make_border("└", "┴", "┘", &widths));

    out
}
