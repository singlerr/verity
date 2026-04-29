use super::leak;
use comrak::nodes::{AstNode, NodeValue};
use ratatui::text::{Line, Span};

/// Collect plain text from inline children of a node.
pub fn collect_inline_text<'a>(node: &'a AstNode<'a>) -> String {
    let mut s = String::new();
    for child in node.children() {
        let b = child.data.borrow();
        match &b.value {
            NodeValue::Text(t) => s.push_str(t),
            NodeValue::Code(c) => s.push_str(&c.literal),
            NodeValue::Strikethrough => {
                drop(b);
                let inner = collect_inline_text(child);
                s.push_str(&format!("~{}~", inner));
            }
            NodeValue::FootnoteReference(fr) => {
                s.push_str(&format!("[^{}]", fr.name));
            }
            _ => {
                drop(b);
                s.push_str(&collect_inline_text(child));
            }
        }
    }
    s
}

/// Render a list (ordered or unordered) with nesting support.
pub fn render_list<'a>(list_node: &'a AstNode<'a>, width: u16, depth: usize) -> Vec<Line<'static>> {
    use comrak::nodes::ListType;
    let mut lines: Vec<Line<'static>> = Vec::new();

    let (list_type, start) = if let NodeValue::List(nl) = &list_node.data.borrow().value {
        (nl.list_type, nl.start)
    } else {
        return lines;
    };

    let indent = "  ".repeat(depth);
    let mut ordinal: usize = start;

    for item in list_node.children() {
        if let NodeValue::Item(_) = &item.data.borrow().value {
            let mut text = String::new();
            for child in item.children() {
                let cb = child.data.borrow();
                match &cb.value {
                    NodeValue::Paragraph => {
                        drop(cb);
                        text.push_str(&collect_inline_text(child));
                    }
                    NodeValue::List(_) => {
                        drop(cb);
                        lines.extend(render_list(child, width, depth + 1));
                    }
                    _ => {
                        drop(cb);
                    }
                }
            }

            let marker = if list_type == ListType::Bullet {
                format!("{}• {}", indent, text)
            } else {
                let m = format!("{}{}. {}", indent, ordinal, text);
                ordinal += 1;
                m
            };

            let inner_w = width as usize;
            let wrapped = super::wrap_text(&marker, inner_w);
            for l in wrapped {
                lines.push(Line::from(vec![Span::raw(leak(l))]));
            }
        }
    }
    lines
}
