use comrak::{Arena, ComrakOptions, parse_document, nodes::{AstNode, NodeValue}};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;
use std::collections::HashSet;

pub(crate) fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub(crate) fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for w in text.split_whitespace() {
        let ww = UnicodeWidthStr::width(w);
        let cw = UnicodeWidthStr::width(cur.as_str());
        if cur.is_empty() {
            cur.push_str(w);
        } else if cw + 1 + ww <= width {
            cur.push(' ');
            cur.push_str(w);
        } else {
            lines.push(cur);
            cur = w.to_string();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Collect plain text from inline children of a node.
fn collect_text<'a>(node: &'a AstNode<'a>) -> String {
    let mut s = String::new();
    for child in node.children() {
        let b = child.data.borrow();
        match &b.value {
            NodeValue::Text(t) => s.push_str(t),
            NodeValue::Code(c) => s.push_str(&c.literal),
            NodeValue::Strikethrough => {
                drop(b);
                let inner = collect_text(child);
                s.push_str(&format!("~{}~", inner));
            }
            NodeValue::FootnoteReference(fr) => {
                s.push_str(&format!("[^{}]", fr.name));
            }
            _ => {
                drop(b);
                s.push_str(&collect_text(child));
            }
        }
    }
    s
}

pub fn render_markdown(md: &str, width: u16) -> Vec<Line<'static>> {
    let arena = Arena::new();
    let mut options = ComrakOptions::default();
    options.extension.strikethrough = true;
    options.extension.footnotes = true;
    let root = parse_document(&arena, md, &options);

    let mut visited: HashSet<usize> = HashSet::new();
    let mut lines: Vec<Line<'static>> = Vec::new();

    for node in root.descendants() {
        let ptr = node as *const _ as usize;
        if !visited.insert(ptr) {
            continue;
        }
        let b = node.data.borrow();
        match &b.value {
            NodeValue::Heading(h) => {
                let level = h.level as usize;
                drop(b);
                let text = collect_text(node);
                let prefix = "#".repeat(level);
                let content = format!("{} {}", prefix, text);
                for l in wrap_text(&content, width as usize) {
                    lines.push(Line::from(vec![Span::raw(leak(l))]));
                }
            }
            NodeValue::Paragraph => {
                drop(b);
                let text = collect_text(node);
                for l in wrap_text(&text, width as usize) {
                    lines.push(Line::from(vec![Span::raw(leak(l))]));
                }
            }
            NodeValue::CodeBlock(cb) => {
                let info = cb.info.clone();
                let lit = cb.literal.clone();
                drop(b);
                lines.push(Line::from(vec![Span::raw(leak(format!("```{}", info)))]));
                for l in lit.lines() {
                    lines.push(Line::from(vec![Span::raw(leak(l.to_string()))]));
                }
                lines.push(Line::from(vec![Span::raw(leak("```".to_string()))]));
            }
            NodeValue::ThematicBreak => {
                lines.push(Line::from(vec![Span::raw(leak("──────".to_string()))]));
            }
            NodeValue::BlockQuote => {
                drop(b);
                for child in node.children() {
                    let cb2 = child.data.borrow();
                    if let NodeValue::Paragraph = &cb2.value {
                        drop(cb2);
                        let text = collect_text(child);
                        let inner_w = (width as usize).saturating_sub(4);
                        for l in wrap_text(&text, inner_w) {
                            lines.push(Line::from(vec![
                                Span::raw(leak(format!("  │ {}", l))),
                            ]));
                        }
                    }
                }
            }
            NodeValue::Table(_) => {
                drop(b);
                lines.extend(table::render_table(node, width));
                // Mark all descendants visited to avoid double-render
                for d in node.descendants() {
                    visited.insert(d as *const _ as usize);
                }
            }
            NodeValue::List(_) => {
                drop(b);
                lines.extend(block::render_list(node, width, 0));
                for d in node.descendants() {
                    visited.insert(d as *const _ as usize);
                }
            }
            NodeValue::FootnoteDefinition(def) => {
                let name = def.name.clone();
                drop(b);
                let text = collect_text(node);
                lines.push(Line::from(vec![
                    Span::raw(leak(format!("[^{}]: {}", name, text))),
                ]));
                for d in node.descendants() {
                    visited.insert(d as *const _ as usize);
                }
            }
            NodeValue::HtmlBlock(hb) => {
                let lit = hb.literal.clone();
                drop(b);
                for l in wrap_text(&lit, width as usize) {
                    lines.push(Line::from(vec![Span::raw(leak(l))]));
                }
            }
            _ => {}
        }
    }
    lines
}

pub mod table;
pub mod block;
