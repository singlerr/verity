use std::sync::Arc;
use crate::app::Source;
use crate::agent::researcher::ResearchDepth;
use crate::llm::provider::{LlmProvider, Message, Role};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citation {
    pub index: usize,
    pub source_url: String,
    pub source_title: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesisOutput {
    pub text: String,
    pub citations: Vec<Citation>,
}

pub struct ResearchSynthesizer {
    provider: Arc<dyn LlmProvider>,
    model: String,
}

impl ResearchSynthesizer {
    pub fn new(provider: Arc<dyn LlmProvider>, model: impl Into<String>) -> Self {
        Self { provider, model: model.into() }
    }

    pub async fn synthesize(
        &self,
        query: &str,
        sources: &[Source],
        depth: ResearchDepth,
    ) -> Result<SynthesisOutput, String> {
        let messages = vec![
            Message { role: Role::System, content: Self::system_prompt(depth) },
            Message { role: Role::User, content: Self::build_context(query, sources) },
        ];
        let chunks = self.provider.stream_completion(&messages, &self.model).await
            .map_err(|e| format!("stream_completion failed: {}", e))?;
        let text: String = chunks.into_iter().map(|c| c.content).collect();
        let citations = Self::extract_citations(&text, sources);
        Ok(SynthesisOutput { text, citations })
    }

    fn system_prompt(depth: ResearchDepth) -> String {
        let base = "You are a research assistant. Every factual claim MUST cite its source using [N] notation, where N is the source index in the provided context.";
        match depth {
            ResearchDepth::Speed => format!("{base} Provide a brief, concise answer."),
            ResearchDepth::Balanced => format!("{base} Provide a balanced, well-reasoned answer."),
            ResearchDepth::Quality => format!("{base} Provide a comprehensive research report, at least 2000 words."),
        }
    }

    fn build_context(query: &str, sources: &[Source]) -> String {
        let mut xml = format!("<query>{}</query>\n<search_results>\n", Self::escape_xml(query));
        for (idx, s) in sources.iter().enumerate() {
            xml.push_str(&format!(
                r#"<result index="{}" url="{}" title="{}">{}</result>"#,
                idx + 1,
                Self::escape_xml(&s.url),
                Self::escape_xml(&s.title),
                Self::escape_xml(&s.snippet),
            ));
            xml.push('\n');
        }
        xml.push_str("</search_results>");
        xml
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
    }

    fn extract_citations(text: &str, sources: &[Source]) -> Vec<Citation> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '[' {
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
                if j > i + 1 && j < chars.len() && chars[j] == ']' {
                    let num: usize = chars[i + 1..j].iter().collect::<String>().parse().unwrap_or(0);
                    if num > 0 {
                        if let Some(source) = sources.get(num - 1) {
                            if seen.insert(num) {
                                out.push(Citation {
                                    index: num,
                                    source_url: source.url.clone(),
                                    source_title: source.title.clone(),
                                });
                            }
                        }
                    }
                }
            }
            i += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source(num: usize) -> Source {
        Source {
            num,
            domain: format!("example{}.com", num),
            title: format!("Title {}", num),
            url: format!("https://example{}.com/page", num),
            snippet: format!("Snippet {}", num),
            quote: format!("Quote {}", num),
        }
    }

    #[test]
    fn citation_parsing() {
        let sources = vec![test_source(1), test_source(2), test_source(3)];
        let text = "According to [1] and [2], this is true. Also [1].";
        let citations = ResearchSynthesizer::extract_citations(text, &sources);
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0].index, 1);
        assert_eq!(citations[1].index, 2);
    }

    #[test]
    fn xml_template_construction() {
        let sources = vec![test_source(1)];
        let xml = ResearchSynthesizer::build_context("test & query", &sources);
        assert!(xml.contains("<query>test &amp; query</query>"));
        assert!(xml.contains(r#"index="1""#));
        assert!(xml.contains("https://example1.com/page"));
        assert!(xml.contains("Title 1"));
    }

    #[test]
    fn depth_prompt_variation() {
        let speed = ResearchSynthesizer::system_prompt(ResearchDepth::Speed);
        let quality = ResearchSynthesizer::system_prompt(ResearchDepth::Quality);
        assert_ne!(speed, quality);
        assert!(quality.contains("2000 words"));
        assert!(speed.contains("brief"));
    }
}
