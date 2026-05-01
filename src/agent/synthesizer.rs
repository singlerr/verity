use std::sync::Arc;

use crate::agent::researcher::{ExtractedFact, ResearchDepth};
use crate::app::Source;
use crate::llm::provider::{LlmProvider, Message, Role};
use tracing::{debug, info, warn};

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
        Self {
            provider,
            model: model.into(),
        }
    }

    pub async fn synthesize(
        &self,
        query: &str,
        sources: &[Source],
        depth: ResearchDepth,
        extracted_facts: &[ExtractedFact],
    ) -> Result<SynthesisOutput, String> {
        let xml_context = Self::build_context(query, sources, extracted_facts);
        debug!("XML context size: {} chars", xml_context.len());
        let messages = vec![
            Message {
                role: Role::System,
                content: Self::system_prompt(depth),
            },
            Message {
                role: Role::User,
                content: xml_context,
            },
        ];
        info!(
            "Synthesizing with {} facts and {} sources",
            extracted_facts.len(),
            sources.len()
        );
        if sources.is_empty() && extracted_facts.is_empty() {
            warn!(
                "No usable sources or facts for synthesis; LLM may produce empty or ungrounded answer"
            );
        }
        let chunks = self
            .provider
            .stream_completion(&messages, &self.model)
            .await
            .map_err(|e| format!("stream_completion failed: {}", e))?;
        let text: String = chunks.into_iter().map(|c| c.content).collect();
        let citations = Self::extract_citations(&text, sources);
        Ok(SynthesisOutput { text, citations })
    }

    fn system_prompt(depth: ResearchDepth) -> String {
        let base = "You are a research assistant. Every factual claim MUST cite its source using [N] notation, where N is the source index in the provided context.";
        match depth {
            ResearchDepth::Speed => format!("{base} Provide a brief, concise answer using the provided facts and sources."),
            ResearchDepth::Balanced => format!("{base} Provide a balanced, well-reasoned answer. Cite sources using [N] notation."),
            ResearchDepth::Quality => format!("{base} Provide a comprehensive research report with full citations. Use [N] notation for every factual claim. Base your answer primarily on the extracted facts."),
        }
    }

    fn build_context(query: &str, sources: &[Source], facts: &[ExtractedFact]) -> String {
        let mut xml = format!("<query>{}</query>\n", Self::escape_xml(query));

        if !facts.is_empty() {
            xml.push_str("<extracted_facts>\n");
            for fact in facts {
                xml.push_str(&format!(
                    r#"<fact source="{}" title="{}">{}</fact>"#,
                    Self::escape_xml(&fact.source_url),
                    Self::escape_xml(&fact.source_title),
                    Self::escape_xml(&fact.content),
                ));
                xml.push('\n');
            }
            xml.push_str("</extracted_facts>\n");
        }

        xml.push_str("<search_results>\n");
        for (idx, s) in sources
            .iter()
            .filter(|s| !s.snippet.trim().is_empty())
            .enumerate()
        {
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
        if sources.iter().filter(|s| !s.snippet.trim().is_empty()).count() == 0 {
            warn!("All search results have empty snippets; no results in synthesis context");
        }
        xml
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
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
                    let num: usize = chars[i + 1..j]
                        .iter()
                        .collect::<String>()
                        .parse()
                        .unwrap_or(0);
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
    fn xml_template_construction_without_facts() {
        let sources = vec![test_source(1)];
        let facts: Vec<ExtractedFact> = vec![];
        let xml = ResearchSynthesizer::build_context("test & query", &sources, &facts);
        assert!(xml.contains("<query>test &amp; query</query>"));
        assert!(xml.contains(r#"index="1""#));
        assert!(xml.contains("https://example1.com/page"));
        assert!(xml.contains("Title 1"));
        assert!(!xml.contains("<extracted_facts>"));
    }

    #[test]
    fn xml_template_construction_with_facts() {
        let sources = vec![test_source(1)];
        let facts = vec![ExtractedFact {
            content: "This is a fact".into(),
            source_url: "https://example.com/source".into(),
            source_title: "Source Title".into(),
        }];
        let xml = ResearchSynthesizer::build_context("test query", &sources, &facts);
        assert!(xml.contains("<query>test query</query>"));
        assert!(xml.contains("<extracted_facts>"));
        assert!(xml.contains(r#"<fact source="https://example.com/source" title="Source Title">This is a fact</fact>"#));
        assert!(xml.contains("<search_results>"));
    }

    #[test]
    fn depth_prompt_variation() {
        let speed = ResearchSynthesizer::system_prompt(ResearchDepth::Speed);
        let balanced = ResearchSynthesizer::system_prompt(ResearchDepth::Balanced);
        let quality = ResearchSynthesizer::system_prompt(ResearchDepth::Quality);
        assert_ne!(speed, quality);
        assert!(speed.contains("facts and sources"));
        assert!(balanced.contains("balanced, well-reasoned"));
        assert!(quality.contains("comprehensive research report"));
        assert!(quality.contains("extracted facts"));
    }

    #[test]
    fn synthesize_uses_facts_in_context() {
        let sources = vec![test_source(1)];
        let facts = vec![ExtractedFact {
            content: "This is a fact".into(),
            source_url: "https://example.com/source".into(),
            source_title: "Source Title".into(),
        }];
        let xml = ResearchSynthesizer::build_context("test query", &sources, &facts);

        assert!(xml.contains("<extracted_facts>"));
        assert!(xml.contains(r#"<fact source="https://example.com/source" title="Source Title">This is a fact</fact>"#));
        assert!(xml.contains("</extracted_facts>"));
        assert!(xml.contains("<search_results>"));
    }

    #[test]
    fn synthesize_fallback_no_facts() {
        let sources = vec![test_source(1)];
        let facts: Vec<ExtractedFact> = vec![];
        let xml = ResearchSynthesizer::build_context("test query", &sources, &facts);

        assert!(!xml.contains("<extracted_facts>"));
        assert!(xml.contains("<query>test query</query>"));
        assert!(xml.contains("<search_results>"));
        assert!(xml.contains(r#"index="1""#));
    }

    #[test]
    fn synthesize_multiple_facts() {
        let sources = vec![test_source(1), test_source(2)];
        let facts = vec![
            ExtractedFact {
                content: "Fact one".into(),
                source_url: "https://example.com/1".into(),
                source_title: "Source 1".into(),
            },
            ExtractedFact {
                content: "Fact two".into(),
                source_url: "https://example.com/2".into(),
                source_title: "Source 2".into(),
            },
        ];
        let xml = ResearchSynthesizer::build_context("multi fact query", &sources, &facts);

        assert!(xml.contains("<extracted_facts>"));
        assert!(xml.contains("Fact one"));
        assert!(xml.contains("Fact two"));
        assert!(xml.contains("https://example.com/1"));
        assert!(xml.contains("https://example.com/2"));
    }

    #[test]
    fn synthesize_xml_escaping() {
        let sources = vec![Source {
            num: 1,
            domain: "example.com".into(),
            title: "Test & Example <tag>".into(),
            url: "https://example.com/page?a=1&b=2".into(),
            snippet: "Snippet with <br> & quotes \"test\"".into(),
            quote: "".into(),
        }];
        let facts = vec![ExtractedFact {
            content: "Content with <special> & chars".into(),
            source_url: "https://example.com?x=1&y=2".into(),
            source_title: "Title & More".into(),
        }];
        let xml = ResearchSynthesizer::build_context("test <&> query", &sources, &facts);

        assert!(xml.contains("&amp;"));
        assert!(!xml.contains("&example"));
        assert!(xml.contains("&lt;special&gt;"));
        assert!(!xml.contains("<special>"));
    }

    #[test]
    fn test_empty_snippet_filtering() {
        let sources = vec![
            Source {
                num: 1,
                domain: "good.com".into(),
                title: "Good Result".into(),
                url: "https://good.com/page".into(),
                snippet: "This has real content".into(),
                quote: "".into(),
            },
            Source {
                num: 2,
                domain: "empty.com".into(),
                title: "Empty Snippet".into(),
                url: "https://empty.com/page".into(),
                snippet: "".into(),
                quote: "".into(),
            },
            Source {
                num: 3,
                domain: "whitespace.com".into(),
                title: "Whitespace Only".into(),
                url: "https://whitespace.com/page".into(),
                snippet: "   ".into(),
                quote: "".into(),
            },
        ];
        let facts: Vec<ExtractedFact> = vec![];
        let xml = ResearchSynthesizer::build_context("test query", &sources, &facts);

        assert!(xml.contains(r#"index="1""#));
        assert!(xml.contains("https://good.com/page"));
        assert!(xml.contains("Good Result"));
        assert!(xml.contains("This has real content"));

        assert!(!xml.contains("https://empty.com/page"));
        assert!(!xml.contains("Empty Snippet"));
        assert!(!xml.contains("https://whitespace.com/page"));
        assert!(!xml.contains("Whitespace Only"));

        let result_count = xml.matches("<result ").count();
        assert_eq!(result_count, 1, "Expected exactly 1 <result> element, found {}", result_count);
    }
}
