use std::sync::Arc;
use std::collections::HashSet;

use serde::Deserialize;

use crate::llm::provider::{LlmProvider, Message, Role, Chunk};
use crate::agent::researcher::ExtractedFact;

// Public extractor interface for use by Task 6 and beyond
// NOTE: We reuse the ExtractedFact type defined in the researcher crate
pub struct ContentExtractor {
    llm: Arc<dyn LlmProvider>,
    model: String,
}

// Extracted facts are defined in the parent researcher module and re-exported
// via crate::agent::researcher::ExtractedFact

// Extractor prompt following Vane's pattern
const EXTRACTOR_PROMPT: &str = r#"You are a precise information extractor. For the provided content chunk and query, extract only factual information relevant to the query.
- Ignore marketing fluff such as "best-in-class" or "seamless".
- Filter noisy UI/headers/footers and non-substantive content.
- Preserve numerical data exactly; do not summarize numbers.
- Output must be in bullet format and in JSON: {"extracted_facts": "- fact1\n- fact2"}.
"#;

// Chunking configuration (as specified in task)
const CHUNK_SIZE: usize = 4000;
const OVERLAP: usize = 500;

impl ContentExtractor {
    pub fn new(llm: Arc<dyn LlmProvider>, model: String) -> Self {
        Self { llm, model }
    }
    pub async fn extract_facts(&self, content: &str, query: &str) -> Vec<ExtractedFact> {
        let chunks = Self::split_text(content, CHUNK_SIZE, OVERLAP);
        let mut facts: Vec<ExtractedFact> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for chunk in chunks {
            // Build a minimal prompt for the LLM. We pass a stable system prompt and the chunk+query in user content.
            let prompt_for_user = format!("Query: {}\nContent: {}", query, chunk);
            let messages = vec![
                Message { role: Role::System, content: EXTRACTOR_PROMPT.to_string() },
                Message { role: Role::User, content: prompt_for_user },
            ];

            match self.llm.stream_completion(&messages, &self.model).await {
                Ok(res) => {
                    // Concatenate all chunk responses into a single string for JSON parsing
                    let json_text: String = res.into_iter().map(|c| c.content).collect();
                    // Expect a JSON payload like {"extracted_facts": "- fact1\n- fact2"}
                    #[derive(Deserialize)]
                    struct JsonWrapper {
                        extracted_facts: String,
                    }
                    let parsed: Result<JsonWrapper, _> = serde_json::from_str(&json_text);
                    match parsed {
                        Ok(wrapper) => {
                            for line in wrapper.extracted_facts.lines() {
                                let t = line.trim();
                                if t.starts_with('-') {
                                    let fact = t.trim_start_matches('-').trim().to_string();
                                    if !fact.is_empty() && seen.insert(fact.clone()) {
                                        facts.push(ExtractedFact {
                                            content: fact,
                                            source_url: String::new(),
                                            source_title: String::new(),
                                        });
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Fallback: use the raw chunk text as a single fact (truncate later if needed)
                            let raw = chunk.clone();
                            let truncated = if raw.len() > 3000 { raw[..3000].to_string() } else { raw };
                            facts.push(ExtractedFact {
                                content: truncated,
                                source_url: String::new(),
                                source_title: String::new(),
                            });
                        }
                    }
                }
                Err(_e) => {
                    // LLM call failed gracefully; fallback to truncated chunk
                    let truncated = if chunk.len() > 3000 { chunk[..3000].to_string() } else { chunk.clone() };
                    facts.push(ExtractedFact {
                        content: truncated,
                        source_url: String::new(),
                        source_title: String::new(),
                    });
                }
            }
        }

        facts
    }

    // Helper: split text into chunks with overlap
    pub(crate) fn split_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks: Vec<String> = Vec::new();
        if chunk_size == 0 || chars.is_empty() {
            chunks.push(text.to_string());
            return chunks;
        }
        let mut start: usize = 0;
        while start < chars.len() {
            let end = std::cmp::min(start + chunk_size, chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);
            if end == chars.len() { break; }
            if end > overlap {
                start = end - overlap;
            } else {
                start = 0;
            }
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Dummy provider that always returns a fixed JSON payload for testing extraction path
    struct DummyProvider;
    impl LlmProvider for DummyProvider {
        fn stream_completion(&self, _messages: &[Message], _model: &str) -> Result<Vec<Chunk>, crate::llm::provider::ProviderError> {
            Ok(vec![Chunk { content: r#"{"extracted_facts": "- Fact A\n- Fact B"}"#.to_string() }])
        }
    }

    #[test]
    fn test_split_text_chunking() {
        let text: String = "a".repeat(10000);
        let chunks = ContentExtractor::split_text(&text, 4000, 500);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 4000);
        assert_eq!(chunks[1].len(), 4000);
        assert_eq!(chunks[2].len(), 3000);
    }

    #[test]
    fn test_extraction_parses_json() {
        let provider = Arc::new(DummyProvider);
        let extractor = ContentExtractor::new(provider, "test-model".to_string());
        let content = "This is some content to test extraction.";
        let facts = futures::executor::block_on(extractor.extract_facts(content, "query"));
        assert!(!facts.is_empty());
        assert_eq!(facts[0].content, "Fact A");
    }

    #[test]
    fn test_fallback_on_bad_json() {
        struct BadJsonProvider;
        impl LlmProvider for BadJsonProvider {
            fn stream_completion(&self, _messages: &[Message], _model: &str) -> Result<Vec<Chunk>, crate::llm::provider::ProviderError> {
                Ok(vec![Chunk { content: "not-json-output".to_string() }])
            }
        }
        let provider = Arc::new(BadJsonProvider);
        let extractor = ContentExtractor::new(provider, "test-model".to_string());
        let content = "Short text.";
        let facts = futures::executor::block_on(extractor.extract_facts(content, "query"));
        // Should produce at least one fact from the fallback
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "not-json-output");
    }
}
