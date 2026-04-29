//! Researcher loop module — context window management and message types.

pub mod context;
pub mod loop_module;
pub mod picker;
pub mod prompt;

use crate::llm::provider::ToolCall;

/// Message variant for the researcher loop.
/// Separate from `Message` in llm/provider.rs (which has Role::System/User/Assistant only).
#[derive(Debug, Clone)]
pub enum ResearcherMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
    },
    AssistantWithToolCalls {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        call_id: String,
        name: String,
        output: String,
    },
}

/// Research depth preset — controls max iterations for the researcher loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResearchDepth {
    Speed, // max 2 iterations
    #[default]
    Balanced, // max 6 iterations (default)
    Quality, // max 25 iterations
}

impl ResearchDepth {
    pub fn max_iterations(&self) -> usize {
        match self {
            Self::Speed => 2,
            Self::Balanced => 6,
            Self::Quality => 25,
        }
    }

    pub fn parse_depth(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "speed" => Self::Speed,
            "quality" => Self::Quality,
            _ => Self::Balanced, // default
        }
    }
}

/// Provider action for mode-specific pipeline behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderAction {
    /// Provider only performs web search, no scraping or extraction.
    SearchOnly,
    /// Provider performs web search and scraping, no fact extraction.
    SearchAndScrape,
    /// Provider performs full pipeline: search, scrape, and fact extraction.
    SearchPickScrapeExtract,
}

/// A single extracted fact from scraped content.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    /// The factual content extracted.
    pub content: String,
    /// URL from which this fact was extracted.
    pub source_url: String,
    /// Title of the source page.
    pub source_title: String,
}

/// Researcher-specific search result with relevance scoring.
/// Distinct from `search::SearchResult` in the search module.
#[derive(Debug, Clone)]
pub struct ResearcherSearchResult {
    /// URL of the search result.
    pub url: String,
    /// Title of the search result.
    pub title: String,
    /// Snippet/summary of the search result.
    pub snippet: String,
    /// Relevance score (0.0 to 1.0).
    pub relevance_score: f32,
}

/// Scraped content with extracted facts.
#[derive(Debug, Clone)]
pub struct ScrapedContent {
    /// URL that was scraped.
    pub url: String,
    /// Title of the scraped page.
    pub title: String,
    /// Raw text content from the page.
    pub raw_text: String,
    /// Facts extracted from the content.
    pub extracted_facts: Vec<ExtractedFact>,
}

// Re-exports
pub use context::ResearcherContext;
pub use loop_module::{ResearcherLoop, ResearcherOutput};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_max_iterations() {
        assert_eq!(ResearchDepth::Speed.max_iterations(), 2);
        assert_eq!(ResearchDepth::Balanced.max_iterations(), 6);
        assert_eq!(ResearchDepth::Quality.max_iterations(), 25);
    }

    #[test]
    fn depth_parse() {
        assert_eq!(ResearchDepth::parse_depth("speed"), ResearchDepth::Speed);
        assert_eq!(
            ResearchDepth::parse_depth("balanced"),
            ResearchDepth::Balanced
        );
        assert_eq!(
            ResearchDepth::parse_depth("quality"),
            ResearchDepth::Quality
        );
        assert_eq!(
            ResearchDepth::parse_depth("unknown"),
            ResearchDepth::Balanced
        );
    }

    #[test]
    fn depth_default() {
        assert_eq!(ResearchDepth::default(), ResearchDepth::Balanced);
    }

    #[test]
    fn researcher_message_variants() {
        let _sys = ResearcherMessage::System {
            content: "sys".into(),
        };
        let _user = ResearcherMessage::User {
            content: "hi".into(),
        };
        let _asst = ResearcherMessage::Assistant {
            content: "hello".into(),
        };
        let _asst_tc = ResearcherMessage::AssistantWithToolCalls {
            content: Some("thinking".into()),
            tool_calls: vec![ToolCall {
                id: "tc1".into(),
                name: "web_search".into(),
                arguments: "{}".into(),
            }],
        };
        let _result = ResearcherMessage::ToolResult {
            call_id: "tc1".into(),
            name: "web_search".into(),
            output: "results".into(),
        };
    }

    #[test]
    fn extracted_fact_construction() {
        let fact = ExtractedFact {
            content: "Rust 1.70 was released in 2023".into(),
            source_url: "https://blog.rust-lang.org/2023/06/01/Rust-1.70.0.html".into(),
            source_title: "Rust 1.70.0".into(),
        };
        assert_eq!(fact.content, "Rust 1.70 was released in 2023");
        assert!(fact.source_url.contains("rust-lang.org"));
        assert_eq!(fact.source_title, "Rust 1.70.0");
    }

    #[test]
    fn researcher_search_result_construction() {
        let result = ResearcherSearchResult {
            url: "https://example.com/article".into(),
            title: "Example Article".into(),
            snippet: "This is a sample snippet.".into(),
            relevance_score: 0.85,
        };
        assert_eq!(result.url, "https://example.com/article");
        assert_eq!(result.title, "Example Article");
        assert_eq!(result.relevance_score, 0.85);
    }

    #[test]
    fn scraped_content_construction() {
        let fact = ExtractedFact {
            content: "Fact content".into(),
            source_url: "https://example.com".into(),
            source_title: "Example".into(),
        };
        let scraped = ScrapedContent {
            url: "https://example.com".into(),
            title: "Example Page".into(),
            raw_text: "Full article text here...".into(),
            extracted_facts: vec![fact],
        };
        assert_eq!(scraped.url, "https://example.com");
        assert!(scraped.raw_text.contains("Full article"));
        assert_eq!(scraped.extracted_facts.len(), 1);
    }

    #[test]
    fn provider_action_variants() {
        assert_eq!(ProviderAction::SearchOnly, ProviderAction::SearchOnly);
        assert_eq!(
            ProviderAction::SearchAndScrape,
            ProviderAction::SearchAndScrape
        );
        assert_eq!(
            ProviderAction::SearchPickScrapeExtract,
            ProviderAction::SearchPickScrapeExtract
        );
    }
}
pub mod extractor;
pub use extractor::ContentExtractor;
