//! Search engine integration module.

pub mod searxng;

use anyhow::Result;

/// A single search result from any search engine.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Title of the search result.
    pub title: String,
    /// URL of the search result.
    pub url: String,
    /// Snippet/summary of the search result.
    pub snippet: String,
    /// Source search engine identifier.
    pub engine: String,
}

/// Trait for search engine implementations.
#[async_trait::async_trait]
pub trait SearchEngine: Send + Sync {
    /// Perform a search query and return results.
    async fn search(&self, query: &str) -> Result<Vec<SearchResult>>;
}

pub use searxng::SearXngClient;
