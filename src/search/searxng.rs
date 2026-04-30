use crate::search::{SearchEngine, SearchResult};
use anyhow::{Context, Result};
use reqwest;
use serde::Deserialize;
use std::time::Duration;
use tracing::{debug, warn};

/// SearXNG API response result item.
#[derive(Deserialize)]
struct SearXngResult {
    title: String,
    url: String,
    content: Option<String>,
    #[serde(default)]
    engine: String,
}

/// SearXNG API response structure.
#[derive(Deserialize)]
struct SearXngResponse {
    results: Vec<SearXngResult>,
}

/// SearXNG search client implementation.
pub struct SearXngClient {
    base_url: String,
    client: reqwest::Client,
}

impl SearXngClient {
    /// Create a new SearXNG client with the specified base URL.
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        }
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[async_trait::async_trait]
impl SearchEngine for SearXngClient {
    async fn search(&self, query: &str, categories: &[&str]) -> Result<Vec<SearchResult>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let categories_str = if categories.is_empty() {
            "general".to_string()
        } else {
            categories.join(",")
        };

        let url = format!("{}/search", self.base_url);

        debug!("SearXNG search request: {} (query: {}, categories: {})", url, query, categories_str);

        let response = self
            .client
            .get(&url)
            .query(&[("q", query), ("format", "json"), ("categories", &categories_str)])
            .send()
            .await
            .with_context(|| format!("Failed to connect to SearXNG at {}", self.base_url))?;

        let status = response.status();
        if status == reqwest::StatusCode::FORBIDDEN {
            anyhow::bail!(
                "SearXNG returned 403 Forbidden. The JSON format may be disabled in your SearXNG configuration. \
                 Enable it in settings.yml: search.formats: [html, json]"
            );
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            warn!("SearXNG returned error status: {} - {}", status, body);
            anyhow::bail!("SearXNG API error: {} - {}", status, body);
        }

        let content_type = response.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if content_type.contains("text/html") {
            anyhow::bail!(
                "SearXNG returned HTML instead of JSON. The JSON format may be disabled in your SearXNG configuration. \
                 Enable it in settings.yml: search.formats: [html, json]"
            );
        }

        let searx_response: SearXngResponse = response
            .json()
            .await
            .with_context(|| "Failed to parse SearXNG JSON response")?;

        debug!("SearXNG returned {} results", searx_response.results.len());

        let results = searx_response
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content.unwrap_or_default(),
                engine: if r.engine.is_empty() {
                    "searxng".to_string()
                } else {
                    r.engine
                },
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_searxng_client_creation() {
        let client = SearXngClient::new("http://localhost:8080");
        assert_eq!(client.base_url(), "http://localhost:8080");
    }

    #[test]
    fn test_searxng_client_url_trimming() {
        let client = SearXngClient::new("http://localhost:8080/");
        assert_eq!(client.base_url(), "http://localhost:8080");
    }

    #[tokio::test]
    async fn test_empty_query_returns_empty_results() {
        let client = SearXngClient::new("http://localhost:8080");
        let results = client.search("", &["general"]).await.unwrap();
        assert!(results.is_empty());
    }
}
