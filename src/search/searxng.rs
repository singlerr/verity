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
            anyhow::bail!("Missing or empty search query");
        }

        let categories_str = if categories.is_empty() {
            "general".to_string()
        } else {
            categories.join(",")
        };

        let url = format!("{}/search", self.base_url);

        debug!(
            "SearXNG search request: {} (query: {}, categories: {})",
            url, query, categories_str
        );

        let response = self
            .client
            .get(&url)
            .query(&[
                ("q", query),
                ("format", "json"),
                ("categories", &categories_str),
            ])
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

        let content_type = response
            .headers()
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
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn mount_search_mock(
        server: &MockServer,
        query: &str,
        categories: &str,
        response: ResponseTemplate,
    ) {
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", query))
            .and(query_param("format", "json"))
            .and(query_param("categories", categories))
            .respond_with(response)
            .mount(server)
            .await;
    }

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
    async fn test_empty_query_is_rejected() {
        let client = SearXngClient::new("http://localhost:8080");
        let err = client.search("", &["general"]).await.unwrap_err();
        assert!(err.to_string().contains("Missing or empty search query"));
    }

    #[tokio::test]
    async fn test_search_success_parses_results_and_sends_expected_query_params() {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                {
                    "title": "Rust",
                    "url": "https://www.rust-lang.org/",
                    "content": "Rust is a language",
                    "engine": "duckduckgo"
                },
                {
                    "title": "The Book",
                    "url": "https://doc.rust-lang.org/book/",
                    "content": null
                }
            ]
        }));

        mount_search_mock(&server, "rust", "general,news", response).await;

        let client = SearXngClient::new(&server.uri());
        let results = client.search("rust", &["general", "news"]).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust");
        assert_eq!(results[0].url, "https://www.rust-lang.org/");
        assert_eq!(results[0].snippet, "Rust is a language");
        assert_eq!(results[0].engine, "duckduckgo");
        assert_eq!(results[1].title, "The Book");
        assert_eq!(results[1].url, "https://doc.rust-lang.org/book/");
        assert_eq!(results[1].snippet, "");
        assert_eq!(results[1].engine, "searxng");
    }

    #[tokio::test]
    async fn test_search_empty_json_results_returns_empty_vec() {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_json(json!({
            "results": []
        }));

        mount_search_mock(&server, "empty", "general", response).await;

        let client = SearXngClient::new(&server.uri());
        let results = client.search("empty", &[]).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_http_error_is_returned() {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(500).set_body_string("internal server error");

        mount_search_mock(&server, "error", "general", response).await;

        let client = SearXngClient::new(&server.uri());
        let err = client.search("error", &[]).await.unwrap_err();
        let message = err.to_string();

        assert!(message.contains("SearXNG API error: 500"), "{message}");
        assert!(message.contains("internal server error"), "{message}");
    }

    #[tokio::test]
    async fn test_search_html_response_is_rejected() {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_raw(
            "<html><body>no json</body></html>".as_bytes().to_vec(),
            "text/html; charset=utf-8",
        );

        mount_search_mock(&server, "html", "general", response).await;

        let client = SearXngClient::new(&server.uri());
        let err = client.search("html", &[]).await.unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("SearXNG returned HTML instead of JSON"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn test_search_malformed_json_is_rejected() {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200)
            .set_body_raw("{not valid json".as_bytes().to_vec(), "application/json");

        mount_search_mock(&server, "bad-json", "general", response).await;

        let client = SearXngClient::new(&server.uri());
        let err = client.search("bad-json", &[]).await.unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("Failed to parse SearXNG JSON response"),
            "{message}"
        );
    }
}
