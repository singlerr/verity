//! Agent tool implementations for search, URL reading, and file reading.

pub mod local;
pub mod registry;
pub use local::{sandbox_path, EditFileTool, GlobTool, GrepTool, ListDirTool, ShellTool, WriteFileTool};
pub use registry::{build_tool_registry, tool_manifest};

use crate::fs::read::read_file;
use crate::search::{SearchEngine, SearchResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::future::join_all;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Trait for agent tools.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, input: &Value) -> Result<Value>;
}

/// Search tool using a SearchEngine implementation.
pub struct SearchTool {
    search_engine: Arc<dyn SearchEngine>,
    categories: Vec<String>,
}

impl SearchTool {
    pub fn new(search_engine: Arc<dyn SearchEngine>) -> Self {
        Self {
            search_engine,
            categories: vec!["general".to_string()],
        }
    }

    pub fn with_categories(search_engine: Arc<dyn SearchEngine>, categories: Vec<String>) -> Self {
        let cats = if categories.is_empty() {
            vec!["general".to_string()]
        } else {
            categories
        };
        Self {
            search_engine,
            categories: cats,
        }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let max_queries = 3;
        let queries: Vec<String> =
            if let Some(queries_array) = input.get("queries").and_then(|v| v.as_array()) {
                queries_array
                    .iter()
                    .take(max_queries)
                    .filter_map(|v| v.as_str().map(String::from))
                    .filter(|s| !s.is_empty())
                    .collect()
            } else if let Some(single_query) = input.get("query").and_then(|v| v.as_str()) {
                vec![single_query.to_string()]
            } else {
                return Err(anyhow::anyhow!(
                    "Missing 'query' or 'queries' field in input"
                ));
            };

        if queries.is_empty() {
            return Err(anyhow::anyhow!("No valid queries provided"));
        }

        // Prefer categories from input args, fall back to self.categories
        let cats: Vec<&str> = if let Some(cats_arr) = input.get("categories").and_then(|v| v.as_array()) {
            cats_arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect()
        } else {
            self.categories.iter().map(|s| s.as_str()).collect()
        };
        let cats_ref: &[&str] = if cats.is_empty() { &["general"] } else { &cats };

        let search_futures: Vec<_> = queries
            .iter()
            .map(|query| self.search_engine.search(query, cats_ref))
            .collect();

        let all_results = join_all(search_futures).await;

        let mut seen_urls: HashSet<String> = HashSet::new();
        let mut merged_results: Vec<SearchResult> = Vec::new();
        let mut error_messages: Vec<String> = Vec::new();

        for (idx, result_set) in all_results.into_iter().enumerate() {
            match result_set {
                Ok(results) => {
                    for result in results {
                        if seen_urls.insert(result.url.clone()) {
                            merged_results.push(result);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Search query failed: {}", e);
                    let query = queries.get(idx).map(|s| s.as_str()).unwrap_or("unknown");
                    error_messages.push(format!("Search query '{}' failed: {}", query, e));
                }
            }
        }

        for msg in error_messages {
            merged_results.push(SearchResult {
                title: "Search Error".to_string(),
                url: String::new(),
                snippet: msg,
                engine: "system".to_string(),
            });
        }

        let formatted: Vec<Value> = merged_results
            .into_iter()
            .take(10)
            .map(|r: SearchResult| {
                json!({
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                })
            })
            .collect();

        Ok(json!({"results": formatted}))
    }
}

/// Tool for reading URL content.
pub struct ReadUrlTool {
    client: reqwest::Client,
}

impl ReadUrlTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    fn extract_title(&self, html: &str) -> Option<String> {
        html.find("<title>").and_then(|start| {
            html.find("</title>")
                .map(|end| html[start + 7..end].trim().to_string())
        })
    }
}

#[async_trait]
impl Tool for ReadUrlTool {
    fn name(&self) -> &str {
        "read_url"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .context("Missing 'url' field in input")?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("Failed to fetch URL")?;

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;
        let title = self.extract_title(&html).unwrap_or_else(|| url.to_string());
        let text = html2text::from_read(html.as_bytes(), 80);
        let truncated: String = text.chars().take(5000).collect();

        Ok(json!({"content": truncated, "title": title}))
    }
}

/// Tool for reading local files.
pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    async fn execute(&self, input: &Value) -> Result<Value> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .context("Missing 'path' field in input")?;

        let safe_path = local::sandbox_path(Path::new(path_str))?;
        let range = input.get("range").and_then(|v| {
            v.as_array().and_then(|arr| {
                if arr.len() == 2 {
                    let start = arr[0].as_u64()? as usize;
                    let end = arr[1].as_u64()? as usize;
                    Some((start, end))
                } else {
                    None
                }
            })
        });

        let content = read_file(&safe_path, range)?;

        // Truncate at 10K chars
        const MAX_CHARS: usize = 10_000;
        let output = if content.len() > MAX_CHARS {
            let total_lines = content.lines().count();
            let truncated: String = content.chars().take(MAX_CHARS).collect();
            format!("{}\n\n[truncated: {} total lines. Use range parameter to read specific sections.]", truncated, total_lines)
        } else {
            content
        };

        Ok(json!({"content": output}))
    }
}

/// Registry for managing and dispatching tools.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct MockSearchEngine {
        results: Mutex<HashMap<String, Vec<SearchResult>>>,
    }

    impl MockSearchEngine {
        fn new() -> Self {
            Self {
                results: Mutex::new(HashMap::new()),
            }
        }

        fn with_result(self, query: &str, results: Vec<SearchResult>) -> Self {
            self.results
                .lock()
                .unwrap()
                .insert(query.to_string(), results);
            self
        }
    }

    #[async_trait]
    impl SearchEngine for MockSearchEngine {
        async fn search(&self, query: &str, _categories: &[&str]) -> Result<Vec<SearchResult>> {
            let results = self.results.lock().unwrap();
            Ok(results.get(query).cloned().unwrap_or_default())
        }
    }

    fn make_result(title: &str, url: &str, snippet: &str) -> SearchResult {
        SearchResult {
            title: title.to_string(),
            url: url.to_string(),
            snippet: snippet.to_string(),
            engine: "mock".to_string(),
        }
    }

    #[tokio::test]
    async fn test_search_single_query() {
        let mock = MockSearchEngine::new().with_result(
            "rust programming",
            vec![
                make_result(
                    "Rust Book",
                    "https://rust-lang.org/book",
                    "The Rust programming language",
                ),
                make_result(
                    "Rust By Example",
                    "https://rust-lang.org/examples",
                    "Learn Rust by example",
                ),
            ],
        );
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({"query": "rust programming"});
        let result = tool.execute(&input).await.unwrap();

        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].get("title").unwrap().as_str().unwrap(),
            "Rust Book"
        );
    }

    #[tokio::test]
    async fn test_search_multiple_queries() {
        let mock = MockSearchEngine::new()
            .with_result(
                "rust programming",
                vec![make_result(
                    "Rust Book",
                    "https://rust-lang.org/book",
                    "The Rust programming language",
                )],
            )
            .with_result(
                "cargo tutorial",
                vec![make_result(
                    "Cargo Guide",
                    "https://doc.rust-lang.org/cargo",
                    "Cargo documentation",
                )],
            );
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({"queries": ["rust programming", "cargo tutorial"]});
        let result = tool.execute(&input).await.unwrap();

        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_deduplicates_by_url() {
        let mock = MockSearchEngine::new()
            .with_result(
                "rust",
                vec![make_result(
                    "Rust Book",
                    "https://rust-lang.org/book",
                    "The Rust programming language",
                )],
            )
            .with_result(
                "rust lang",
                vec![
                    make_result(
                        "Rust Book Duplicate",
                        "https://rust-lang.org/book",
                        "Same URL different title",
                    ),
                    make_result("Rust Blog", "https://blog.rust-lang.org", "The Rust blog"),
                ],
            );
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({"queries": ["rust", "rust lang"]});
        let result = tool.execute(&input).await.unwrap();

        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_caps_at_3_queries() {
        let mock = MockSearchEngine::new()
            .with_result(
                "q1",
                vec![make_result("R1", "https://example.com/1", "Snippet 1")],
            )
            .with_result(
                "q2",
                vec![make_result("R2", "https://example.com/2", "Snippet 2")],
            )
            .with_result(
                "q3",
                vec![make_result("R3", "https://example.com/3", "Snippet 3")],
            )
            .with_result(
                "q4",
                vec![make_result("R4", "https://example.com/4", "Snippet 4")],
            );
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({"queries": ["q1", "q2", "q3", "q4"]});
        let result = tool.execute(&input).await.unwrap();

        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_search_returns_max_10_results() {
        let many_results: Vec<SearchResult> = (0..15)
            .map(|i| {
                make_result(
                    &format!("Title {}", i),
                    &format!("https://example.com/{}", i),
                    "Snippet",
                )
            })
            .collect();

        let mock = MockSearchEngine::new().with_result("many", many_results);
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({"query": "many"});
        let result = tool.execute(&input).await.unwrap();

        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 10);
    }

    #[tokio::test]
    async fn test_search_missing_query_field() {
        let mock = MockSearchEngine::new();
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({});
        let result = tool.execute(&input).await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Missing 'query' or 'queries' field"));
    }

    #[tokio::test]
    async fn test_search_empty_queries_array() {
        let mock = MockSearchEngine::new();
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({"queries": []});
        let result = tool.execute(&input).await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No valid queries provided"));
    }

    #[tokio::test]
    async fn test_search_backward_compatibility() {
        let mock = MockSearchEngine::new().with_result(
            "test query",
            vec![make_result("Result", "https://example.com", "Test snippet")],
        );
        let tool = SearchTool::new(Arc::new(mock));

        let input = json!({"query": "test query"});
        let result = tool.execute(&input).await.unwrap();

        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn sandbox_allows_nested_creation() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let nested = tmpdir.path().join("sub").join("new_file.txt");
        std::fs::create_dir(tmpdir.path().join("sub")).unwrap();
        let result = local::sandbox_path(&nested);
        assert!(result.is_ok(), "Creating new files in existing CWD subdirectories should be allowed");

        drop(tmpdir);
    }

    #[tokio::test]
    async fn read_file_truncates_large_output() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("large.txt");
        let large_content: String = "x".repeat(15_000);
        std::fs::write(&test_file, &large_content).unwrap();

        let tool = ReadFileTool::new();
        let input = json!({"path": test_file.to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let content = result.get("content").unwrap().as_str().unwrap();
        assert!(content.contains("[truncated:"), "Large files should be truncated");
        assert!(content.len() < 15_000, "Output should be shorter than original");

        drop(tmpdir);
    }

    #[tokio::test]
    async fn read_file_small_file_not_truncated() {
        let cwd = std::env::current_dir().unwrap();
        let tmpdir = tempfile::tempdir_in(&cwd).unwrap();
        let test_file = tmpdir.path().join("small.txt");
        std::fs::write(&test_file, "Hello World").unwrap();

        let tool = ReadFileTool::new();
        let input = json!({"path": test_file.to_str().unwrap()});
        let result = tool.execute(&input).await.unwrap();

        let content = result.get("content").unwrap().as_str().unwrap();
        assert_eq!(content, "Hello World");
        assert!(!content.contains("[truncated:"), "Small files should not be truncated");

        drop(tmpdir);
    }

    #[tokio::test]
    async fn test_search_error_propagation() {
        struct FailingSearchEngine;
        #[async_trait]
        impl SearchEngine for FailingSearchEngine {
            async fn search(&self, _query: &str, _categories: &[&str]) -> Result<Vec<SearchResult>> {
                anyhow::bail!("connection refused")
            }
        }
        let tool = SearchTool::new(Arc::new(FailingSearchEngine));
        let input = json!({"query": "test"});
        let result = tool.execute(&input).await.unwrap();
        let results = result.get("results").unwrap().as_array().unwrap();
        assert!(!results.is_empty());
        let error_result = &results[0];
        assert_eq!(error_result.get("title").unwrap().as_str().unwrap(), "Search Error");
        assert!(error_result.get("snippet").unwrap().as_str().unwrap().contains("connection refused"));
    }

    #[tokio::test]
    async fn test_search_with_categories_constructor() {
        struct CategoryRecordingEngine {
            categories: Mutex<Vec<String>>,
            results: Mutex<HashMap<String, Vec<SearchResult>>>,
        }
        impl CategoryRecordingEngine {
            fn new() -> Self {
                Self {
                    categories: Mutex::new(Vec::new()),
                    results: Mutex::new(HashMap::new()),
                }
            }
            fn with_result(self, query: &str, results: Vec<SearchResult>) -> Self {
                self.results.lock().unwrap().insert(query.to_string(), results);
                self
            }
        }
        #[async_trait]
        impl SearchEngine for CategoryRecordingEngine {
            async fn search(&self, query: &str, categories: &[&str]) -> Result<Vec<SearchResult>> {
                let mut cats = self.categories.lock().unwrap();
                cats.clear();
                cats.extend(categories.iter().map(|&s| s.to_string()));
                let results = self.results.lock().unwrap();
                Ok(results.get(query).cloned().unwrap_or_default())
            }
        }

        let mock = CategoryRecordingEngine::new()
            .with_result("test query", vec![make_result("Result", "https://example.com", "Test snippet")]);
        let tool = SearchTool::with_categories(Arc::new(mock), vec!["news".to_string()]);

        let input = json!({"query": "test query"});
        let _result = tool.execute(&input).await.unwrap();
    }

    #[tokio::test]
    async fn test_search_categories_from_input_override() {
        struct CategoryRecordingEngine {
            categories: Mutex<Vec<String>>,
            results: Mutex<HashMap<String, Vec<SearchResult>>>,
        }
        impl CategoryRecordingEngine {
            fn new() -> Self {
                Self {
                    categories: Mutex::new(Vec::new()),
                    results: Mutex::new(HashMap::new()),
                }
            }
            fn with_result(self, query: &str, results: Vec<SearchResult>) -> Self {
                self.results.lock().unwrap().insert(query.to_string(), results);
                self
            }
            fn get_categories(&self) -> Vec<String> {
                self.categories.lock().unwrap().clone()
            }
        }
        #[async_trait]
        impl SearchEngine for CategoryRecordingEngine {
            async fn search(&self, query: &str, categories: &[&str]) -> Result<Vec<SearchResult>> {
                let mut cats = self.categories.lock().unwrap();
                cats.clear();
                cats.extend(categories.iter().map(|&s| s.to_string()));
                let results = self.results.lock().unwrap();
                Ok(results.get(query).cloned().unwrap_or_default())
            }
        }

        let mock = Arc::new(CategoryRecordingEngine::new()
            .with_result("test query", vec![make_result("Result", "https://example.com", "Test snippet")]));
        let tool = SearchTool::with_categories(mock.clone(), vec!["general".to_string()]);

        let input = json!({"query": "test query", "categories": ["news"]});
        let _result = tool.execute(&input).await.unwrap();

        let received_categories = mock.get_categories();
        assert_eq!(received_categories, vec!["news"]);
    }
}
