use std::sync::Arc;

use serde::Deserialize;

use crate::llm::provider::{LlmProvider, Message, Role};

/// Local representation of a search result suitable for picking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerSearchResult {
    pub url: String,
    pub title: String,
    pub snippet: String,
}

/// A lightweight LLM-based result picker used in Quality mode.
pub struct SearchResultPicker {
    pub provider: Arc<dyn LlmProvider + Send + Sync>,
    pub model: String,
}

impl SearchResultPicker {
    pub fn new(provider: Arc<dyn LlmProvider + Send + Sync>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// Pick the best subset of results using an LLM. Returns up to 3 items.
    pub async fn pick_best_urls(
        &self,
        results: &[PickerSearchResult],
        query: &str,
    ) -> Vec<PickerSearchResult> {
        // Build a compact, explicit prompt that includes the candidate results.
        let mut results_block = String::new();
        for (i, r) in results.iter().enumerate() {
            results_block.push_str(&format!("{}. {} - {}\n", i, r.title, r.snippet));
        }

        // Primary instruction for the picker.
        let system_prompt = r#"You are a high-quality search result picker. Given a list of results (title, snippet, and URL), select 2-3 results that best match the user's query. Prioritize relevance, content quality, reputable sources, and diversity. Return a JSON object with a single field: {"picked_indices": [0, 2]} and ensure at most 3 indices. Do not include anything else in the response. If the answer cannot be produced reliably, fall back to selecting the first 3 results."#;

        let user_prompt = format!("Query: {}\nResults:\n{}", query, results_block);
        let messages = vec![
            Message {
                role: Role::System,
                content: system_prompt.to_string(),
            },
            Message {
                role: Role::User,
                content: user_prompt,
            },
        ];

        // Ask the provider to stream a completion. We don't rely on a final token here.
        let chunks = match self
            .provider
            .stream_completion(&messages, &self.model)
            .await
        {
            Ok(ch) => ch,
            Err(_) => {
                // If the provider fails, fall back to heuristic (top-3 by position).
                return Self::fallback_top3(results);
            }
        };

        let raw = chunks.into_iter().map(|c| c.content).collect::<String>();
        let mut json_text = raw.trim().to_string();
        // Support fenced code blocks sometimes returned by LLMs.
        if json_text.starts_with("```json") {
            json_text = json_text
                .trim_start_matches("```json")
                .trim_end_matches("```")
                .trim()
                .to_string();
        }

        #[derive(Deserialize)]
        struct PickerResponse {
            picked_indices: Vec<usize>,
        }

        if let Ok(resp) = serde_json::from_str::<PickerResponse>(&json_text) {
            // Build the selected list while guarding bounds and duplicates.
            let mut chosen: Vec<PickerSearchResult> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for idx in resp.picked_indices.into_iter() {
                if idx < results.len() && !seen.contains(&idx) {
                    seen.insert(idx);
                    chosen.push(results[idx].clone());
                    if chosen.len() == 3 {
                        break;
                    }
                }
            }
            if !chosen.is_empty() {
                return chosen;
            }
            // If for some reason nothing was chosen, fall back to top-3.
            return Self::fallback_top3(results);
        }

        // Fallback: invalid JSON -> top-3 by position
        Self::fallback_top3(results)
    }

    /// Simple heuristic picker for Speed/Balanced modes: top 3 by position (no LLM).
    pub fn pick_by_heuristic(results: &[PickerSearchResult]) -> Vec<PickerSearchResult> {
        results.iter().cloned().take(3).collect()
    }

    fn fallback_top3(results: &[PickerSearchResult]) -> Vec<PickerSearchResult> {
        results.iter().cloned().take(3).collect()
    }
}

/// Re-export for convenience in tests or other modules.
pub type PickerResult = PickerSearchResult;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider::{Message, ProviderError, Role};
    use std::error::Error;

    struct MockProvider {
        // The JSON to return as the content of a single Chunk
        response: String,
        // simple name
        name: String,
    }

    #[async_trait::async_trait]
    impl LlmProvider for MockProvider {
        async fn stream_completion(
            &self,
            _messages: &[Message],
            _model: &str,
        ) -> Result<Vec<crate::llm::provider::Chunk>, ProviderError> {
            Ok(vec![crate::llm::provider::Chunk {
                content: self.response.clone(),
            }])
        }
        async fn complete_with_tools(
            &self,
            _messages: &[Message],
            _tools: &[crate::llm::provider::ToolDefinition],
            _model: &str,
        ) -> Result<crate::llm::provider::ToolResponse, ProviderError> {
            Ok(crate::llm::provider::ToolResponse {
                content: None,
                tool_calls: vec![],
                finish_reason: crate::llm::provider::FinishReason::Stop,
                usage: None,
            })
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn is_authenticated(&self) -> bool {
            true
        }
        async fn authenticate(&mut self, _api_key: &str) -> Result<(), ProviderError> {
            Ok(())
        }
        async fn deauthenticate(&mut self) -> Result<(), ProviderError> {
            Ok(())
        }
        async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
            Ok(vec!["mock-model".to_string()])
        }
    }

    #[tokio::test]
    async fn picker_llm_parses_valid_json() {
        let results = vec![
            PickerSearchResult {
                url: "https://a.example/1".into(),
                title: "Alpha".into(),
                snippet: "first result".into(),
            },
            PickerSearchResult {
                url: "https://b.example/2".into(),
                title: "Beta".into(),
                snippet: "second result".into(),
            },
            PickerSearchResult {
                url: "https://c.example/3".into(),
                title: "Gamma".into(),
                snippet: "third result".into(),
            },
        ];
        let mock = MockProvider {
            response: r#"{"picked_indices": [0,2]}"#.to_string(),
            name: "mock".to_string(),
        };
        let picker = SearchResultPicker::new(Arc::new(mock), "mock-model");
        let picked = picker.pick_best_urls(&results, "query").await;
        assert_eq!(picked.len(), 2);
        assert_eq!(picked[0].title, "Alpha");
        assert_eq!(picked[1].title, "Gamma");
    }

    #[tokio::test]
    async fn picker_llm_invalid_json_falls_back() {
        let results = vec![
            PickerSearchResult {
                url: "a".into(),
                title: "A".into(),
                snippet: "a".into(),
            },
            PickerSearchResult {
                url: "b".into(),
                title: "B".into(),
                snippet: "b".into(),
            },
            PickerSearchResult {
                url: "c".into(),
                title: "C".into(),
                snippet: "c".into(),
            },
            PickerSearchResult {
                url: "d".into(),
                title: "D".into(),
                snippet: "d".into(),
            },
        ];
        let mock = MockProvider {
            response: b"not json"
                .to_vec()
                .iter()
                .map(|&b| b as char)
                .collect::<String>(),
            name: "mock".to_string(),
        };
        // Build a picker
        let picker = SearchResultPicker::new(Arc::new(mock), "mock-model");
        let picked = picker.pick_best_urls(&results, "query").await;
        // Should fallback to top 3 by position
        assert_eq!(picked.len(), 3);
        assert_eq!(picked[0].title, "A");
        assert_eq!(picked[1].title, "B");
        assert_eq!(picked[2].title, "C");
    }

    #[test]
    fn heuristic_picker_limits() {
        let results = vec![
            PickerSearchResult {
                url: "1".into(),
                title: "t1".into(),
                snippet: "s1".into(),
            },
            PickerSearchResult {
                url: "2".into(),
                title: "t2".into(),
                snippet: "s2".into(),
            },
            PickerSearchResult {
                url: "3".into(),
                title: "t3".into(),
                snippet: "s3".into(),
            },
            PickerSearchResult {
                url: "4".into(),
                title: "t4".into(),
                snippet: "s4".into(),
            },
        ];
        let picked = SearchResultPicker::pick_by_heuristic(&results);
        assert_eq!(picked.len(), 3);
        assert_eq!(picked[0].url, "1");
        assert_eq!(picked[1].url, "2");
        assert_eq!(picked[2].url, "3");
    }
}
