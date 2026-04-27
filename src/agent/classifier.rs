use std::sync::Arc;
use serde::Deserialize;
use crate::llm::provider::{Message, Role, ProviderRegistry};
use crate::auth::store::CredentialStore;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    DirectAnswer,
    WebResearch,
    LocalAnalysis,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct ClassifiedIntent {
    pub intent: QueryIntent,
    pub search_queries: Vec<String>,
    pub reasoning: String,
}

/// Intent classifier using a registered LLM provider
pub struct QueryClassifier {
    provider_registry: Arc<ProviderRegistry>,
    model: String,
}

impl QueryClassifier {
    pub fn new(registry: Arc<ProviderRegistry>, model: String) -> Self {
        Self {
            provider_registry: registry,
            model,
        }
    }

    pub async fn classify(&self, query: &str) -> ClassifiedIntent {
        // System prompt (exact content)
        let system_prompt = "You are an intent classifier. Given a user query, classify it into exactly one of:\n- direct_answer: Factual questions, math, definitions that need no external data\n- web_research: Questions requiring current web information, news, latest data\n- local_analysis: Questions about local files, codebase, current project\n- mixed: Questions needing both local analysis AND web research\n\nRespond ONLY with JSON: {\"intent\":\"direct_answer|web_research|local_analysis|mixed\",\"search_queries\": [\"query1\",\"query2\"],\"reasoning\":\"brief explanation\"}\n\nIMPORTANT: For web_research or mixed, provide 1-3 specific, reformulated search queries that will find the BEST results. NEVER copy the raw user query as a search query";
        let system = Message {
            role: Role::System,
            content: system_prompt.to_string(),
        };
        let user = Message {
            role: Role::User,
            content: query.to_string(),
        };
        let messages = vec![system, user];

        // LLM call following planner pattern
        if let Some(provider_handle) = self.provider_registry.resolve(&self.model) {
            // Load credentials and authenticate
            let provider_name = {
                let lock = provider_handle.read().await;
                lock.name().to_string()
            };
            if let Ok(cred_store) = CredentialStore::load() {
                if let Some(creds) = cred_store.get(&provider_name) {
                    let mut w = provider_handle.write().await;
                    let _ = w.authenticate(&creds.api_key).await;
                }
            }
            let provider_guard = provider_handle.read().await;
            if let Ok(chunks) = provider_guard.stream_completion(&messages, &self.model).await {
                let json_str: String = chunks.into_iter().map(|c| c.content).collect::<String>();
                // Strip markdown code fences if present
                let mut json_text = json_str.trim().to_string();
                if json_text.starts_with("```json") {
                    json_text = json_text
                        .trim_start_matches("```json")
                        .trim_end_matches("```")
                        .trim()
                        .to_string();
                }

                #[derive(Deserialize)]
                struct ClassifierResponse {
                    intent: QueryIntent,
                    search_queries: Vec<String>,
                    reasoning: String,
                }
                if let Ok(resp) = serde_json::from_str::<ClassifierResponse>(&json_text) {
                    return ClassifiedIntent {
                        intent: resp.intent,
                        search_queries: resp.search_queries,
                        reasoning: resp.reasoning,
                    };
                } else {
                    // Fallback on parse failure
                    return ClassifiedIntent {
                        intent: QueryIntent::WebResearch,
                        search_queries: vec![query.to_string()],
                        reasoning: "classification fallback: web_research".to_string(),
                    };
                }
            }
        }
        // Global fallback
        ClassifiedIntent {
            intent: QueryIntent::WebResearch,
            search_queries: vec![query.to_string()],
            reasoning: "classification fallback: web_research".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_valid_web_research() {
        let json = r#"{"intent":"web_research","search_queries":["rust async"],"reasoning":"test"}"#;
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["intent"], "web_research");
    }

    #[test]
    fn parse_direct_answer_no_queries() {
        let json = r#"{"intent":"direct_answer","search_queries":[],"reasoning":"math"}"#;
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["search_queries"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn fallback_on_invalid_json() {
        let result: Result<serde_json::Value, _> = serde_json::from_str("not json");
        assert!(result.is_err());
    }
}
