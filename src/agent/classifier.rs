use std::sync::Arc;
use serde::Deserialize;
use crate::llm::provider::{Message, Role, ProviderRegistry};
use crate::auth::store::CredentialStore;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    General,
    News,
    Science,
    It,
}

impl SourceType {
    pub fn as_searxng_category(&self) -> &'static str {
        match self {
            SourceType::General => "general",
            SourceType::News => "news",
            SourceType::Science => "science",
            SourceType::It => "it",
        }
    }
}

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
    pub skip_search: bool,
    pub source_types: Vec<SourceType>,
    pub quality: bool,
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
        let system_prompt = "You are an intent classifier. Given a user query, classify it into exactly one of:
- direct_answer: Factual questions, math, definitions that need no external data
- web_research: Questions requiring current web information, news, latest data
- local_analysis: Questions about local files, codebase, current project
- mixed: Questions needing both local analysis AND web research

For web_research or mixed, you can also set \"quality\": true when the query requires deep, multi-angle research with comprehensive coverage (e.g., complex topics needing thorough investigation, comparisons, or detailed analysis).

Respond ONLY with JSON: {\"intent\":\"direct_answer|web_research|local_analysis|mixed\",\"search_queries\":[\"query1\",\"query2\"],\"reasoning\":\"brief explanation\",\"skip_search\":false,\"source_types\":[\"general\",\"news\",\"science\",\"it\"],\"quality\":false}

IMPORTANT: For web_research or mixed, provide 1-3 specific, reformulated search queries. NEVER copy the raw user query. Set skip_search to true only for direct_answer. Set source_types to filter SearXNG categories (general, news, science, it). Set quality to true for complex research-heavy queries needing deep coverage.";
        let system = Message {
            role: Role::System,
            content: system_prompt.to_string(),
        };
        let user = Message {
            role: Role::User,
            content: query.to_string(),
        };
        let messages = vec![system, user];

        if let Some(provider_handle) = self.provider_registry.resolve(&self.model) {
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
                    skip_search: Option<bool>,
                    source_types: Option<Vec<SourceType>>,
                    quality: Option<bool>,
                }
                if let Ok(resp) = serde_json::from_str::<ClassifierResponse>(&json_text) {
                    return ClassifiedIntent {
                        intent: resp.intent,
                        search_queries: resp.search_queries,
                        reasoning: resp.reasoning,
                        skip_search: resp.skip_search.unwrap_or(false),
                        source_types: resp.source_types.unwrap_or_else(|| vec![SourceType::General]),
                        quality: resp.quality.unwrap_or(false),
                    };
                } else {
                    return ClassifiedIntent {
                        intent: QueryIntent::WebResearch,
                        search_queries: vec![query.to_string()],
                        reasoning: "classification fallback: web_research".to_string(),
                        skip_search: false,
                        source_types: vec![SourceType::General],
                        quality: false,
                    };
                }
            }
        }
        ClassifiedIntent {
            intent: QueryIntent::WebResearch,
            search_queries: vec![query.to_string()],
            reasoning: "classification fallback: web_research".to_string(),
            skip_search: false,
            source_types: vec![SourceType::General],
            quality: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_type_mapping() {
        assert_eq!(SourceType::General.as_searxng_category(), "general");
        assert_eq!(SourceType::News.as_searxng_category(), "news");
        assert_eq!(SourceType::Science.as_searxng_category(), "science");
        assert_eq!(SourceType::It.as_searxng_category(), "it");
    }

    #[test]
    fn fallback_defaults() {
        let fallback = ClassifiedIntent {
            intent: QueryIntent::WebResearch,
            search_queries: vec!["test".to_string()],
            reasoning: "fallback".to_string(),
            skip_search: false,
            source_types: vec![SourceType::General],
            quality: false,
        };
        assert!(!fallback.skip_search);
        assert_eq!(fallback.source_types, vec![SourceType::General]);
        assert!(!fallback.quality);
    }

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
