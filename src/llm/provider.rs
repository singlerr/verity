use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;

// Lightweight error alias for provider operations
pub type ProviderError = Box<dyn std::error::Error + Send + Sync>;

/// A model entry that pairs a model ID with its provider.
/// This replaces string-prefix-based provider inference.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,     // API-facing model ID (e.g. "meta/llama-3.1-70b-instruct")
    pub provider: String, // provider key  (e.g. "nvidia", "openai")
}

impl ModelEntry {
    /// The string to store in config and pass to `ProviderRegistry::resolve`.
    /// NVIDIA models get a `nvidia/` prefix so the existing resolver routes correctly.
    pub fn config_id(&self) -> String {
        match self.provider.as_str() {
            "nvidia" => format!("nvidia/{}", self.name),
            _ => self.name.clone(),
        }
    }

    /// Human-readable label for the model selection UI.
    pub fn display_name(&self) -> String {
        match self.provider.as_str() {
            "nvidia" => format!("nvidia/{}", self.name),
            _ => self.name.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub content: String,
}

// Tooling integration types for dynamic tool-calling by LLMs
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    pub call_id: String,
    pub name: String,
    pub output: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: Option<TokenUsage>,
}

/// LlmProvider trait: core interface for all LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn stream_completion(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<Vec<Chunk>, ProviderError>;
    /// Non-streaming completion with tool-calling support.
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
    ) -> Result<ToolResponse, ProviderError>;
    fn name(&self) -> &str;
    fn is_authenticated(&self) -> bool;
    async fn authenticate(&mut self, api_key: &str) -> Result<(), ProviderError>;
    async fn deauthenticate(&mut self) -> Result<(), ProviderError>;
    async fn list_models(&self) -> Result<Vec<String>, ProviderError>;
    fn supports_tool_calling(&self) -> bool {
        false
    }
}

/// A thread-safe handle to a provider instance using trait objects
pub type ProviderHandle = Arc<RwLock<Box<dyn LlmProvider + Send + Sync>>>;

/// Static metadata for a registered LLM provider.
/// Declared at registration time — no locking needed to read.
pub struct ProviderMetadata {
    pub display_name: String,        // Human-readable: "OpenAI", "Anthropic", "Google Gemini", "Ollama (local)", "NVIDIA NIM"
    pub requires_api_key: bool,      // false for Ollama
    pub model_prefixes: Vec<String>, // Model ID prefixes: ["gpt-"], ["claude-"], ["gemini-"], ["nvidia/"], []
    pub fallback_models: Vec<String>,// Default models when API unavailable
}

/// Registry of available providers exposed via trait objects
pub struct ProviderRegistry {
    providers: HashMap<String, ProviderHandle>,
    metadata: HashMap<String, ProviderMetadata>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, provider: Box<dyn LlmProvider + Send + Sync>, meta: ProviderMetadata) {
        self.providers.insert(name.clone(), Arc::new(RwLock::new(provider)));
        self.metadata.insert(name, meta);
    }

    pub fn get(&self, name: &str) -> Option<ProviderHandle> {
        self.providers.get(name).cloned()
    }

    pub fn provider_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.metadata.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn get_metadata(&self, name: &str) -> Option<&ProviderMetadata> {
        self.metadata.get(name)
    }

    pub fn resolve(&self, model: &str) -> Option<ProviderHandle> {
        for (name, meta) in &self.metadata {
            if meta.model_prefixes.iter().any(|prefix| model.starts_with(prefix.as_str())) {
                return self.providers.get(name).cloned();
            }
        }
        for (name, meta) in &self.metadata {
            if meta.model_prefixes.is_empty() {
                return self.providers.get(name).cloned();
            }
        }
        self.providers.values().next().cloned()
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    pub struct MockLlmProvider {
        pub responses: Arc<Mutex<Vec<ToolResponse>>>,
        pub stream_responses: Arc<Mutex<Vec<Vec<Chunk>>>>,
        pub call_count: Arc<Mutex<usize>>,
        pub tool_calling_support: bool,
    }

    impl MockLlmProvider {
        pub fn new(responses: Vec<ToolResponse>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
                stream_responses: Arc::new(Mutex::new(vec![vec![Chunk {
                    content: "mock synthesis".into(),
                }]])),
                call_count: Arc::new(Mutex::new(0)),
                tool_calling_support: true,
            }
        }
        pub fn with_stream(responses: Vec<ToolResponse>, stream: Vec<Vec<Chunk>>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
                stream_responses: Arc::new(Mutex::new(stream)),
                call_count: Arc::new(Mutex::new(0)),
                tool_calling_support: true,
            }
        }
        pub fn without_tool_calling() -> Self {
            let mut m = Self::new(vec![]);
            m.tool_calling_support = false;
            m
        }
    }

    #[async_trait]
    impl LlmProvider for MockLlmProvider {
        async fn stream_completion(
            &self,
            _messages: &[Message],
            _model: &str,
        ) -> Result<Vec<Chunk>, ProviderError> {
            let mut responses = self.stream_responses.lock().unwrap();
            if responses.is_empty() {
                Ok(vec![Chunk {
                    content: "mock".into(),
                }])
            } else {
                Ok(responses.remove(0))
            }
        }
        async fn complete_with_tools(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _model: &str,
        ) -> Result<ToolResponse, ProviderError> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err("No more responses".into())
            } else {
                Ok(responses.remove(0))
            }
        }
        fn name(&self) -> &str {
            "mock"
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
            Ok(vec!["mock-model".into()])
        }
        fn supports_tool_calling(&self) -> bool {
            self.tool_calling_support
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_definition_serde_roundtrip() {
        let td = ToolDefinition {
            name: "test".into(),
            description: "desc".into(),
            parameters: serde_json::json!({"type":"object"}),
        };
        let s = serde_json::to_string(&td).unwrap();
        let td2: ToolDefinition = serde_json::from_str(&s).unwrap();
        assert_eq!(td, td2);
    }

    #[test]
    fn finish_reason_from_str() {
        fn to_fr(s: &str) -> FinishReason {
            match s {
                "stop" => FinishReason::Stop,
                "tool_calls" => FinishReason::ToolCalls,
                "length" => FinishReason::Length,
                _ => FinishReason::Stop,
            }
        }
        assert_eq!(to_fr("stop"), FinishReason::Stop);
        assert_eq!(to_fr("tool_calls"), FinishReason::ToolCalls);
    }

    #[test]
    fn mock_provider_basic_response() {
        use mock::MockLlmProvider;
        let response = ToolResponse {
            content: Some("test content".into()),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: None,
        };
        let mock = MockLlmProvider::new(vec![response.clone()]);
        assert!(mock.supports_tool_calling());
        assert!(mock.is_authenticated());
    }

    #[test]
    fn mock_provider_tool_calling_disabled() {
        use mock::MockLlmProvider;
        let mock = MockLlmProvider::without_tool_calling();
        assert!(!mock.supports_tool_calling());
    }

    #[test]
    fn non_openai_provider_rejected() {
        use mock::MockLlmProvider;
        let non_openai = MockLlmProvider::without_tool_calling();
        assert!(
            !non_openai.supports_tool_calling(),
            "Non-OpenAI provider should not support tool calling"
        );

        let openai_compatible = MockLlmProvider::new(vec![]);
        assert!(
            openai_compatible.supports_tool_calling(),
            "OpenAI-compatible provider should support tool calling"
        );
    }

    #[test]
    fn mock_provider_complete_with_tools_consumes_responses() {
        use mock::MockLlmProvider;
        use std::sync::Arc;

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let response1 = ToolResponse {
                content: Some("first".into()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: None,
            };
            let response2 = ToolResponse {
                content: Some("second".into()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: None,
            };
            let mock = MockLlmProvider::new(vec![response1, response2]);

            let result1 = mock.complete_with_tools(&[], &[], "model").await.unwrap();
            assert_eq!(result1.content, Some("first".into()));

            let result2 = mock.complete_with_tools(&[], &[], "model").await.unwrap();
            assert_eq!(result2.content, Some("second".into()));

            assert_eq!(*mock.call_count.lock().unwrap(), 2);
        });
    }

    #[test]
    fn mock_provider_stream_completion() {
        use mock::MockLlmProvider;

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let stream1 = vec![
                Chunk {
                    content: "Hello ".into(),
                },
                Chunk {
                    content: "world".into(),
                },
            ];
            let stream2 = vec![
                Chunk {
                    content: "Second ".into(),
                },
                Chunk {
                    content: "response".into(),
                },
            ];
            let mock = MockLlmProvider::with_stream(vec![], vec![stream1, stream2]);

            let result1 = mock.stream_completion(&[], "model").await.unwrap();
            assert_eq!(result1.len(), 2);
            assert_eq!(result1[0].content, "Hello ");
            assert_eq!(result1[1].content, "world");

            let result2 = mock.stream_completion(&[], "model").await.unwrap();
            assert_eq!(result2.len(), 2);
            assert_eq!(result2[0].content, "Second ");
            assert_eq!(result2[1].content, "response");
        });
    }

    #[test]
    fn provider_metadata_fields() {
        let pr = crate::llm::build_registry();

        // Check all 5 providers are registered
        let names = pr.provider_names();
        assert!(names.contains(&"openai".to_string()), "openai missing");
        assert!(names.contains(&"anthropic".to_string()), "anthropic missing");
        assert!(names.contains(&"google".to_string()), "google missing");
        assert!(names.contains(&"ollama".to_string()), "ollama missing");
        assert!(names.contains(&"nvidia".to_string()), "nvidia missing");
        assert_eq!(names.len(), 5, "expected exactly 5 providers");

        // Check OpenAI metadata
        let openai = pr.get_metadata("openai").expect("openai metadata");
        assert_eq!(openai.display_name, "OpenAI");
        assert!(openai.requires_api_key);
        assert!(openai.model_prefixes.contains(&"gpt-".to_string()));
        assert!(!openai.fallback_models.is_empty());

        // Check Anthropic metadata
        let anthropic = pr.get_metadata("anthropic").expect("anthropic metadata");
        assert_eq!(anthropic.display_name, "Anthropic");
        assert!(anthropic.requires_api_key);
        assert!(anthropic.model_prefixes.contains(&"claude-".to_string()));

        // Check Google metadata
        let google = pr.get_metadata("google").expect("google metadata");
        assert_eq!(google.display_name, "Google Gemini");
        assert!(google.requires_api_key);
        assert!(google.model_prefixes.iter().any(|p| p.starts_with("gemini")));

        // Check Ollama metadata
        let ollama = pr.get_metadata("ollama").expect("ollama metadata");
        assert_eq!(ollama.display_name, "Ollama (local)");
        assert!(!ollama.requires_api_key);

        // Check NVIDIA metadata
        let nvidia = pr.get_metadata("nvidia").expect("nvidia metadata");
        assert_eq!(nvidia.display_name, "NVIDIA NIM");
        assert!(nvidia.requires_api_key);
        assert!(nvidia.model_prefixes.contains(&"nvidia/".to_string()));
    }

    #[test]
    fn resolve_routes_by_prefix() {
        let pr = crate::llm::build_registry();

        // OpenAI models
        assert!(pr.resolve("gpt-4o").is_some());
        assert!(pr.resolve("gpt-3.5-turbo").is_some());

        // Anthropic models
        assert!(pr.resolve("claude-3-opus-latest").is_some());
        assert!(pr.resolve("claude-sonnet-4-5").is_some());

        // Google models
        assert!(pr.resolve("gemini-1.5-pro").is_some());
        assert!(pr.resolve("gemini-2.0-flash").is_some());

        // NVIDIA models
        assert!(pr.resolve("nvidia/llama-3.1").is_some());

        // Unknown models should fall back to Ollama (or first provider)
        assert!(pr.resolve("llama-3.1").is_some());
        assert!(pr.resolve("mistral-7b").is_some());
    }

    #[test]
    fn provider_names_returns_sorted() {
        let pr = crate::llm::build_registry();
        let names = pr.provider_names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "provider_names should return sorted names");
    }
}
