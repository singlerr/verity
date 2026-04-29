//! LLM provider module.

pub mod anthropic;
pub mod google;
pub mod nvidia;
pub mod ollama;
pub mod openai;
pub mod provider;

pub use anthropic::AnthropicProvider;
pub use google::GoogleProvider;
pub use nvidia::NvidiaProvider;
pub use ollama::OllamaProvider;
pub use openai::OpenAiProvider;
pub use provider::{
    Chunk, FinishReason, LlmProvider, Message, ProviderError, ProviderHandle, ProviderMetadata,
    ProviderRegistry, Role, TokenUsage, ToolCall, ToolDefinition, ToolResponse, ToolResult,
};

/// Build a registry with all available providers registered.
pub fn build_registry() -> ProviderRegistry {
    let mut reg = ProviderRegistry::new();
    reg.register("openai".to_string(), Box::new(OpenAiProvider::new()), ProviderMetadata {
        display_name: "OpenAI".into(),
        requires_api_key: true,
        model_prefixes: vec!["gpt-".into()],
        fallback_models: vec!["gpt-4o".into(), "gpt-4o-mini".into(), "gpt-4-turbo".into(), "gpt-3.5-turbo".into()],
    });
    reg.register("anthropic".to_string(), Box::new(AnthropicProvider::new()), ProviderMetadata {
        display_name: "Anthropic".into(),
        requires_api_key: true,
        model_prefixes: vec!["claude-".into()],
        fallback_models: vec!["claude-opus-4-5".into(), "claude-sonnet-4-5".into(), "claude-haiku-4-5".into(), "claude-3-5-sonnet-latest".into(), "claude-3-5-haiku-latest".into(), "claude-3-opus-latest".into()],
    });
    reg.register("google".to_string(), Box::new(GoogleProvider::new()), ProviderMetadata {
        display_name: "Google Gemini".into(),
        requires_api_key: true,
        model_prefixes: vec!["gemini-".into()],
        fallback_models: vec!["gemini-2.0-flash".into(), "gemini-1.5-pro".into(), "gemini-1.5-flash".into()],
    });
    reg.register("ollama".to_string(), Box::new(OllamaProvider::new()), ProviderMetadata {
        display_name: "Ollama (local)".into(),
        requires_api_key: false,
        model_prefixes: vec![],
        fallback_models: vec![],
    });
    reg.register("nvidia".to_string(), Box::new(NvidiaProvider::new()), ProviderMetadata {
        display_name: "NVIDIA NIM".into(),
        requires_api_key: true,
        model_prefixes: vec!["nvidia/".into()],
        fallback_models: vec![],
    });
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_registry_registers_all_providers() {
        let reg = build_registry();
        assert_eq!(reg.provider_names().len(), 5);
        for name in &["openai", "anthropic", "google", "ollama", "nvidia"] {
            assert!(reg.get(name).is_some(), "provider '{}' not registered", name);
            assert!(reg.get_metadata(name).is_some(), "metadata for '{}' missing", name);
        }
    }
}
