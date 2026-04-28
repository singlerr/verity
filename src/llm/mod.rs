//! LLM provider module.

pub mod anthropic;
pub mod google;
pub mod ollama;
pub mod openai;
pub mod provider;

pub use anthropic::AnthropicProvider;
pub use google::GoogleProvider;
pub use openai::OpenAiProvider;
pub use ollama::OllamaProvider;
pub use provider::{
    Chunk, LlmProvider, Message, ProviderError, ProviderHandle, ProviderRegistry, Role,
    ToolDefinition, ToolCall, ToolResponse, ToolResult, FinishReason, TokenUsage,
};

/// Build a registry with all available providers registered.
pub fn build_registry() -> ProviderRegistry {
    let mut reg = ProviderRegistry::new();
    reg.register("openai".to_string(), Box::new(OpenAiProvider::new()));
    reg.register("anthropic".to_string(), Box::new(AnthropicProvider::new()));
    reg.register("google".to_string(), Box::new(GoogleProvider::new()));
    reg.register("ollama".to_string(), Box::new(OllamaProvider::new()));
    reg
}
