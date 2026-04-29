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
    Chunk, FinishReason, LlmProvider, Message, ProviderError, ProviderHandle, ProviderRegistry,
    Role, TokenUsage, ToolCall, ToolDefinition, ToolResponse, ToolResult,
};

/// Build a registry with all available providers registered.
pub fn build_registry() -> ProviderRegistry {
    let mut reg = ProviderRegistry::new();
    reg.register("openai".to_string(), Box::new(OpenAiProvider::new()));
    reg.register("anthropic".to_string(), Box::new(AnthropicProvider::new()));
    reg.register("google".to_string(), Box::new(GoogleProvider::new()));
    reg.register("ollama".to_string(), Box::new(OllamaProvider::new()));
    reg.register("nvidia".to_string(), Box::new(NvidiaProvider::new()));
    reg
}
