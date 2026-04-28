use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;

// Lightweight error alias for provider operations
pub type ProviderError = Box<dyn std::error::Error + Send + Sync>;

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
    async fn stream_completion(&self, messages: &[Message], model: &str) -> Result<Vec<Chunk>, ProviderError>;
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
}

/// A thread-safe handle to a provider instance using trait objects
pub type ProviderHandle = Arc<RwLock<Box<dyn LlmProvider + Send + Sync>>>;

/// Registry of available providers exposed via trait objects
pub struct ProviderRegistry {
  providers: HashMap<String, ProviderHandle>,
}

impl Default for ProviderRegistry {
  fn default() -> Self {
      Self::new()
  }
}

impl ProviderRegistry {
  pub fn new() -> Self {
    Self { providers: HashMap::new() }
  }

  // Register a concrete provider behind a trait object
  pub fn register(&mut self, name: String, provider: Box<dyn LlmProvider + Send + Sync>) {
    self.providers.insert(name, Arc::new(RwLock::new(provider)));
  }

  pub fn get(&self, name: &str) -> Option<ProviderHandle> {
    self.providers.get(name).cloned()
  }

  // Resolve a provider based on a model prefix routing rule
  pub fn resolve(&self, model: &str) -> Option<ProviderHandle> {
    let key = if model.starts_with("gpt-") {
      "openai"
    } else if model.starts_with("claude-") {
      "anthropic"
    } else if model.starts_with("gemini-") {
      "google"
    } else {
      "ollama"
    };
    self.providers.get(key).cloned()
  }
}

/// Convenience: build an empty registry (no providers registered yet)
pub fn build_registry() -> ProviderRegistry {
  ProviderRegistry::new()
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn tool_definition_serde_roundtrip() {
    let td = ToolDefinition { name: "test".into(), description: "desc".into(), parameters: serde_json::json!({"type":"object"}) };
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
}
