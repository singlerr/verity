use std::collections::HashMap;
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

/// LlmProvider trait: core interface for all LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
  async fn stream_completion(&self, messages: &[Message], model: &str) -> Result<Vec<Chunk>, ProviderError>;
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
