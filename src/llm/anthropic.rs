use crate::llm::provider::{Chunk, LlmProvider, Message, ProviderError, Role};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Serialize)]
struct AnthropicRequest<'a> {
  model: &'a str,
  max_tokens: u32,
  #[serde(skip_serializing_if = "Option::is_none")]
  system: Option<String>,
  messages: Vec<AnthropicMessage>,
  stream: bool,
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
  role: String,
  content: String,
}

#[derive(Deserialize)]
struct AnthropicEvent {
  #[serde(rename = "type")]
  event_type: String,
  #[serde(default)]
  delta: Option<Delta>,
}

#[derive(Deserialize)]
struct Delta {
  #[serde(rename = "type")]
  delta_type: String,
  #[serde(default)]
  text: Option<String>,
}

pub struct AnthropicProvider {
  api_key: Option<String>,
  client: reqwest::Client,
}

impl AnthropicProvider {
  pub fn new() -> Self {
    Self {
      api_key: None,
      client: reqwest::Client::new(),
    }
  }

  fn build_request<'a>(&self, messages: &[Message], model: &'a str) -> AnthropicRequest<'a> {
    let mut system = None;
    let anthropic_messages: Vec<AnthropicMessage> = messages
      .iter()
      .filter_map(|msg| match msg.role {
        Role::System => {
          system = Some(msg.content.clone());
          None
        }
        Role::User => Some(AnthropicMessage {
          role: "user".to_string(),
          content: msg.content.clone(),
        }),
        Role::Assistant => Some(AnthropicMessage {
          role: "assistant".to_string(),
          content: msg.content.clone(),
        }),
      })
      .collect();

    AnthropicRequest {
      model,
      max_tokens: 4096,
      system,
      messages: anthropic_messages,
      stream: true,
    }
  }

  async fn parse_sse_stream(&self, response: reqwest::Response) -> Result<Vec<Chunk>, ProviderError> {
    let mut chunks = Vec::new();
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
      let bytes = chunk?;
      buffer.push_str(&String::from_utf8_lossy(&bytes));

      while let Some(pos) = buffer.find('\n') {
        let line = buffer[..pos].trim().to_string();
        buffer = buffer[pos + 1..].to_string();

        if let Some(data) = line.strip_prefix("data: ") {
          if data == "[DONE]" {
            continue;
          }
          if let Ok(event) = serde_json::from_str::<AnthropicEvent>(data) {
            if event.event_type == "content_block_delta" {
              if let Some(delta) = event.delta {
                if delta.delta_type == "text_delta" {
                  if let Some(text) = delta.text {
                    chunks.push(Chunk { content: text });
                  }
                }
              }
            }
          }
        }
      }
    }

    Ok(chunks)
  }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
  async fn stream_completion(&self, messages: &[Message], model: &str) -> Result<Vec<Chunk>, ProviderError> {
    let api_key = self.api_key.as_ref().ok_or_else(|| {
      Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "API key not set"))
        as Box<dyn std::error::Error + Send + Sync>
    })?;

    let request_body = self.build_request(messages, model);

    let response = self
      .client
      .post(ANTHROPIC_API_URL)
      .header("x-api-key", api_key)
      .header("anthropic-version", ANTHROPIC_VERSION)
      .header("Content-Type", "application/json")
      .json(&request_body)
      .send()
      .await?;

    let status = response.status();
    if !status.is_success() {
      let error_text = response.text().await?;
        return Err(Box::new(std::io::Error::other(
        format!("Anthropic API error {}: {}", status, error_text),
      )));
    }

    self.parse_sse_stream(response).await
  }

  fn name(&self) -> &str {
    "anthropic"
  }

  fn is_authenticated(&self) -> bool {
    self.api_key.is_some()
  }

  async fn authenticate(&mut self, api_key: &str) -> Result<(), ProviderError> {
    self.api_key = Some(api_key.to_string());
    Ok(())
  }

  async fn deauthenticate(&mut self) -> Result<(), ProviderError> {
    self.api_key = None;
    Ok(())
  }

  async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
    Ok(vec![
      "claude-3-5-sonnet-latest".into(),
      "claude-3-5-haiku-latest".into(),
      "claude-3-opus-latest".into(),
    ])
  }
}

impl Default for AnthropicProvider {
  fn default() -> Self {
    Self::new()
  }
}
