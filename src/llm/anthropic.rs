use crate::llm::provider::{
    Chunk, FinishReason, LlmProvider, Message, ProviderError, Role, TokenUsage, ToolCall,
    ToolDefinition, ToolResponse,
};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicToolDef>>,
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

// --- Tool calling types ---

#[derive(Serialize)]
struct AnthropicToolDef {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicToolResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
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
            tools: None,
        }
    }

    async fn parse_sse_stream(
        &self,
        response: reqwest::Response,
    ) -> Result<Vec<Chunk>, ProviderError> {
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
    async fn stream_completion(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<Vec<Chunk>, ProviderError> {
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "API key not set",
            )) as Box<dyn std::error::Error + Send + Sync>
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
            return Err(Box::new(std::io::Error::other(format!(
                "Anthropic API error {}: {}",
                status, error_text
            ))));
        }

        self.parse_sse_stream(response).await
    }

    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
    ) -> Result<ToolResponse, ProviderError> {
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "API key not set",
            )) as Box<dyn std::error::Error + Send + Sync>
        })?;

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

        let tool_defs: Vec<AnthropicToolDef> = tools
            .iter()
            .map(|t| AnthropicToolDef {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect();

        let request_body = AnthropicRequest {
            model,
            max_tokens: 4096,
            system,
            messages: anthropic_messages,
            stream: false,
            tools: Some(tool_defs),
        };

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
            return Err(Box::new(std::io::Error::other(format!(
                "Anthropic API error {}: {}",
                status, error_text
            ))));
        }

        let body: AnthropicToolResponse = response.json().await.map_err(|e| {
            Box::new(std::io::Error::other(format!("Parse error: {}", e)))
                as Box<dyn std::error::Error + Send + Sync>
        })?;

        let mut content: Option<String> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in &body.content {
            match block.block_type.as_str() {
                "text" => {
                    if let Some(text) = &block.text {
                        content = Some(text.clone());
                    }
                }
                "tool_use" => {
                    tool_calls.push(ToolCall {
                        id: block.id.clone().unwrap_or_default(),
                        name: block.name.clone().unwrap_or_default(),
                        arguments: block
                            .input
                            .clone()
                            .and_then(|v| serde_json::to_string(&v).ok())
                            .unwrap_or_default(),
                    });
                }
                _ => {}
            }
        }

        let finish_reason = match body.stop_reason.as_deref() {
            Some("tool_use") => FinishReason::ToolCalls,
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            _ => FinishReason::Stop,
        };

        let usage = body.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens.zip(u.output_tokens).map(|(i, o)| i + o),
        });

        Ok(ToolResponse {
            content,
            tool_calls,
            finish_reason,
            usage,
        })
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

    fn supports_tool_calling(&self) -> bool {
        true
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}
