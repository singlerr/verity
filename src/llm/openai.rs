use crate::auth::store::{CredentialStore, Credentials};
use crate::llm::provider::{Chunk, LlmProvider, Message, ProviderError, Role, ToolDefinition, ToolResponse, TokenUsage, ToolCall, FinishReason};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const PROVIDER_NAME: &str = "openai";

pub struct OpenAiProvider {
    api_key: Option<String>,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)] struct OpenAiMessage { role: String, content: String, }

#[derive(Debug, Deserialize)] struct ChatCompletionChunk { choices: Vec<Choice>, }
#[derive(Debug, Deserialize)] struct Choice { delta: Delta, }
#[derive(Debug, Deserialize, Default)] struct Delta { #[serde(default)] content: Option<String>, }

impl OpenAiProvider {
    pub fn new() -> Self {
        Self { api_key: None, client: reqwest::Client::new() }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self { api_key: Some(api_key), client: reqwest::Client::new() }
    }

    fn build_headers(&self) -> Result<HeaderMap, ProviderError> {
        let api_key = self.api_key.as_ref().ok_or("API key not set")?;
        let mut headers = HeaderMap::new();
        let auth_value = HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| format!("Invalid API key: {}", e))?;
        headers.insert(AUTHORIZATION, auth_value);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    fn map_http_error(status: reqwest::StatusCode) -> ProviderError {
        let msg = match status.as_u16() {
            401 => "Authentication failed: Invalid API key (401)",
            429 => "Rate limit exceeded (429)",
            500..=599 => "OpenAI server error",
            _ => "Request failed",
        };
        std::io::Error::other(msg).into()
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn stream_completion(&self, messages: &[Message], model: &str) -> Result<Vec<Chunk>, ProviderError> {
        let headers = self.build_headers()?;

        let openai_messages: Vec<OpenAiMessage> = messages.iter().map(|m| OpenAiMessage { role: match m.role { Role::System => "system".into(), Role::User => "user".into(), Role::Assistant => "assistant".into(), }, content: m.content.clone() }).collect();

        let body = json!({"model": model, "messages": openai_messages, "stream": true});

        let response = self.client.post(OPENAI_API_URL).headers(headers).json(&body).send().await.map_err(|e| format!("Request failed: {}", e))?;

        let status = response.status();
        if !status.is_success() { return Err(Self::map_http_error(status)); }

        let mut chunks = Vec::new();
        let mut stream = response.bytes_stream();

        while let Some(item) = stream.next().await {
            let bytes = item.map_err(|e| -> ProviderError { format!("Stream error: {}", e).into() })?;
            let text = String::from_utf8_lossy(&bytes);
            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" { return Ok(chunks); }
                    if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(data) {
                        for choice in chunk.choices {
                            if let Some(content) = choice.delta.content { chunks.push(Chunk { content }); }
                        }
                    }
                }
            }
        }
        Ok(chunks)
    }

    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
    ) -> Result<ToolResponse, ProviderError> {
        let headers = self.build_headers()?;
        let openai_messages: Vec<OpenAiMessage> = messages.iter().map(|m| OpenAiMessage {
            role: match m.role {
                Role::System => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: m.content.clone(),
        }).collect();

        // Build tools payload
        let tool_entries: Vec<serde_json::Value> = tools.iter().map(|t| {
            serde_json::json!({"type":"function","function":{"name": t.name, "description": t.description, "parameters": t.parameters}})
        }).collect();

        let body = serde_json::json!({
            "model": model,
            "messages": openai_messages,
            "stream": false,
            "tools": tool_entries
        });

        let mut req_builder = self.client.post(OPENAI_API_URL);
        if let Some(ref key) = self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }
        let response = req_builder.headers(headers).json(&body).send().await.map_err(|e| format!("Request failed: {}", e))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Self::map_http_error(status));
        }

        // Non-streaming: read full JSON response and extract tool calls if any
        let v: serde_json::Value = response.json().await.map_err(|e| format!("Parse error: {}", e))?;
        let mut content: Option<String> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut finish_reason = FinishReason::Stop;
        if let Some(choice) = v.get("choices").and_then(|c| c.as_array()).and_then(|arr| arr.first()) {
            if let Some(msg) = choice.get("message") {
                if let Some(c) = msg.get("content").and_then(|x| x.as_str()) { content = Some(c.to_string()); }
                if let Some(tcs) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for tc in tcs {
                        let id = tc.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                        let mut name = tc.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string();
                        let mut arguments = tc.get("arguments").and_then(|s| s.as_str()).unwrap_or("").to_string();
                        if let Some(func) = tc.get("function") {
                          if name.is_empty() { if let Some(n) = func.get("name").and_then(|s| s.as_str()) { name = n.to_string(); } }
                          if arguments.is_empty() { if let Some(a) = func.get("arguments").and_then(|s| s.as_str()) { arguments = a.to_string(); } }
                        }
                        tool_calls.push(ToolCall { id, name, arguments });
                    }
                }
            }
            if let Some(fr) = choice.get("finish_reason").and_then(|s| s.as_str()) {
                finish_reason = match fr {
                    "stop" => FinishReason::Stop,
                    "tool_calls" => FinishReason::ToolCalls,
                    "length" => FinishReason::Length,
                    _ => FinishReason::Stop,
                };
            }
        }
        let usage = v.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|x| x.as_u64()).map(|n| n as u32),
            completion_tokens: u.get("completion_tokens").and_then(|x| x.as_u64()).map(|n| n as u32),
            total_tokens: u.get("total_tokens").and_then(|x| x.as_u64()).map(|n| n as u32),
        });
        Ok(ToolResponse { content, tool_calls, finish_reason, usage })
    }

    fn name(&self) -> &str { PROVIDER_NAME }
    fn is_authenticated(&self) -> bool { self.api_key.is_some() }

    async fn authenticate(&mut self, api_key: &str) -> Result<(), ProviderError> {
        self.api_key = Some(api_key.to_string());
        let mut store = CredentialStore::load().map_err(|e| format!("Failed to load store: {}", e))?;
        store.set(PROVIDER_NAME.to_string(), Credentials { api_key: api_key.to_string() });
        store.save().map_err(|e| format!("Failed to save: {}", e))?;
        Ok(())
    }

    async fn deauthenticate(&mut self) -> Result<(), ProviderError> {
        self.api_key = None;
        let mut store = CredentialStore::load().map_err(|e| format!("Failed to load store: {}", e))?;
        store.remove(PROVIDER_NAME);
        store.save().map_err(|e| format!("Failed to save: {}", e))?;
        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let headers = self.build_headers()?;
        let response = self.client.get("https://api.openai.com/v1/models")
            .headers(headers).send().await.map_err(|e| format!("Request failed: {}", e))?;
        if !response.status().is_success() {
            return Err(Self::map_http_error(response.status()));
        }
        #[derive(Deserialize)]
        struct Model { id: String }
        #[derive(Deserialize)]
        struct ModelsResponse { data: Vec<Model> }
        let body: ModelsResponse = response.json().await.map_err(|e| format!("Parse error: {}", e))?;
        Ok(body.data.into_iter().map(|m| m.id).collect())
    }
}

impl Default for OpenAiProvider {
    fn default() -> Self { Self::new() }
}
