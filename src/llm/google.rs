//! Google Gemini provider implementation.
use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::llm::provider::{
    Chunk, FinishReason, LlmProvider, Message, ProviderError, Role, TokenUsage, ToolCall,
    ToolDefinition, ToolResponse,
};

pub struct GoogleProvider {
    api_key: Option<String>,
    client: Client,
}

#[derive(Serialize, Debug)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize, Debug)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize, Debug)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Debug)]
#[allow(non_snake_case)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    systemInstruction: Option<GeminiSystemInstruction>,
}

#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Deserialize, Debug)]
struct GeminiCandidate {
    content: Option<GeminiCandidateContent>,
}

#[derive(Deserialize, Debug)]
struct GeminiCandidateContent {
    parts: Option<Vec<GeminiPart>>,
}

// --- Function calling types ---

#[derive(Serialize, Debug)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize, Debug)]
struct GeminiToolDecl {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Deserialize, Debug)]
struct GeminiFunctionCall {
    name: String,
    args: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
#[allow(non_snake_case)]
struct GeminiToolResponse {
    candidates: Option<Vec<GeminiToolCandidate>>,
    usageMetadata: Option<GeminiUsageMetadata>,
}

#[derive(Deserialize, Debug)]
struct GeminiToolCandidate {
    content: Option<GeminiToolContent>,
}

#[derive(Deserialize, Debug)]
struct GeminiToolContent {
    parts: Option<Vec<GeminiToolPart>>,
}

#[derive(Deserialize, Debug)]
struct GeminiToolPart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    function_call: Option<GeminiFunctionCall>,
}

#[derive(Deserialize, Debug)]
#[allow(non_snake_case)]
struct GeminiUsageMetadata {
    promptTokenCount: Option<u32>,
    candidatesTokenCount: Option<u32>,
    totalTokenCount: Option<u32>,
}

impl Default for GoogleProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GoogleProvider {
    pub fn new() -> Self {
        Self {
            api_key: None,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LlmProvider for GoogleProvider {
    async fn stream_completion(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<Vec<Chunk>, ProviderError> {
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No API key set",
            )) as ProviderError
        })?;

        let mut system_text = String::new();
        let mut contents: Vec<GeminiContent> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    if !system_text.is_empty() {
                        system_text.push('\n');
                    }
                    system_text.push_str(&msg.content);
                }
                Role::User => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: msg.content.clone(),
                        }],
                    });
                }
                Role::Assistant => {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts: vec![GeminiPart {
                            text: msg.content.clone(),
                        }],
                    });
                }
            }
        }

        let request_body = GeminiRequest {
            contents,
            systemInstruction: if system_text.is_empty() {
                None
            } else {
                Some(GeminiSystemInstruction {
                    parts: vec![GeminiPart { text: system_text }],
                })
            },
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            model, api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Box::new(e) as ProviderError)?;

        match response.status() {
            StatusCode::OK => {}
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "Invalid or expired API key",
                )));
            }
            status if status.as_u16() >= 400 && status.as_u16() < 500 => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Client error: {}", status),
                )));
            }
            status if status.as_u16() >= 500 => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::ConnectionAborted,
                    format!("Server error: {}", status),
                )));
            }
            _ => {}
        }

        let body = response
            .text()
            .await
            .map_err(|e| Box::new(e) as ProviderError)?;

        let mut chunks: Vec<Chunk> = Vec::new();

        for line in body.lines() {
            let line = line.trim();
            if !line.starts_with("data: ") {
                continue;
            }
            let json_str = &line[6..];
            if json_str.is_empty() || json_str == "[DONE]" {
                continue;
            }

            match serde_json::from_str::<GeminiResponse>(json_str) {
                Ok(response) => {
                    if let Some(candidates) = response.candidates {
                        if let Some(first) = candidates.first() {
                            if let Some(content) = &first.content {
                                if let Some(parts) = &content.parts {
                                    for part in parts {
                                        chunks.push(Chunk {
                                            content: part.text.clone(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to parse response: {}", e),
                    )));
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
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No API key set",
            )) as ProviderError
        })?;

        let mut system_text = String::new();
        let mut contents: Vec<GeminiContent> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    if !system_text.is_empty() {
                        system_text.push('\n');
                    }
                    system_text.push_str(&msg.content);
                }
                Role::User => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: msg.content.clone(),
                        }],
                    });
                }
                Role::Assistant => {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts: vec![GeminiPart {
                            text: msg.content.clone(),
                        }],
                    });
                }
            }
        }

        let function_declarations: Vec<GeminiFunctionDeclaration> = tools
            .iter()
            .map(|t| GeminiFunctionDeclaration {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            })
            .collect();

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, api_key
        );

        let mut body = serde_json::json!({
            "contents": contents,
        });

        if !system_text.is_empty() {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{ "text": system_text }]
            });
        }

        if !function_declarations.is_empty() {
            body["tools"] = serde_json::json!([{
                "function_declarations": function_declarations
            }]);
        }

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Box::new(e) as ProviderError)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "<no body>".into());
            return Err(Box::new(std::io::Error::other(format!(
                "Gemini API error {}: {}",
                status, error_text
            ))));
        }

        let tool_response: GeminiToolResponse = response
            .json()
            .await
            .map_err(|e| Box::new(std::io::Error::other(format!("Parse error: {}", e))) as ProviderError)?;

        let mut content: Option<String> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        if let Some(candidates) = tool_response.candidates {
            if let Some(first) = candidates.first() {
                if let Some(content_data) = &first.content {
                    if let Some(parts) = &content_data.parts {
                        for part in parts {
                            if let Some(text) = &part.text {
                                if content.is_none() {
                                    content = Some(text.clone());
                                }
                            }
                            if let Some(fc) = &part.function_call {
                                tool_calls.push(ToolCall {
                                    id: format!("call_{}", tool_calls.len()),
                                    name: fc.name.clone(),
                                    arguments: fc.args
                                        .clone()
                                        .and_then(|v| serde_json::to_string(&v).ok())
                                        .unwrap_or_default(),
                                });
                            }
                        }
                    }
                }
            }
        }

        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        };

        let usage = tool_response.usageMetadata.map(|u| TokenUsage {
            prompt_tokens: u.promptTokenCount,
            completion_tokens: u.candidatesTokenCount,
            total_tokens: u.totalTokenCount,
        });

        Ok(ToolResponse {
            content,
            tool_calls,
            finish_reason,
            usage,
        })
    }

    fn name(&self) -> &str {
        "google"
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
            "gemini-1.5-pro-latest".into(),
            "gemini-1.5-flash-latest".into(),
        ])
    }

    fn supports_tool_calling(&self) -> bool {
        true
    }
}
