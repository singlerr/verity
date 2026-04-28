use crate::llm::provider::{Chunk, LlmProvider, Message, ProviderError, Role, ToolDefinition, ToolResponse};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const PROVIDER_NAME: &str = "ollama";
pub struct OllamaProvider {
    base_url: String,
    client: reqwest::Client,
    authenticated: bool,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}
#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(rename = "model")]
    _model: String,
    message: ResponseMessage,
    done: bool,
}
#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(rename = "role")]
    _role: String,
    content: String,
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_URL.to_string(),
            client: reqwest::Client::new(),
            authenticated: false,
        }
    }

    pub fn with_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
            authenticated: false,
        }
    }

    fn role_to_string(role: &Role) -> String {
        match role {
            Role::System => "system".to_string(),
            Role::User => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
        }
    }

    fn map_error(e: reqwest::Error) -> ProviderError {
        let msg = if e.is_connect() {
            "Connection failed: Is Ollama running?"
        } else if e.is_timeout() {
            "Request timed out"
        } else {
            "Request failed"
        };
        Box::new(std::io::Error::other(msg))
    }
}
#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn stream_completion(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<Vec<Chunk>, ProviderError> {
        let ollama_messages: Vec<OllamaMessage> = messages
            .iter()
            .map(|m| OllamaMessage {
                role: Self::role_to_string(&m.role),
                content: m.content.clone(),
            })
            .collect();

        let body = json!({
            "model": model,
            "messages": ollama_messages,
            "stream": true
        });

        let url = format!("{}/api/chat", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(Self::map_error)?;

        if !response.status().is_success() {
            let msg = format!("Ollama returned error: {}", response.status());
            return Err(Box::new(std::io::Error::other(
                msg,
            )));
        }

        let mut chunks = Vec::new();
        let mut stream = response.bytes_stream();

        while let Some(item) = stream.next().await {
            let bytes = item.map_err(|e| format!("Stream error: {}", e))?;
            let text = String::from_utf8_lossy(&bytes);

            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<ChatResponse>(line) {
                    Ok(response) => {
                        if !response.message.content.is_empty() {
                            chunks.push(Chunk {
                                content: response.message.content,
                            });
                        }
                        if response.done {
                            return Ok(chunks);
                        }
                    }
                    Err(e) => {
                        return Err(format!("Failed to parse response: {}", e).into());
                    }
                }
            }
        }

        Ok(chunks)
    }
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn is_authenticated(&self) -> bool {
        self.authenticated
    }

    async fn authenticate(&mut self, _api_key: &str) -> Result<(), ProviderError> {
        let _ = self.list_models().await?;
        self.authenticated = true;
        Ok(())
    }

    async fn deauthenticate(&mut self) -> Result<(), ProviderError> {
        self.authenticated = false;
        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await.map_err(Self::map_error)?;
        if !response.status().is_success() {
            return Err(Box::new(std::io::Error::other("Failed to connect to Ollama")));
        }
        #[derive(Deserialize)]
        struct ModelTag { name: String }
        #[derive(Deserialize)]
        struct TagsResponse { models: Vec<ModelTag> }
        let body: TagsResponse = response.json().await.map_err(|e| format!("Parse error: {}", e))?;
        Ok(body.models.into_iter().map(|m| m.name).collect())
    }

    async fn complete_with_tools(&self, _messages: &[Message], _tools: &[ToolDefinition], _model: &str) -> Result<ToolResponse, ProviderError> { Err("Tool calling not supported by this provider".into()) }
}
impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new_provider() {
        let provider = OllamaProvider::new();
        assert!(!provider.is_authenticated());
        assert_eq!(provider.name(), "ollama");
        assert_eq!(provider.base_url, "http://localhost:11434");
    }
    #[test]
    fn test_with_custom_url() {
        let provider = OllamaProvider::with_url("http://192.168.1.100:11434");
        assert_eq!(provider.base_url, "http://192.168.1.100:11434");
    }
}
