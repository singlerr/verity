use std::sync::Arc;
use anyhow::Result;
use crate::app::{PlanStep, StepStatus, Tool};
use crate::llm::provider::{Message, Role, ProviderRegistry};
use crate::agent::classifier::{ClassifiedIntent, QueryIntent};
use serde::Deserialize;

pub struct AgentPlanner {
    _provider_registry: Arc<ProviderRegistry>,
    _model: String,
}

impl AgentPlanner {
    pub fn new(registry: Arc<ProviderRegistry>, model: String) -> Self {
        Self {
            _provider_registry: registry,
            _model: model,
        }
    }

    pub fn plan_prompt(query: &str, intent: Option<&ClassifiedIntent>) -> Vec<Message> {
        let manifest = crate::agent::tool_manifest();
        let mut system = format!(
            "You are an AI coding and research assistant with access to tools.\n{manifest}\n\n\
             Break the user's request into concrete steps. The \"title\" field must be the ACTUAL INPUT to the tool:\n\
             - search: exact search query string (NEVER copy raw user query)\n\
             - read_file: exact file path to read\n\
             - list_dir: directory path (or \".\" for current)\n\
             - shell: exact shell command to run\n\
             - read: URL to fetch\n\
             - think: description of what to reason about\n\n\
             Respond ONLY with a JSON array. Each element must have:\n             {{\"id\":\"1\",\"title\":\"<tool input>\",\"tool\":\"search|read|think|edit|shell|read_file|list_dir\",\"status\":\"queued\",\"duration\":null,\"thoughts\":[]}}"
        );

        if let Some(classified) = intent {
            match classified.intent {
                QueryIntent::WebResearch => {
                    system.push_str(&format!(
                        "\n\nIMPORTANT: Query classified as WEB RESEARCH. Use reformulated search queries: {}. NEVER copy raw user query as search title.",
                        classified.search_queries.join(", ")));
                }
                QueryIntent::LocalAnalysis => {
                    system.push_str("\n\nIMPORTANT: Query classified as LOCAL ANALYSIS. Use ONLY list_dir, read_file, shell, think tools. NO search steps.");
                }
                QueryIntent::Mixed => {
                    system.push_str(&format!(
                        "\n\nIMPORTANT: Query needs BOTH local analysis AND web research. Do local steps FIRST, then web search. Reformulated queries: {}. NEVER use raw query as search title.",
                        classified.search_queries.join(", ")));
                }
                QueryIntent::DirectAnswer => {
                    system.push_str("\n\nIMPORTANT: Direct factual question. Use only think tool to reason through the answer.");
                }
            }
        }
        let user_content = if let Some(classified) = intent {
            format!("Intent: {:?}. Search queries: {:?}. User's question: {}", classified.intent, classified.search_queries, query)
        } else {
            query.to_string()
        };
        vec![
            Message { role: Role::System, content: system },
            Message { role: Role::User, content: user_content },
        ]
    }

    pub async fn plan(&self, query: &str, intent: Option<&ClassifiedIntent>) -> Result<Vec<PlanStep>> {
        let messages = Self::plan_prompt(query, intent);

        // Attempt to fetch a plan from a registered LLM provider
        if let Some(provider_handle) = self._provider_registry.resolve(&self._model) {
            // Load and apply stored credentials
            let provider_name = {
                let lock = provider_handle.read().await;
                lock.name().to_string()
            };
            if let Ok(cred_store) = crate::auth::store::CredentialStore::load() {
                if let Some(creds) = cred_store.get(&provider_name) {
                    let mut w = provider_handle.write().await;
                    let _ = w.authenticate(&creds.api_key).await;
                }
            }
            // Acquire read access to the provider (tokio::RwLock)
            let provider_guard = provider_handle.read().await;
            // Call the provider to stream completion (plan in JSON)
            if let Ok(chunks) = provider_guard.stream_completion(&messages, &self._model).await {
                // Join chunk content into a single JSON string
                let json_str: String = chunks.into_iter().map(|c| c.content).collect::<String>();
                    // Parse JSON into intermediary struct and map to PlanStep
                    #[derive(Deserialize)]
                    struct StepJson {
                        id: String,
                        title: String,
                        tool: String,
                        status: String,
                        duration: Option<f64>,
                        thoughts: Vec<String>,
                    }
                    let mut json_text = json_str.trim().to_string();
                    if json_text.starts_with("```json") {
                        json_text = json_text.trim_start_matches("```json").trim_end_matches("```").trim().to_string();
                    } else if json_text.starts_with("```") {
                        json_text = json_text.trim_start_matches("```").trim_end_matches("```").trim().to_string();
                    }
                    if let Ok(steps_json) = serde_json::from_str::<Vec<StepJson>>(&json_text) {
                        let mut steps = Vec::<PlanStep>::with_capacity(steps_json.len());
                        for s in steps_json {
                            let tool = match s.tool.as_str() {
                                "Search"|"search" => Tool::Search,
                                "Read"|"read" => Tool::Read,
                                "Think"|"think" => Tool::Think,
                                "Edit"|"edit" => Tool::Edit,
                                "Shell"|"shell" => Tool::Shell,
                                "ReadFile"|"read_file" => Tool::ReadFile,
                                "ListDir"|"list_dir" => Tool::ListDir,
                                _ => Tool::Search,
                            };
                            let status = match s.status.to_lowercase().as_str() {
                                "queued" => StepStatus::Queued,
                                "running" => StepStatus::Running,
                                "done" | "completed" => StepStatus::Done,
                                _ => StepStatus::Queued,
                            };
                            steps.push(PlanStep {
                                id: s.id,
                                title: s.title,
                                tool,
                                status,
                                duration: s.duration,
                                thoughts: s.thoughts,
                            });
                        }
                        if !steps.is_empty() {
                            return Ok(steps);
                        }
                    }
                }
        }

        let fallback = match intent.map(|i| &i.intent) {
            Some(QueryIntent::LocalAnalysis) | None => vec![
                PlanStep { id: "1".to_string(), title: ".".to_string(), tool: Tool::ReadFile, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
                PlanStep { id: "2".to_string(), title: "think".to_string(), tool: Tool::Think, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
            ],
            Some(QueryIntent::WebResearch) => {
                let q = intent.and_then(|i| i.search_queries.first()).map(|s| s.as_str()).unwrap_or(query);
                vec![
                    PlanStep { id: "1".to_string(), title: q.to_string(), tool: Tool::Search, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
                    PlanStep { id: "2".to_string(), title: "think".to_string(), tool: Tool::Think, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
                ]
            }
            Some(QueryIntent::Mixed) => {
                let q = intent.and_then(|i| i.search_queries.first()).map(|s| s.as_str()).unwrap_or(query);
                vec![
                    PlanStep { id: "1".to_string(), title: ".".to_string(), tool: Tool::ReadFile, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
                    PlanStep { id: "2".to_string(), title: q.to_string(), tool: Tool::Search, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
                    PlanStep { id: "3".to_string(), title: "think".to_string(), tool: Tool::Think, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
                ]
            }
            Some(QueryIntent::DirectAnswer) => vec![
                PlanStep { id: "1".to_string(), title: query.to_string(), tool: Tool::Think, status: StepStatus::Queued, duration: None, thoughts: Vec::new() },
            ],
        };
        Ok(fallback)
    }
}
