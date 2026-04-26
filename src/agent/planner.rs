use std::sync::Arc;

use anyhow::Result;

use crate::app::{PlanStep, StepStatus, Tool};
use crate::llm::provider::{Message, Role, ProviderRegistry};
use serde::Deserialize;

/// Agent planner that uses LLM function calling to break user queries
/// into concrete research plan steps. This is a lightweight stub that
/// preserves the intended interface and enables compilation in tests.
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

    /// Build a plan prompt for the given query. This mirrors the intended
    /// LLM prompt structure used by the real system.
    pub fn plan_prompt(query: &str) -> Vec<Message> {
        let system = Message {
            role: Role::System,
            content: "You are a research assistant. Break the user's query into concrete research steps. Available tools: search(query) - search the web, read_url(url) - read web page content, read_file(path) - read local file. Respond with a numbered plan, one step per line. Format: [TOOL] description".to_string(),
        };
        let user = Message {
            role: Role::User,
            content: query.to_string(),
        };
        vec![system, user]
    }

    /// Produce a concrete plan for the given query.
    /// Try to generate the plan via an LLM function call. If the LLM is unavailable
    /// or fails to return a valid JSON, fall back to a hardcoded 3-step plan.
    pub async fn plan(&self, query: &str) -> Result<Vec<PlanStep>> {
        // Build an LLM prompt that asks for a JSON array of steps
        let messages = Self::plan_prompt(query);

        // Attempt to fetch a plan from a registered LLM provider
        if let Some(provider_handle) = self._provider_registry.resolve(&self._model) {
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
                    if let Ok(steps_json) = serde_json::from_str::<Vec<StepJson>>(&json_str) {
                        let mut steps = Vec::<PlanStep>::with_capacity(steps_json.len());
                        for s in steps_json {
                            let tool = match s.tool.as_str() {
                                "Search"|"search" => Tool::Search,
                                "Read"|"read" => Tool::Read,
                                "Think"|"think" => Tool::Think,
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

        // Fallback hardcoded plan (Search, Read, Think)
        let mut steps = Vec::<PlanStep>::new();
        steps.push(PlanStep {
            id: "1".to_string(),
            title: format!("Perform web search for: {}", query),
            tool: Tool::Search,
            status: StepStatus::Queued,
            duration: None,
            thoughts: Vec::<String>::new(),
        });
        steps.push(PlanStep {
            id: "2".to_string(),
            title: "Read top results to gather evidence".to_string(),
            tool: Tool::Read,
            status: StepStatus::Queued,
            duration: None,
            thoughts: Vec::<String>::new(),
        });
        steps.push(PlanStep {
            id: "3".to_string(),
            title: "Synthesize evidence and propose a concrete plan".to_string(),
            tool: Tool::Think,
            status: StepStatus::Queued,
            duration: None,
            thoughts: Vec::<String>::new(),
        });
        Ok(steps)
    }
}
