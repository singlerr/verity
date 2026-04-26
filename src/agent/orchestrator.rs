use std::sync::Arc;
use std::time::Instant;

use serde_json::json;

use crate::app::{PlanStep, Source, AnswerChunk, Tool as PlanTool};
use crate::agent::planner::AgentPlanner;
use crate::agent::tools::ToolRegistry;
use crate::llm::provider::{ProviderRegistry, Role};

use std::sync::mpsc;

/// Public answer container.
#[derive(Debug, Clone)]
pub struct Answer {
    pub text: String,
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    PlanReady(Vec<PlanStep>),
    StepStarted(usize),
    StepProgress(usize, String),
    StepDone(usize),
    StepFailed(usize, String),
    SourceFound(Source),
    AnswerChunk(AnswerChunk),
    Done(Answer),
    Error(String),
    ModelListReady(Vec<String>),
}

pub struct AgentOrchestrator {
    planner: AgentPlanner,
    tools: ToolRegistry,
    provider_registry: Arc<ProviderRegistry>,
    model: String,
}

impl AgentOrchestrator {
    pub fn new(planner: AgentPlanner, tools: ToolRegistry, provider_registry: Arc<ProviderRegistry>, model: String) -> Self {
        Self { planner, tools, provider_registry, model }
    }

    pub async fn run(&self, query: &str, tx: mpsc::Sender<AgentEvent>) {
        // 1) Plan
        let steps = match self.planner.plan(query).await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(AgentEvent::Error(format!("Planner failed: {:?}", e)));
                return;
            }
        };
        let steps_clone = steps.clone();
        let _ = tx.send(AgentEvent::PlanReady(steps_clone));

        // 2) Execute steps sequentially
        let mut all_sources: Vec<Source> = Vec::new();
        for (idx, step) in steps.into_iter().enumerate() {
            let _ = tx.send(AgentEvent::StepStarted(idx));
            let _start = Instant::now();

            match step.tool {
                PlanTool::Search => {
                    if let Some(tool) = self.tools.get("search") {
                        let input = json!({"query": step.title});
                        match tool.execute(&input).await {
                            Ok(val) => {
                                if let Some(results) = val.as_array() {
                                    for (i, item) in results.iter().enumerate() {
                                        let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
                                        let url = item.get("url").and_then(|v| v.as_str()).unwrap_or("");
                                        let snippet = item.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                                        let domain = extract_domain(url);
                                        let src = Source {
                                            num: all_sources.len() + i + 1,
                                            domain,
                                            title: title.to_string(),
                                            url: url.to_string(),
                                            snippet: snippet.to_string(),
                                            quote: String::new(),
                                        };
                                        all_sources.push(src.clone());
                                        let _ = tx.send(AgentEvent::SourceFound(src));
                                    }
                                } else {
                                    let _ = tx.send(AgentEvent::StepFailed(idx, "Search: unexpected response".to_string()));
                                }
                                let _ = tx.send(AgentEvent::StepDone(idx));
                            }
                            Err(e) => {
                                let _ = tx.send(AgentEvent::StepFailed(idx, format!("Search tool failed: {}", e)));
                            }
                        }
                    } else {
                        let _ = tx.send(AgentEvent::StepFailed(idx, "Search tool not found".to_string()));
                    }
                }
                PlanTool::Read => {
                    if let Some(tool) = self.tools.get("read_url") {
                        if let Some(first) = all_sources.first() {
                            let input = json!({"url": first.url});
                            match tool.execute(&input).await {
                                Ok(_v) => {
                                    let _ = tx.send(AgentEvent::StepDone(idx));
                                }
                                Err(e) => {
                                    let _ = tx.send(AgentEvent::StepFailed(idx, format!("Read tool failed: {}", e)));
                                }
                            }
                        } else {
                            let _ = tx.send(AgentEvent::StepProgress(idx, "No sources to read".to_string()));
                            let _ = tx.send(AgentEvent::StepDone(idx));
                        }
                    } else {
                        let _ = tx.send(AgentEvent::StepFailed(idx, "Read tool not found".to_string()));
                    }
                }
                PlanTool::Think => {
                    let _ = tx.send(AgentEvent::StepProgress(idx, "Thinking...".to_string()));
                    let _ = tx.send(AgentEvent::StepDone(idx));
                }
                PlanTool::Edit | PlanTool::Shell => {
                    let _ = tx.send(AgentEvent::StepProgress(idx, "Skipped".to_string()));
                    let _ = tx.send(AgentEvent::StepDone(idx));
                }
            }

            let _dur = _start.elapsed();
            let _ = _dur;
        }

        // 3) Synthesis via LLM provider
        if let Some(provider_handle) = self.provider_registry.resolve(&self.model) {
            let provider_lock = provider_handle.write().await;
            let mut messages: Vec<crate::llm::provider::Message> = Vec::new();
            messages.push(crate::llm::provider::Message {
                role: Role::System,
                content: "You are an AI research assistant.".to_string(),
            });
            let mut synth = format!("Query: {}\nSources:\n", query);
            for s in &all_sources {
                synth.push_str(&format!("- {} {}\n", s.title, s.url));
            }
            messages.push(crate::llm::provider::Message { role: Role::User, content: synth });

            match provider_lock.stream_completion(&messages, &self.model).await {
                Ok(chunks) => {
                    let mut final_text = String::new();
                    for c in &chunks {
                        final_text.push_str(&c.content);
                    }
                    for c in chunks {
                        let chunk = AnswerChunk {
                            text: c.content,
                            is_code: false,
                            is_bold: false,
                            is_em: false,
                            citations: vec![],
                        };
                        let _ = tx.send(AgentEvent::AnswerChunk(chunk));
                    }
                    let answer = Answer { text: final_text, sources: all_sources };
                    let _ = tx.send(AgentEvent::Done(answer));
                }
                Err(e) => {
                    let _ = tx.send(AgentEvent::Error(format!("LLM synthesis failed: {:?}", e)));
                }
            }
        } else {
            let _ = tx.send(AgentEvent::Error("No provider for model".to_string()));
        }
    }
}

fn extract_domain(url: &str) -> String {
    if let Some(pos) = url.find("://") {
        let rest = &url[(pos+3)..];
        rest.split('/').next().unwrap_or("").to_string()
    } else {
        String::new()
    }
}
