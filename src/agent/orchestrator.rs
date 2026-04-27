use std::sync::Arc;

use crate::app::{PlanStep, Source, AnswerChunk};
use crate::agent::planner::AgentPlanner;
use crate::agent::tools::ToolRegistry;
use crate::llm::provider::{ProviderRegistry, Role};
use tokio_util::sync::CancellationToken;
use crate::agent::classifier::QueryClassifier;

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
    Classified(crate::agent::classifier::QueryIntent),
    SearchingIteration { current: u8, max: u8, query: String },
}

pub struct AgentOrchestrator {
    planner: AgentPlanner,
    tools: ToolRegistry,
    provider_registry: Arc<ProviderRegistry>,
    model: String,
    classifier: QueryClassifier,
}

impl AgentOrchestrator {
    pub fn new(
        planner: AgentPlanner,
        tools: ToolRegistry,
        provider_registry: Arc<ProviderRegistry>,
        model: String,
    ) -> Self {
        // Lightweight default classifier construction; allows main.rs to compile with 4 args
        let classifier = QueryClassifier::new(provider_registry.clone(), model.clone());
        Self { planner, tools, provider_registry, model, classifier }
    }

    pub async fn run(&self, query: &str, tx: mpsc::Sender<AgentEvent>, cancel_token: CancellationToken) {
        // Phase 1: classify the query
        let classified = self.classifier.classify(query).await;
        let _ = tx.send(AgentEvent::Classified(classified.intent.clone()));
        // 1) Plan
        let steps = match self.planner.plan(query, Some(&classified)).await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(AgentEvent::Error(format!("Planner failed: {:?}", e)));
                return;
            }
        };
        let steps_clone = steps.clone();
        let _ = tx.send(AgentEvent::PlanReady(steps_clone));

        // Check cancellation after planning
        if cancel_token.is_cancelled() {
            let _ = tx.send(AgentEvent::Error("Query cancelled".to_string()));
            return;
        }

        // 2) Execute steps sequentially via the extracted executor.
        let mut all_sources: Vec<Source> = Vec::new();
        let mut step_outputs: Vec<String> = Vec::new();
        for (idx, step) in steps.into_iter().enumerate() {
            if cancel_token.is_cancelled() {
                let _ = tx.send(AgentEvent::Error("Query cancelled".to_string()));
                return;
            }
            if let Some(out) = crate::agent::executor::execute_step(&step, idx, &self.tools, &mut all_sources, &tx).await {
                step_outputs.push(out);
            }
        }

        // Check cancellation before synthesis
        if cancel_token.is_cancelled() {
            let _ = tx.send(AgentEvent::Error("Query cancelled".to_string()));
            return;
        }

        // 3) Synthesis via LLM provider
        if let Some(provider_handle) = self.provider_registry.resolve(&self.model) {
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
            let provider_lock = provider_handle.write().await;
            let manifest = crate::agent::tool_manifest();
            let mut messages: Vec<crate::llm::provider::Message> = Vec::new();
            messages.push(crate::llm::provider::Message {
                role: Role::System,
                content: format!(
                    "You are an AI coding and research assistant.\n{manifest}\n\
                     Answer concisely and accurately.\n\
                     Every factual claim MUST be backed by a citation: \
                     use `[filename]` for local files and `[url]` for web sources. \
                     If no evidence was gathered, say so explicitly.",
                ),
            });
            let mut synth = format!("Query: {}\n", query);
            if !step_outputs.is_empty() {
                synth.push_str("\nGathered context:\n");
                for out in &step_outputs {
                    synth.push_str(out);
                    synth.push('\n');
                }
            }
            if !all_sources.is_empty() {
                synth.push_str("\nWeb sources:\n");
                for s in &all_sources {
                    synth.push_str(&format!("- {} {}\n", s.title, s.url));
                }
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
