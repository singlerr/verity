use std::sync::mpsc;
use std::sync::Arc;

use crate::agent::classifier::{QueryClassifier, QueryIntent};
use crate::agent::planner::AgentPlanner;
use crate::agent::researcher::{ResearchDepth, ResearcherLoop, ResearcherOutput};
use crate::agent::synthesizer::ResearchSynthesizer;
use crate::agent::tools::ToolRegistry;
use crate::app::{AnswerChunk, PlanStep, Source};
use crate::llm::provider::{
    Chunk, LlmProvider, Message, ModelEntry, ProviderError, ProviderHandle, ProviderRegistry,
    ToolDefinition, ToolResponse,
};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

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
    ModelListReady(Vec<ModelEntry>),
    Classified(QueryIntent),
    SearchingIteration { current: u8, max: u8, query: String },
}

/// Adapter wrapping ProviderHandle to implement LlmProvider.
/// Required because ResearcherLoop/Synthesizer expect Arc<dyn LlmProvider>.
struct ProviderHandleAdapter {
    handle: ProviderHandle,
    name_str: String,
    supports_tool_calling: bool,
}

impl ProviderHandleAdapter {
    fn new(handle: ProviderHandle) -> Self {
        Self {
            handle,
            name_str: "provider".to_string(),
            supports_tool_calling: false,
        }
    }
    fn new_with_name(handle: ProviderHandle, name: String) -> Self {
        Self {
            handle,
            name_str: name,
            supports_tool_calling: false,
        }
    }
    fn new_with_support(handle: ProviderHandle, name: String, supports_tool_calling: bool) -> Self {
        Self {
            handle,
            name_str: name,
            supports_tool_calling,
        }
    }
}

#[async_trait]
impl LlmProvider for ProviderHandleAdapter {
    async fn stream_completion(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<Vec<Chunk>, ProviderError> {
        let lock = self.handle.read().await;
        lock.stream_completion(messages, model).await
    }
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
    ) -> Result<ToolResponse, ProviderError> {
        let lock = self.handle.read().await;
        lock.complete_with_tools(messages, tools, model).await
    }
    fn name(&self) -> &str {
        &self.name_str
    }
    fn is_authenticated(&self) -> bool {
        true
    } // authenticated before adapter creation
    async fn authenticate(&mut self, api_key: &str) -> Result<(), ProviderError> {
        let mut lock = self.handle.write().await;
        lock.authenticate(api_key).await
    }
    async fn deauthenticate(&mut self) -> Result<(), ProviderError> {
        let mut lock = self.handle.write().await;
        lock.deauthenticate().await
    }
    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let lock = self.handle.read().await;
        lock.list_models().await
    }
    fn supports_tool_calling(&self) -> bool {
        self.supports_tool_calling
    }
}

pub struct AgentOrchestrator {
    #[allow(dead_code)] // kept for API compat; Task 7 removes dead code
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
        let classifier = QueryClassifier::new(provider_registry.clone(), model.clone());
        Self {
            planner,
            tools,
            provider_registry,
            model,
            classifier,
        }
    }

    pub async fn run(
        &self,
        query: &str,
        tx: mpsc::Sender<AgentEvent>,
        cancel_token: CancellationToken,
    ) {
        // Phase 1: Classify
        let classified = self.classifier.classify(query).await;
        let _ = tx.send(AgentEvent::Classified(classified.intent.clone()));
        if cancel_token.is_cancelled() {
            let _ = tx.send(AgentEvent::Error("Cancelled".into()));
            return;
        }

        // LocalAnalysis, WebResearch, and Mixed all route through the researcher loop.
        // LocalAnalysis benefits from local tools (read_file, list_dir, grep, glob, edit_file)
        // which are now available in all research modes.
        let depth = match classified.intent {
            QueryIntent::DirectAnswer => ResearchDepth::Speed,
            _ if classified.quality => ResearchDepth::Quality,
            _ => ResearchDepth::Balanced,
        };

        // Resolve and authenticate provider
        let provider_handle = match self.provider_registry.resolve(&self.model) {
            Some(h) => h,
            None => {
                let _ = tx.send(AgentEvent::Error("No provider for model".into()));
                return;
            }
        };
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
        let tool_calling = {
            let lock = provider_handle.read().await;
            lock.supports_tool_calling()
        };
        let provider: Arc<dyn LlmProvider> = Arc::new(ProviderHandleAdapter::new_with_support(
            provider_handle,
            provider_name.clone(),
            tool_calling,
        ));

        // Check tool calling support for non-trivial research
        if !classified.skip_search && !provider.supports_tool_calling() {
            let _ = tx.send(AgentEvent::Error(
                "Deep Research requires a model with tool calling support. Please select a compatible model (e.g., gpt-4o, claude-3-5-sonnet-latest, gemini-2.0-flash). You can use /model to change models.".into()
            ));
            return;
        }

        // Phase 2: Research (skip if DirectAnswer)
        let output = if classified.skip_search {
            ResearcherOutput {
                answer: String::new(),
                sources: Vec::new(),
                iterations_used: 0,
                extracted_facts: Vec::new(),
            }
        } else {
            let researcher = ResearcherLoop::new(
                provider.clone(),
                self.model.clone(),
                self.tools.clone(),
                cancel_token.clone(),
            );
            match researcher.run(query, depth, &tx).await {
                Ok(out) => out,
                Err(e) => {
                    let _ = tx.send(AgentEvent::Error(e));
                    return;
                }
            }
        };

        if cancel_token.is_cancelled() {
            let _ = tx.send(AgentEvent::Error("Cancelled".into()));
            return;
        }

        // Phase 3: Synthesize
        let synthesizer = ResearchSynthesizer::new(provider, &self.model);
        let (final_text, final_sources) = match synthesizer
            .synthesize(query, &output.sources, depth, &output.extracted_facts)
            .await
        {
            Ok(synth) => (synth.text, output.sources),
            Err(_) => (output.answer, output.sources), // fallback to researcher answer
        };

        // Emit answer chunks (split by paragraph for progressive rendering)
        for chunk_text in final_text.split("\n\n") {
            let chunk = AnswerChunk {
                text: chunk_text.to_string(),
                is_code: false,
                is_bold: false,
                is_em: false,
                citations: vec![],
            };
            let _ = tx.send(AgentEvent::AnswerChunk(chunk));
        }

        let _ = tx.send(AgentEvent::Done(Answer {
            text: final_text,
            sources: final_sources,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider::mock::MockLlmProvider;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[test]
    fn provider_adapter_tool_calling_check() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mock = MockLlmProvider::new(vec![]);
            let handle = Arc::new(RwLock::new(
                Box::new(mock) as Box<dyn LlmProvider + Send + Sync>
            ));
            let adapter = ProviderHandleAdapter::new_with_support(handle, "openai".to_string(), true);
            assert!(
                adapter.supports_tool_calling(),
                "openai provider should support tool calling"
            );

            let mock2 = MockLlmProvider::without_tool_calling();
            let handle2 = Arc::new(RwLock::new(
                Box::new(mock2) as Box<dyn LlmProvider + Send + Sync>
            ));
            let adapter2 = ProviderHandleAdapter::new_with_support(handle2, "ollama".to_string(), false);
            assert!(
                !adapter2.supports_tool_calling(),
                "non-tool-calling provider should not support tool calling"
            );

            let mock3 = MockLlmProvider::new(vec![]);
            let handle3 = Arc::new(RwLock::new(
                Box::new(mock3) as Box<dyn LlmProvider + Send + Sync>
            ));
            let adapter3 = ProviderHandleAdapter::new_with_support(handle3, "nvidia".to_string(), true);
            assert!(
                adapter3.supports_tool_calling(),
                "nvidia provider should support tool calling"
            );
        });
    }

    #[test]
    fn orchestrator_rejects_non_tool_calling_provider() {
        let mock = MockLlmProvider::without_tool_calling();
        assert!(
            !mock.supports_tool_calling(),
            "Mock without tool calling returns false"
        );
    }
}
