use std::sync::Arc;
use std::sync::mpsc;

use async_trait::async_trait;
use crate::app::{PlanStep, Source, AnswerChunk};
use crate::agent::planner::AgentPlanner;
use crate::agent::tools::ToolRegistry;
use crate::agent::classifier::{QueryClassifier, QueryIntent};
use crate::agent::researcher::{ResearcherLoop, ResearchDepth, ResearcherOutput};
use crate::agent::synthesizer::ResearchSynthesizer;
use crate::llm::provider::{
    LlmProvider, ProviderRegistry, ProviderHandle, ProviderError,
    Message, Chunk, ToolDefinition, ToolResponse,
};
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
    ModelListReady(Vec<String>),
    Classified(QueryIntent),
    SearchingIteration { current: u8, max: u8, query: String },
}

/// Adapter wrapping ProviderHandle to implement LlmProvider.
/// Required because ResearcherLoop/Synthesizer expect Arc<dyn LlmProvider>.
struct ProviderHandleAdapter {
    handle: ProviderHandle,
    name_str: String,
}

impl ProviderHandleAdapter {
    fn new(handle: ProviderHandle) -> Self {
        Self { handle, name_str: "provider".to_string() }
    }
}

#[async_trait]
impl LlmProvider for ProviderHandleAdapter {
    async fn stream_completion(&self, messages: &[Message], model: &str) -> Result<Vec<Chunk>, ProviderError> {
        let lock = self.handle.read().await;
        lock.stream_completion(messages, model).await
    }
    async fn complete_with_tools(
        &self, messages: &[Message], tools: &[ToolDefinition], model: &str,
    ) -> Result<ToolResponse, ProviderError> {
        let lock = self.handle.read().await;
        lock.complete_with_tools(messages, tools, model).await
    }
    fn name(&self) -> &str { &self.name_str }
    fn is_authenticated(&self) -> bool { true } // authenticated before adapter creation
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
        Self { planner, tools, provider_registry, model, classifier }
    }

    pub async fn run(&self, query: &str, tx: mpsc::Sender<AgentEvent>, cancel_token: CancellationToken) {
        // Phase 1: Classify
        let classified = self.classifier.classify(query).await;
        let _ = tx.send(AgentEvent::Classified(classified.intent.clone()));
        if cancel_token.is_cancelled() {
            let _ = tx.send(AgentEvent::Error("Cancelled".into()));
            return;
        }

        let depth = match classified.intent {
            QueryIntent::DirectAnswer => ResearchDepth::Speed,
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
        let provider: Arc<dyn LlmProvider> =
            Arc::new(ProviderHandleAdapter::new(provider_handle));

        // Phase 2: Research (skip if DirectAnswer)
        let output = if classified.skip_search {
            ResearcherOutput { answer: String::new(), sources: Vec::new(), iterations_used: 0 }
        } else {
            let researcher = ResearcherLoop::new(
                provider.clone(), self.model.clone(), self.tools.clone(), cancel_token.clone(),
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
        let (final_text, final_sources) = match synthesizer.synthesize(query, &output.sources, depth).await {
            Ok(synth) => (synth.text, output.sources),
            Err(_) => (output.answer, output.sources), // fallback to researcher answer
        };

        // Emit answer chunks (split by paragraph for progressive rendering)
        for chunk_text in final_text.split("\n\n") {
            let chunk = AnswerChunk {
                text: chunk_text.to_string(), is_code: false, is_bold: false, is_em: false, citations: vec![],
            };
            let _ = tx.send(AgentEvent::AnswerChunk(chunk));
        }

        let _ = tx.send(AgentEvent::Done(Answer { text: final_text, sources: final_sources }));
    }
}