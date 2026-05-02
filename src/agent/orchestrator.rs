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
use tracing::debug;

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

        let setup_trace =
            Self::research_setup_trace(&classified.intent, depth, classified.skip_search);
        debug!("{}", setup_trace);
        let _ = tx.send(AgentEvent::StepProgress(0, setup_trace));

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
        let tool_support_trace = Self::tool_support_trace(tool_calling);
        debug!("{}", tool_support_trace);
        let _ = tx.send(AgentEvent::StepProgress(0, tool_support_trace));
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
            let categories: Vec<String> = if classified.source_types.is_empty() {
                vec!["general".to_string()]
            } else {
                classified
                    .source_types
                    .iter()
                    .map(|st| st.as_searxng_category().to_string())
                    .collect()
            };
            let suggested_web_queries =
                Self::suggested_web_queries(&classified.intent, &classified.search_queries, query);
            let suggested_query_count_trace =
                Self::suggested_query_count_trace(suggested_web_queries.len());
            debug!("{}", suggested_query_count_trace);
            let _ = tx.send(AgentEvent::StepProgress(0, suggested_query_count_trace));
            let researcher = ResearcherLoop::new(
                provider.clone(),
                self.model.clone(),
                self.tools.clone(),
                cancel_token.clone(),
                categories,
                suggested_web_queries,
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

        let final_trace =
            Self::final_counts_trace(output.sources.len(), output.extracted_facts.len());
        debug!("{}", final_trace);
        let _ = tx.send(AgentEvent::StepProgress(0, final_trace));

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

    fn research_setup_trace(
        intent: &QueryIntent,
        depth: ResearchDepth,
        skip_search: bool,
    ) -> String {
        format!(
            "research:classify={} depth={} skip_search={}",
            Self::intent_label(intent),
            Self::depth_label(depth),
            skip_search
        )
    }

    fn tool_support_trace(tool_support: bool) -> String {
        format!("research:tool_support={}", tool_support)
    }

    fn final_counts_trace(sources: usize, facts: usize) -> String {
        format!("research:final sources={} facts={}", sources, facts)
    }

    fn suggested_query_count_trace(count: usize) -> String {
        format!("research:suggested_web_queries={}", count)
    }

    fn suggested_web_queries(
        intent: &QueryIntent,
        classified_queries: &[String],
        raw_query: &str,
    ) -> Vec<String> {
        match intent {
            QueryIntent::WebResearch | QueryIntent::Mixed => {
                let queries: Vec<String> = classified_queries
                    .iter()
                    .map(|query| query.trim())
                    .filter(|query| !query.is_empty())
                    .take(3)
                    .map(Self::truncate_suggestion)
                    .collect();
                if queries.is_empty() {
                    let raw_query = raw_query.trim();
                    if raw_query.is_empty() {
                        vec![]
                    } else {
                        vec![Self::truncate_suggestion(raw_query)]
                    }
                } else {
                    queries
                }
            }
            QueryIntent::LocalAnalysis | QueryIntent::DirectAnswer => vec![],
        }
    }

    fn truncate_suggestion(query: &str) -> String {
        query.chars().take(200).collect()
    }

    fn intent_label(intent: &QueryIntent) -> &'static str {
        match intent {
            QueryIntent::DirectAnswer => "direct_answer",
            QueryIntent::WebResearch => "web_research",
            QueryIntent::LocalAnalysis => "local_analysis",
            QueryIntent::Mixed => "mixed",
        }
    }

    fn depth_label(depth: ResearchDepth) -> &'static str {
        match depth {
            ResearchDepth::Speed => "speed",
            ResearchDepth::Balanced => "balanced",
            ResearchDepth::Quality => "quality",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider::mock::MockLlmProvider;
    use crate::llm::provider::{Chunk, FinishReason, ProviderMetadata, ToolCall, ToolResponse};
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tokio_util::sync::CancellationToken;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn provider_registry_with_mock(provider: MockLlmProvider) -> Arc<ProviderRegistry> {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "mock".into(),
            Box::new(provider),
            ProviderMetadata {
                display_name: "mock".into(),
                requires_api_key: false,
                model_prefixes: vec!["gpt-".into()],
                fallback_models: vec![],
            },
        );
        Arc::new(registry)
    }

    fn make_orchestrator(provider: MockLlmProvider, searxng_url: &str) -> AgentOrchestrator {
        let registry = provider_registry_with_mock(provider);
        let planner = AgentPlanner::new(registry.clone(), "gpt-test".into());
        let tools = crate::agent::tools::build_tool_registry(searxng_url);
        AgentOrchestrator::new(planner, tools, registry, "gpt-test".into())
    }

    fn mock_stream_responses(final_answer: &str) -> Vec<Vec<Chunk>> {
        vec![
            vec![Chunk {
                content: json!({
                    "intent": "web_research",
                    "search_queries": ["verity research plumbing"],
                    "reasoning": "needs live search",
                    "skip_search": false,
                    "source_types": ["general"],
                    "quality": false
                })
                .to_string(),
            }],
            vec![Chunk {
                content: final_answer.to_string(),
            }],
        ]
    }

    fn mock_research_response(content: &str) -> ToolResponse {
        ToolResponse {
            content: Some(content.into()),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: None,
        }
    }

    fn mock_web_search_response() -> ToolResponse {
        ToolResponse {
            content: None,
            tool_calls: vec![
                ToolCall {
                    id: "tc_reasoning".into(),
                    name: "__reasoning_preamble".into(),
                    arguments: json!({"thoughts": "Use the suggested web query now."}).to_string(),
                },
                ToolCall {
                    id: "tc_web_search".into(),
                    name: "web_search".into(),
                    arguments: json!({"queries": ["verity research plumbing"]}).to_string(),
                },
            ],
            finish_reason: FinishReason::ToolCalls,
            usage: None,
        }
    }

    async fn run_research_case(search_response: ResponseTemplate) -> Vec<AgentEvent> {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "verity research plumbing"))
            .and(query_param("format", "json"))
            .and(query_param("categories", "general"))
            .respond_with(search_response)
            .mount(&server)
            .await;

        let provider = MockLlmProvider::with_stream(
            vec![
                mock_web_search_response(),
                mock_research_response("research loop answer 2"),
            ],
            mock_stream_responses("Final synthesized answer"),
        );
        let orchestrator = make_orchestrator(provider, &server.uri());
        let (tx, rx) = mpsc::channel();

        orchestrator
            .run("verity research plumbing", tx, CancellationToken::new())
            .await;

        rx.try_iter().collect()
    }

    fn step_progress_messages(events: &[AgentEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::StepProgress(_, msg) => Some(msg.clone()),
                _ => None,
            })
            .collect()
    }

    fn done_event(events: &[AgentEvent]) -> Option<&Answer> {
        events.iter().find_map(|event| match event {
            AgentEvent::Done(answer) => Some(answer),
            _ => None,
        })
    }

    fn source_found_count(events: &[AgentEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::SourceFound(_)))
            .count()
    }

    #[test]
    fn provider_adapter_tool_calling_check() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mock = MockLlmProvider::new(vec![]);
            let handle = Arc::new(RwLock::new(
                Box::new(mock) as Box<dyn LlmProvider + Send + Sync>
            ));
            let adapter =
                ProviderHandleAdapter::new_with_support(handle, "openai".to_string(), true);
            assert!(
                adapter.supports_tool_calling(),
                "openai provider should support tool calling"
            );

            let mock2 = MockLlmProvider::without_tool_calling();
            let handle2 = Arc::new(RwLock::new(
                Box::new(mock2) as Box<dyn LlmProvider + Send + Sync>
            ));
            let adapter2 =
                ProviderHandleAdapter::new_with_support(handle2, "ollama".to_string(), false);
            assert!(
                !adapter2.supports_tool_calling(),
                "non-tool-calling provider should not support tool calling"
            );

            let mock3 = MockLlmProvider::new(vec![]);
            let handle3 = Arc::new(RwLock::new(
                Box::new(mock3) as Box<dyn LlmProvider + Send + Sync>
            ));
            let adapter3 =
                ProviderHandleAdapter::new_with_support(handle3, "nvidia".to_string(), true);
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

    #[test]
    fn research_diagnostics_are_concise() {
        let setup = AgentOrchestrator::research_setup_trace(
            &QueryIntent::WebResearch,
            ResearchDepth::Quality,
            false,
        );
        assert_eq!(
            setup,
            "research:classify=web_research depth=quality skip_search=false"
        );

        let support = AgentOrchestrator::tool_support_trace(true);
        assert_eq!(support, "research:tool_support=true");

        let final_counts = AgentOrchestrator::final_counts_trace(3, 2);
        assert_eq!(final_counts, "research:final sources=3 facts=2");

        let suggested_count = AgentOrchestrator::suggested_query_count_trace(2);
        assert_eq!(suggested_count, "research:suggested_web_queries=2");

        let local_setup = AgentOrchestrator::research_setup_trace(
            &QueryIntent::LocalAnalysis,
            ResearchDepth::Balanced,
            false,
        );
        assert_eq!(
            local_setup,
            "research:classify=local_analysis depth=balanced skip_search=false"
        );

        let skip_setup = AgentOrchestrator::research_setup_trace(
            &QueryIntent::DirectAnswer,
            ResearchDepth::Speed,
            true,
        );
        assert_eq!(
            skip_setup,
            "research:classify=direct_answer depth=speed skip_search=true"
        );

        let zero_counts = AgentOrchestrator::final_counts_trace(0, 0);
        assert_eq!(zero_counts, "research:final sources=0 facts=0");
    }

    #[test]
    fn web_research_provides_classifier_suggested_queries() {
        let classified_queries = vec!["rust async".to_string(), "tokio runtime".to_string()];

        let suggested_queries = AgentOrchestrator::suggested_web_queries(
            &QueryIntent::WebResearch,
            &classified_queries,
            "raw rust query",
        );

        assert_eq!(suggested_queries, classified_queries);
    }

    #[test]
    fn web_research_suggestions_fall_back_to_raw_query_when_classifier_queries_are_empty() {
        let classified_queries = vec![" ".to_string(), "".to_string()];

        let suggested_queries = AgentOrchestrator::suggested_web_queries(
            &QueryIntent::Mixed,
            &classified_queries,
            "raw mixed query",
        );

        assert_eq!(suggested_queries, vec!["raw mixed query".to_string()]);
    }

    #[test]
    fn local_analysis_does_not_provide_web_query_suggestions() {
        let classified_queries = vec!["should not search web".to_string()];

        let suggested_queries = AgentOrchestrator::suggested_web_queries(
            &QueryIntent::LocalAnalysis,
            &classified_queries,
            "inspect local files",
        );

        assert!(suggested_queries.is_empty());
    }

    #[tokio::test]
    async fn research_plumbing_regression_covers_success_zero_results_and_errors() {
        let success_events = run_research_case(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                {
                    "title": "Verity research plumbing",
                    "url": "https://example.com/verity",
                    "content": "Search snippet"
                }
            ]
        })))
        .await;
        let success_progress = step_progress_messages(&success_events);

        assert!(success_events.iter().any(|event| matches!(
            event,
            AgentEvent::SearchingIteration {
                current: 1,
                query,
                ..
            } if query == "verity research plumbing"
        )));
        assert!(success_progress
            .iter()
            .any(|msg| msg == "research:suggested_web_queries=1"));
        assert_eq!(source_found_count(&success_events), 1);
        assert!(success_progress
            .iter()
            .any(|msg| msg == "search:results=1 error_kinds=none"));
        let success_done = done_event(&success_events).expect("done event");
        assert!(!success_done.text.trim().is_empty());
        assert_eq!(success_done.sources.len(), 1);

        let zero_events = run_research_case(ResponseTemplate::new(200).set_body_json(json!({
            "results": []
        })))
        .await;
        let zero_progress = step_progress_messages(&zero_events);

        assert_eq!(source_found_count(&zero_events), 0);
        assert!(zero_progress
            .iter()
            .any(|msg| msg == "search:results=0 error_kinds=none"));
        assert!(!done_event(&zero_events)
            .expect("done event")
            .text
            .is_empty());

        let error_events =
            run_research_case(ResponseTemplate::new(500).set_body_string("boom")).await;
        let error_progress = step_progress_messages(&error_events);

        assert_eq!(source_found_count(&error_events), 0);
        assert!(error_progress
            .iter()
            .any(|msg| { msg == "search:results=0 error_kinds=search_result_row:1" }));
        assert!(!done_event(&error_events)
            .expect("done event")
            .text
            .is_empty());
    }
}
