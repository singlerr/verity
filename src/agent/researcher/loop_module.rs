//! Core iterative LLM tool-calling research loop engine.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc;

use tokio_util::sync::CancellationToken;

use crate::app::Source;
use crate::agent::orchestrator::AgentEvent;
use crate::agent::tools::ToolRegistry;
use crate::llm::provider::{FinishReason, LlmProvider, Message, Role, ToolCall, ToolDefinition};
use super::{ResearchDepth, ResearcherContext, ResearcherMessage};
use super::prompt;

/// Output of a completed research loop.
#[derive(Debug, Clone)]
pub struct ResearcherOutput {
    pub answer: String,
    pub sources: Vec<Source>,
    pub iterations_used: usize,
}

/// Iterative LLM tool-calling research loop.
/// The LLM decides which tools to call each iteration; results feed back
/// until the LLM calls `done()` or max iterations are reached.
pub struct ResearcherLoop {
    provider: Arc<dyn LlmProvider>,
    model: String,
    tool_defs: Vec<ToolDefinition>,
    tool_registry: ToolRegistry,
    cancel_token: CancellationToken,
}

impl ResearcherLoop {
    pub fn new(
        provider: Arc<dyn LlmProvider>, model: String,
        tool_registry: ToolRegistry, cancel_token: CancellationToken,
    ) -> Self {
        let tool_defs = prompt::get_tool_definitions();
        Self { provider, model, tool_defs, tool_registry, cancel_token }
    }

    /// Run the iterative research loop.
    pub async fn run(
        &self, query: &str, depth: ResearchDepth, tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<ResearcherOutput, String> {
        let max_iters = depth.max_iterations();
        let mut ctx = ResearcherContext::new();
        for msg in prompt::build_initial_messages(query, depth) { ctx.push_message(msg); }
        let mut sources: Vec<Source> = Vec::new();
        let mut seen_urls: HashSet<String> = HashSet::new();
        let mut answer = String::new();
        let mut iterations_used = 0;

        for i in 0..max_iters {
            if self.cancel_token.is_cancelled() { return Err("Cancelled".into()); }
            if ctx.is_over_budget() { ctx.truncate_oldest(); }
            let api_msgs = to_api_messages(&ctx);
            let response = self.provider
                .complete_with_tools(&api_msgs, &self.tool_defs, &self.model)
                .await.map_err(|e| format!("LLM call failed: {:?}", e))?;
            match response.finish_reason {
                FinishReason::Stop | FinishReason::Length => {
                    answer = response.content.unwrap_or_default();
                    break;
                }
                FinishReason::ToolCalls => {
                    ctx.push_message(ResearcherMessage::AssistantWithToolCalls {
                        content: response.content.clone(), tool_calls: response.tool_calls.clone(),
                    });
                    for tc in &response.tool_calls {
                        let result = self.execute_tool_call(
                            tc, &mut sources, &mut seen_urls, i, max_iters, tx,
                        ).await;
                        ctx.push_message(ResearcherMessage::ToolResult {
                            call_id: tc.id.clone(), name: tc.name.clone(), output: result,
                        });
                    }
                }
            }
            iterations_used = i + 1;
        }
        if answer.is_empty() { answer = "Research completed without a final summary.".into(); }
        Ok(ResearcherOutput { answer, sources, iterations_used })
    }

    async fn execute_tool_call(
        &self, tc: &ToolCall, sources: &mut Vec<Source>, seen_urls: &mut HashSet<String>,
        iteration: usize, max_iters: usize, tx: &mpsc::Sender<AgentEvent>,
    ) -> String {
        let args: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or_default();
        match tc.name.as_str() {
            "web_search" => {
                let query = args.get("query").and_then(|q| q.as_str()).unwrap_or("").to_string();
                let _ = tx.send(AgentEvent::SearchingIteration {
                    current: (iteration + 1) as u8, max: max_iters as u8, query: query.clone(),
                });
                match self.tool_registry.get("search") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            if let Some(results) = val.get("results").and_then(|r| r.as_array()) {
                                for item in results.iter() {
                                    let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string();
                                    if !url.is_empty() && seen_urls.insert(url.clone()) {
                                        sources.push(Source {
                                            num: sources.len() + 1, domain: extract_domain(&url),
                                            title: item.get("title").and_then(|t| t.as_str()).unwrap_or("").into(),
                                            url, snippet: item.get("snippet").and_then(|s| s.as_str()).unwrap_or("").into(),
                                            quote: String::new(),
                                        });
                                        if let Some(s) = sources.last().cloned() { let _ = tx.send(AgentEvent::SourceFound(s)); }
                                    }
                                }
                            }
                            val.to_string()
                        }
                        Err(e) => format!("Search error: {}", e),
                    },
                    None => "Search tool not available".into(),
                }
            }
            "scrape_url" => match self.tool_registry.get("read_url") {
                Some(tool) => match tool.execute(&args).await {
                    Ok(val) => val.to_string(),
                    Err(e) => format!("Scrape error: {}", e),
                },
                None => "Read URL tool not available".into(),
            },
            "__reasoning_preamble" => args.get("thoughts")
                .and_then(|t| t.as_str()).map(String::from)
                .unwrap_or_else(|| "Reasoning acknowledged".into()),
            "done" => args.get("summary")
                .and_then(|s| s.as_str()).map(String::from)
                .unwrap_or_default(),
            _ => format!("Unknown tool: {}", tc.name),
        }
    }
}

/// Convert ResearcherContext messages to provider API Messages.
fn to_api_messages(ctx: &ResearcherContext) -> Vec<Message> {
    ctx.messages.iter().map(|msg| match msg {
        ResearcherMessage::System { content } => Message { role: Role::System, content: content.clone() },
        ResearcherMessage::User { content } => Message { role: Role::User, content: content.clone() },
        ResearcherMessage::Assistant { content } => Message { role: Role::Assistant, content: content.clone() },
        ResearcherMessage::AssistantWithToolCalls { content, tool_calls } => {
            let tc_json = serde_json::to_string(tool_calls).unwrap_or_default();
            let body = content.as_ref()
                .map(|c| format!("{}\n\nTool calls: {}", c, tc_json))
                .unwrap_or_else(|| format!("Tool calls: {}", tc_json));
            Message { role: Role::Assistant, content: body }
        }
        ResearcherMessage::ToolResult { call_id, name, output } => {
            Message { role: Role::User, content: format!("Tool {} ({}): {}", name, call_id, output) }
        }
    }).collect()
}

fn extract_domain(url: &str) -> String {
    url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url).split('/').next().unwrap_or(url).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn researcher_loop_new_compiles() {
        let cancel = CancellationToken::new();
        let registry = ToolRegistry::new();
        let provider: Arc<dyn LlmProvider> = Arc::new(crate::llm::openai::OpenAiProvider::new());
        let _rl = ResearcherLoop::new(provider, "gpt-4".into(), registry, cancel);
    }
    #[test]
    fn researcher_output_construction() {
        let out = ResearcherOutput { answer: "test".into(), sources: vec![], iterations_used: 3 };
        assert_eq!(out.answer, "test");
        assert!(out.sources.is_empty());
        assert_eq!(out.iterations_used, 3);
    }
    #[test]
    fn to_api_messages_conversion() {
        let mut ctx = ResearcherContext::new();
        ctx.push_message(ResearcherMessage::System { content: "sys".into() });
        ctx.push_message(ResearcherMessage::User { content: "hi".into() });
        ctx.push_message(ResearcherMessage::Assistant { content: "hello".into() });
        let msgs = to_api_messages(&ctx);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert_eq!(msgs[2].role, Role::Assistant);
    }
    #[test]
    fn extract_domain_basic() {
        assert_eq!(extract_domain("https://example.com/path"), "example.com");
        assert_eq!(extract_domain("http://test.org/"), "test.org");
    }
}