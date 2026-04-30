//! Core iterative LLM tool-calling research loop engine.

use std::collections::HashSet;
use std::sync::mpsc;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use super::{ExtractedFact, ResearchDepth, ResearcherContext, ResearcherMessage};
use crate::agent::orchestrator::AgentEvent;
use crate::agent::researcher::prompt;
use crate::agent::researcher::ContentExtractor;
use crate::agent::tools::ToolRegistry;
use crate::app::Source;
use crate::llm::provider::{FinishReason, LlmProvider, Message, Role, ToolCall, ToolDefinition};

/// Output of a completed research loop.
#[derive(Debug, Clone)]
pub struct ResearcherOutput {
    pub answer: String,
    pub sources: Vec<Source>,
    pub iterations_used: usize,
    pub extracted_facts: Vec<ExtractedFact>,
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
        provider: Arc<dyn LlmProvider>,
        model: String,
        tool_registry: ToolRegistry,
        cancel_token: CancellationToken,
    ) -> Self {
        let tool_defs = prompt::get_tool_definitions();
        Self {
            provider,
            model,
            tool_defs,
            tool_registry,
            cancel_token,
        }
    }

    /// Run the iterative research loop (delegates to mode-specific pipelines).
    pub async fn run(
        &self,
        query: &str,
        depth: ResearchDepth,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<ResearcherOutput, String> {
        match depth {
            ResearchDepth::Speed => self.run_speed_mode(query, tx).await,
            ResearchDepth::Balanced => self.run_balanced_mode(query, tx).await,
            ResearchDepth::Quality => self.run_quality_mode(query, tx).await,
        }
    }

    /// Speed mode: minimal iterations, direct search → scrape top URLs and extract facts.
    async fn run_speed_mode(
        &self,
        query: &str,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<ResearcherOutput, String> {
        let max_iters = 2usize;
        let mut ctx = ResearcherContext::new();
        for msg in prompt::build_initial_messages(query, ResearchDepth::Speed) {
            ctx.push_message(msg);
        }
        let mut sources: Vec<Source> = Vec::new();
        let mut seen_urls: HashSet<String> = HashSet::new();
        let mut answer = String::new();
        let mut extracted_facts: Vec<ExtractedFact> = Vec::new();

        let extractor = ContentExtractor::new(self.provider.clone(), self.model.clone());

        for i in 0..max_iters {
            if self.cancel_token.is_cancelled() {
                return Err("Cancelled".into());
            }
            if ctx.is_over_budget() {
                ctx.truncate_oldest();
            }
            let api_msgs = to_api_messages(&ctx);
            let response = self
                .provider
                .complete_with_tools(&api_msgs, &self.tool_defs, &self.model)
                .await
                .map_err(|e| format!("LLM call failed: {:?}", e))?;
            match response.finish_reason {
                FinishReason::Stop | FinishReason::Length => {
                    answer = response.content.unwrap_or_default();
                    break;
                }
                FinishReason::ToolCalls => {
                    ctx.push_message(ResearcherMessage::AssistantWithToolCalls {
                        content: response.content.clone(),
                        tool_calls: response.tool_calls.clone(),
                    });
                    for tc in &response.tool_calls {
                        // reuse existing behavior to populate sources for web_search results
                        let _ = self
                            .execute_tool_call(tc, &mut sources, &mut seen_urls, i, max_iters, tx)
                            .await;
                    }
                }
            }

            // After tool-calls, scrape top N results directly
            let top_n = std::cmp::min(3, sources.len());
            if top_n > 0 {
                if let Some(read_url) = self.tool_registry.get("read_url") {
                    for idx in 0..top_n {
                        let url = sources[idx].url.clone();
                        let val = read_url
                            .execute(&serde_json::json!({"url": url}))
                            .await
                            .unwrap_or_default();
                        let content = val
                            .get("content")
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string();
                        // Run extraction on content
                        let facts = extractor.extract_facts(&content, query).await;
                        extracted_facts.extend(facts);
                    }
                }
            }

            // Iterate complete; report progress (best-effort)
            if i + 1 >= max_iters {
                break;
            }
        }

        if answer.is_empty() {
            answer = "Research completed without a final summary.".into();
        }
        Ok(ResearcherOutput {
            answer,
            sources,
            iterations_used: max_iters,
            extracted_facts,
        })
    }

    /// Balanced mode: more iterations with explicit reasoning on each tool call and a heuristic top-3 result pick.
    async fn run_balanced_mode(
        &self,
        query: &str,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<ResearcherOutput, String> {
        let max_iters = 6usize;
        let mut ctx = ResearcherContext::new();
        for msg in prompt::build_initial_messages(query, ResearchDepth::Balanced) {
            ctx.push_message(msg);
        }
        let mut sources: Vec<Source> = Vec::new();
        let mut seen_urls: HashSet<String> = HashSet::new();
        let mut answer = String::new();
        let mut extracted_facts: Vec<ExtractedFact> = Vec::new();

        let extractor = ContentExtractor::new(self.provider.clone(), self.model.clone());

        for i in 0..max_iters {
            if self.cancel_token.is_cancelled() {
                return Err("Cancelled".into());
            }
            if ctx.is_over_budget() {
                ctx.truncate_oldest_with_priority();
            }
            let api_msgs = to_api_messages(&ctx);
            let response = self
                .provider
                .complete_with_tools(&api_msgs, &self.tool_defs, &self.model)
                .await
                .map_err(|e| format!("LLM call failed: {:?}", e))?;
            match response.finish_reason {
                FinishReason::Stop | FinishReason::Length => {
                    let min_iter = Self::min_iterations_for_depth(ResearchDepth::Balanced);
                    if i + 1 < min_iter {
                        ctx.push_message(ResearcherMessage::System {
                            content: format!("You must continue researching. This is iteration {}/{}, minimum {} iterations required.", i + 1, max_iters, min_iter),
                        });
                        continue;
                    }
                    answer = response.content.unwrap_or_default();
                    break;
                }
                FinishReason::ToolCalls => {
                    ctx.push_message(ResearcherMessage::AssistantWithToolCalls {
                        content: response.content.clone(),
                        tool_calls: response.tool_calls.clone(),
                    });
                    let has_reasoning = Self::has_reasoning_preamble(&response.tool_calls);
                    for tc in &response.tool_calls {
                        if tc.name != "__reasoning_preamble" && !has_reasoning {
                            ctx.push_message(ResearcherMessage::System {
                                content: "You MUST call __reasoning_preamble before any tool call. Your tool call was skipped.".into()
                            });
                            continue;
                        }
                        let _ = self
                            .execute_tool_call(tc, &mut sources, &mut seen_urls, i, max_iters, tx)
                            .await;
                    }
                    // After tool calls, pick top 3 (heuristic) and scrape them
                    let top_results = Self::select_top_results(&sources);
                    for s in &top_results {
                        if let Some(read) = self.tool_registry.get("read_url") {
                            if let Ok(val) = read.execute(&serde_json::json!({"url": &s.url})).await
                            {
                                if let Some(content) = val.get("content").and_then(|c| c.as_str()) {
                                    let facts = extractor.extract_facts(content, query).await;
                                    extracted_facts.extend(facts);
                                }
                            }
                        }
                    }
                }
            }
            // Progress indicator
            let _ = tx.send(AgentEvent::StepProgress(
                i + 1,
                format!("Iteration {} complete", i + 1),
            ));
        }
        if answer.is_empty() {
            answer = "Research completed without a final summary.".into();
        }
        Ok(ResearcherOutput {
            answer,
            sources,
            iterations_used: max_iters,
            extracted_facts,
        })
    }

    /// Quality mode: iterative, angle-diverse search with LLM-based picker and repeat scraping/extraction.
    async fn run_quality_mode(
        &self,
        query: &str,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<ResearcherOutput, String> {
        let max_iters = 25usize;
        let mut ctx = ResearcherContext::new();
        for msg in prompt::build_initial_messages(query, ResearchDepth::Quality) {
            ctx.push_message(msg);
        }
        let mut sources: Vec<Source> = Vec::new();
        let mut seen_urls: HashSet<String> = HashSet::new();
        let mut answer = String::new();
        let mut extracted_facts: Vec<ExtractedFact> = Vec::new();

        let extractor = ContentExtractor::new(self.provider.clone(), self.model.clone());

        for i in 0..max_iters {
            if self.cancel_token.is_cancelled() {
                return Err("Cancelled".into());
            }
            if ctx.is_over_budget() {
                ctx.truncate_oldest_with_priority();
            }
            let api_msgs = to_api_messages(&ctx);
            let response = self
                .provider
                .complete_with_tools(&api_msgs, &self.tool_defs, &self.model)
                .await
                .map_err(|e| format!("LLM call failed: {:?}", e))?;
            match response.finish_reason {
                FinishReason::Stop | FinishReason::Length => {
                    let min_iter = Self::min_iterations_for_depth(ResearchDepth::Quality);
                    if i + 1 < min_iter {
                        ctx.push_message(ResearcherMessage::System {
                            content: format!("You must continue researching. This is iteration {}/{}, minimum {} iterations required.", i + 1, max_iters, min_iter),
                        });
                        continue;
                    }
                    answer = response.content.unwrap_or_default();
                    break;
                }
                FinishReason::ToolCalls => {
                    ctx.push_message(ResearcherMessage::AssistantWithToolCalls {
                        content: response.content.clone(),
                        tool_calls: response.tool_calls.clone(),
                    });
                    let has_reasoning = Self::has_reasoning_preamble(&response.tool_calls);
                    for tc in &response.tool_calls {
                        if tc.name != "__reasoning_preamble" && !has_reasoning {
                            ctx.push_message(ResearcherMessage::System {
                                content: "You MUST call __reasoning_preamble before any tool call. Your tool call was skipped.".into()
                            });
                            continue;
                        }
                        let _ = self
                            .execute_tool_call(tc, &mut sources, &mut seen_urls, i, max_iters, tx)
                            .await;
                    }
                    // After tool calls, pick best URLs (placeholder heuristic if picker not wired)
                    let picked = Self::select_top_results(&sources);
                    for s in &picked {
                        if let Some(read) = self.tool_registry.get("read_url") {
                            if let Ok(val) = read.execute(&serde_json::json!({"url": &s.url})).await
                            {
                                if let Some(content) = val.get("content").and_then(|c| c.as_str()) {
                                    let facts = extractor.extract_facts(content, query).await;
                                    extracted_facts.extend(facts);
                                }
                            }
                        }
                    }
                }
            }
            // Emit progress
            let _ = tx.send(AgentEvent::StepProgress(i + 1, String::from("progress")));
        }
        if answer.is_empty() {
            answer = "Research completed without a final summary.".into();
        }
        Ok(ResearcherOutput {
            answer,
            sources,
            iterations_used: max_iters,
            extracted_facts,
        })
    }

    /// Simple helper to pick top 3 results from the current sources by their order.
    fn select_top_results(sources: &[Source]) -> Vec<Source> {
        sources.iter().cloned().take(3).collect()
    }

    /// Check if any tool call in the list is a reasoning preamble.
    fn has_reasoning_preamble(tool_calls: &[ToolCall]) -> bool {
        tool_calls
            .iter()
            .any(|tc| tc.name == "__reasoning_preamble")
    }

    fn min_iterations_for_depth(depth: ResearchDepth) -> usize {
        match depth {
            ResearchDepth::Speed => 1,
            ResearchDepth::Balanced => 2,
            ResearchDepth::Quality => 3,
        }
    }

    async fn execute_tool_call(
        &self,
        tc: &ToolCall,
        sources: &mut Vec<Source>,
        seen_urls: &mut HashSet<String>,
        iteration: usize,
        max_iters: usize,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> String {
        let args: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or_default();
        match tc.name.as_str() {
            "web_search" => {
                let query = args
                    .get("query")
                    .and_then(|q| q.as_str())
                    .unwrap_or("")
                    .to_string();
                let _ = tx.send(AgentEvent::SearchingIteration {
                    current: (iteration + 1) as u8,
                    max: max_iters as u8,
                    query: query.clone(),
                });
                match self.tool_registry.get("search") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            if let Some(results) = val.get("results").and_then(|r| r.as_array()) {
                                for item in results.iter() {
                                    let url = item
                                        .get("url")
                                        .and_then(|u| u.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    if !url.is_empty() && seen_urls.insert(url.clone()) {
                                        sources.push(Source {
                                            num: sources.len() + 1,
                                            domain: extract_domain(&url),
                                            title: item
                                                .get("title")
                                                .and_then(|t| t.as_str())
                                                .unwrap_or("")
                                                .into(),
                                            url,
                                            snippet: item
                                                .get("snippet")
                                                .and_then(|s| s.as_str())
                                                .unwrap_or("")
                                                .into(),
                                            quote: String::new(),
                                        });
                                        if let Some(s) = sources.last().cloned() {
                                            let _ = tx.send(AgentEvent::SourceFound(s));
                                        }
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
            "__reasoning_preamble" => {
                let thoughts = args
                    .get("thoughts")
                    .and_then(|t| t.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| "Reasoning acknowledged".into());
                let _ = tx.send(AgentEvent::StepProgress(iteration + 1, thoughts.clone()));
                thoughts
            }
            "done" => args
                .get("summary")
                .and_then(|s| s.as_str())
                .map(String::from)
                .unwrap_or_default(),
            "read_file" => {
                let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let _ = tx.send(AgentEvent::StepProgress(
                    iteration + 1,
                    format!("Reading file: {}", path),
                ));
                match self.tool_registry.get("read_file") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            let content = val.get("content").and_then(|c| c.as_str()).unwrap_or("");
                            content.to_string()
                        }
                        Err(e) => format!("Read file error: {}", e),
                    },
                    None => "Read file tool not available".into(),
                }
            }
            "list_dir" => {
                let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let _ = tx.send(AgentEvent::StepProgress(
                    iteration + 1,
                    format!("Listing directory: {}", path),
                ));
                match self.tool_registry.get("list_dir") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            let tree = val.get("tree").and_then(|t| t.as_str()).unwrap_or("");
                            tree.to_string()
                        }
                        Err(e) => format!("List dir error: {}", e),
                    },
                    None => "List dir tool not available".into(),
                }
            }
            "write_file" => {
                let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let _ = tx.send(AgentEvent::StepProgress(
                    iteration + 1,
                    format!("Writing file: {}", path),
                ));
                match self.tool_registry.get("write_file") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            let success = val.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
                            let result_path = val.get("path").and_then(|p| p.as_str()).unwrap_or("");
                            if success {
                                format!("Successfully wrote file: {}", result_path)
                            } else {
                                format!("Failed to write file: {}", result_path)
                            }
                        }
                        Err(e) => format!("Write file error: {}", e),
                    },
                    None => "Write file tool not available".into(),
                }
            }
            "shell" => {
                let command = args.get("command").and_then(|c| c.as_str()).unwrap_or("");
                let _ = tx.send(AgentEvent::StepProgress(
                    iteration + 1,
                    format!("Running shell command: {}", command),
                ));
                match self.tool_registry.get("shell") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            let stdout = val.get("stdout").and_then(|s| s.as_str()).unwrap_or("");
                            let stderr = val.get("stderr").and_then(|s| s.as_str()).unwrap_or("");
                            let _exit_code = val.get("exit_code").and_then(|c| c.as_i64()).unwrap_or(0);
                            if stderr.is_empty() {
                                format!("Output:\n{}", stdout)
                            } else {
                                format!("Output:\n{}\nStderr:\n{}", stdout, stderr)
                            }
                        }
                        Err(e) => format!("Shell error: {}", e),
                    },
                    None => "Shell tool not available".into(),
                }
            }
            "grep" => {
                let pattern = args.get("pattern").and_then(|p| p.as_str()).unwrap_or("");
                let _ = tx.send(AgentEvent::StepProgress(
                    iteration + 1,
                    format!("Searching for: {}", pattern),
                ));
                match self.tool_registry.get("grep") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            let matches = val.get("matches").and_then(|m| m.as_str()).unwrap_or("");
                            matches.to_string()
                        }
                        Err(e) => format!("Grep error: {}", e),
                    },
                    None => "Grep tool not available".into(),
                }
            }
            "glob" => {
                let pattern = args.get("pattern").and_then(|p| p.as_str()).unwrap_or("");
                let _ = tx.send(AgentEvent::StepProgress(
                    iteration + 1,
                    format!("Finding files: {}", pattern),
                ));
                match self.tool_registry.get("glob") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            let files = val.get("files").and_then(|f| f.as_array())
                                .map(|arr| arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join("\n"))
                                .unwrap_or_default();
                            files
                        }
                        Err(e) => format!("Glob error: {}", e),
                    },
                    None => "Glob tool not available".into(),
                }
            }
            "edit_file" => {
                let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let _ = tx.send(AgentEvent::StepProgress(
                    iteration + 1,
                    format!("Editing file: {}", path),
                ));
                match self.tool_registry.get("edit_file") {
                    Some(tool) => match tool.execute(&args).await {
                        Ok(val) => {
                            let success = val.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
                            let replacements = val.get("replacements").and_then(|r| r.as_i64()).unwrap_or(0);
                            if success {
                                format!("Successfully made {} replacement(s)", replacements)
                            } else {
                                "Edit failed".to_string()
                            }
                        }
                        Err(e) => format!("Edit file error: {}", e),
                    },
                    None => "Edit file tool not available".into(),
                }
            }
            _ => format!("Unknown tool: {}", tc.name),
        }
    }
}

/// Convert ResearcherContext messages to provider API Messages.
fn to_api_messages(ctx: &ResearcherContext) -> Vec<Message> {
    ctx.messages
        .iter()
        .map(|msg| match msg {
            ResearcherMessage::System { content } => Message {
                role: Role::System,
                content: content.clone(),
            },
            ResearcherMessage::User { content } => Message {
                role: Role::User,
                content: content.clone(),
            },
            ResearcherMessage::Assistant { content } => Message {
                role: Role::Assistant,
                content: content.clone(),
            },
            ResearcherMessage::AssistantWithToolCalls {
                content,
                tool_calls,
            } => {
                let tc_json = serde_json::to_string(tool_calls).unwrap_or_default();
                let body = content
                    .as_ref()
                    .map(|c| format!("{}\n\nTool calls: {}", c, tc_json))
                    .unwrap_or_else(|| format!("Tool calls: {}", tc_json));
                Message {
                    role: Role::Assistant,
                    content: body,
                }
            }
            ResearcherMessage::ToolResult {
                call_id,
                name,
                output,
            } => Message {
                role: Role::User,
                content: format!("Tool {} ({}): {}", name, call_id, output),
            },
        })
        .collect()
}

fn extract_domain(url: &str) -> String {
    url.strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or(url)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider::mock::MockLlmProvider;
    use crate::llm::provider::ToolResponse;

    fn make_tool_call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: format!("tc_{}", name),
            name: name.into(),
            arguments: args.into(),
        }
    }

    fn make_stop_response(content: &str) -> ToolResponse {
        ToolResponse {
            content: Some(content.into()),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: None,
        }
    }

    fn make_tool_response(tool_calls: Vec<ToolCall>) -> ToolResponse {
        ToolResponse {
            content: None,
            tool_calls,
            finish_reason: FinishReason::ToolCalls,
            usage: None,
        }
    }

    fn make_reasoning_preamble(thoughts: &str) -> ToolCall {
        make_tool_call(
            "__reasoning_preamble",
            &format!("{{\"thoughts\": \"{}\"}}", thoughts),
        )
    }

    #[test]
    fn researcher_loop_new_compiles() {
        let cancel = CancellationToken::new();
        let registry = ToolRegistry::new();
        let provider: Arc<dyn LlmProvider> = Arc::new(crate::llm::openai::OpenAiProvider::new());
        let _rl = ResearcherLoop::new(provider, "gpt-4".into(), registry, cancel);
    }

    #[test]
    fn researcher_output_construction() {
        let out = ResearcherOutput {
            answer: "test".into(),
            sources: vec![],
            iterations_used: 3,
            extracted_facts: vec![],
        };
        assert_eq!(out.answer, "test");
        assert!(out.sources.is_empty());
        assert_eq!(out.iterations_used, 3);
        assert!(out.extracted_facts.is_empty());
    }

    #[test]
    fn to_api_messages_conversion() {
        let mut ctx = ResearcherContext::new();
        ctx.push_message(ResearcherMessage::System {
            content: "sys".into(),
        });
        ctx.push_message(ResearcherMessage::User {
            content: "hi".into(),
        });
        ctx.push_message(ResearcherMessage::Assistant {
            content: "hello".into(),
        });
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

    #[test]
    fn min_iterations_for_depth_values() {
        assert_eq!(
            ResearcherLoop::min_iterations_for_depth(ResearchDepth::Speed),
            1
        );
        assert_eq!(
            ResearcherLoop::min_iterations_for_depth(ResearchDepth::Balanced),
            2
        );
        assert_eq!(
            ResearcherLoop::min_iterations_for_depth(ResearchDepth::Quality),
            3
        );
    }

    #[test]
    fn has_reasoning_preamble_detection() {
        let with_preamble = vec![
            make_reasoning_preamble("thinking"),
            make_tool_call("web_search", r#"{"query": "test"}"#),
        ];
        assert!(ResearcherLoop::has_reasoning_preamble(&with_preamble));

        let without_preamble = vec![make_tool_call("web_search", r#"{"query": "test"}"#)];
        assert!(!ResearcherLoop::has_reasoning_preamble(&without_preamble));
    }

    #[test]
    fn speed_mode_no_reasoning_check() {
        let cancel = CancellationToken::new();
        let registry = ToolRegistry::new();

        let responses = vec![make_stop_response("quick answer")];
        let mock = MockLlmProvider::new(responses);

        let provider: Arc<dyn LlmProvider> = Arc::new(mock);
        let _loop_ref = ResearcherLoop::new(provider, "gpt-4".into(), registry, cancel);

        assert_eq!(
            ResearcherLoop::min_iterations_for_depth(ResearchDepth::Speed),
            1
        );

        let tool_calls = vec![make_tool_call("web_search", r#"{"query": "test"}"#)];
        assert!(!ResearcherLoop::has_reasoning_preamble(&tool_calls));
    }

    #[test]
    fn reasoning_enforcement_logic() {
        let tool_calls_no_preamble = vec![make_tool_call("web_search", r#"{"query": "test"}"#)];
        let has_reasoning = ResearcherLoop::has_reasoning_preamble(&tool_calls_no_preamble);
        assert!(!has_reasoning);
    }

    #[test]
    fn select_top_results_basic() {
        let sources = vec![
            Source {
                num: 1,
                domain: "a.com".into(),
                title: "A".into(),
                url: "https://a.com".into(),
                snippet: "snippet a".into(),
                quote: "".into(),
            },
            Source {
                num: 2,
                domain: "b.com".into(),
                title: "B".into(),
                url: "https://b.com".into(),
                snippet: "snippet b".into(),
                quote: "".into(),
            },
            Source {
                num: 3,
                domain: "c.com".into(),
                title: "C".into(),
                url: "https://c.com".into(),
                snippet: "snippet c".into(),
                quote: "".into(),
            },
            Source {
                num: 4,
                domain: "d.com".into(),
                title: "D".into(),
                url: "https://d.com".into(),
                snippet: "snippet d".into(),
                quote: "".into(),
            },
        ];
        let top = ResearcherLoop::select_top_results(&sources);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].url, "https://a.com");
        assert_eq!(top[1].url, "https://b.com");
        assert_eq!(top[2].url, "https://c.com");
    }

    #[test]
    fn mock_provider_call_counting() {
        let responses = vec![make_stop_response("first"), make_stop_response("second")];
        let mock = MockLlmProvider::new(responses);

        assert_eq!(*mock.call_count.lock().unwrap(), 0);
    }

    #[test]
    fn context_truncation_preserves_reasoning_preamble() {
        let mut ctx = ResearcherContext::with_budget(200);
        ctx.push_message(ResearcherMessage::System {
            content: "system prompt".into(),
        });

        for i in 0..5 {
            ctx.push_message(ResearcherMessage::ToolResult {
                call_id: format!("rp{}", i),
                name: "__reasoning_preamble".into(),
                output: format!("reasoning thought {}", i),
            });
        }

        for i in 0..10 {
            ctx.push_message(ResearcherMessage::ToolResult {
                call_id: format!("tr{}", i),
                name: "web_search".into(),
                output: "y".repeat(100),
            });
        }

        for i in 0..5 {
            ctx.push_message(ResearcherMessage::User {
                content: "x".repeat(100),
            });
        }

        assert!(ctx.is_over_budget());
        ctx.truncate_oldest_with_priority();

        assert!(matches!(
            ctx.messages.first(),
            Some(ResearcherMessage::System { .. })
        ));

        let reasoning_count = ctx.messages.iter().filter(|m| {
            matches!(m, ResearcherMessage::ToolResult { name, .. } if name == "__reasoning_preamble")
        }).count();
        assert!(
            reasoning_count > 0,
            "Reasoning preambles should be preserved during truncation"
        );
    }
}
