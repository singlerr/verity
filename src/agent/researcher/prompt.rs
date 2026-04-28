//! Researcher prompts — mode-specific system prompts and tool definitions.

use crate::llm::provider::ToolDefinition;
use super::{ResearchDepth, ResearcherMessage};

/// Build the system prompt for the researcher loop based on depth mode.
pub fn get_system_prompt(depth: ResearchDepth) -> String {
    let base = "You are an AI research assistant with access to tools for searching the web and reading web pages.

RULES:
- Use the web_search tool to find information. Provide a focused, specific query.
- Use the scrape_url tool to read the full content of a web page.
- Use the __reasoning_preamble tool to think through your approach BEFORE searching.
- Use the done tool when you have gathered enough information to answer comprehensively.
- Every factual claim in your final answer MUST cite its source using [N] notation where N is the result index.
- Do NOT hallucinate URLs. Only cite URLs you actually visited via scrape_url.
- Respond with tool calls to gather information. Call done() when ready to synthesize.
- If a search returns no results, try a different query.
- If a URL fails to load, try another source.";

    let mode_specific = match depth {
        ResearchDepth::Speed => "\n\nMODE: SPEED
- You have at most 2 iterations. Be direct and efficient.
- Go straight to the most relevant search. Skip reasoning_preamble.
- After reading 1-2 sources, call done() immediately.
- Keep your final answer concise (2-3 paragraphs).",
        ResearchDepth::Balanced => "\n\nMODE: BALANCED
- You have at most 6 iterations. Be thorough but focused.
- Start with __reasoning_preamble to plan your approach.
- Search with reformulated queries, not the raw user question.
- Read 3-5 sources to get diverse perspectives.
- Call done() when you have enough evidence for a well-rounded answer.",
        ResearchDepth::Quality => "\n\nMODE: QUALITY (Deep Research)
- You have at most 25 iterations. Be exhaustive.
- ALWAYS start with __reasoning_preamble to break down the question.
- Search multiple angles: definitions, recent developments, expert opinions, counterarguments.
- Read 8+ sources. Prioritize authoritative sources (academic, government, established publications).
- Verify claims across multiple sources.
- Your final answer MUST be at least 2000 words, comprehensive, and well-structured.
- Include background, analysis, and conclusions with full citations.",
    };

    format!("{}\n{}", base, mode_specific)
}

/// Build the user prompt for a specific iteration.
pub fn get_user_prompt(query: &str, iteration: usize, max_iterations: usize) -> String {
    if iteration == 0 {
        format!(
            "Research this question: {}\n\nStart by thinking through your approach, then search for information.",
            query
        )
    } else if iteration >= max_iterations - 1 {
        format!(
            "This is your last iteration. You MUST call done() now with a comprehensive answer based on all information gathered.\n\nOriginal question: {}",
            query
        )
    } else {
        format!(
            "Iteration {}/{}: Continue researching: {}.\nIf you have enough information, call done(). Otherwise, search for more details.",
            iteration + 1,
            max_iterations,
            query
        )
    }
}

/// Get the standard tool definitions for the researcher loop.
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "web_search".into(),
            description: "Search the web for information using a search query. Returns a list of results with titles, URLs, and snippets.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and use reformulated terms, not the raw user question."
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "scrape_url".into(),
            description: "Read and extract the full text content from a web page URL. Use this to get detailed information from search results.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to read."
                    }
                },
                "required": ["url"]
            }),
        },
        ToolDefinition {
            name: "__reasoning_preamble".into(),
            description: "Think through the research question and plan your approach before searching. Use this at the start of research.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "thoughts": {
                        "type": "string",
                        "description": "Your reasoning about what to search for and how to approach the question."
                    }
                },
                "required": ["thoughts"]
            }),
        },
        ToolDefinition {
            name: "done".into(),
            description: "Signal that you have gathered enough information. Provide a comprehensive summary answering the user's question with [N] citations.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Your comprehensive answer with [N] citation markers referencing the sources."
                    }
                },
                "required": ["summary"]
            }),
        },
    ]
}

/// Build initial messages for the researcher loop.
pub fn build_initial_messages(query: &str, depth: ResearchDepth) -> Vec<ResearcherMessage> {
    let system = get_system_prompt(depth);
    let user = get_user_prompt(query, 0, depth.max_iterations());
    vec![
        ResearcherMessage::System { content: system },
        ResearcherMessage::User { content: user },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_count() {
        let tools = get_tool_definitions();
        assert_eq!(tools.len(), 4);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"scrape_url"));
        assert!(names.contains(&"__reasoning_preamble"));
        assert!(names.contains(&"done"));
    }

    #[test]
    fn system_prompt_varies_by_depth() {
        let speed = get_system_prompt(ResearchDepth::Speed);
        let balanced = get_system_prompt(ResearchDepth::Balanced);
        let quality = get_system_prompt(ResearchDepth::Quality);
        assert!(speed.contains("SPEED"));
        assert!(balanced.contains("BALANCED"));
        assert!(quality.contains("QUALITY"));
        assert!(quality.contains("2000 words"));
    }

    #[test]
    fn user_prompt_last_iteration_forces_done() {
        let prompt = get_user_prompt("test", 1, 2);
        assert!(prompt.contains("MUST call done()"));
    }

    #[test]
    fn build_initial_messages_has_system_and_user() {
        let msgs = build_initial_messages("test query", ResearchDepth::Balanced);
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0], ResearcherMessage::System { .. }));
        assert!(matches!(msgs[1], ResearcherMessage::User { .. }));
    }

    #[test]
    fn tool_definitions_have_valid_json_schema() {
        for tool in get_tool_definitions() {
            assert!(tool.parameters.is_object());
            assert!(tool.parameters.get("type").is_some());
        }
    }
}
