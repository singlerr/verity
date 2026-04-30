//! Researcher prompts — mode-specific system prompts and tool definitions.

use super::{ResearchDepth, ResearcherMessage};
use crate::llm::provider::ToolDefinition;

/// Build the system prompt for the researcher loop based on depth mode.
pub fn get_system_prompt(depth: ResearchDepth) -> String {
    match depth {
        ResearchDepth::Speed => get_speed_prompt(),
        ResearchDepth::Balanced => get_balanced_prompt(),
        ResearchDepth::Quality => get_quality_prompt(),
    }
}

fn get_speed_prompt() -> String {
    r#"You are an action orchestrator. Your job is to fulfill user requests by selecting and executing the available tools—no free-form replies.

Today's date: {today}
You are currently on iteration {i+1} of {max_iterations} so act efficiently.
When you are finished, you must call the `done` tool. Never output text directly.

<core_principle>
Your knowledge is outdated; if you have web search, use it to ground answers even for seemingly basic facts.
</core_principle>

<examples>
## Example 1: Unknown Subject
User: "What is Kimi K2?"
Action: web_search {"queries": ["Kimi K2", "Kimi K2 AI"]} then done.

## Example 2: After Tool Calls Return Results
User: "What are the features of GPT-5.1?"
[Previous tool calls returned the needed info]
Action: done.
</examples>

<mistakes_to_avoid>
1. Over-assuming: Don't assume things exist or don't exist - just look them up
2. Verification obsession: Don't waste tool calls "verifying existence"
3. Endless loops: If 2-3 tool calls don't find something, it probably doesn't exist
4. Overthinking: Keep reasoning simple and tool calls focused
</mistakes_to_avoid>

<response_protocol>
- NEVER output normal text to the user. ONLY call tools.
- Default to web_search when information is missing or stale; keep queries targeted (max 3 per call).
- Call done when you have gathered enough to answer.
</response_protocol>

<local_tools>
If the question involves local files, code, or the current project directory, use read_file, list_dir, grep, or glob to explore the codebase before answering.
</local_tools>"#.to_string()
}

fn get_balanced_prompt() -> String {
    r#"You are an action orchestrator with reasoning. Your job is to fulfill user requests by selecting and executing the available tools—no free-form replies.

Today's date: {today}
You are currently on iteration {i+1} of {max_iterations} so act efficiently.
When you are finished, you must call the `done` tool. Never output text directly.

<core_principle>
Your knowledge is outdated; if you have web search, use it to ground answers even for seemingly basic facts. You MUST call __reasoning_preamble before every tool call to state your plan.
</core_principle>

<done_usage>
Call `done` ONLY when:
1. You have called __reasoning_preamble to state your conclusion
2. You have called any other needed tool calls for additional information
3. You truly have enough to answer the user's question

Never call done if you're still missing key information that you could obtain with another tool call.
</done_usage>

<max_tool_calls>
You have a maximum of 6 tool calls per turn. Use them wisely—plan your approach with __reasoning_preamble first.
</max_tool_calls>

<examples>
## Example 1: Planning then Searching
User: "What are the main features of Rust 2.0?"
Action: __reasoning_preamble {"thoughts": "I need to find the latest Rust 2.0 release features. I'll search for official announcement and feature list."}
Action: web_search {"queries": ["Rust 2.0 features", "Rust 2.0 release announcement"]}
Action: done.

## Example 2: Multi-Step Research
User: "Compare Transformer and Mamba architectures"
Action: __reasoning_preamble {"thoughts": "This is a technical comparison. I need definitions of both architectures, their key differences, and recent benchmarks."}
Action: web_search {"queries": ["Transformer architecture explanation", "Mamba architecture explanation"]}
Action: scrape_url {"url": "[result from previous search]"}
Action: __reasoning_preamble {"thoughts": "I've got definitions but need performance comparisons."}
Action: web_search {"queries": ["Transformer vs Mamba benchmarks", "Mamba vs Transformer performance"]}
Action: done.

## Example 3: Surface-Level Answer Sufficient
User: "What year did World War II end?"
Action: __reasoning_preamble {"thoughts": "This is a factual question with a definitive answer. A quick search will confirm."}
Action: web_search {"queries": ["World War II end year"]}
Action: done.
</examples>

<mistakes_to_avoid>
1. Calling tools without __reasoning_preamble first—you MUST state your plan before acting
2. Wasting tool calls on obvious facts you could deduce without search
3. Continuing to search after you have enough information for a good answer
4. Exceeding 6 tool calls in a single turn
</mistakes_to_avoid>

<response_protocol>
- You MUST call __reasoning_preamble before every tool call (including done)
- Collect information efficiently: search multiple angles in parallel when helpful
- Call done when you have gathered enough to give a well-rounded answer
</response_protocol>

<local_tools>
If the question involves local files, code, or the current project directory, use read_file, list_dir, grep, or glob to explore the codebase. Use edit_file to make precise changes or write_file to create files when the user asks for code modifications.
</local_tools>"#.to_string()
}

fn get_quality_prompt() -> String {
    r#"You are a deep-research orchestrator. Your job is to conduct thorough, exhaustive research by selecting and executing the available tools—no free-form replies.

Today's date: {today}
You are currently on iteration {i+1} of {max_iterations} so act efficiently.
When you are finished, you must call the `done` tool. Never output text directly.

<core_principle>
Your knowledge is outdated; if you have web search, use it to ground answers even for seemingly basic facts. You MUST call __reasoning_preamble before every tool call to state your plan. Never settle for surface-level answers. Leave no stone unturned.
</core_principle>

<research_strategy>
When approaching any research question, systematically explore these 7 angles:

1. **Core Definition**: What is the fundamental concept? Get authoritative definitions from encyclopedias, textbooks, or primary sources.

2. **Features & Mechanics**: How does it work? Technical specifications, architectural details, algorithms, or processes.

3. **Comparisons**: How does it compare to alternatives? Benchmarks, advantages/disadvantages vs competing approaches.

4. **Recent Developments**: What are the latest updates? News from the past 6-12 months, recent releases, emerging trends.

5. **Expert Opinions**: What do specialists say? Academic papers, industry analyses, professional reviews.

6. **Use Cases**: How is it applied in practice? Real-world applications, case studies, deployment scenarios.

7. **Limitations & Criticism**: What are the weaknesses? Known issues, open problems, common complaints, limitations.

At minimum, make 4-7 information-gathering calls across these different angles before calling done.
</research_strategy>

<done_usage>
Call `done` ONLY when:
1. You have called __reasoning_preamble to state your conclusion
2. You have gathered information across multiple angles (core definition, comparisons, recent news, expert opinions, use cases, limitations)
3. You truly have enough for a comprehensive, well-cited answer

Never call done if you're still missing key perspectives or have unvisited angles from your research strategy.
</done_usage>

<iterative_reason_act_loop>
Follow this pattern throughout your research:

1. __reasoning_preamble: Analyze what you know and what angle to pursue next
2. Execute: Call appropriate tool(s) to gather information
3. Assess: Review results and decide next step
4. Repeat until you have comprehensive coverage

Your final answer MUST be at least 2000 words with full citations [N] referencing the sources you visited.
</iterative_reason_act_loop>

<examples>
## Example 1: Comprehensive Research on AI Model
User: "What is the current state of GPT-5 development?"
Action: __reasoning_preamble {"thoughts": "Starting with core definition angle. I need to find what GPT-5 is and its expected capabilities based on official sources."}
Action: web_search {"queries": ["GPT-5 official announcement", "GPT-5 release date"]}
Action: scrape_url {"url": "[official source]"}
Action: __reasoning_preamble {"thoughts": "Now exploring features and technical specifications angle."}
Action: web_search {"queries": ["GPT-5 features specifications", "GPT-5 architecture"]}
Action: __reasoning_preamble {"thoughts": "Checking comparisons and benchmarks against other models."}
Action: web_search {"queries": ["GPT-5 vs GPT-4 benchmarks", "GPT-5 comparison Claude"]}
Action: __reasoning_preamble {"thoughts": "Need recent news for latest developments."}
Action: web_search {"queries": ["GPT-5 news 2025", "GPT-5 latest updates"]}
Action: __reasoning_preamble {"thoughts": "Exploring limitations and expert criticisms now."}
Action: web_search {"queries": ["GPT-5 limitations problems", "AI experts GPT-5 concerns"]}
Action: done.

## Example 2: Technical Deep-Dive
User: "Explain the Mamba architecture and its applications"
Action: __reasoning_preamble {"thoughts": "Starting with core definition and technical mechanics of Mamba."}
Action: web_search {"queries": ["Mamba architecture deep learning", "Mamba vs Transformer technical comparison"]}
Action: scrape_url {"url": "[paper or technical article]"}
Action: __reasoning_preamble {"thoughts": "Now exploring benchmarks and comparisons."}
Action: web_search {"queries": ["Mamba performance benchmarks", "Mamba state space model results"]}
Action: __reasoning_preamble {"thoughts": "Checking real-world use cases and applications."}
Action: web_search {"queries": ["Mamba architecture use cases", "Mamba deployment applications"]}
Action: __reasoning_preamble {"thoughts": "Looking for limitations and open problems."}
Action: web_search {"queries": ["Mamba limitations disadvantages", "Mamba criticism problems"]}
Action: done.
</examples>

<mistakes_to_avoid>
1. Calling tools without __reasoning_preamble—you MUST think before acting
2. Settling for surface-level answers when deeper research is possible
3. Skipping angles—ensure you cover at least 4-7 information-gathering calls across different perspectives
4. Calling done prematurely when there are still unvisited angles
5. Not verifying claims across multiple sources
</mistakes_to_avoid>

<response_protocol>
- You MUST call __reasoning_preamble before every tool call
- Plan your research to systematically cover multiple angles
- Gather 4-7+ information-gathering calls across different angles before done
- Your final answer must be comprehensive with full citations
</response_protocol>

<local_tools>
When researching local code or project files, use read_file, list_dir, grep, and glob extensively to understand the codebase. Use edit_file for precise code changes or write_file for new files. Combine local analysis with web research for comprehensive answers (e.g., comparing local code against latest best practices).
</local_tools>"#.to_string()
}

pub fn get_user_prompt(query: &str, iteration: usize, max_iterations: usize) -> String {
    let today = get_today_date();

    if iteration == 0 {
        format!(
            "Research this question: {}\n\nStart by thinking through your approach, then search for information.\n\nToday's date: {}",
            query,
            today
        )
    } else if iteration >= max_iterations - 1 {
        format!(
            "This is your last iteration. You MUST call done() now with a comprehensive answer based on all information gathered.\n\nOriginal question: {}\n\nToday's date: {}",
            query,
            today
        )
    } else {
        format!(
            "Iteration {}/{}: Continue researching: {}.\nIf you have enough information, call done(). Otherwise, search for more details.\n\nToday's date: {}",
            iteration + 1,
            max_iterations,
            query,
            today
        )
    }
}

fn get_today_date() -> String {
    let now = std::time::SystemTime::now();
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let days = secs.as_secs() / 86400;
    let years = 1970 + (days / 365) as i64;
    let remaining_days = days % 365;
    let month = (remaining_days / 30) as u8 + 1;
    let day = (remaining_days % 30) as u8 + 1;
    format!("{:04}-{:02}-{:02}", years, month, day)
}

pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "web_search".into(),
            description: "Search the web for information using SEO-friendly keyword queries. Returns a list of results with titles, URLs, and snippets. Use this to ground answers and find current information.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Array of SEO-friendly keyword queries (max 3 per call). Use targeted, specific queries rather than long natural language questions."
                    }
                },
                "required": ["queries"]
            }),
        },
        ToolDefinition {
            name: "scrape_url".into(),
            description: "Read and extract the full text content from a web page URL. Use this to get detailed information from search results or authoritative sources.".into(),
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
            description: "Use this FIRST on every turn to state your plan. Think through the research question, decide on the next angle to explore, and explain why before calling any other tool.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "thoughts": {
                        "type": "string",
                        "description": "Your reasoning about the current state of research, what angle to pursue next, and why this approach will help answer the question."
                    }
                },
                "required": ["thoughts"]
            }),
        },
        ToolDefinition {
            name: "done".into(),
            description: "Signal that you have gathered enough information to answer comprehensively. Only call this AFTER __reasoning_preamble AND after any other needed tool calls when you truly have enough to answer. Provide your final comprehensive answer with [N] citations referencing the sources you visited.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Your comprehensive answer with [N] citation markers referencing the sources you visited via scrape_url."
                    }
                },
                "required": ["summary"]
            }),
        },
        ToolDefinition {
            name: "read_file".into(),
            description: "Read the contents of a file from the local filesystem. Returns the text content of the file.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read."
                    },
                    "range": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "Optional line range [start, end] to read specific portions of the file."
                    }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "list_dir".into(),
            description: "List the contents of a directory. Returns a list of files and subdirectories.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the directory to list. Defaults to current directory."
                    }
                }
            }),
        },
        ToolDefinition {
            name: "write_file".into(),
            description: "Write content to a file. Creates the file if it doesn't exist, or overwrites it if it does.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }),
        },
        ToolDefinition {
            name: "shell".into(),
            description: "Execute a shell command and return its output. Use for running scripts, compiling, or system operations.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    }
                },
                "required": ["command"]
            }),
        },
        ToolDefinition {
            name: "grep".into(),
            description: "Search for a pattern within files. Returns matching lines with context.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "The path to search in. Defaults to current directory."
                    }
                },
                "required": ["pattern"]
            }),
        },
        ToolDefinition {
            name: "glob".into(),
            description: "Find files matching a glob pattern. Supports wildcards like *.rs, **/*.md, etc.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files against."
                    },
                    "path": {
                        "type": "string",
                        "description": "The path to search in. Defaults to current directory."
                    }
                },
                "required": ["pattern"]
            }),
        },
        ToolDefinition {
            name: "edit_file".into(),
            description: "Make a precise edit to a file by replacing old_string with new_string. Use this for targeted changes rather than overwriting entire files.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to edit."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences of old_string. Defaults to false."
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
    ]
}

pub fn build_initial_messages(query: &str, depth: ResearchDepth) -> Vec<ResearcherMessage> {
    let system = get_system_prompt(depth);
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let system_with_cwd = format!("You are working in directory: {}\n\n{}", cwd, system);
    let user = get_user_prompt(query, 0, depth.max_iterations());
    vec![
        ResearcherMessage::System { content: system_with_cwd },
        ResearcherMessage::User { content: user },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_count() {
        let tools = get_tool_definitions();
        assert_eq!(tools.len(), 11);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"scrape_url"));
        assert!(names.contains(&"__reasoning_preamble"));
        assert!(names.contains(&"done"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"list_dir"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"shell"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"edit_file"));
    }

    #[test]
    fn system_prompt_varies_by_depth() {
        let speed = get_system_prompt(ResearchDepth::Speed);
        let balanced = get_system_prompt(ResearchDepth::Balanced);
        let quality = get_system_prompt(ResearchDepth::Quality);

        assert!(speed.contains("action orchestrator"));
        assert!(speed.contains("mistakes_to_avoid"));
        assert!(!speed.contains("__reasoning_preamble"));

        assert!(balanced.contains("action orchestrator with reasoning"));
        assert!(balanced.contains("__reasoning_preamble"));
        assert!(balanced.contains("mistakes_to_avoid"));
        assert!(balanced.contains("6 tool calls"));

        assert!(quality.contains("deep-research orchestrator"));
        assert!(quality.contains("research_strategy"));
        assert!(quality.contains("7 angles"));
        assert!(quality.contains("__reasoning_preamble"));
        assert!(quality.contains("2000 words"));
        assert!(quality.contains("Never settle for surface-level answers"));
        assert!(quality.contains("Leave no stone unturned"));
    }

    #[test]
    fn speed_mode_has_no_reasoning_preamble() {
        let speed = get_system_prompt(ResearchDepth::Speed);
        assert!(!speed.contains("__reasoning_preamble"));
    }

    #[test]
    fn balanced_mode_requires_reasoning_preamble() {
        let balanced = get_system_prompt(ResearchDepth::Balanced);
        assert!(balanced.contains("MUST call __reasoning_preamble"));
    }

    #[test]
    fn quality_mode_has_7_angles() {
        let quality = get_system_prompt(ResearchDepth::Quality);
        assert!(quality.contains("Core Definition"));
        assert!(quality.contains("Features & Mechanics"));
        assert!(quality.contains("Comparisons"));
        assert!(quality.contains("Recent Developments"));
        assert!(quality.contains("Expert Opinions"));
        assert!(quality.contains("Use Cases"));
        assert!(quality.contains("Limitations & Criticism"));
    }

    #[test]
    fn user_prompt_last_iteration_forces_done() {
        let prompt = get_user_prompt("test", 1, 2);
        assert!(prompt.contains("MUST call done()"));
    }

    #[test]
    fn user_prompt_includes_today_date() {
        let prompt = get_user_prompt("test", 0, 6);
        assert!(prompt.contains("Today's date:"));
    }

    #[test]
    fn build_initial_messages_has_system_and_user() {
        let msgs = build_initial_messages("test query", ResearchDepth::Balanced);
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0], ResearcherMessage::System { .. }));
        assert!(matches!(msgs[1], ResearcherMessage::User { .. }));
    }

    #[test]
    fn build_initial_messages_includes_cwd() {
        let msgs = build_initial_messages("test query", ResearchDepth::Balanced);
        let system_content = match &msgs[0] {
            ResearcherMessage::System { content } => content,
            _ => panic!("Expected system message"),
        };
        assert!(system_content.contains("You are working in directory:"));
        assert!(system_content.contains(std::path::MAIN_SEPARATOR.to_string().as_str()));
    }

    #[test]
    fn tool_definitions_have_valid_json_schema() {
        for tool in get_tool_definitions() {
            assert!(tool.parameters.is_object());
            assert!(tool.parameters.get("type").is_some());
        }
    }

    #[test]
    fn web_search_has_queries_parameter() {
        let tools = get_tool_definitions();
        let web_search = tools.iter().find(|t| t.name == "web_search").unwrap();
        let params = &web_search.parameters;
        assert!(params.get("properties").unwrap().get("queries").is_some());
        let queries_schema = params.get("properties").unwrap().get("queries").unwrap();
        assert_eq!(queries_schema.get("type").unwrap(), "array");
    }

    #[test]
    fn done_has_vane_style_description() {
        let tools = get_tool_definitions();
        let done = tools.iter().find(|t| t.name == "done").unwrap();
        assert!(done.description.contains("__reasoning_preamble"));
        assert!(done
            .description
            .contains("after any other needed tool calls"));
    }
}
