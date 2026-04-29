//! ResearcherContext — context window management for the researcher loop.

use super::{ResearchDepth, ResearcherMessage};

const DEFAULT_MAX_BUDGET: usize = 100_000; // ~100k tokens

/// Context window for the researcher loop.
/// Tracks messages and enforces a token budget via truncation.
#[derive(Debug, Clone)]
pub struct ResearcherContext {
    /// Ordered message history.
    pub messages: Vec<ResearcherMessage>,
    max_budget: usize,
}

impl ResearcherContext {
    /// Create a new context with the default budget (~100k tokens).
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_budget: DEFAULT_MAX_BUDGET,
        }
    }

    /// Create a new context with a custom max budget (in tokens).
    pub fn with_budget(max_budget: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_budget,
        }
    }

    /// Push a message onto the context.
    pub fn push_message(&mut self, msg: ResearcherMessage) {
        self.messages.push(msg);
    }

    /// Estimate token count using a rough character-length heuristic.
    /// Divides total content length by 4.
    pub fn token_estimate(&self) -> usize {
        let mut total = 0usize;
        for msg in &self.messages {
            total += match msg {
                ResearcherMessage::System { content } => content.len(),
                ResearcherMessage::User { content } => content.len(),
                ResearcherMessage::Assistant { content } => content.len(),
                ResearcherMessage::AssistantWithToolCalls {
                    content,
                    tool_calls,
                } => {
                    let mut n = content.as_ref().map(|c| c.len()).unwrap_or(0);
                    for tc in tool_calls {
                        n += tc.arguments.len() + tc.name.len();
                    }
                    n
                }
                ResearcherMessage::ToolResult { output, name, .. } => output.len() + name.len(),
            };
        }
        total / 4
    }

    /// Returns true if the estimated token count exceeds the budget.
    pub fn is_over_budget(&self) -> bool {
        self.token_estimate() > self.max_budget
    }

    /// Truncate oldest tool results (and then user/assistant pairs if needed)
    /// to bring the context back within budget.
    /// NEVER removes the first message (system prompt) at index 0.
    pub fn truncate_oldest(&mut self) {
        self.truncate_oldest_with_priority();
    }

    /// Truncate with priority: preserves reasoning preambles.
    /// Phase 1: remove oldest ToolResult messages (except __reasoning_preamble)
    /// Phase 2: remove oldest User/Assistant pairs if still over budget
    pub fn truncate_oldest_with_priority(&mut self) {
        if self.messages.len() <= 1 {
            return;
        }

        // Phase 1: remove oldest tool results first, but skip reasoning preambles
        let mut i = 1;
        while i < self.messages.len() && self.is_over_budget() {
            if let ResearcherMessage::ToolResult { name, .. } = &self.messages[i] {
                if name != "__reasoning_preamble" {
                    self.messages.remove(i);
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        // Phase 2: remove oldest user/assistant pairs if still over budget
        let mut i = 1;
        while i < self.messages.len() && self.is_over_budget() {
            match &self.messages[i] {
                ResearcherMessage::User { .. } | ResearcherMessage::Assistant { .. } => {
                    self.messages.remove(i);
                }
                _ => {
                    i += 1;
                }
            }
        }
    }

    /// Returns the maximum number of research iterations for a given depth.
    pub fn max_iterations_for_depth(depth: ResearchDepth) -> usize {
        depth.max_iterations()
    }
}

impl Default for ResearcherContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::researcher::ResearcherMessage;

    #[test]
    fn context_truncation_preserves_system() {
        let mut ctx = ResearcherContext::with_budget(200); // very small budget
        ctx.push_message(ResearcherMessage::System {
            content: "You are helpful.".into(),
        });
        for i in 0..20 {
            ctx.push_message(ResearcherMessage::User {
                content: "x".repeat(100),
            });
            ctx.push_message(ResearcherMessage::ToolResult {
                call_id: format!("c{}", i),
                name: "search".into(),
                output: "y".repeat(100),
            });
        }
        assert!(ctx.is_over_budget());
        ctx.truncate_oldest();
        // System message preserved
        assert!(matches!(
            ctx.messages.first(),
            Some(ResearcherMessage::System { .. })
        ));
        // Budget respected (or at least reduced)
        assert!(!ctx.is_over_budget() || ctx.messages.len() < 41);
    }

    #[test]
    fn researcher_message_variants() {
        let _sys = ResearcherMessage::System {
            content: "sys".into(),
        };
        let _user = ResearcherMessage::User {
            content: "hi".into(),
        };
        let _asst = ResearcherMessage::Assistant {
            content: "hello".into(),
        };
        let _asst_tc = ResearcherMessage::AssistantWithToolCalls {
            content: Some("thinking".into()),
            tool_calls: vec![crate::llm::provider::ToolCall {
                id: "tc1".into(),
                name: "web_search".into(),
                arguments: "{}".into(),
            }],
        };
        let _result = ResearcherMessage::ToolResult {
            call_id: "tc1".into(),
            name: "web_search".into(),
            output: "results".into(),
        };
    }

    #[test]
    fn token_estimate_simple() {
        let mut ctx = ResearcherContext::with_budget(1000);
        ctx.push_message(ResearcherMessage::User {
            content: "hello".into(), // 5 chars
        });
        // 5 chars / 4 ~= 1 token
        assert_eq!(ctx.token_estimate(), 1);
    }

    #[test]
    fn depth_max_iterations() {
        assert_eq!(ResearchDepth::Speed.max_iterations(), 2);
        assert_eq!(ResearchDepth::Balanced.max_iterations(), 6);
        assert_eq!(ResearchDepth::Quality.max_iterations(), 25);
    }

    #[test]
    fn context_with_budget() {
        let ctx = ResearcherContext::with_budget(500);
        assert_eq!(ctx.token_estimate(), 0);
        assert!(!ctx.is_over_budget());
    }

    #[test]
    fn truncate_preserves_reasoning_preamble() {
        let mut ctx = ResearcherContext::with_budget(300);
        ctx.push_message(ResearcherMessage::System {
            content: "You are helpful.".into(),
        });

        for i in 0..3 {
            ctx.push_message(ResearcherMessage::ToolResult {
                call_id: format!("rp{}", i),
                name: "__reasoning_preamble".into(),
                output: format!("reasoning step {}", i),
            });
        }

        for i in 0..10 {
            ctx.push_message(ResearcherMessage::User {
                content: "x".repeat(100),
            });
            ctx.push_message(ResearcherMessage::ToolResult {
                call_id: format!("c{}", i),
                name: "search".into(),
                output: "y".repeat(100),
            });
        }

        assert!(ctx.is_over_budget());

        let reasoning_before = ctx.messages.iter().filter(|m| {
            matches!(m, ResearcherMessage::ToolResult { name, .. } if name == "__reasoning_preamble")
        }).count();
        assert_eq!(reasoning_before, 3);

        ctx.truncate_oldest_with_priority();

        assert!(matches!(
            ctx.messages.first(),
            Some(ResearcherMessage::System { .. })
        ));

        let reasoning_after = ctx.messages.iter().filter(|m| {
            matches!(m, ResearcherMessage::ToolResult { name, .. } if name == "__reasoning_preamble")
        }).count();
        assert_eq!(
            reasoning_after, 3,
            "All reasoning preambles should be preserved"
        );
    }
}
