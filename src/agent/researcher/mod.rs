//! Researcher loop module — context window management and message types.

pub mod context;
pub mod prompt;
pub mod loop_module;

use crate::llm::provider::ToolCall;

/// Message variant for the researcher loop.
/// Separate from `Message` in llm/provider.rs (which has Role::System/User/Assistant only).
#[derive(Debug, Clone)]
pub enum ResearcherMessage {
    System { content: String },
    User { content: String },
    Assistant { content: String },
    AssistantWithToolCalls {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        call_id: String,
        name: String,
        output: String,
    },
}

/// Research depth preset — controls max iterations for the researcher loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResearchDepth {
    Speed,         // max 2 iterations
    #[default]
    Balanced,      // max 6 iterations (default)
    Quality,       // max 25 iterations
}

impl ResearchDepth {
    pub fn max_iterations(&self) -> usize {
        match self {
            Self::Speed => 2,
            Self::Balanced => 6,
            Self::Quality => 25,
        }
    }

    pub fn parse_depth(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "speed" => Self::Speed,
            "quality" => Self::Quality,
            _ => Self::Balanced, // default
        }
    }
}

// Re-exports
pub use context::ResearcherContext;
pub use loop_module::{ResearcherLoop, ResearcherOutput};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_max_iterations() {
        assert_eq!(ResearchDepth::Speed.max_iterations(), 2);
        assert_eq!(ResearchDepth::Balanced.max_iterations(), 6);
        assert_eq!(ResearchDepth::Quality.max_iterations(), 25);
    }

    #[test]
    fn depth_parse() {
        assert_eq!(ResearchDepth::parse_depth("speed"), ResearchDepth::Speed);
        assert_eq!(ResearchDepth::parse_depth("balanced"), ResearchDepth::Balanced);
        assert_eq!(ResearchDepth::parse_depth("quality"), ResearchDepth::Quality);
        assert_eq!(ResearchDepth::parse_depth("unknown"), ResearchDepth::Balanced);
    }

    #[test]
    fn depth_default() {
        assert_eq!(ResearchDepth::default(), ResearchDepth::Balanced);
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
            tool_calls: vec![ToolCall {
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
}
