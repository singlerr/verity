// Module root
pub mod planner;
pub use planner::AgentPlanner;

pub mod tools;

pub use tools::{Tool, ToolRegistry, SearchTool, ReadUrlTool, ReadFileTool, WriteFileTool, ListDirTool, ShellTool, build_tool_registry, tool_manifest};

pub mod classifier;
pub mod executor;

pub use classifier::{QueryIntent, ClassifiedIntent};

pub mod researcher;
pub use researcher::{ResearcherContext, ResearcherMessage, ResearchDepth};

// Expose orchestrator and common event types for MVP
pub mod synthesizer;

pub mod orchestrator;
pub use orchestrator::{AgentOrchestrator, AgentEvent, Answer};
