// Module root
pub mod planner;
pub use planner::AgentPlanner;

pub mod tools;

pub use tools::{
    build_tool_registry, tool_manifest, EditFileTool, GrepTool, ListDirTool, ReadFileTool,
    ReadUrlTool, SearchTool, ShellTool, Tool, ToolRegistry, WriteFileTool,
};

pub mod classifier;
pub mod executor;

pub use classifier::{ClassifiedIntent, QueryIntent};

pub mod researcher;
pub use researcher::{ResearchDepth, ResearcherContext, ResearcherMessage};

// Expose orchestrator and common event types for MVP
pub mod synthesizer;

pub mod orchestrator;
pub use orchestrator::{AgentEvent, AgentOrchestrator, Answer};
