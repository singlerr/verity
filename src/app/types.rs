//! Type definitions for the app UI state.

#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Research,
    Code,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Queued,
    Running,
    Done,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tool {
    Search,
    Read,
    Think,
    Edit,
    Shell,
    ReadFile,
    ListDir,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LineKind {
    Cmd,
    Out,
    Ok,
    Dim,
}

// Global UI state for the agent-driven workflow
#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    Idle,
    Classifying,
    Planning,
    Researching,
    AnswerReady,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: String,
    pub title: String,
    pub tool: Tool,
    pub status: StepStatus,
    pub duration: Option<f64>,
    pub thoughts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Source {
    pub num: usize,
    pub domain: String,
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub quote: String,
}

#[derive(Debug, Clone)]
pub struct AnswerChunk {
    pub text: String,
    pub is_code: bool,
    pub is_bold: bool,
    pub is_em: bool,
    pub citations: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct TerminalLine {
    pub kind: LineKind,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Focus {
    Left,
    Right,
    Command,
}
