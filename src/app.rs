use std::time::Instant;

// Import AgentEvent type for wiring agent events into the UI
use crate::agent::orchestrator::AgentEvent;
use crate::ui::spinner::Spinner;

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

#[derive(Debug, Clone)]
pub struct App {
    pub mode: Mode,
    pub plan_steps: Vec<PlanStep>,
    pub sources: Vec<Source>,
    pub answer_chunks: Vec<AnswerChunk>,
    pub trace_lines: Vec<TerminalLine>,
    pub state: AppState,
    pub focus: Focus,
    pub query: String,
    pub submitted_query: String,
    pub selected_source: Option<usize>,
    pub running: bool,
    pub start_time: Option<Instant>,
    pub spinner: Spinner,
    pub autocomplete_idx: usize,
    pub model_select_open: bool,
    pub model_list: Vec<String>,
    pub selected_model_idx: usize,
    pub active_model: String,
}

impl App {
    pub fn new() -> Self {
        let config = crate::config::Config::load().unwrap_or_default();
        App {
            mode: Mode::Research,
            plan_steps: Vec::new(),
            sources: Vec::new(),
            answer_chunks: Vec::new(),
            trace_lines: Vec::new(),
            state: AppState::Idle,
            focus: Focus::Left,
            query: String::new(),
            submitted_query: String::new(),
            selected_source: None,
            running: true,
            start_time: None,
            spinner: Spinner::new(),
            autocomplete_idx: 0,
            model_select_open: false,
            model_list: Vec::new(),
            selected_model_idx: 0,
            active_model: config.active_model,
        }
    }

    // Handle incoming AgentEvent from the agent orchestrator and mutate app state
    pub fn handle_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::PlanReady(steps) => {
                self.plan_steps = steps;
                self.state = AppState::Researching;
            }
            AgentEvent::StepStarted(idx) => {
                if idx < self.plan_steps.len() {
                    self.plan_steps[idx].status = StepStatus::Running;
                }
            }
            AgentEvent::StepDone(idx) => {
                if idx < self.plan_steps.len() {
                    self.plan_steps[idx].status = StepStatus::Done;
                }
            }
            AgentEvent::StepFailed(idx, msg) => {
                if idx < self.plan_steps.len() {
                    self.plan_steps[idx].status = StepStatus::Done;
                }
                self.trace_lines.push(TerminalLine {
                    kind: LineKind::Dim,
                    text: format!("Step {} failed: {}", idx, msg),
                });
            }
            AgentEvent::StepProgress(_idx, msg) => {
                self.trace_lines.push(TerminalLine {
                    kind: LineKind::Out,
                    text: msg,
                });
            }
            AgentEvent::SourceFound(src) => {
                self.sources.push(src);
            }
            AgentEvent::AnswerChunk(chunk) => {
                self.answer_chunks.push(chunk);
            }
            AgentEvent::Done(_answer) => {
                self.state = AppState::AnswerReady;
            }
            AgentEvent::Error(msg) => {
                self.state = AppState::Error(msg);
            }
            AgentEvent::ModelListReady(models) => {
                self.model_list = models;
            }
        }
    }

    /// Advance the spinner by one frame (called on tick events).
    pub fn on_tick(&mut self) {
        self.spinner.tick();
    }

    pub fn submit_query(&mut self) {
        if !self.query.is_empty() {
            self.submitted_query = std::mem::take(&mut self.query);
            self.state = AppState::Planning;
            self.sources.clear();
            self.answer_chunks.clear();
            self.trace_lines.clear();
            self.plan_steps.clear();
            self.start_time = Some(Instant::now());
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
