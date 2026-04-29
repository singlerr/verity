use std::time::{Duration, Instant};

// Import AgentEvent type for wiring agent events into the UI
use crate::agent::orchestrator::AgentEvent;
use crate::agent::researcher::ResearchDepth;
use crate::llm::provider::ModelEntry;
use crate::ui::spinner::Spinner;

pub mod types;
pub use types::*;

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
    pub elapsed: Option<Duration>,
    pub spinner: Spinner,
    pub autocomplete_idx: usize,
    pub model_select_open: bool,
    pub model_list: Vec<ModelEntry>,
    pub selected_model_idx: usize,
    pub active_model: String,
    pub answer_scroll: u16,
    pub research_depth: ResearchDepth,
    pub provider_display_names: std::collections::HashMap<String, String>,
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
            elapsed: None,
            spinner: Spinner::new(),
            autocomplete_idx: 0,
            model_select_open: false,
            model_list: Vec::new(),
            selected_model_idx: 0,
            active_model: config.active_model,
            answer_scroll: 0,
            research_depth: ResearchDepth::default(),
            provider_display_names: std::collections::HashMap::new(),
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
                self.elapsed = self.start_time.map(|t| t.elapsed());
                self.start_time = None;
                self.state = AppState::AnswerReady;
            }
            AgentEvent::Error(msg) => {
                self.state = AppState::Error(msg);
            }
            AgentEvent::ModelListReady(models) => {
                self.model_list = models;
            }
            AgentEvent::Classified(_intent) => {
                self.state = AppState::Classifying;
            }
            AgentEvent::SearchingIteration {
                current,
                max,
                query,
            } => {
                self.state = AppState::Researching;
                self.trace_lines.push(TerminalLine {
                    kind: LineKind::Out,
                    text: format!("Searching ({}/{}) for: {}", current, max, query),
                });
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
            self.elapsed = None;
            self.answer_scroll = 0;
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
