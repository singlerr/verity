//! Spinner animation for streaming states.
/// Braille spinner frames cycling at 80ms intervals.
pub const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Animated spinner state.
#[derive(Debug, Clone)]
pub struct Spinner {
    tick_count: usize,
}

impl Spinner {
    pub fn new() -> Self {
        Spinner { tick_count: 0 }
    }

    /// Advance the spinner by one frame.
    pub fn tick(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1) % 10;
    }

    /// Current spinner frame.
    pub fn frame(&self) -> &'static str {
        SPINNER_FRAMES[self.tick_count % 10]
    }

    /// Reset spinner to initial frame.
    pub fn reset(&mut self) {
        self.tick_count = 0;
    }
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}
