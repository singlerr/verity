use crate::llm::provider::ProviderRegistry;
use std::sync::Arc;

pub struct AgentPlanner {
    _provider_registry: Arc<ProviderRegistry>,
    _model: String,
}

impl AgentPlanner {
    pub fn new(registry: Arc<ProviderRegistry>, model: String) -> Self {
        Self {
            _provider_registry: registry,
            _model: model,
        }
    }
}
