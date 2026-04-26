//! Credential store for API authentication.
//!
//! Stores credentials in plain TOML format at `~/.verity/credentials.toml`.
//! File permissions are set to 0600 on Unix for security.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Per-provider credentials.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Credentials {
    pub api_key: String,
}

/// Authentication status for a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthStatus {
    Authenticated,
    NotAuthenticated,
    Expired,
}

/// Credential store managing API keys per provider.
///
/// Persists to `~/.verity/credentials.toml` in plain TOML format:
///
/// ```toml
/// [openai]
/// api_key = "sk-..."
///
/// [anthropic]
/// api_key = "sk-ant-..."
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CredentialStore {
    credentials: HashMap<String, Credentials>,
}

impl CredentialStore {
    /// Returns the path to the credentials file.
    fn credentials_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("verity")
            .join("credentials.toml")
    }

    /// Loads the credential store from disk.
    ///
    /// Returns an empty store if the file does not exist.
    pub fn load() -> Result<Self> {
        let path = Self::credentials_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path)?;
        let store: CredentialStore = toml::from_str(&content)?;
        Ok(store)
    }

    /// Saves the credential store to disk.
    pub fn save(&self) -> Result<()> {
        let path = Self::credentials_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(&path, content)?;

        #[cfg(unix)]
        {
            // Set file permissions to owner-read/write only (0600)
            use std::os::unix::fs::PermissionsExt;
            if let Ok(mut perms) = std::fs::metadata(&path).map(|m| m.permissions()) {
                perms.set_mode(0o600);
                std::fs::set_permissions(&path, perms)?;
            }
        }

        Ok(())
    }

    /// Gets credentials for a provider.
    pub fn get(&self, provider: &str) -> Option<&Credentials> {
        self.credentials.get(provider)
    }

    /// Gets the authentication status for a provider.
    pub fn status(&self, provider: &str) -> AuthStatus {
        match self.credentials.get(provider) {
            Some(creds) if !creds.api_key.is_empty() => AuthStatus::Authenticated,
            _ => AuthStatus::NotAuthenticated,
        }
    }

    /// Sets credentials for a provider.
    pub fn set(&mut self, provider: String, credentials: Credentials) {
        self.credentials.insert(provider, credentials);
    }

    /// Removes credentials for a provider.
    pub fn remove(&mut self, provider: &str) {
        self.credentials.remove(provider);
    }

    /// Returns an iterator over all provider names.
    pub fn providers(&self) -> impl Iterator<Item = &String> {
        self.credentials.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_status_ordering() {
        assert_ne!(AuthStatus::Authenticated, AuthStatus::NotAuthenticated);
        assert_ne!(AuthStatus::Authenticated, AuthStatus::Expired);
        assert_ne!(AuthStatus::NotAuthenticated, AuthStatus::Expired);
    }

    #[test]
    fn test_empty_store() {
        let store = CredentialStore::default();
        assert!(store.get("openai").is_none());
        assert_eq!(store.status("openai"), AuthStatus::NotAuthenticated);
        assert!(store.providers().next().is_none());
    }

    #[test]
    fn test_set_and_get() {
        let mut store = CredentialStore::default();
        store.set(
            "openai".to_string(),
            Credentials {
                api_key: "sk-test".to_string(),
            },
        );
        assert_eq!(
            store.get("openai").expect("should exist").api_key,
            "sk-test"
        );
        assert_eq!(store.status("openai"), AuthStatus::Authenticated);
    }

    #[test]
    fn test_remove() {
        let mut store = CredentialStore::default();
        store.set(
            "openai".to_string(),
            Credentials {
                api_key: "sk-test".to_string(),
            },
        );
        store.remove("openai");
        assert!(store.get("openai").is_none());
    }
}
