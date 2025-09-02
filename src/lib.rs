//! Afiyah core library
//!
//! This crate exposes minimal public APIs so the repository builds in CI.
//! Subsequent edits will implement biologically grounded modules per CONTRIBUTING.md.

/// Returns the current crate semantic version used by the CLI and tests.
pub fn version() -> &'static str {
    "0.1.0"
}

