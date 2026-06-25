use super::types::{ConfigError, SmosConfig};

// ---------------------------------------------------------------------------
// SmosConfig loading
// ---------------------------------------------------------------------------

impl SmosConfig {
    /// Load from a TOML file (overridden by `SMOS__*` environment variables).
    /// Returns defaults when the file is missing so the proxy runs
    /// out-of-the-box without a config file; sections absent from a partial
    /// file also fall back to their defaults via `#[serde(default)]`.
    ///
    /// Environment overrides use the `SMOS__` prefix and a `__` section
    /// separator.
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let mut builder = ::config::Config::builder();
        if std::path::Path::new(path).exists() {
            // The existence pre-check is load-bearing: `File::with_name`
            // treats its argument as a *stem* and, when the exact file is
            // absent, probes `<name>.toml` (the only registered extension
            // under the `features = ["toml"]` build) — so
            // `with_name("smos.toml")` would search for `smos.toml.toml`.
            // Skipping the source when the path does not exist avoids that
            // probe entirely. Do not remove the pre-check without
            // replacing it; `File::from(Path)` does NOT bypass the probe
            // in config 0.14.x (it funnels through the same `find_file`
            // extension-search path as `with_name`).
            builder = builder.add_source(::config::File::with_name(path));
        }
        builder = builder.add_source(::config::Environment::with_prefix("SMOS").separator("__"));
        let cfg: SmosConfig = builder.build()?.try_deserialize()?;
        // Fail-fast on invalid config: an operator who ships a config with a
        // bad confidence range or a missing embedding dimension should hear
        // about it at startup, not on the first request that hits the broken
        // path. `validate` collects EVERY problem in one pass so a single
        // startup error is enough to fix a half-broken TOML.
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load directly from a TOML string. Used by `smos init`'s self-test
    /// so the canonical default config (shipped as an inline literal) is
    /// validated without going through the file system. Environment
    /// overrides are NOT applied here — the caller already controls the
    /// input verbatim.
    pub fn load_from_str(toml: &str) -> Result<Self, ConfigError> {
        let cfg: SmosConfig = ::config::Config::builder()
            .add_source(::config::File::from_str(toml, ::config::FileFormat::Toml))
            .build()?
            .try_deserialize()?;
        cfg.validate()?;
        Ok(cfg)
    }
}
