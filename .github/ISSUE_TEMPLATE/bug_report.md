---
name: Bug report
about: Report something that is broken or behaves incorrectly
title: "[bug] "
labels: bug
---

## Summary

A one-sentence description of what is wrong.

## Steps to reproduce

1.
2.
3.

## Expected behaviour

What you expected to happen.

## Actual behaviour

What happened instead.

## Environment

- SMOS version: (`smos --version` or the git commit)
- Rust toolchain: (`rustc --version`)
- OS:
- GPU feature enabled (if any): `nli-cuda` / `nli-directml` / `nli-metal` / `nli-webgpu` / none
- Ollama version (if relevant):
- Extraction model:
- Embedding model:

## Logs

Paste the relevant SMOS server logs (redact secrets / API keys). If the issue is NLI-related, include the lines around the watcher startup (`NLI backend started` or `NLI backend failed to start`).

```
<paste here>
```

## Reproducible config

If the bug depends on `smos.toml`, paste the relevant sections (redact `api_key` fields):

```toml
<paste here>
```

## Smoke test status

If you have run `smos doctor`, paste the result:

```
<paste here>
```
