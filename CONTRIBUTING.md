# Contributing to SNDR Core Engine

Thanks for your interest! Full guidelines live in
**[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** — this root file exists so
GitHub auto-links it. The essentials:

- **TDD.** New behaviour starts with a failing test (`make test`). Every change
  is verified by running it, not by inspection.
- **Quality gates.** `make gates` must pass (21 audit gates: doc-sync,
  lifecycle, i18n, english-only, gui-contract, …). CI runs the full suite on
  py3.10 + py3.12.
- **English-only in code.** Comments, docstrings, log/error strings, `help=`
  text — English. Markdown docs may be in any language.
- **Patches** live under `sndr/engines/vllm/patches/<family>/` and are declared
  in `sndr/dispatcher/registry.py` with a lifecycle, env flag, and
  `vllm_version_range`. Run `make preflight` before a pin bump.
- **Security.** See [SECURITY.md](SECURITY.md) for disclosure. Don't add
  unauthenticated network surfaces; gate mutating/exec paths behind
  `SNDR_ENABLE_APPLY` / `SNDR_ENABLE_EXEC`.
- **No AI attribution** in commit messages or public artifacts.

License: Apache-2.0 ([LICENSE](LICENSE)).
