# Contributing to EVSE Power-Telemetry Intrusion Detection

Thank you for your interest in contributing! This document outlines the
guidelines for contributing to this project.

## How to Contribute

### 1. Reporting Issues
- Check the [issues](../../issues) list first to avoid duplicates.
- Include a clear, minimal title and a step-by-step description.
- Provide the environment (OS, Python version, package versions) and any
  relevant error output.

### 2. Suggesting Enhancements
- Open an issue describing the proposed enhancement, its motivation, and
  possible implementation approach.

### 3. Pull Requests
1. **Fork** the repository and create a feature branch
   (`git checkout -b feature/my-feature`).
2. **Follow the structure:**
   - Experiments live in `notebooks/` and are numbered in reading order.
   - Keep analysis prose in `docs/`.
   - Store run artifacts in `reports/` and `figures/`.
3. **Preserve reproducibility:** always fix random seeds and document any new
   dependency in `requirements.txt`.
4. **Test your changes** by re-running the affected notebook(s).
5. Push the branch and open a pull request with a clear description of the
   motivation, change, and expected effect on results.

## Reproducibility Guidelines
- Never commit raw datasets or large generated artifacts
  (`data/`, `reports/`, `figures/` are git-ignored).
- Keep every experiment self-contained and deterministic.
- If you change the evaluation strategy, document it in
  `docs/methodology.md` and update `docs/results.md`.

## Code of Conduct
Be respectful and constructive. Harassment or discrimination of any kind will
not be tolerated.

## License
By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
