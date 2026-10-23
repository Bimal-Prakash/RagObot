# Contributing to RagObot

Thank you for your interest in contributing to RagObot! We welcome contributions from developers worldwide to improve multi-agent orchestration, grounding verification, retrieval performance, and user experience.

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful and constructive in all issues, pull requests, and discussions.

---

## Getting Started

1. **Fork the Repository**: Create a fork of [Bimal-Prakash/RagObot](https://github.com/Bimal-Prakash/RagObot) on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/RagObot.git
   cd RagObot
   ```
3. **Set up the Python Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
4. **Set up the Frontend**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```
5. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your preferred endpoints and keys:
   ```bash
   cp .env.example .env
   ```

---

## Development Workflow

### Architecture Principles

- **Deterministic Grounding**: Every candidate answer must be supported by retrieved evidence or uploaded documents. Unsupported factual claims or hallucinated numerical figures must fail-closed to an abstention.
- **Fail-Closed Security**: Evaluator and validation parsers must never default to permissive paths on ambiguous outputs. Developer mode must never expose internal chain-of-thought or worker drafts without header authorization (`X-Dev-Key` or `Authorization: Bearer`).
- **Clean Separation**: Keep LangGraph orchestration, retrieval logic, LLM workers, and HTTP route handling decoupled.
- **Cross-Platform Compatibility**: While RagObot supports low-latency Windows SAPI speech, all core logic and Neural TTS must gracefully degrade on Linux and macOS without crashing.

### Running Backend Tests

Before submitting any code, verify that all backend tests pass:
```bash
python -m unittest discover tests
```

### Verifying Frontend Builds

Ensure the React frontend compiles cleanly with zero production bundle errors:
```bash
cd frontend
npm run build
npm test # if tests configured
cd ..
```

---

## Submitting Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, documented, and tested code.
3. Add unit tests in `tests/` covering new features or bug fixes.
4. Keep commits atomic and descriptive.
5. Push to your fork and submit a Pull Request targeting `main`.
6. Ensure all CI automated checks pass on your PR.

---

## Reporting Security Issues

If you discover a potential security vulnerability (e.g. prompt injection, unauthorized data exposure, path traversal, or credential leakage), **please do NOT open a public GitHub issue**.

Instead, please send a responsible disclosure email directly to the project maintainers with steps to reproduce the issue. We will review and patch verified vulnerabilities promptly.
