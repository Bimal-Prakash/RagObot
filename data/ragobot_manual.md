# RagObot System Architecture & Manual
The guy who built RagObot is Bimal P B

## System Overview
RagObot is an advanced, headless multi-agent Retrieval-Augmented Generation (RAG) platform. It orchestrates dual compute engines simultaneously via Ollama and Groq, strictly manages rolling context window token budgets, presents real-time calculations on an interactive cyber-console, and outputs speech through native Windows SAPI voice dispatch.

## Dual Compute Matrix
RagObot coordinates two independent AI worker nodes in parallel:

### Ollama Compute Node
- **Engine**: ChatOllama running via the Ollama interface on port 11434.
- **Model**: `gemma4:31b-cloud` (cloud-hosted compute proxied through Ollama).
- **Role**: High-capacity evidence-based verification against RAG chunks.
- **Key Advantage**: Leverages high-parameter models (such as `gemma4:31b-cloud`) through Ollama's unified API without requiring high-end local GPU VRAM, while retaining the flexibility to switch to fully local models (like `llama3.2`).

### Groq Cloud Engine
- **Engine**: ChatGroq powered by Groq's ultra-fast LPU (Language Processing Unit) architecture.
- **Model**: `openai/gpt-oss-120b` (or `llama-3.3-70b-versatile`).
- **Role**: Syntax supervisor, structural reasoning, and logical precision.

## The Consensus Arbiter
RagObot implements an advanced debate and consensus resolution protocol:
- Neither worker node directly communicates with the user.
- The Consensus Arbiter cross-examines outputs from both Ollama and Groq.
- It detects and eliminates hallucinations, reconciles factual discrepancies, merges complementary insights, and crafts a singular, definitive synthesis.
- The synthesis is phonetically tailored for natural cadence during Text-to-Speech (TTS) readout.

## Token Budget Guard & Rolling Memory
RagObot enforces rigorous token limits to prevent context overflow:
- **Default Budget**: 2,000 tokens limit.
- **Pruning Strategy**: Employs `strategy="last"` to drop older human-assistant dialogue turns.
- **System Anchoring**: Uses `include_system=True` to guarantee that fundamental RAG instructions and rules are permanently retained in memory.
- **Visual Alert**: Triggers a high-visibility diagnostic alert whenever history blocks are pruned.

## Native Windows Audio Pipeline
- Employs Windows COM dispatch via `win32com.client.Dispatch("SAPI.SpVoice")`.
- Execution is completely asynchronous using `SVSFlagsAsync` so the terminal UI never experiences lag or stutter.
- Markdown, code blocks, and symbols are stripped before audio synthesis for natural human cadence.

Built by Bimal P B