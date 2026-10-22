import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { X, Copy, Check, Scale, Bot, Cpu, Sparkles, Database, CheckCircle2 } from 'lucide-react';

export default function AgentDebateModal({
  isOpen,
  onClose,
  ollamaText = '',
  groqText = '',
  arbiterDebate = '',
  finalResponse = '',
  isStreaming = false,
  status = {}
}) {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const handleCopyAll = () => {
    const textToCopy = `=== OLLAMA AGENT PERSPECTIVE ===\n${ollamaText}\n\n=== GROQ AGENT PERSPECTIVE ===\n${groqText}\n\n=== ARBITER DEBATE & RESOLUTION ===\n${arbiterDebate}\n\n=== FINAL CONSENSUS VERDICT ===\n${finalResponse}`;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const localModelName = status.ollama_model || 'Ollama Model';
  const cloudModelName = status.groq_model || 'Groq LPU';

  return (
    <div className="debate-modal-backdrop" onClick={onClose}>
      <div
        className="debate-modal-container"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="debate-modal-header">
          <div className="debate-header-left">
            <div className="debate-badge">
              <Scale size={14} />
              <span>Agent Debate Arena</span>
            </div>
            <div className="debate-subtitle">
              Dual-agent cross-examination & consensus convergence backed by ChromaDB (Nomic embeddings)
            </div>
          </div>

          <div className="debate-header-actions">
            <button
              className="debate-action-btn"
              onClick={handleCopyAll}
              title="Copy complete debate & perspectives"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
              <span>{copied ? 'Copied' : 'Copy Debate'}</span>
            </button>
            <button
              className="debate-close-btn"
              onClick={onClose}
              title="Close window"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* 2-Window Split View */}
        <div className="debate-split-grid">
          {/* Window 1: Ollama Agent */}
          <div className="agent-window">
            <div className="agent-window-bar">
              <div className="agent-title-group">
                <Cpu size={14} />
                <span className="agent-name">Window 1: Ollama Agent</span>
                <span className="agent-tag">{localModelName}</span>
              </div>
              <div className="agent-status-indicator">
                {isStreaming && !ollamaText ? (
                  <span className="status-live-dot pulse" />
                ) : (
                  <span className="status-live-dot" />
                )}
                <span>{isStreaming ? 'Computing' : 'Settled'}</span>
              </div>
            </div>

            <div className="agent-window-body">
              {ollamaText ? (
                <div className="stream-text-content">
                  <ReactMarkdown>{ollamaText}</ReactMarkdown>
                </div>
              ) : (
                <div className="stream-empty-state">
                  <span className="pulse-dot" />
                  <span>Waiting for Ollama localized reasoning stream...</span>
                </div>
              )}
            </div>
          </div>

          {/* Window 2: Groq Agent */}
          <div className="agent-window">
            <div className="agent-window-bar">
              <div className="agent-title-group">
                <Sparkles size={14} />
                <span className="agent-name">Window 2: Groq Agent</span>
                <span className="agent-tag">{cloudModelName}</span>
              </div>
              <div className="agent-status-indicator">
                {isStreaming && !groqText ? (
                  <span className="status-live-dot pulse" />
                ) : (
                  <span className="status-live-dot" />
                )}
                <span>{isStreaming ? 'Computing' : 'Settled'}</span>
              </div>
            </div>

            <div className="agent-window-body">
              {groqText ? (
                <div className="stream-text-content">
                  <ReactMarkdown>{groqText}</ReactMarkdown>
                </div>
              ) : (
                <div className="stream-empty-state">
                  <span className="pulse-dot" />
                  <span>Waiting for Groq cloud LPU reasoning stream...</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Convergence & Cross-Examination Window */}
        <div className="debate-resolution-section">
          <div className="resolution-header">
            <div className="resolution-title">
              <Database size={15} />
              <span>Arbiter Cross-Examination & Consensus Convergence</span>
            </div>
            <div className="resolution-meta">
              <span>Ground Truth: ChromaDB Vector Collection</span>
              <span className="meta-bullet">•</span>
              <span>Model: nomic-embed-text</span>
            </div>
          </div>

          <div className="resolution-body">
            {/* The Active Debate & Critique */}
            {arbiterDebate && (
              <div className="debate-critique-box">
                <div className="critique-header">
                  <Scale size={13} />
                  <span>Cross-Examination & Points of Contention</span>
                </div>
                <div className="critique-text markdown-body">
                  <ReactMarkdown>{arbiterDebate}</ReactMarkdown>
                </div>
              </div>
            )}

            {/* The Final Consensus Point */}
            {finalResponse && (
              <div className="debate-verdict-box">
                <div className="verdict-header">
                  <CheckCircle2 size={14} />
                  <span>Convergence Point (Final Agreed Verdict)</span>
                </div>
                <div className="verdict-text markdown-body">
                  <ReactMarkdown>{finalResponse}</ReactMarkdown>
                </div>
              </div>
            )}

            {!arbiterDebate && !finalResponse && isStreaming && (
              <div className="resolution-waiting">
                <span className="pulse-dot" />
                <span>Arbiter awaiting completion of parallel agents before initiating cross-examination...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
