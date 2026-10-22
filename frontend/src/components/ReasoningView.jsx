import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChevronDown, ChevronUp, Scale, Database, ShieldCheck, Loader2 } from 'lucide-react';

export default function ReasoningView({
  ollamaText = '',
  groqText = '',
  arbiterDebate = '',
  isStreaming = false,
  stageLabel = '',
  status = {},
  onOpenDebate
}) {
  const [isOpen, setIsOpen] = useState(false);

  // During streaming, show stage progress indicator without leaking ungrounded raw text
  if (!ollamaText && !groqText && !isStreaming) return null;

  return (
    <div className="reasoning-box">
      <div className="reasoning-trigger-bar">
        <button
          type="button"
          className="reasoning-trigger"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
        >
          <span style={{ fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            {isStreaming ? (
              <>
                <Loader2 size={13} className="spin-animation" style={{ color: 'var(--text-cyan)' }} />
                <span>{stageLabel || 'Pipeline executing & verifying grounding...'}</span>
              </>
            ) : (
              <>
                <ShieldCheck size={13} style={{ color: 'var(--color-primary, #06b6d4)' }} />
                <span>View Audited Reasoning & Debate</span>
              </>
            )}
          </span>
          {isOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {!isStreaming && onOpenDebate && (
          <button
            type="button"
            className="debate-arena-pill-btn"
            onClick={onOpenDebate}
            title="Open 2-window verified agent debate arena"
          >
            <Scale size={12} />
            <span>2-Window Arena</span>
          </button>
        )}
      </div>

      {isOpen && (
        <div className="reasoning-content">
          {isStreaming ? (
            <div style={{ padding: '12px', fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Worker outputs quarantined behind validation boundary until Consensus Arbiter and Grounding Validator finish.
            </div>
          ) : (
            <>
              <div className="reasoning-models-grid">
                {ollamaText && (
                  <div className="model-column">
                    <div className="model-col-header">
                      <span>Window 1: Ollama Output</span>
                      <span className="col-subtag">{status.ollama_model || 'Ollama'}</span>
                    </div>
                    <div className="model-stream-text">{ollamaText}</div>
                  </div>
                )}

                {groqText && (
                  <div className="model-column">
                    <div className="model-col-header">
                      <span>Window 2: Groq Output</span>
                      <span className="col-subtag">{status.groq_model || 'Groq LPU'}</span>
                    </div>
                    <div className="model-stream-text">{groqText}</div>
                  </div>
                )}
              </div>

              {arbiterDebate && (
                <div className="inline-debate-card">
                  <div className="inline-debate-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Scale size={13} />
                      <span>Arbiter Cross-Examination & Debate Resolution</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <Database size={11} />
                      <span>ChromaDB (Nomic)</span>
                    </div>
                  </div>
                  <div className="inline-debate-body markdown-body">
                    <ReactMarkdown>{arbiterDebate}</ReactMarkdown>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
