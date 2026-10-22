import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChevronDown, ChevronUp, Scale, Database } from 'lucide-react';

export default function ReasoningView({
  ollamaText = '',
  groqText = '',
  arbiterDebate = '',
  isStreaming = false,
  status = {},
  onOpenDebate
}) {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (isStreaming) {
      setIsOpen(true);
    }
  }, [isStreaming]);

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
          <span style={{ fontWeight: 500 }}>
            {isStreaming ? 'Agents Computing & Debating...' : 'View Reasoning & Debate'}
          </span>
          {isOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {onOpenDebate && (
          <button
            type="button"
            className="debate-arena-pill-btn"
            onClick={onOpenDebate}
            title="Open 2-window live agent debate arena"
          >
            <Scale size={12} />
            <span>2-Window Arena</span>
          </button>
        )}
      </div>

      {isOpen && (
        <div className="reasoning-content">
          <div className="reasoning-models-grid">
            {ollamaText && (
              <div className="model-column">
                <div className="model-col-header">
                  <span>Window 1: Ollama Stream</span>
                  <span className="col-subtag">{status.ollama_model || 'Ollama'}</span>
                </div>
                <div className="model-stream-text">{ollamaText}</div>
              </div>
            )}

            {groqText && (
              <div className="model-column">
                <div className="model-col-header">
                  <span>Window 2: Groq Stream</span>
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
        </div>
      )}
    </div>
  );
}
