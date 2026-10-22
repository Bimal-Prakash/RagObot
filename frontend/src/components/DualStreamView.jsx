import React, { useEffect, useRef } from 'react';

export default function DualStreamView({
  ollamaText = '',
  groqText = '',
  isStreaming = false,
  status = {}
}) {
  const ollamaBodyRef = useRef(null);
  const groqBodyRef = useRef(null);

  // Auto-scroll stream containers as new tokens stream in
  useEffect(() => {
    if (ollamaBodyRef.current) {
      ollamaBodyRef.current.scrollTop = ollamaBodyRef.current.scrollHeight;
    }
  }, [ollamaText]);

  useEffect(() => {
    if (groqBodyRef.current) {
      groqBodyRef.current.scrollTop = groqBodyRef.current.scrollHeight;
    }
  }, [groqText]);

  return (
    <div className="dual-stream-grid" id="dual-stream-grid">
      {/* Column A: Ollama Local Matrix */}
      <div className={`glass-panel stream-panel ollama ${isStreaming && ollamaText ? 'active' : ''}`} id="panel-ollama">
        <div className="stream-header">
          <div className="stream-title cyan">
            <span>⚡</span>
            <span>Ollama Brain // Local Matrix</span>
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-cyan)' }}>
            {status.ollama_model || 'llama3.2:3b'}
          </div>
        </div>

        <div className={`stream-body ${isStreaming ? 'active' : ''}`} ref={ollamaBodyRef} id="stream-ollama-body">
          {ollamaText ? (
            <>
              {ollamaText}
              {isStreaming && <span className="typing-cursor" />}
            </>
          ) : (
            <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Waiting for user query to initiate localized compute verification stream...
            </span>
          )}
        </div>
      </div>

      {/* Column B: Groq Cloud LPU */}
      <div className={`glass-panel stream-panel groq ${isStreaming && groqText ? 'active' : ''}`} id="panel-groq">
        <div className="stream-header">
          <div className="stream-title yellow">
            <span>🚀</span>
            <span>Groq Engine // Cloud LPU</span>
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-yellow)' }}>
            {status.groq_model || 'llama-3.3-70b'}
          </div>
        </div>

        <div className={`stream-body ${isStreaming ? 'active' : ''}`} ref={groqBodyRef} id="stream-groq-body">
          {groqText ? (
            <>
              {groqText}
              {isStreaming && <span className="typing-cursor yellow" />}
            </>
          ) : (
            <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Waiting for user query to initiate ultra-fast structural reasoning stream...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
