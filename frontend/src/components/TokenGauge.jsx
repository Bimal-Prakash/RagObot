import React from 'react';

export default function TokenGauge({
  tokenCount = 0,
  maxTokens = 2000,
  wasTrimmed = false,
  status = {}
}) {
  const percentage = Math.min(100, Math.round((tokenCount / maxTokens) * 100));
  const isWarning = percentage > 80 || wasTrimmed;

  return (
    <div className="glass-panel token-dashboard" id="token-dashboard">
      <div className="system-status-grid">
        {/* Ollama Local Matrix */}
        <div className="glass-panel status-card ollama" id="card-ollama">
          <div className="status-card-icon">⚡</div>
          <div>
            <div className="status-card-title">1. Local Matrix (Ollama)</div>
            <div className="status-card-value">
              {status.ollama_model || 'llama3.2:3b'}
              <span className="status-badge online">ONLINE</span>
            </div>
          </div>
        </div>

        {/* Groq Cloud LPU */}
        <div className="glass-panel status-card groq" id="card-groq">
          <div className="status-card-icon">🚀</div>
          <div>
            <div className="status-card-title">2. Cloud LPU (Groq)</div>
            <div className="status-card-value">
              {status.groq_model || 'llama-3.3-70b'}
              <span className={`status-badge ${status.groq_ready ? 'online' : 'standby'}`}>
                {status.groq_ready ? 'READY' : 'KEY REQ'}
              </span>
            </div>
          </div>
        </div>

        {/* TTS Native Audio */}
        <div className="glass-panel status-card voice" id="card-voice">
          <div className="status-card-icon">🔊</div>
          <div>
            <div className="status-card-title">3. Windows Native Voice</div>
            <div className="status-card-value">
              WinSDK SAPI
              <span className={`status-badge ${status.tts_enabled ? 'online' : 'standby'}`}>
                {status.tts_enabled ? 'DISPATCH ON' : 'MUTED'}
              </span>
            </div>
          </div>
        </div>

        {/* RAG Source Chunks */}
        <div className="glass-panel status-card rag" id="card-rag">
          <div className="status-card-icon">📄</div>
          <div>
            <div className="status-card-title">4. Structured Markdown</div>
            <div className="status-card-value">
              {status.indexed_chunks || 0} Chunks
              <span className="status-badge online">INDEXED</span>
            </div>
          </div>
        </div>
      </div>

      {/* Rolling Token Budget Progress Bar */}
      <div className="token-dashboard-header" style={{ marginTop: '12px' }}>
        <div className="token-label-wrap">
          <span>📊 Rolling Context Window:</span>
          <strong>{tokenCount.toLocaleString()} / {maxTokens.toLocaleString()} Tokens ({percentage}%)</strong>
        </div>

        {wasTrimmed ? (
          <div className="prune-alert" id="prune-alert-badge">
            <span>✂️</span>
            <span>ALERT: CONTEXT PRUNED (ROLLING RECOVERY ACTIVE)</span>
          </div>
        ) : (
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-green)' }}>
            ✓ Context Stable
          </div>
        )}
      </div>

      <div className="token-track">
        <div
          className={`token-fill ${isWarning ? 'warning' : ''}`}
          style={{ width: `${Math.max(2, percentage)}%` }}
          id="token-progress-bar"
        />
      </div>
    </div>
  );
}
