import React from 'react';
import { Bot, RefreshCw, Volume2, VolumeX, Plus, Database } from 'lucide-react';

export default function Header({
  status = {},
  tokenCount = 0,
  maxTokens = 2000,
  wasTrimmed = false,
  audioMode = 'hybrid',
  setAudioMode,
  onReloadDocs,
  onResetSession,
  isReloading = false
}) {
  const percentage = Math.min(100, Math.round((tokenCount / maxTokens) * 100));
  const isTokenWarning = percentage > 80 || wasTrimmed;

  const cycleAudioMode = () => {
    const modes = ['hybrid', 'backend', 'browser', 'muted'];
    const currentIndex = modes.indexOf(audioMode);
    const nextMode = modes[(currentIndex + 1) % modes.length];
    setAudioMode(nextMode);
  };

  return (
    <header className="app-header" id="app-header">
      <div className="header-left">
        {/* Brand identity */}
        <div className="brand-badge">
          <div className="brand-icon-box">
            <Bot size={20} />
          </div>
          <div className="brand-info">
            <div className="brand-name">
              <span>RagObot</span>
              <span className="brand-tag">ASSISTANT</span>
            </div>
            <span className="brand-sub">Multi-Model Knowledge Synthesis</span>
          </div>
        </div>
      </div>

      <div className="header-right">
        {/* Context Window / Token Budget Pill */}
        <div className="token-gauge-pill" title={`Rolling Context Window: ${tokenCount} / ${maxTokens} tokens`}>
          <span>Context:</span>
          <div className="token-gauge-bar">
            <div
              className={`token-gauge-fill ${isTokenWarning ? 'warning' : ''}`}
              style={{ width: `${Math.max(4, percentage)}%` }}
            />
          </div>
          <span>{tokenCount}t</span>
        </div>

        {/* Knowledge Base Status & Re-index Button */}
        <button
          className="header-action-btn"
          onClick={onReloadDocs}
          disabled={isReloading}
          title="Re-scan and index Markdown documents in data/"
        >
          <Database size={13} color="var(--accent-primary)" />
          <span>{status.indexed_chunks || 0} Chunks</span>
          <RefreshCw size={12} className={isReloading ? 'spin-anim' : ''} />
        </button>

        {/* Audio Output Source Selector */}
        <button
          className={`header-action-btn ${audioMode !== 'muted' ? 'active' : ''}`}
          onClick={cycleAudioMode}
          title="Toggle TTS audio dispatch source (Hybrid, Backend, Browser, Muted)"
        >
          {audioMode === 'muted' ? <VolumeX size={13} /> : <Volume2 size={13} />}
          <span>Audio: {audioMode.toUpperCase()}</span>
        </button>

        {/* New Chat Button */}
        <button
          className="header-action-btn new-chat"
          onClick={onResetSession}
          title="Start a new conversation thread"
        >
          <Plus size={14} />
          <span>New Chat</span>
        </button>
      </div>
    </header>
  );
}
