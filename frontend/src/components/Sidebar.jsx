import React from 'react';
import { Plus, Database, RefreshCw, Volume2, VolumeX, PanelLeftClose, MessageSquare, Trash2 } from 'lucide-react';
import Logo from './Logo';

export default function Sidebar({
  isOpen,
  onToggle,
  status = {},
  tokenCount = 0,
  maxTokens = 2000,
  audioMode = 'active',
  setAudioMode,
  onReloadDocs,
  onResetSession,
  isReloading = false,
  threads = [],
  activeThreadId = null,
  onSelectThread = () => {},
  onDeleteThread = () => {}
}) {
  const percentage = Math.min(100, Math.round((tokenCount / maxTokens) * 100));

  const toggleAudio = () => {
    setAudioMode(audioMode === 'muted' ? 'active' : 'muted');
  };

  return (
    <aside className={`app-sidebar ${isOpen ? '' : 'collapsed'}`} id="app-sidebar">
      <div className="sidebar-top">
        {/* Brand header */}
        <div className="sidebar-brand">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Logo size={22} />
            <span className="brand-title-text">RagObot</span>
          </div>
          <button
            className="sidebar-icon-btn"
            onClick={onToggle}
            title="Collapse sidebar"
          >
            <PanelLeftClose size={16} />
          </button>
        </div>

        {/* New Chat Button */}
        <button
          className="sidebar-btn-new-chat"
          onClick={onResetSession}
        >
          <Plus size={15} />
          <span>New Chat</span>
        </button>

        {/* Recent Chats / Threads Section (ChatGPT Style) */}
        <div className="sidebar-threads-container">
          <div className="sidebar-section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '4px 0 6px 0' }}>
            <span>Recent Chats</span>
            <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {threads.length}
            </span>
          </div>

          <div className="sidebar-threads-scroll">
            {threads.length === 0 ? (
              <div className="sidebar-threads-empty">
                <span>No past chats yet</span>
              </div>
            ) : (
              threads.map((t) => {
                const isActive = t.id === activeThreadId;
                return (
                  <div
                    key={t.id}
                    className={`thread-item ${isActive ? 'active' : ''}`}
                    onClick={() => onSelectThread(t.id)}
                    title={t.title}
                  >
                    <MessageSquare size={13} className="thread-item-icon" />
                    <span className="thread-title">{t.title || 'Untitled Chat'}</span>
                    <button
                      className="thread-delete-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteThread(t.id);
                      }}
                      title="Delete chat"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Knowledge Base Section */}
        <div className="sidebar-section-title">Knowledge Base</div>
        <div className="sidebar-knowledge-card">
          <div className="sidebar-knowledge-info">
            <Database size={14} color="#FFFFFF" />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
              <span>{status.indexed_chunks || 0} Chunks</span>
              <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>ChromaDB • nomic</span>
            </div>
          </div>
          <button
            className="sidebar-icon-btn"
            onClick={onReloadDocs}
            disabled={isReloading}
            title="Re-index ChromaDB documents"
          >
            <RefreshCw size={12} className={isReloading ? 'spin-anim' : ''} />
          </button>
        </div>
      </div>

      <div className="sidebar-bottom">
        {/* Context Window Gauge */}
        <div className="sidebar-token-meter">
          <div className="token-meter-label">
            <span>Context</span>
            <span>{tokenCount} / {maxTokens}t</span>
          </div>
          <div className="token-meter-track">
            <div
              className="token-meter-bar"
              style={{ width: `${Math.max(2, percentage)}%` }}
            />
          </div>
        </div>

        {/* Audio Output Row */}
        <div className="sidebar-audio-row">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {audioMode === 'muted' ? <VolumeX size={14} /> : <Volume2 size={14} />}
            <span>Voice Audio</span>
          </div>
          <button
            className="audio-toggle-btn"
            onClick={toggleAudio}
          >
            {audioMode === 'muted' ? 'MUTED' : 'ACTIVE'}
          </button>
        </div>
      </div>
    </aside>
  );
}
