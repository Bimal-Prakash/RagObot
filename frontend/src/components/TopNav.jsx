import React from 'react';
import { PanelLeft, Plus, Volume2, VolumeX } from 'lucide-react';
import Logo from './Logo';

export default function TopNav({
  sidebarOpen,
  onToggleSidebar,
  audioMode,
  setAudioMode,
  onResetSession
}) {
  return (
    <header className="top-navbar">
      <div className="top-navbar-left">
        {!sidebarOpen && (
          <button
            className="nav-toggle-btn"
            onClick={onToggleSidebar}
            title="Open sidebar"
          >
            <PanelLeft size={16} />
          </button>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {!sidebarOpen && <Logo size={20} />}
          <span style={{ fontSize: '14px', fontWeight: 600, color: '#FFFFFF' }}>
            RagObot
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <button
          className="nav-toggle-btn"
          onClick={() => setAudioMode(audioMode === 'muted' ? 'active' : 'muted')}
          title="Toggle audio"
        >
          {audioMode === 'muted' ? <VolumeX size={14} /> : <Volume2 size={14} />}
        </button>
        {!sidebarOpen && (
          <button
            className="nav-toggle-btn"
            onClick={onResetSession}
            title="New Chat"
          >
            <Plus size={14} />
          </button>
        )}
      </div>
    </header>
  );
}
