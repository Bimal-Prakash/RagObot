import React, { useState, useEffect } from 'react';
import { Cpu, ChevronDown, Check, Sparkles, Shield, Cloud, HardDrive, Settings2, X } from 'lucide-react';

export default function ModelSelector({
  modelMode = 'dual', // 'dual' | 'local' | 'cloud'
  setModelMode,
  localModel = 'llama3.2:3b',
  setLocalModel,
  cloudModel = 'llama-3.3-70b-versatile',
  setCloudModel,
  status = {}
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [availableModels, setAvailableModels] = useState({
    local_models: ['llama3.2:3b', 'deepseek-r1:8b', 'qwen2.5:3b', 'mistral:latest'],
    cloud_models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
    groq_ready: false
  });
  const [customLocal, setCustomLocal] = useState('');
  const [customCloud, setCustomCloud] = useState('');

  // Fetch available models from backend
  useEffect(() => {
    fetch('/api/models')
      .then((res) => res.json())
      .then((data) => {
        if (data.local_models && data.local_models.length > 0) {
          setAvailableModels(data);
          if (!localModel && data.default_local) setLocalModel(data.default_local);
          if (!cloudModel && data.default_cloud) setCloudModel(data.default_cloud);
        }
      })
      .catch(() => {
        // Fallback default model list
      });
  }, []);

  // Mode display label
  const getModeLabel = () => {
    if (modelMode === 'dual') return 'Dual Engine (Consensus)';
    if (modelMode === 'local') return `Local (${localModel})`;
    if (modelMode === 'cloud') return `Cloud (${cloudModel.split('-')[0] || 'LPU'})`;
    return 'Select Engine';
  };

  return (
    <div style={{ position: 'relative' }}>
      {/* Trigger Button in Header */}
      <button
        type="button"
        className="header-action-btn active"
        onClick={() => setIsOpen(!isOpen)}
        title="Select model and execution engine"
        style={{ gap: '8px', padding: '0 12px' }}
      >
        {modelMode === 'dual' ? (
          <Sparkles size={14} color="var(--accent-primary)" />
        ) : modelMode === 'local' ? (
          <HardDrive size={14} color="var(--accent-cyan)" />
        ) : (
          <Cloud size={14} color="var(--accent-amber)" />
        )}
        <span style={{ fontWeight: 600 }}>{getModeLabel()}</span>
        <ChevronDown size={12} color="var(--text-muted)" />
      </button>

      {/* Model Selection Modal / Popover */}
      {isOpen && (
        <div className="modal-overlay" onClick={() => setIsOpen(false)}>
          <div
            className="modal-card"
            style={{ maxWidth: '520px' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div className="modal-title-wrap">
                <div className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Settings2 size={16} color="var(--accent-primary)" />
                  <span>Choose Model & Engine</span>
                </div>
                <div className="modal-subtitle">
                  Configure which models RagObot uses to answer your queries
                </div>
              </div>
              <button className="modal-close-btn" onClick={() => setIsOpen(false)}>
                <X size={16} />
              </button>
            </div>

            <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* 1. Execution Mode Selection */}
              <div>
                <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '8px', display: 'block' }}>
                  Execution Mode
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                  {/* Dual Mode */}
                  <div
                    className={`suggestion-card ${modelMode === 'dual' ? 'active' : ''}`}
                    style={{
                      padding: '12px',
                      borderRadius: 'var(--radius-md)',
                      borderColor: modelMode === 'dual' ? 'var(--accent-primary)' : 'var(--border-subtle)',
                      background: modelMode === 'dual' ? 'var(--accent-primary-subtle)' : 'var(--bg-surface)'
                    }}
                    onClick={() => setModelMode('dual')}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Sparkles size={16} color="var(--accent-primary)" />
                      {modelMode === 'dual' && <Check size={14} color="var(--accent-primary)" />}
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '6px' }}>
                      Dual Consensus
                    </div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                      Cross-examines local & cloud
                    </div>
                  </div>

                  {/* Local Only */}
                  <div
                    className={`suggestion-card ${modelMode === 'local' ? 'active' : ''}`}
                    style={{
                      padding: '12px',
                      borderRadius: 'var(--radius-md)',
                      borderColor: modelMode === 'local' ? 'var(--accent-cyan)' : 'var(--border-subtle)',
                      background: modelMode === 'local' ? 'rgba(6, 182, 212, 0.12)' : 'var(--bg-surface)'
                    }}
                    onClick={() => setModelMode('local')}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <HardDrive size={16} color="var(--accent-cyan)" />
                      {modelMode === 'local' && <Check size={14} color="var(--accent-cyan)" />}
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '6px' }}>
                      Local Only
                    </div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                      Runs on device hardware
                    </div>
                  </div>

                  {/* Cloud Only */}
                  <div
                    className={`suggestion-card ${modelMode === 'cloud' ? 'active' : ''}`}
                    style={{
                      padding: '12px',
                      borderRadius: 'var(--radius-md)',
                      borderColor: modelMode === 'cloud' ? 'var(--accent-amber)' : 'var(--border-subtle)',
                      background: modelMode === 'cloud' ? 'rgba(245, 158, 11, 0.12)' : 'var(--bg-surface)'
                    }}
                    onClick={() => setModelMode('cloud')}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Cloud size={16} color="var(--accent-amber)" />
                      {modelMode === 'cloud' && <Check size={14} color="var(--accent-amber)" />}
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '6px' }}>
                      Cloud Only
                    </div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                      Ultra-fast Groq LPU
                    </div>
                  </div>
                </div>
              </div>

              {/* 2. Local Model Picker (Shown if Dual or Local) */}
              {(modelMode === 'dual' || modelMode === 'local') && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Select Local Model (Ollama)
                    </label>
                    <span style={{ fontSize: '11px', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span className="status-dot online" /> Connected
                    </span>
                  </div>
                  <select
                    value={localModel}
                    onChange={(e) => setLocalModel(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-surface)',
                      border: '1px solid var(--border-medium)',
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '13px',
                      outline: 'none'
                    }}
                  >
                    {availableModels.local_models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* 3. Cloud Model Picker (Shown if Dual or Cloud) */}
              {(modelMode === 'dual' || modelMode === 'cloud') && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Select Cloud Model (Groq)
                    </label>
                    <span style={{ fontSize: '11px', color: availableModels.groq_ready ? 'var(--accent-emerald)' : 'var(--accent-amber)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span className={`status-dot ${availableModels.groq_ready ? 'online' : 'warning'}`} />
                      {availableModels.groq_ready ? 'API Key Active' : 'Key Req / Fallback'}
                    </span>
                  </div>
                  <select
                    value={cloudModel}
                    onChange={(e) => setCloudModel(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-surface)',
                      border: '1px solid var(--border-medium)',
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '13px',
                      outline: 'none'
                    }}
                  >
                    {availableModels.cloud_models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Confirm / Close Button */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '10px' }}>
                <button
                  type="button"
                  className="header-action-btn new-chat"
                  onClick={() => setIsOpen(false)}
                  style={{ padding: '0 18px', height: '36px' }}
                >
                  Apply Selection
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
