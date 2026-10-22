import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, Square, ChevronDown, X, Paperclip, FileText, Check, Loader2, Eye, EyeOff, Key, Save } from 'lucide-react';

const DEFAULT_SUGGESTIONS = [
  "How does RagObot synthesize answers?",
  "How are local and cloud models coordinated?",
  "Explain the context window and token limits.",
  "How does the Windows native voice playback work?"
];

const GROQ_PRESETS = [
  "llama-3.3-70b-versatile",
  "llama-3.1-8b-instant",
  "mixtral-8x7b-32768"
];

export default function PromptInput({
  onSubmit,
  onStop,
  onUploadSuccess,
  isStreaming,
  showSuggestions = false,
  modelMode = 'dual',
  setModelMode,
  localModel = '',
  setLocalModel,
  cloudModel = '',
  setCloudModel,
  status = {},
  activeThreadId = null
}) {
  const [query, setQuery] = useState('');
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);

  // Groq configuration state
  const [groqModelInput, setGroqModelInput] = useState(cloudModel || 'llama-3.3-70b-versatile');
  const [groqApiKeyInput, setGroqApiKeyInput] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isSavingGroq, setIsSavingGroq] = useState(false);
  const [groqSaveFeedback, setGroqSaveFeedback] = useState(null);
  const [showDualGroqSettings, setShowDualGroqSettings] = useState(false);

  const [availableModels, setAvailableModels] = useState({
    local_models: [],
    cloud_models: [],
    groq_ready: false
  });

  const textareaRef = useRef(null);
  const dropdownRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch('/api/models')
      .then((res) => res.json())
      .then((data) => {
        if (data.local_models && data.local_models.length > 0) {
          setAvailableModels(data);
          if (data.default_local) setLocalModel(data.default_local);
          if (!cloudModel && data.default_cloud) {
            setCloudModel(data.default_cloud);
            setGroqModelInput(data.default_cloud);
          }
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (cloudModel) {
      setGroqModelInput(cloudModel);
    }
  }, [cloudModel]);

  const handleSaveGroqSettings = async (e) => {
    if (e) e.preventDefault();
    setIsSavingGroq(true);
    setGroqSaveFeedback(null);

    const payload = {};
    if (groqModelInput.trim()) {
      payload.model = groqModelInput.trim();
    }
    if (groqApiKeyInput.trim()) {
      payload.api_key = groqApiKeyInput.trim();
    }

    try {
      const res = await fetch('/api/settings/groq', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        const data = await res.json();
        if (data.status === 'success') {
          setGroqSaveFeedback({ type: 'success', message: 'Saved directly to .env' });
          if (data.groq_model) {
            setCloudModel(data.groq_model);
            setGroqModelInput(data.groq_model);
          }
          setAvailableModels((prev) => ({
            ...prev,
            groq_ready: data.groq_ready,
            cloud_models: Array.from(new Set([data.groq_model, ...(prev.cloud_models || [])]))
          }));
          setGroqApiKeyInput('');
          setTimeout(() => setGroqSaveFeedback(null), 3500);
        } else {
          setGroqSaveFeedback({ type: 'error', message: data.error || 'Failed to save' });
        }
      } else {
        setGroqSaveFeedback({ type: 'error', message: 'Server communication error' });
      }
    } catch (err) {
      setGroqSaveFeedback({ type: 'error', message: err.message });
    } finally {
      setIsSavingGroq(false);
    }
  };

  const handleGroqKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSaveGroqSettings();
    }
  };

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsModelDropdownOpen(false);
      }
    };
    if (isModelDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isModelDropdownOpen]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (query.trim() && !isStreaming) {
      onSubmit(query.trim(), uploadedFile ? uploadedFile.name : null);
      setQuery('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleSuggestionClick = (text) => {
    if (!isStreaming) {
      onSubmit(text, uploadedFile ? uploadedFile.name : null);
    }
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    if (activeThreadId) {
      formData.append('thread_id', activeThreadId);
    }

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        if (data.status === 'success') {
          setUploadedFile({ name: file.name, chunks: data.indexed_chunks });
          if (onUploadSuccess) onUploadSuccess(data.indexed_chunks, file.name);
        } else {
          alert(`Upload failed: ${data.error}`);
        }
      } else {
        alert('File upload failed. Verify the server is running.');
      }
    } catch (err) {
      alert(`Error uploading file: ${err.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const getModelButtonLabel = () => {
    if (modelMode === 'dual') return 'Dual';
    if (modelMode === 'local') {
      const shortName = localModel ? localModel.split(':')[0] : 'Ollama';
      return `Ollama (${shortName})`;
    }
    if (modelMode === 'cloud') {
      const shortName = cloudModel ? cloudModel.split('-')[0] : 'Groq';
      return `Groq (${shortName})`;
    }
    return 'Model';
  };

  return (
    <footer className="input-dock" id="prompt-dock">
      <div className="input-dock-inner">
        {/* Uploaded File Chip Banner */}
        {uploadedFile && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div className="attached-file-chip">
              <FileText size={12} />
              <span>{uploadedFile.name} (Indexed)</span>
              <button
                type="button"
                style={{ background: 'transparent', border: 'none', color: '#999999', cursor: 'pointer', display: 'flex' }}
                onClick={() => setUploadedFile(null)}
              >
                <X size={12} />
              </button>
            </div>
          </div>
        )}

        {showSuggestions && (
          <div className="quick-tags-row">
            {DEFAULT_SUGGESTIONS.map((suggestion, idx) => (
              <button
                key={idx}
                type="button"
                className="quick-tag-chip"
                onClick={() => handleSuggestionClick(suggestion)}
                disabled={isStreaming}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        <div className="input-card" ref={dropdownRef}>
          {/* File Upload / Attachment Paperclip Button */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept=".pdf,.md,.txt"
            style={{ display: 'none' }}
          />

          <button
            type="button"
            className="attach-file-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Upload PDF, Markdown, or TXT document to knowledge base"
            disabled={isStreaming || isUploading}
          >
            {isUploading ? (
              <Loader2 size={15} className="spin-anim" />
            ) : (
              <Paperclip size={15} />
            )}
          </button>

          <textarea
            ref={textareaRef}
            className="prompt-textarea"
            placeholder={isStreaming ? "Thinking..." : "Ask a question about your indexed documents..."}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isStreaming}
          />

          <div className="input-right-actions">
            {/* Model Selector Button inside Chat Box */}
            <button
              type="button"
              className={`model-picker-btn ${isModelDropdownOpen ? 'active' : ''}`}
              onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
              title="Choose model"
              disabled={isStreaming}
            >
              <span>{getModelButtonLabel()}</span>
              <ChevronDown size={11} />
            </button>

            {/* Model Selection Dropdown Popup */}
            {isModelDropdownOpen && (
              <div className="model-picker-dropdown">
                <div className="model-picker-header">
                  <div className="model-picker-title">
                    <span>Select Model</span>
                  </div>
                  <button
                    type="button"
                    style={{ background: 'transparent', border: 'none', color: '#777777', cursor: 'pointer' }}
                    onClick={() => setIsModelDropdownOpen(false)}
                  >
                    <X size={13} />
                  </button>
                </div>

                <div className="model-mode-tabs">
                  <button
                    type="button"
                    className={`model-mode-tab ${modelMode === 'dual' ? 'active' : ''}`}
                    onClick={() => setModelMode('dual')}
                  >
                    Dual
                  </button>
                  <button
                    type="button"
                    className={`model-mode-tab ${modelMode === 'local' ? 'active' : ''}`}
                    onClick={() => setModelMode('local')}
                  >
                    Ollama
                  </button>
                  <button
                    type="button"
                    className={`model-mode-tab ${modelMode === 'cloud' ? 'active' : ''}`}
                    onClick={() => setModelMode('cloud')}
                  >
                    Groq
                  </button>
                </div>

                {/* Ollama local model selector */}
                {(modelMode === 'dual' || modelMode === 'local') && (
                  <div className="model-select-group">
                    <div className="model-select-label-row">
                      <span className="model-select-label">Ollama Model</span>
                      <span className="model-select-sublabel">Local compute</span>
                    </div>
                    <select
                      className="model-select-dropdown"
                      value={localModel}
                      onChange={(e) => setLocalModel(e.target.value)}
                    >
                      {availableModels.local_models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Groq Cloud Model & API Key Configuration */}
                {modelMode === 'cloud' && (
                  <div className="groq-settings-container">
                    <div className="model-select-group">
                      <div className="model-select-label-row">
                        <span className="model-select-label">Groq Model Name</span>
                        <span className="model-select-sublabel">Type any model or pick</span>
                      </div>
                      <input
                        type="text"
                        className="model-text-input"
                        placeholder="e.g. llama-3.3-70b-versatile"
                        value={groqModelInput}
                        onChange={(e) => setGroqModelInput(e.target.value)}
                        onKeyDown={handleGroqKeyDown}
                      />
                      <div className="model-chips-row">
                        {GROQ_PRESETS.map((m) => (
                          <button
                            key={m}
                            type="button"
                            className={`model-chip-btn ${groqModelInput === m ? 'active' : ''}`}
                            onClick={() => setGroqModelInput(m)}
                          >
                            {m.split('-')[0] + '-' + m.split('-')[1]}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="model-select-group">
                      <div className="model-select-label-row">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                          <Key size={11} />
                          <span className="model-select-label">Groq API Key</span>
                        </div>
                        <span className={`key-status-indicator ${availableModels.groq_ready ? 'active' : ''}`}>
                          {availableModels.groq_ready ? '● Active in .env' : '○ Missing in .env'}
                        </span>
                      </div>
                      <div className="key-input-wrapper">
                        <input
                          type={showApiKey ? 'text' : 'password'}
                          className="model-text-input key-input"
                          placeholder={availableModels.groq_ready ? "Key Active in .env (enter to update)" : "gsk_..."}
                          value={groqApiKeyInput}
                          onChange={(e) => setGroqApiKeyInput(e.target.value)}
                          onKeyDown={handleGroqKeyDown}
                        />
                        <button
                          type="button"
                          className="key-toggle-btn"
                          onClick={() => setShowApiKey(!showApiKey)}
                          title={showApiKey ? "Hide key" : "Show key"}
                        >
                          {showApiKey ? <EyeOff size={12} /> : <Eye size={12} />}
                        </button>
                      </div>
                    </div>

                    <div className="groq-actions-row">
                      <button
                        type="button"
                        className="save-env-btn"
                        onClick={handleSaveGroqSettings}
                        disabled={isSavingGroq}
                        title="Save model & API key directly to .env (or press Enter)"
                      >
                        {isSavingGroq ? (
                          <>
                            <Loader2 size={12} className="spin-anim" />
                            <span>Saving...</span>
                          </>
                        ) : (
                          <>
                            <Save size={12} />
                            <span>Save to .env</span>
                            <span className="kbd-shortcut">↵ Enter</span>
                          </>
                        )}
                      </button>
                    </div>

                    {groqSaveFeedback && (
                      <div className={`env-save-feedback ${groqSaveFeedback.type}`}>
                        {groqSaveFeedback.type === 'success' && <Check size={12} />}
                        <span>{groqSaveFeedback.message}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Groq in Dual Mode */}
                {modelMode === 'dual' && (
                  <div className="groq-dual-section">
                    <div className="model-select-group">
                      <div className="model-select-label-row">
                        <span className="model-select-label">Groq Model</span>
                        <button
                          type="button"
                          className="toggle-groq-config-btn"
                          onClick={() => setShowDualGroqSettings(!showDualGroqSettings)}
                        >
                          {showDualGroqSettings ? 'Hide .env config' : 'Configure in .env'}
                        </button>
                      </div>
                      <select
                        className="model-select-dropdown"
                        value={cloudModel}
                        onChange={(e) => {
                          setCloudModel(e.target.value);
                          setGroqModelInput(e.target.value);
                        }}
                      >
                        {availableModels.cloud_models.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>

                    {showDualGroqSettings && (
                      <div className="groq-settings-container dual-groq-box">
                        <div className="model-select-group">
                          <span className="model-select-label">Custom Model Name</span>
                          <input
                            type="text"
                            className="model-text-input"
                            placeholder="e.g. llama-3.3-70b-versatile"
                            value={groqModelInput}
                            onChange={(e) => setGroqModelInput(e.target.value)}
                            onKeyDown={handleGroqKeyDown}
                          />
                        </div>

                        <div className="model-select-group">
                          <div className="model-select-label-row">
                            <span className="model-select-label">Groq API Key</span>
                            <span className={`key-status-indicator ${availableModels.groq_ready ? 'active' : ''}`}>
                              {availableModels.groq_ready ? '● Active' : '○ Not Set'}
                            </span>
                          </div>
                          <div className="key-input-wrapper">
                            <input
                              type={showApiKey ? 'text' : 'password'}
                              className="model-text-input key-input"
                              placeholder={availableModels.groq_ready ? "Key Active in .env (enter new to update)" : "gsk_..."}
                              value={groqApiKeyInput}
                              onChange={(e) => setGroqApiKeyInput(e.target.value)}
                              onKeyDown={handleGroqKeyDown}
                            />
                            <button
                              type="button"
                              className="key-toggle-btn"
                              onClick={() => setShowApiKey(!showApiKey)}
                            >
                              {showApiKey ? <EyeOff size={12} /> : <Eye size={12} />}
                            </button>
                          </div>
                        </div>

                        <div className="groq-actions-row">
                          <button
                            type="button"
                            className="save-env-btn"
                            onClick={handleSaveGroqSettings}
                            disabled={isSavingGroq}
                          >
                            {isSavingGroq ? (
                              <>
                                <Loader2 size={12} className="spin-anim" />
                                <span>Saving...</span>
                              </>
                            ) : (
                              <>
                                <Save size={12} />
                                <span>Save to .env</span>
                                <span className="kbd-shortcut">↵ Enter</span>
                              </>
                            )}
                          </button>
                        </div>

                        {groqSaveFeedback && (
                          <div className={`env-save-feedback ${groqSaveFeedback.type}`}>
                            {groqSaveFeedback.type === 'success' && <Check size={12} />}
                            <span>{groqSaveFeedback.message}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {isStreaming ? (
              <button
                type="button"
                className="send-action-btn"
                onClick={onStop}
                title="Stop"
              >
                <Square size={13} fill="#000000" />
              </button>
            ) : (
              <button
                type="button"
                className="send-action-btn"
                onClick={handleSend}
                disabled={!query.trim() || isStreaming}
                title="Send"
              >
                <ArrowUp size={15} strokeWidth={2.5} />
              </button>
            )}
          </div>
        </div>

        <div style={{ textAlign: 'center', fontSize: '11px', color: '#555555', marginTop: '2px' }}>
          RagObot answers are grounded in your indexed documentation.
        </div>
      </div>
    </footer>
  );
}
