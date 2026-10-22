import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import TopNav from './components/TopNav';
import ChatMessage from './components/ChatMessage';
import PromptInput from './components/PromptInput';
import Logo from './components/Logo';
import { browserSpeech } from './utils/speech';

const SUGGESTIONS = [
  {
    title: "Document Synthesis",
    desc: "How does RagObot verify and synthesize facts?",
    query: "How does RagObot synthesize answers from documents?"
  },
  {
    title: "Dual Compute Coordination",
    desc: "How do local and cloud models run in parallel?",
    query: "How do local and cloud models work together in RagObot?"
  },
  {
    title: "Context Window Guard",
    desc: "How does the token budget manage conversation history?",
    query: "Explain the context window and token budget management."
  },
  {
    title: "Windows Voice Audio",
    desc: "How does asynchronous SAPI speech handoff work?",
    query: "How does the Windows native voice playback work?"
  }
];

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [status, setStatus] = useState({});
  const [backendOnline, setBackendOnline] = useState(true);
  const [audioMode, setAudioMode] = useState('active'); // 'active' | 'muted'
  const [modelMode, setModelMode] = useState('dual'); // 'dual' | 'local' | 'cloud'
  const [localModel, setLocalModel] = useState('');
  const [cloudModel, setCloudModel] = useState('');
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [activeSpeakingMsgId, setActiveSpeakingMsgId] = useState(null);
  const [isReloading, setIsReloading] = useState(false);

  const [tokenCount, setTokenCount] = useState(0);
  const [wasTrimmed, setWasTrimmed] = useState(false);

  // Multi-Thread Conversation State (ChatGPT Style)
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [lastUploadedDoc, setLastUploadedDoc] = useState(null);

  const chatBottomRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Fetch threads list from backend
  const fetchThreads = async () => {
    try {
      const res = await fetch('/api/threads');
      if (res.ok) {
        const data = await res.json();
        const threadList = data.threads || [];
        setThreads(threadList);
        return threadList;
      }
    } catch (e) {
      console.error('Failed to load threads', e);
    }
    return [];
  };

  // Switch to an existing conversation thread
  const handleSelectThread = async (threadId) => {
    if (isStreaming) return;
    stopAllAudio();
    setActiveThreadId(threadId);

    try {
      const res = await fetch(`/api/threads/${threadId}`);
      if (res.ok) {
        const threadData = await res.json();
        setMessages(threadData.messages || []);
        setTokenCount(threadData.token_count || 0);
        setWasTrimmed(threadData.was_trimmed || false);
        setLastUploadedDoc(threadData.last_uploaded_doc || null);
      }
    } catch (e) {
      console.error('Failed to switch thread', e);
    }
  };

  // Create a brand new conversation thread
  const handleNewChat = async () => {
    if (isStreaming) return;
    stopAllAudio();
    try {
      const res = await fetch('/api/threads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Chat' })
      });
      if (res.ok) {
        const newThread = await res.json();
        setActiveThreadId(newThread.id);
        setMessages([]);
        setTokenCount(0);
        setWasTrimmed(false);
        setLastUploadedDoc(null);
        setThreads((prev) => [
          {
            id: newThread.id,
            title: newThread.title,
            updated_at: newThread.updated_at,
            message_count: 0,
            token_count: 0
          },
          ...prev.filter((t) => t.id !== newThread.id)
        ]);
      }
    } catch (e) {
      const fallbackId = `thread-${Date.now()}`;
      setActiveThreadId(fallbackId);
      setMessages([]);
      setTokenCount(0);
      setWasTrimmed(false);
      setLastUploadedDoc(null);
    }
  };

  // Delete a conversation thread
  const handleDeleteThread = async (threadId) => {
    try {
      await fetch(`/api/threads/${threadId}`, { method: 'DELETE' });
      setThreads((prev) => {
        const remaining = prev.filter((t) => t.id !== threadId);
        if (activeThreadId === threadId) {
          if (remaining.length > 0) {
            handleSelectThread(remaining[0].id);
          } else {
            handleNewChat();
          }
        }
        return remaining;
      });
    } catch (e) {
      console.error('Failed to delete thread', e);
    }
  };

  // Fetch initial backend status and initialize threads
  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        setBackendOnline(true);
        if (data.ollama_model) setLocalModel(data.ollama_model);
        if (data.groq_model) setCloudModel(data.groq_model);
        if (data.current_tokens !== undefined) setTokenCount(data.current_tokens);
        if (data.was_trimmed !== undefined) setWasTrimmed(data.was_trimmed);
      } else {
        setBackendOnline(false);
      }
    } catch (err) {
      setBackendOnline(false);
    }
  };

  useEffect(() => {
    const initApp = async () => {
      await fetchStatus();
      const threadList = await fetchThreads();
      if (threadList.length > 0) {
        // Automatically restore the most recent active thread
        handleSelectThread(threadList[0].id);
      } else {
        // Start fresh with a new thread
        handleNewChat();
      }
    };
    initApp();
  }, []);

  const scrollToBottom = (smooth = true) => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' });
    }
  };

  useEffect(() => {
    scrollToBottom(false);
  }, [messages.length]);

  // Audio Playback
  const handlePlaySpeech = (msgId, text) => {
    stopAllAudio();
    setActiveSpeakingMsgId(msgId);
    browserSpeech.speak(
      text,
      () => setIsSpeaking(true),
      () => {
        setIsSpeaking(false);
        setActiveSpeakingMsgId(null);
      }
    );
  };

  const stopAllAudio = () => {
    browserSpeech.stop();
    setIsSpeaking(false);
    setActiveSpeakingMsgId(null);
    fetch('/api/voice/stop', { method: 'POST' }).catch(() => {});
  };

  // Submit Query
  const handleQuery = async (queryText, uploadedDocContext = null) => {
    if (!queryText.trim() || isStreaming) return;

    stopAllAudio();

    let currentThreadId = activeThreadId;
    if (!currentThreadId) {
      currentThreadId = `thread-${Date.now()}`;
      setActiveThreadId(currentThreadId);
    }

    const effectiveDoc = uploadedDocContext || lastUploadedDoc;
    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const userMessage = {
      id: userMsgId,
      role: 'user',
      content: queryText,
      timestamp
    };

    const assistantMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      sources: [],
      ollamaStream: '',
      groqStream: '',
      arbiterDebate: '',
      isStreaming: true,
      timestamp
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          thread_id: currentThreadId,
          uploaded_doc: effectiveDoc,
          trigger_backend_tts: false,
          model_mode: modelMode,
          local_model: localModel,
          cloud_model: cloudModel
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      setBackendOnline(true);
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (!part.trim()) continue;

          let eventType = 'message';
          let eventData = '';

          const lines = part.split('\n');
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6);
            }
          }

          if (!eventData) continue;

          try {
            const parsed = JSON.parse(eventData);

            if (eventType === 'docs') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId ? { ...msg, sources: parsed } : msg
                )
              );
            } else if (eventType === 'stage') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? {
                        ...msg,
                        stageLabel: parsed.label || parsed.stage
                      }
                    : msg
                )
              );
            } else if (eventType === 'ollama_token') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? { ...msg, ollamaStream: (msg.ollamaStream || '') + (parsed.token || '') }
                    : msg
                )
              );
            } else if (eventType === 'groq_token') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? { ...msg, groqStream: (msg.groqStream || '') + (parsed.token || '') }
                    : msg
                )
              );
            } else if (eventType === 'evidence_status') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? {
                        ...msg,
                        evidenceAvailable: parsed.evidence_available,
                        evidenceReason: parsed.evidence_reason
                      }
                    : msg
                )
              );
            } else if (eventType === 'validation_status') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? {
                        ...msg,
                        answerGrounded: parsed.answer_grounded,
                        groundingReason: parsed.grounding_reason,
                        isAbstained: parsed.is_abstained
                      }
                    : msg
                )
              );
            } else if (eventType === 'consensus') {
              const synthesis = parsed.final_response || '';
              const isAbstained = parsed.is_abstained || false;
              setTokenCount(parsed.token_count || 0);
              setWasTrimmed(parsed.was_trimmed || false);

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? {
                        ...msg,
                        content: synthesis,
                        arbiterDebate: parsed.arbiter_debate || '',
                        ollamaStream: parsed.ollama_output || msg.ollamaStream,
                        groqStream: parsed.groq_output || msg.groqStream,
                        evidenceAvailable: parsed.evidence_available,
                        evidenceReason: parsed.evidence_reason,
                        answerGrounded: parsed.answer_grounded,
                        groundingReason: parsed.grounding_reason,
                        isAbstained: isAbstained,
                        abstentionReason: parsed.abstention_reason || '',
                        tokenCount: parsed.token_count,
                        wasTrimmed: parsed.was_trimmed,
                        stageLabel: null,
                        isStreaming: false
                      }
                    : msg
                )
              );

              // Update thread tracking with thread_id and thread_title
              if (parsed.thread_id) {
                setActiveThreadId(parsed.thread_id);
              }
              if (parsed.thread_title) {
                setThreads((prev) => {
                  const existingIndex = prev.findIndex((t) => t.id === parsed.thread_id);
                  if (existingIndex >= 0) {
                    const updated = [...prev];
                    updated[existingIndex] = {
                      ...updated[existingIndex],
                      title: parsed.thread_title,
                      updated_at: Date.now()
                    };
                    return updated.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
                  } else {
                    return [
                      {
                        id: parsed.thread_id,
                        title: parsed.thread_title,
                        updated_at: Date.now(),
                        message_count: 2,
                        token_count: parsed.token_count || 0
                      },
                      ...prev
                    ];
                  }
                });
              }

              // Audio synthesis (TTS) occurs on verified grounded response when not muted
              if (!isAbstained && audioMode !== 'muted' && synthesis) {
                handlePlaySpeech(assistantMsgId, synthesis);
              }
            } else if (eventType === 'error') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? {
                        ...msg,
                        content: `**Error:** ${parsed.error}`,
                        isStreaming: false
                      }
                    : msg
                )
              );
            }
          } catch (e) {
            console.warn('JSON parse error in SSE chunk:', e);
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Query execution error:', err);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  content: `> **Connection Error:** Could not connect to backend server (${err.message}).\n\nEnsure FastAPI is running: \`.venv\\Scripts\\python -m uvicorn api:app --reload --port 8000\``,
                  isStreaming: false
                }
              : msg
          )
        );
        setBackendOnline(false);
      }
    } finally {
      setIsStreaming(false);
      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantMsgId ? { ...msg, isStreaming: false } : msg))
      );
      fetchStatus();
    }
  };

  const handleStopStream = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsStreaming(false);
  };

  const handleReloadDocs = async () => {
    setIsReloading(true);
    try {
      const res = await fetch('/api/reload-docs', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setStatus((prev) => ({ ...prev, indexed_chunks: data.indexed_chunks }));
      }
    } catch (err) {
      console.error('Reload docs error:', err);
    } finally {
      setIsReloading(false);
    }
  };

  return (
    <div className="app-layout" id="ragobot-root">
      {/* 1. Left Minimalist Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        status={status}
        tokenCount={tokenCount}
        maxTokens={status.token_budget || 2000}
        audioMode={audioMode}
        setAudioMode={setAudioMode}
        onReloadDocs={handleReloadDocs}
        onResetSession={handleNewChat}
        isReloading={isReloading}
        threads={threads}
        activeThreadId={activeThreadId}
        onSelectThread={handleSelectThread}
        onDeleteThread={handleDeleteThread}
      />

      {/* 2. Main Chat Workspace */}
      <div className="main-chat-wrapper">
        <TopNav
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen(true)}
          audioMode={audioMode}
          setAudioMode={setAudioMode}
          onResetSession={handleNewChat}
        />

        <main className="chat-viewport">
          <div className="chat-container">
            {!backendOnline && (
              <div className="offline-banner">
                <span>Backend offline. Run <code>.venv\Scripts\python -m uvicorn api:app --reload</code> on port 8000.</span>
                <button
                  className="sidebar-icon-btn"
                  style={{ color: '#FFFFFF', fontSize: '11px' }}
                  onClick={fetchStatus}
                >
                  Retry
                </button>
              </div>
            )}

            {messages.length === 0 ? (
              <div className="empty-state">
                <div style={{ marginBottom: '20px' }}>
                  <Logo size={52} inverted={true} />
                </div>
                <h1 className="empty-title">RagObot</h1>
                <p className="empty-subtitle">
                  Ask questions about your indexed documents.
                </p>

                <div className="suggestions-grid">
                  {SUGGESTIONS.map((s, idx) => (
                    <div
                      key={idx}
                      className="suggestion-card"
                      onClick={() => handleQuery(s.query)}
                    >
                      <div className="suggestion-card-title">{s.title}</div>
                      <div className="suggestion-card-desc">{s.desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  status={status}
                  isSpeaking={isSpeaking}
                  activeSpeakingMsgId={activeSpeakingMsgId}
                  onPlaySpeech={handlePlaySpeech}
                  onStopSpeech={stopAllAudio}
                />
              ))
            )}

            <div ref={chatBottomRef} style={{ height: '1px' }} />
          </div>
        </main>

        {/* 3. Floating Prompt Input Dock with In-Chat-Box Model Selector */}
        <PromptInput
          onSubmit={handleQuery}
          onStop={handleStopStream}
          onUploadSuccess={(count, filename) => {
            setStatus((prev) => ({ ...prev, indexed_chunks: count }));
            if (filename) setLastUploadedDoc(filename);
          }}
          isStreaming={isStreaming}
          showSuggestions={messages.length > 0}
          modelMode={modelMode}
          setModelMode={setModelMode}
          localModel={localModel}
          setLocalModel={setLocalModel}
          cloudModel={cloudModel}
          setCloudModel={setCloudModel}
          status={status}
          activeThreadId={activeThreadId}
        />
      </div>
    </div>
  );
}
