import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User, Copy, Check, Volume2, Square, Scale, ShieldAlert, CheckCircle2 } from 'lucide-react';
import SourcesList from './SourcesList';
import ReasoningView from './ReasoningView';
import AgentDebateModal from './AgentDebateModal';

export default function ChatMessage({
  message,
  status = {},
  isSpeaking = false,
  activeSpeakingMsgId = null,
  onPlaySpeech,
  onStopSpeech
}) {
  const [copied, setCopied] = useState(false);
  const [isDebateModalOpen, setIsDebateModalOpen] = useState(false);
  const isUser = message.role === 'user';
  const isThisSpeaking = isSpeaking && activeSpeakingMsgId === message.id;

  const handleCopy = () => {
    if (message.content) {
      navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const hasDualOutputs = message.ollamaStream || message.groqStream || message.isStreaming;

  return (
    <div className="message-row">
      <div className={`message-avatar ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? <User size={16} /> : <Bot size={18} />}
      </div>

      <div className="message-body">
        <div className="message-header">
          <span className="message-sender">{isUser ? 'You' : 'RagObot'}</span>
          <span className="message-time">{message.timestamp || 'Just now'}</span>
        </div>

        {isUser ? (
          <div className="user-bubble">
            {message.content}
          </div>
        ) : (
          <div className="assistant-bubble">
            {/* 1. Retrieved Source Citations */}
            {message.sources && message.sources.length > 0 && (
              <SourcesList sources={message.sources} />
            )}

            {/* 2. Collapsible Dual-Agent Reasoning & Inline Debate */}
            {hasDualOutputs && (
              <ReasoningView
                ollamaText={message.ollamaStream}
                groqText={message.groqStream}
                arbiterDebate={message.arbiterDebate}
                isStreaming={message.isStreaming}
                stageLabel={message.stageLabel}
                status={status}
                onOpenDebate={() => setIsDebateModalOpen(true)}
              />
            )}

            {/* 3. Workflow Grounding / Abstention Status Badge */}
            {message.isAbstained && (
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '4px 10px',
                marginBottom: '10px',
                borderRadius: '4px',
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid rgba(239, 68, 68, 0.35)',
                color: '#F87171',
                fontSize: '11px',
                fontWeight: 600
              }}>
                <ShieldAlert size={13} />
                <span>Workflow Abstention (Evidence Missing or Ungrounded)</span>
              </div>
            )}

            {!message.isAbstained && message.answerGrounded && (
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '4px 10px',
                marginBottom: '10px',
                borderRadius: '4px',
                background: 'rgba(34, 197, 94, 0.12)',
                border: '1px solid rgba(34, 197, 94, 0.35)',
                color: '#4ADE80',
                fontSize: '11px',
                fontWeight: 600
              }}>
                <CheckCircle2 size={13} />
                <span>Verified & Grounded in Sources</span>
              </div>
            )}

            {/* 4. Synthesized Final Answer with Markdown */}
            <div className="markdown-body">
              {message.content ? (
                <ReactMarkdown>{message.content}</ReactMarkdown>
              ) : message.isStreaming ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
                  <span className="pulse-dot" style={{ backgroundColor: '#FFFFFF' }} />
                  <span>Generating consensus response...</span>
                </div>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>No response generated.</span>
              )}
            </div>

            {/* 4. Action Bar (Copy, Audio Playback, 2-Window Debate Arena, Tokens) */}
            {!message.isStreaming && message.content && (
              <div className="message-actions">
                <div className="action-buttons-group">
                  <button
                    className="msg-action-btn"
                    onClick={handleCopy}
                    title="Copy response to clipboard"
                  >
                    {copied ? <Check size={13} color="#FFFFFF" /> : <Copy size={13} />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>

                  {/* 2-Window Agent Debate Button */}
                  {hasDualOutputs && (
                    <button
                      className="msg-action-btn debate-arena-action-btn"
                      onClick={() => setIsDebateModalOpen(true)}
                      title="Inspect 2-window agent debate, cross-examination, and convergence"
                    >
                      <Scale size={13} />
                      <span>Agent Debate (2-Window)</span>
                    </button>
                  )}

                  {isThisSpeaking ? (
                    <button
                      className="msg-action-btn active"
                      onClick={onStopSpeech}
                      title="Stop speaking"
                    >
                      <Square size={13} fill="currentColor" />
                      <span>Stop Voice</span>
                      <span className="speaking-wave">
                        <span className="wave-bar" />
                        <span className="wave-bar" />
                        <span className="wave-bar" />
                        <span className="wave-bar" />
                      </span>
                    </button>
                  ) : (
                    <button
                      className="msg-action-btn"
                      onClick={() => onPlaySpeech(message.id, message.content)}
                      title="Read aloud with voice synthesis"
                    >
                      <Volume2 size={13} />
                      <span>Read Aloud</span>
                    </button>
                  )}
                </div>

                {message.tokenCount ? (
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {message.tokenCount} tokens
                  </span>
                ) : null}
              </div>
            )}

            {/* 5. Dedicated 2-Window Debate Arena Modal */}
            <AgentDebateModal
              isOpen={isDebateModalOpen}
              onClose={() => setIsDebateModalOpen(false)}
              ollamaText={message.ollamaStream}
              groqText={message.groqStream}
              arbiterDebate={message.arbiterDebate}
              finalResponse={message.content}
              isStreaming={message.isStreaming}
              status={status}
            />
          </div>
        )}
      </div>
    </div>
  );
}
