import React from 'react';

export default function ConsensusCard({
  consensusText = '',
  isSpeaking = false,
  onPlaySpeech,
  onStopSpeech
}) {
  if (!consensusText) {
    return null;
  }

  return (
    <div className="glass-panel consensus-card" id="consensus-card">
      <div className="consensus-header">
        <div className="consensus-title">
          <span>⚖️</span>
          <span>Consensus Arbiter Synthesis</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {isSpeaking && (
            <div className="audio-status-pill" id="audio-waveform-pill">
              <div className="waveform">
                <div className="waveform-bar" />
                <div className="waveform-bar" />
                <div className="waveform-bar" />
                <div className="waveform-bar" />
              </div>
              <span>VOICE DISPATCH ACTIVE</span>
            </div>
          )}

          {isSpeaking ? (
            <button
              id="stop-speech-btn"
              className="cyber-btn danger"
              onClick={onStopSpeech}
              title="Stop audio playback"
            >
              <span>⏹️</span>
              <span>Stop Audio</span>
            </button>
          ) : (
            <button
              id="play-speech-btn"
              className="cyber-btn"
              onClick={onPlaySpeech}
              title="Replay synthesis voice"
            >
              <span>🔊</span>
              <span>Speak Output</span>
            </button>
          )}
        </div>
      </div>

      <div className="consensus-content" id="consensus-content-text">
        {consensusText}
      </div>
    </div>
  );
}
