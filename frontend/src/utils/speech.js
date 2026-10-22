// Hybrid Neural Voice Synthesizer
// Uses studio-grade Neural TTS with automatic Web Speech natural-voice fallback

export function cleanTextForSpeech(text) {
  if (!text) return '';
  return text
    .replace(/```[\s\S]*?```/g, '') // code blocks
    .replace(/`([^`]+)`/g, '$1')   // inline code
    .replace(/#+\s*(.+)/g, '$1. ')  // headers converted to sentence pauses
    .replace(/[*_~]+/g, '')         // bold/italic
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // links
    .replace(/^\s*[-*+]\s+/gm, '')  // bullets
    .replace(/^\s*\d+\.\s+/gm, '')   // numbering
    .replace(/[^\x00-\x7F]+/g, ' ')  // emojis/symbols
    .replace(/\s+/g, ' ')
    .trim();
}

class BrowserSpeechEngine {
  constructor() {
    this.synth = typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis : null;
    this.currentUtterance = null;
    this.currentAudio = null;
    this.activeVoice = null;
    this.cachedVoices = [];
    this.speakSessionId = 0;

    if (this.synth) {
      this.initVoices();
      if (this.synth.onvoiceschanged !== undefined) {
        this.synth.onvoiceschanged = () => this.initVoices();
      }
    }
  }

  initVoices() {
    if (!this.synth) return;
    this.cachedVoices = this.synth.getVoices();
    // Prioritize high-quality human/natural voices over old mechanical desktop voices
    const naturalVoice = this.cachedVoices.find(v => 
      (v.name.includes('Natural') || v.name.includes('Online') || v.name.includes('Neural')) && v.lang.startsWith('en')
    ) || this.cachedVoices.find(v => 
      v.name.includes('Google') && v.lang.startsWith('en')
    ) || this.cachedVoices.find(v => 
      v.name.includes('Zira') || v.name.includes('Samantha') || (v.lang.startsWith('en') && !v.name.includes('David'))
    );

    if (naturalVoice) {
      this.activeVoice = naturalVoice;
    }
  }

  async speak(text, onStart, onEnd) {
    this.stop();
    const currentSession = ++this.speakSessionId;

    const cleaned = cleanTextForSpeech(text);
    if (!cleaned) {
      if (onEnd) onEnd();
      return;
    }

    // 1. Try High-Definition Studio Neural Voice via backend API first
    try {
      const response = await fetch('/api/voice/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: cleaned, voice: 'en-US-JennyNeural' })
      });

      // Abort if session was superseded by stop() or another speak() call
      if (this.speakSessionId !== currentSession) return;

      if (response.ok) {
        const blob = await response.blob();
        if (this.speakSessionId !== currentSession) return;

        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        this.currentAudio = audio;

        audio.onplay = () => {
          if (this.speakSessionId === currentSession && onStart) onStart();
        };

        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          if (this.currentAudio === audio) this.currentAudio = null;
          if (this.speakSessionId === currentSession && onEnd) onEnd();
        };

        audio.onerror = () => {
          URL.revokeObjectURL(audioUrl);
          if (this.currentAudio === audio) this.currentAudio = null;
          if (this.speakSessionId === currentSession) {
            this._fallbackWebSpeech(cleaned, onStart, onEnd, currentSession);
          }
        };

        try {
          await audio.play();
          return;
        } catch (playErr) {
          // If browser policy blocked audio.play(), fallback to Web Speech
          URL.revokeObjectURL(audioUrl);
          if (this.currentAudio === audio) this.currentAudio = null;
          if (this.speakSessionId === currentSession) {
            this._fallbackWebSpeech(cleaned, onStart, onEnd, currentSession);
          }
          return;
        }
      }
    } catch (e) {
      // Backend neural synthesis unavailable or network error -> use natural web speech fallback
    }

    if (this.speakSessionId === currentSession) {
      this._fallbackWebSpeech(cleaned, onStart, onEnd, currentSession);
    }
  }

  _fallbackWebSpeech(cleaned, onStart, onEnd, currentSession) {
    if (!this.synth) {
      if (onEnd) onEnd();
      return;
    }

    const utterance = new SpeechSynthesisUtterance(cleaned);
    
    // Refresh voices if not yet initialized
    if (!this.activeVoice) {
      this.initVoices();
    }
    if (this.activeVoice) {
      utterance.voice = this.activeVoice;
    }

    // Calibrated natural human cadence: natural speed and warm, expressive pitch
    utterance.rate = 0.96;
    utterance.pitch = 1.02;

    utterance.onstart = () => {
      if ((!currentSession || this.speakSessionId === currentSession) && onStart) onStart();
    };

    utterance.onend = () => {
      this.currentUtterance = null;
      if ((!currentSession || this.speakSessionId === currentSession) && onEnd) onEnd();
    };

    utterance.onerror = () => {
      this.currentUtterance = null;
      if ((!currentSession || this.speakSessionId === currentSession) && onEnd) onEnd();
    };

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
  }

  stop() {
    this.speakSessionId++;
    if (this.currentAudio) {
      try {
        this.currentAudio.pause();
        this.currentAudio.currentTime = 0;
      } catch (e) {}
      this.currentAudio = null;
    }
    if (this.synth) {
      try {
        this.synth.cancel();
      } catch (e) {}
    }
    this.currentUtterance = null;
  }

  isSpeaking() {
    const isAudioPlaying = this.currentAudio && !this.currentAudio.paused;
    const isSynthSpeaking = this.synth ? this.synth.speaking : false;
    return Boolean(isAudioPlaying || isSynthSpeaking);
  }
}

export const browserSpeech = new BrowserSpeechEngine();
