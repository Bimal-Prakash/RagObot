import re
import os
import ctypes
import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Optional

try:
    import pythoncom
    import win32com.client
    HAS_WIN32 = True
except (ImportError, AttributeError):
    HAS_WIN32 = False
    pythoncom = None
    win32com = None

import config

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

# SAPI 5.x flags
SVSFlagsAsync = 1
SVSFPurgeBeforeSpeak = 2

def clean_text_for_speech(text: str) -> str:
    """
    Format and clean text for natural human speech synthesis.
    Preserves natural inflection, sentence pauses, and emotional rhythm.
    """
    if not text:
        return ""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Convert markdown headers to natural pauses
    text = re.sub(r"#+\s*(.+)", r"\1.", text)
    # Remove bold/italic markers while preserving the words
    text = re.sub(r"[*_~]+", "", text)
    # Convert markdown links [text](url) to just text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Convert bullet points into natural pause items
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove emojis and unprintable symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Normalize excessive punctuation but keep question marks and periods for inflection
    text = re.sub(r"\s+", " ", text).strip()
    return text

class NativeVoicePipeline:
    """
    Hyper-Realistic Neural Voice Pipeline.
    Uses Microsoft Edge Neural Voices (e.g. en-US-JennyNeural, en-US-GuyNeural)
    for expressive, emotional human speech, with seamless SAPI fallback.
    """
    _instance: Optional["NativeVoicePipeline"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
            
    def __init__(self):
        if self._initialized:
            return
        self.enabled = config.TTS_ENABLED
        self.voice = getattr(config, "TTS_VOICE", "en-US-JennyNeural")
        self.rate = config.TTS_RATE
        self.volume = config.TTS_VOLUME
        self._speaker = None
        self._current_audio_alias = "ragobot_speech"
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "winmm"):
            self._mci = ctypes.windll.winmm.mciSendStringW
        else:
            self._mci = None
        self._temp_dir = Path(tempfile.gettempdir()) / "ragobot_audio"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        
    def _get_sapi_speaker(self):
        """Thread-safe SAPI speaker initialization as fallback."""
        if not HAS_WIN32 or pythoncom is None or win32com is None:
            return None
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        if self._speaker is None:
            self._speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self._speaker.Rate = self.rate
            self._speaker.Volume = self.volume
        return self._speaker

    async def generate_neural_audio_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        """Generates raw MP3 bytes using Edge Neural TTS for frontend or playback."""
        target_voice = voice or self.voice
        cleaned = clean_text_for_speech(text)
        if not cleaned:
            return b""
        
        # Calculate rate offset string (e.g. "+5%")
        rate_str = f"{self.rate:+d}%" if self.rate != 0 else "+0%"
        communicate = edge_tts.Communicate(cleaned, voice=target_voice, rate=rate_str)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    def speak(self, text: str, voice: Optional[str] = None) -> None:
        """
        Dispatches neural speech asynchronously.
        Plays high-definition natural voice via Windows MCI with fallback to SAPI.
        """
        if not self.enabled or not text:
            return
            
        cleaned = clean_text_for_speech(text)
        if not cleaned:
            return
            
        target_voice = voice or self.voice
        
        def _dispatch_worker(speech_text: str):
            self.stop()
            if HAS_EDGE_TTS:
                try:
                    # Run async edge_tts in a dedicated loop
                    temp_mp3 = self._temp_dir / f"speech_{threading.get_ident()}.mp3"
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    rate_str = f"{self.rate:+d}%" if self.rate != 0 else "+0%"
                    communicate = edge_tts.Communicate(speech_text, voice=target_voice, rate=rate_str)
                    loop.run_until_complete(communicate.save(str(temp_mp3)))
                    loop.close()
                    
                    # Play via Windows MCI if available
                    if self._mci is not None:
                        alias = f"ragobot_{threading.get_ident()}"
                        self._current_audio_alias = alias
                        self._mci(f'open "{temp_mp3}" type mpegvideo alias {alias}', None, 0, None)
                        self._mci(f"play {alias} wait", None, 0, None)
                        self._mci(f"close {alias}", None, 0, None)
                    
                    try:
                        temp_mp3.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return
                except Exception:
                    # Fallback to SAPI
                    pass

            # Fallback to SAPI 5.x
            try:
                speaker = self._get_sapi_speaker()
                if speaker is not None:
                    speaker.Speak(speech_text, SVSFlagsAsync | SVSFPurgeBeforeSpeak)
            except Exception:
                pass
                
        thread = threading.Thread(target=_dispatch_worker, args=(cleaned,), daemon=True)
        thread.start()
        
    def stop(self) -> None:
        """Immediately abort any active neural audio or SAPI speech."""
        try:
            if self._mci is not None and self._current_audio_alias:
                self._mci(f"stop {self._current_audio_alias}", None, 0, None)
                self._mci(f"close {self._current_audio_alias}", None, 0, None)
        except Exception:
            pass
        try:
            if self._speaker is not None:
                self._speaker.Speak("", SVSFPurgeBeforeSpeak)
        except Exception:
            pass

    def toggle(self) -> bool:
        """Toggle TTS on/off and return new state."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.stop()
        return self.enabled

# Global singleton
tts_engine = NativeVoicePipeline()
