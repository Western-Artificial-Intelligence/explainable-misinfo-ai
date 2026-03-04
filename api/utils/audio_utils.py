"""
Audio analysis utilities for detecting silence and voice presence.

This module provides functions for:
- RMS (Root Mean Square) energy calculation
- Silence detection
- Audio level analysis
- Voice activity detection (VAD) support
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysis:
    """Result of audio analysis"""
    average_energy: float
    rms_peak: float
    rms_minimum: float
    silence_ratio: float
    is_silent: bool
    has_voice: bool
    confidence: float
    details: dict


class AudioAnalyzer:
    """Audio analysis for silence and voice detection"""
    
    # Thresholds for silence detection (RMS-based)
    SILENCE_THRESHOLD_RMS = 0.02  # ~-34 dB (very quiet)
    VOICE_THRESHOLD_RMS = 0.05  # ~-26 dB (conversational speech)
    
    # Energy thresholds
    MIN_ENERGY_DB = -50
    MAX_ENERGY_DB = 0
    
    # For detecting "no meaningful voice"
    VOICE_PRESENCE_RATIO = 0.1  # At least 10% of audio should be voice
    MIN_VOICE_DURATION_SEC = 0.5  # At least 0.5s of voice content
    
    @staticmethod
    def calculate_rms(audio_data: np.ndarray) -> float:
        """
        Calculate RMS (Root Mean Square) energy of audio.
        
        Args:
            audio_data: Audio samples (float32, range typically -1.0 to 1.0)
            
        Returns:
            RMS value (0 to 1)
        """
        if len(audio_data) == 0:
            return 0.0
        
        return float(np.sqrt(np.mean(audio_data ** 2)))

    @staticmethod
    def audio_to_db(rms: float, reference: float = 1.0) -> float:
        """
        Convert RMS to dB scale.
        
        Args:
            rms: RMS value
            reference: Reference value (typically 1.0)
            
        Returns:
            dB level
        """
        if rms <= 0:
            return float('-inf')
        
        return 20 * np.log10(rms / reference)

    @staticmethod
    def db_to_audio(db: float, reference: float = 1.0) -> float:
        """Convert dB to RMS"""
        return reference * (10 ** (db / 20))

    @staticmethod
    def detect_voice_frames(
        audio_data: np.ndarray,
        sample_rate: int,
        frame_duration_ms: int = 20,
        threshold_rms: float = None,
    ) -> Tuple[List[bool], float]:
        """
        Detect which frames contain voice/non-silence.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            frame_duration_ms: Frame length in milliseconds
            threshold_rms: RMS threshold for voice detection
            
        Returns:
            Tuple of (voice_flags, mean_silence_ratio)
        """
        if threshold_rms is None:
            threshold_rms = AudioAnalyzer.VOICE_THRESHOLD_RMS
        
        frame_samples = int(sample_rate * frame_duration_ms / 1000)
        voice_frames = []
        silence_count = 0
        
        for i in range(0, len(audio_data), frame_samples):
            frame = audio_data[i:i + frame_samples]
            if len(frame) == 0:
                continue
            
            rms = AudioAnalyzer.calculate_rms(frame)
            is_voice = rms > threshold_rms
            voice_frames.append(is_voice)
            
            if not is_voice:
                silence_count += 1
        
        silence_ratio = silence_count / len(voice_frames) if voice_frames else 1.0
        
        return voice_frames, silence_ratio

    @staticmethod
    def is_silent(
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        threshold_rms: float = None,
        min_voice_ratio: float = None,
        min_duration_sec: float = None,
    ) -> bool:
        """
        Determine if audio is essentially silent/no meaningful voice.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            threshold_rms: Voice threshold
            min_voice_ratio: Minimum voice presence ratio
            min_duration_sec: Minimum voice duration
            
        Returns:
            True if audio is silent/no voice
        """
        if threshold_rms is None:
            threshold_rms = AudioAnalyzer.VOICE_THRESHOLD_RMS
        
        if min_voice_ratio is None:
            min_voice_ratio = AudioAnalyzer.VOICE_PRESENCE_RATIO
        
        if min_duration_sec is None:
            min_duration_sec = AudioAnalyzer.MIN_VOICE_DURATION_SEC
        
        audio_duration = len(audio_data) / sample_rate
        
        # Very short audio is more likely to be silence
        if audio_duration < 0.5:
            return True
        
        # Calculate RMS energy
        rms = AudioAnalyzer.calculate_rms(audio_data)
        
        # If RMS is below threshold, definitely silent
        if rms < threshold_rms:
            return True
        
        # Detect voice frames
        voice_frames, silence_ratio = AudioAnalyzer.detect_voice_frames(
            audio_data,
            sample_rate,
            threshold_rms=threshold_rms
        )
        
        voice_ratio = 1 - silence_ratio
        voice_duration = voice_ratio * audio_duration
        
        # Check if voice presence meets minimum requirements
        if voice_ratio < min_voice_ratio or voice_duration < min_duration_sec:
            return True
        
        return False

    @staticmethod
    def analyze_audio(
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
    ) -> AudioAnalysis:
        """
        Comprehensive audio analysis.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            frame_duration_ms: Frame length for analysis
            
        Returns:
            AudioAnalysis object with detailed metrics
        """
        # Calculate overall RMS
        rms = AudioAnalyzer.calculate_rms(audio_data)
        rms_db = AudioAnalyzer.audio_to_db(rms)
        
        # Detect voice frames
        voice_frames, silence_ratio = AudioAnalyzer.detect_voice_frames(
            audio_data,
            sample_rate,
            frame_duration_ms=frame_duration_ms
        )
        
        # Calculate frame-level statistics
        frame_rms_values = []
        frame_samples = int(sample_rate * frame_duration_ms / 1000)
        
        for i in range(0, len(audio_data), frame_samples):
            frame = audio_data[i:i + frame_samples]
            if len(frame) > 0:
                frame_rms_values.append(AudioAnalyzer.calculate_rms(frame))
        
        rms_peak = max(frame_rms_values) if frame_rms_values else 0.0
        rms_minimum = min(frame_rms_values) if frame_rms_values else 0.0
        
        # Determine if silent
        is_silent = AudioAnalyzer.is_silent(audio_data, sample_rate)
        
        # Voice presence
        voice_ratio = 1 - silence_ratio
        has_voice = voice_ratio > AudioAnalyzer.VOICE_PRESENCE_RATIO
        
        # Confidence score
        if is_silent:
            confidence = min(1.0, silence_ratio + 0.2)
        elif has_voice:
            confidence = min(1.0, voice_ratio)
        else:
            confidence = 0.5
        
        details = {
            'rms_db': float(rms_db),
            'frame_count': len(voice_frames),
            'voice_frames': sum(1 for f in voice_frames if f),
            'silence_frames': sum(1 for f in voice_frames if not f),
            'duration_seconds': len(audio_data) / sample_rate,
            'voice_ratio': voice_ratio,
            'peak_rms_db': float(AudioAnalyzer.audio_to_db(rms_peak)),
            'min_rms_db': float(AudioAnalyzer.audio_to_db(rms_minimum)) if rms_minimum > 0 else float('-inf'),
        }
        
        return AudioAnalysis(
            average_energy=rms,
            rms_peak=rms_peak,
            rms_minimum=rms_minimum,
            silence_ratio=silence_ratio,
            is_silent=is_silent,
            has_voice=has_voice,
            confidence=confidence,
            details=details,
        )


def audio_bytes_to_numpy(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    num_channels: int = 1,
    sample_width: int = 2,  # 16-bit = 2 bytes
) -> np.ndarray:
    """
    Convert raw audio bytes to numpy array.
    
    Args:
        audio_bytes: Raw audio bytes
        sample_rate: Expected sample rate
        num_channels: Number of channels
        sample_width: Bytes per sample
        
    Returns:
        Numpy array of audio data (float32, normalized to -1..1)
    """
    # Convert bytes to int array
    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        dtype = np.int16
    
    audio_array = np.frombuffer(audio_bytes, dtype=dtype)
    
    # Normalize to float32 (-1..1 range)
    if dtype == np.int16:
        audio_float = audio_array.astype(np.float32) / 32768.0
    elif dtype == np.int32:
        audio_float = audio_array.astype(np.float32) / 2147483648.0
    else:
        audio_float = audio_array.astype(np.float32)
    
    # Handle multi-channel (mix down to mono if needed)
    if num_channels > 1:
        audio_float = audio_float.reshape(-1, num_channels)
        audio_float = np.mean(audio_float, axis=1)
    
    return audio_float
