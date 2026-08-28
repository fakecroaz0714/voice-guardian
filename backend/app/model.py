import numpy as np
import librosa

class AudioDeepfakeDetector:
    def __init__(self):
        self.sample_rate = 16000

    def predict(self, raw_bytes: bytes) -> dict:
        try:
            # 1. Convert incoming PCM bytes to 16-bit float array
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_data) < 256:
                return {"is_synthetic": False, "confidence": 0.0, "status": "Silence / Ambient"}

            # 2. Voice Activity Detection (Ignore low background noise)
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.012:
                return {"is_synthetic": False, "confidence": 0.0, "status": "Silence / Ambient"}

            # 3. Frequency Band Ratio (Spectral Distribution)
            stft = np.abs(librosa.stft(audio_data, n_fft=512))
            
            # Vocal fundamentals (80 Hz to 1000 Hz) vs High TTS Artifacts (> 3000 Hz)
            low_energy = np.sum(stft[:35, :]) + 1e-6
            high_energy = np.sum(stft[100:, :]) + 1e-6
            freq_ratio = high_energy / low_energy

            # 4. Zero Crossing Rate (Phase continuity check)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

            # 5. Pitch Stability Analysis (F0 Micro-variations)
            # Human speech has natural pitch fluctuations; AI voices are unnaturally flat/uniform
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate, n_fft=512)
            valid_pitches = pitches[pitches > 0]
            pitch_std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0.0

            # 6. Multi-Factor Scoring Engine
            synthetic_score = 10.0  # Clean base human score

            # Calibrated Frequency Ratio (Prevents false alarms on 'S' and 'T' sounds)
            if freq_ratio > 0.42:
                synthetic_score += 45.0
            elif freq_ratio > 0.28:
                synthetic_score += 25.0

            # ZCR Phase Shift Anomaly
            if zcr > 0.22 or zcr < 0.015:
                synthetic_score += 20.0

            # Pitch Stability Anomaly (Synthetic voices maintain unnaturally low pitch variance)
            if len(valid_pitches) > 5 and pitch_std < 12.0:
                synthetic_score += 20.0

            threat_confidence = float(np.clip(synthetic_score, 5.0, 98.0))
            is_synthetic = threat_confidence >= 50.0

            return {
                "is_synthetic": is_synthetic,
                "confidence": round(threat_confidence, 2),
                "threat_confidence": round(threat_confidence, 2),
                "status": "Synthetic Voice Detected" if is_synthetic else "Authentic Voice"
            }

        except Exception as e:
            return {"is_synthetic": False, "confidence": 0.0, "threat_confidence": 0.0, "status": f"Processing Error: {str(e)}"}
