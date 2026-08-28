import numpy as np
import librosa

class AudioDeepfakeDetector:
    def __init__(self):
        self.sample_rate = 16000

    def extract_features(self, audio_bytes: bytes) -> tuple:
        # Convert raw 16-bit PCM bytes to floating-point array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Guard clause for short silent chunks
        if len(audio_np) < 512 or np.max(np.abs(audio_np)) < 0.01:
            return None, 0.0

        # Feature 1: MFCC / LFCC spectral representation
        mfccs = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate, n_mfcc=13)
        mfcc_variance = float(np.var(mfccs))

        # Feature 2: High-frequency spectral rollover (Vocoders miss continuous high-band noise)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=self.sample_rate)[0]
        mean_rolloff = float(np.mean(spectral_rolloff))

        return mfcc_variance, mean_rolloff

    def predict(self, audio_bytes: bytes) -> dict:
        mfcc_var, rolloff = self.extract_features(audio_bytes)
        
        if mfcc_var is None:
            return {
                "is_synthetic": False,
                "confidence": 0.0,
                "status": "Silence / Low Energy"
            }

        # Dynamic heuristic scoring simulating vocoder artifact boundaries
        # Synthetic speech exhibits unnaturally flat spectral variance in micro-windows
        anomaly_score = 0.0
        if mfcc_var < 15.0:
            anomaly_score += 0.55
        if rolloff > 6000.0:
            anomaly_score += 0.35

        confidence = min(99.9, max(5.0, anomaly_score * 100))
        is_synthetic = confidence > 50.0

        return {
            "is_synthetic": is_synthetic,
            "confidence": round(confidence, 1),
            "status": "Active Analysis"
        }

