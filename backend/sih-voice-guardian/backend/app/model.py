import numpy as np
import librosa

class AudioDeepfakeDetector:
    def __init__(self):
        self.sample_rate = 16000

    def extract_features(self, audio_bytes: bytes) -> np.ndarray:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_np) < 160:
            return np.zeros((20, 1))
        mfcc = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate, n_mfcc=20)
        return mfcc

    def predict(self, audio_bytes: bytes) -> dict:
        features = self.extract_features(audio_bytes)
        confidence_score = float(np.mean(np.abs(features)) % 1.0) 
        is_synthetic = confidence_score > 0.5

        return {
            "is_synthetic": is_synthetic,
            "confidence": round(confidence_score * 100, 2)
        }
