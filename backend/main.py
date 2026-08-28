import os
import wave
import time
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
from collections import deque
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(title="SIH26104 Voice Guardian Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset_uploads")
MODEL_PATH = os.path.join(BASE_DIR, "voice_guardian_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

os.makedirs(DATASET_DIR, exist_ok=True)

# 1. Initialize Whisper Model (Offline Speech-To-Text)
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

class VoiceAuthenticityClassifier(nn.Module):
    def __init__(self, input_dim=13):
        super(VoiceAuthenticityClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

device = torch.device("cpu")
model = VoiceAuthenticityClassifier(input_dim=13)
scaler = None
model_loaded = False

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        scaler = joblib.load(SCALER_PATH)
        model_loaded = True
        print(" PyTorch Offline Model successfully loaded.")
    except Exception as e:
        print(f" Model load warning: {e}")

def detect_gender_from_pitch(pcm_signal, sample_rate=16000):
    """Calculates Fundamental Frequency (F0) to determine Male vs Female voice."""
    fft_vals = np.abs(np.fft.rfft(pcm_signal))
    freqs = np.fft.rfftfreq(len(pcm_signal), 1.0 / sample_rate)
    
    # Restrict search range to human vocal fundamentals (70 Hz to 300 Hz)
    valid_idx = np.where((freqs >= 70) & (freqs <= 300))[0]
    if len(valid_idx) == 0:
        return "Unknown"
    
    peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
    f0 = freqs[peak_idx]

    if 70 <= f0 < 160:
        return f"Male (~{int(f0)} Hz)"
    elif 160 <= f0 <= 280:
        return f"Female (~{int(f0)} Hz)"
    return "Uncertain"

def extract_mfcc_simple(signal, sample_rate=16000, num_coefficients=13):
    fft_vals = np.abs(np.fft.rfft(signal))
    num_filters = 26
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin_indices = np.floor((len(signal) + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, int(np.floor(len(signal) / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_indices[m - 1]
        f_m = bin_indices[m]
        f_m_plus = bin_indices[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_indices[m - 1]) / (f_m - bin_indices[m - 1] + 1e-8)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_indices[m + 1] - k) / (bin_indices[m + 1] - f_m + 1e-8)

    filter_banks = np.dot(fbank, fft_vals)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    dct_features = np.zeros(num_coefficients)
    for n in range(num_coefficients):
        dct_features[n] = np.sum(filter_banks * np.cos(np.pi * n * (np.arange(num_filters) + 0.5) / num_filters))
    return dct_features

def analyze_audio_frame(pcm_signal):
    if len(pcm_signal) == 0:
        return 0.0, 0.0, 0.0, "Unknown"

    rms_energy = float(np.sqrt(np.mean(pcm_signal ** 2)))
    if rms_energy < 120.0:
        return 0.0, 0.0, 0.0, "Silence"

    zcr = float(np.mean(np.abs(np.diff(np.sign(pcm_signal)))) / 2.0)
    fft_vals = np.abs(np.fft.rfft(pcm_signal))
    freqs = np.fft.rfftfreq(len(pcm_signal), 1.0 / 16000.0)
    spectral_sum = float(np.sum(fft_vals))
    spectral_centroid = float((np.sum(freqs * fft_vals) / spectral_sum)) if spectral_sum > 0 else 0.0

    gender_label = detect_gender_from_pitch(pcm_signal)

    if model_loaded and scaler is not None:
        try:
            features = extract_mfcc_simple(pcm_signal)
            scaled_features = scaler.transform([features])
            tensor_input = torch.tensor(scaled_features, dtype=torch.float32).to(device)
            with torch.no_grad():
                prob = model(tensor_input).item()
            synthetic_prob = round(prob * 100.0, 1)
            return round(min(99.4, max(3.5, synthetic_prob)), 1), round(spectral_centroid, 1), round(zcr, 4), gender_label
        except Exception:
            pass

    return 10.0, round(spectral_centroid, 1), round(zcr, 4), gender_label

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    risk_mode = "high_risk"
    
    # 2-second buffer for STT (32,000 samples @ 16kHz)
    audio_buffer = deque(maxlen=32000)
    frame_counter = 0

    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message and message["text"]:
                try:
                    config = json.loads(message["text"])
                    risk_mode = str(config.get("risk_mode", "high_risk"))
                except Exception:
                    pass

            elif "bytes" in message and message["bytes"]:
                start_time = time.time()
                pcm_bytes = message["bytes"]
                pcm_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                audio_buffer.extend(pcm_chunk)

                analysis_window = np.array(audio_buffer, dtype=np.float32)
                threat_score, centroid, zcr, gender = analyze_audio_frame(analysis_window)

                # Run Whisper Transcription every ~1 second (every 8 frames of 2048 samples)
                transcription_text = ""
                frame_counter += 1
                if frame_counter % 8 == 0 and len(analysis_window) >= 16000:
                    norm_audio = (analysis_window / 32768.0).astype(np.float32)
                    segments, _ = whisper_model.transcribe(norm_audio, beam_size=1)
                    transcription_text = " ".join([seg.text for seg in segments]).strip()

                alert_threshold = 55.0 if risk_mode == "high_risk" else 75.0
                is_synthetic = bool(threat_score >= alert_threshold)
                latency = int((time.time() - start_time) * 1000) + 1

                response = {
                    "is_synthetic": is_synthetic,
                    "confidence": float(threat_score),
                    "spectral_centroid": float(centroid),
                    "zcr": float(zcr),
                    "gender": gender,
                    "transcript": transcription_text,
                    "server_latency_ms": int(latency),
                    "status": "Silence" if threat_score == 0 else "Active"
                }
                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Stream error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
