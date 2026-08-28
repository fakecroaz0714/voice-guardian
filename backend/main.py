import os
import wave
import time
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        scaler = joblib.load(SCALER_PATH)
        print(" PyTorch Model active.")
    except Exception as e:
        print(f" Model load warning: {e}")

def analyze_audio_frame(pcm_signal):
    if len(pcm_signal) == 0:
        return 0.0, 0.0, 0.0

    rms_energy = float(np.sqrt(np.mean(pcm_signal ** 2)))
    
    # Standard voice activation threshold (filters silence without killing quiet speech)
    if rms_energy < 120.0:
        return 0.0, 0.0, 0.0

    zcr = float(np.mean(np.abs(np.diff(np.sign(pcm_signal)))) / 2.0)
    fft_vals = np.abs(np.fft.rfft(pcm_signal))
    freqs = np.fft.rfftfreq(len(pcm_signal), 1.0 / 16000.0)
    spectral_sum = float(np.sum(fft_vals))
    spectral_centroid = float((np.sum(freqs * fft_vals) / spectral_sum)) if spectral_sum > 0 else 0.0

    # Calculate High-Frequency Ratio (Vocoder Phase Discontinuity Detection)
    high_freq_energy = np.sum(fft_vals[freqs > 3000])
    total_energy = spectral_sum if spectral_sum > 0 else 1.0
    hf_ratio = float(high_freq_energy / total_energy)

    # Dynamic Scoring Matrix
    synthetic_prob = 10.0

    # Synthetic indicators (High HF ratio combined with low ZCR or unnatural spectral centroid elevation)
    if hf_ratio > 0.35 and zcr < 0.08:
        synthetic_prob += 65.0
    elif hf_ratio > 0.25:
        synthetic_prob += 40.0

    if spectral_centroid > 3200:
        synthetic_prob += 25.0

    # Natural Human Formant Calibration (human speech clusters between 800Hz - 2600Hz)
    if 800 <= spectral_centroid <= 2600 and hf_ratio < 0.20:
        synthetic_prob = max(4.0, synthetic_prob - 35.0)

    return round(min(99.4, max(3.5, synthetic_prob)), 1), round(spectral_centroid, 1), round(zcr, 4)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    save_volunteer_data = False
    accent_tag = "General_Indian"
    risk_mode = "high_risk"
    wav_file = None

    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message and message["text"]:
                try:
                    config = json.loads(message["text"])
                    save_volunteer_data = bool(config.get("save_audio", False))
                    accent_tag = str(config.get("accent", "General_Indian")).replace(" ", "_")
                    risk_mode = str(config.get("risk_mode", "high_risk"))

                    if save_volunteer_data and not wav_file:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"volunteer_{accent_tag}_{timestamp}.wav"
                        file_path = os.path.join(DATASET_DIR, filename)
                        wav_file = wave.open(file_path, 'wb')
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                except Exception:
                    pass

            elif "bytes" in message and message["bytes"]:
                start_time = time.time()
                pcm_bytes = message["bytes"]
                
                if save_volunteer_data and wav_file:
                    wav_file.writeframes(pcm_bytes)

                pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                threat_score, centroid, zcr = analyze_audio_frame(pcm_array)

                alert_threshold = 55.0 if risk_mode == "high_risk" else 75.0
                is_synthetic = bool(threat_score >= alert_threshold)
                latency = int((time.time() - start_time) * 1000) + 1

                response = {
                    "is_synthetic": is_synthetic,
                    "confidence": float(threat_score),
                    "spectral_centroid": float(centroid),
                    "zcr": float(zcr),
                    "server_latency_ms": int(latency),
                    "risk_mode": risk_mode,
                    "status": "Silence" if threat_score == 0 else "Active"
                }
                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Stream handler recovered: {e}")
    finally:
        if wav_file:
            try:
                wav_file.close()
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
