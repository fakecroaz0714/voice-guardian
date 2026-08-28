import time
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.model import AudioDeepfakeDetector

app = FastAPI(title="SIH Voice Guardian Real-Time Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = AudioDeepfakeDetector()
score_buffer = deque(maxlen=6)

@app.get("/")
def read_root():
    return {"status": "Voice Guardian Deepfake Engine Online"}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    score_buffer.clear()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            start_time = time.time()
            
            result = detector.predict(data)
            
            if result["status"] != "Silence / Ambient":
                val = result.get("threat_confidence", result.get("confidence", 0.0))
                score_buffer.append(val)
                smoothed_score = sum(score_buffer) / len(score_buffer)
                
                result["confidence"] = round(smoothed_score, 2)
                result["threat_confidence"] = round(smoothed_score, 2)
                return_is_synthetic = smoothed_score >= 50.0
                result["is_synthetic"] = return_is_synthetic
                result["status"] = "ALERT: Synthetic Voice Detected!" if return_is_synthetic else "Authentic Voice"
            
            latency_ms = round((time.time() - start_time) * 1000, 2)
            result["server_latency_ms"] = latency_ms
            result["processing_latency"] = latency_ms
            
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("Client disconnected cleanly.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
