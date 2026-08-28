import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.model import AudioDeepfakeDetector

app = FastAPI(title="Voice Cloning Detection Engine")
detector = AudioDeepfakeDetector()

@app.get("/")
def root():
    return {"status": "Voice Guardian Engine Online"}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to real-time audio pipeline.")
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            start_time = time.time()
            result = detector.predict(audio_chunk)
            processing_time_ms = round((time.time() - start_time) * 1000, 2)
            result["server_latency_ms"] = processing_time_ms
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print("Client disconnected.")
