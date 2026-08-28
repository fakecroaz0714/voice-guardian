#!/bin/bash

echo "Starting SIH Voice Guardian..."

# Start FastAPI Backend on Port 8000
cd ~/sih-voice-guardian/backend
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Start Frontend Server on Port 3000
cd ~/sih-voice-guardian/frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!

echo "---------------------------------------"
echo "Backend running on http://127.0.0.1:8000"
echo "Frontend running on http://127.0.0.1:3000"
echo "---------------------------------------"
echo "Press CTRL+C to stop both servers."

# Trap CTRL+C to kill both background processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait
