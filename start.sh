#!/bin/bash

echo "Starting SIH Voice Guardian..."

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Resolve Python from backend venv, root venv, or system python
if [ -f "$DIR/backend/venv/bin/python" ]; then
    PYTHON_BIN="$DIR/backend/venv/bin/python"
elif [ -f "$DIR/venv/bin/python" ]; then
    PYTHON_BIN="$DIR/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Start FastAPI Backend on Port 8000
cd "$DIR/backend"
"$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Start Frontend Server on Port 3000
cd "$DIR/frontend"
"$PYTHON_BIN" -m http.server 3000 &
FRONTEND_PID=$!

echo "---------------------------------------"
echo "Backend running on http://127.0.0.1:8000"
echo "Frontend running on http://127.0.0.1:3000"
echo "---------------------------------------"
echo "Press CTRL+C to stop both servers."

# Trap CTRL+C to kill both background processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait
