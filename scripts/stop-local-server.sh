#!/usr/bin/env bash
# Stop the locally running inference server.
# Reads the PID file written by start-local-server.sh.

PID_FILE=".server.pid"

# --- No PID file ---
if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE — server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

# --- Process already gone ---
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running. Removing stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

# --- Graceful stop ---
echo "Stopping server (PID $PID)..."
kill "$PID"

# Wait up to 10s for graceful shutdown
for i in $(seq 1 10); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Server stopped."
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# --- Force kill if still alive ---
echo "Server did not stop gracefully — force killing..."
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Server force stopped."
