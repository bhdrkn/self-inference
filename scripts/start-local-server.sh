#!/usr/bin/env bash
# Start the inference server locally and wait until it's ready.
# Writes a PID file so stop-local-server.sh can shut it down cleanly.
# Use MOCK_MODE=true for fast startup without loading a model.

PID_FILE=".server.pid"
LOG_FILE=".server.log"
TIMEOUT=300  # 5 minutes — first run downloads model weights (~2 GB)

# --- Check if already running ---
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Server is already running (PID $PID)."
        echo "Use scripts/stop-local-server.sh to stop it first."
        exit 1
    else
        echo "Removing stale PID file (process $PID no longer exists)."
        rm -f "$PID_FILE"
    fi
fi

# --- Start server ---
echo "Starting server..."
echo "(Logs: $LOG_FILE)"
.venv/bin/python src/server.py >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"

# --- Wait for startup ---
ELAPSED=0
LAST_SHOWN=""

while [ $ELAPSED -lt $TIMEOUT ]; do
    # Process died — startup failed
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: Server process exited unexpectedly. Last log output:"
        echo "---"
        tail -20 "$LOG_FILE"
        echo "---"
        rm -f "$PID_FILE"
        exit 1
    fi

    # Successful startup
    if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "Server ready on http://localhost:8000 (PID $SERVER_PID)"
        exit 0
    fi

    # Show the latest log line if it changed
    CURRENT_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null || true)
    if [ "$CURRENT_LINE" != "$LAST_SHOWN" ] && [ -n "$CURRENT_LINE" ]; then
        echo "[${ELAPSED}s] $CURRENT_LINE"
        LAST_SHOWN="$CURRENT_LINE"
    fi

    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

echo ""
echo "Timed out after ${TIMEOUT}s waiting for server to start."
echo "It may still be downloading model weights — check $LOG_FILE for progress."
echo "Leave it running or kill PID $SERVER_PID manually."
exit 1
