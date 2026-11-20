#!/bin/bash
# Supervisor for kl_realtime_v3.py — restart on any exit so cycles never stall.
export ENABLE_POLYMARKET_LIVE=true
cd /home/ubuntu/kalshi_superbot

# Prevent multiple instances
LOCKFILE="/tmp/kl_supervise.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "Another supervisor instance is running"; exit 1; }

while true; do
  python3 -u kl_realtime_v3.py >> /tmp/kl_realtime_v3.log 2>&1
  echo "$(date -u +%FT%TZ) supervisor: bot exited rc=$?, restarting in 5s" >> /tmp/kl_realtime_v3.log
  sleep 5
done
