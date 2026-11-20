#!/usr/bin/env python3
"""
heartbeat.py — Broadcast agent heartbeat to MemPalace logstream.

Usage: python3 scripts/heartbeat.py --cycles 42 --task "polymarket_backtest"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

AGENT_IDENTITY = "ubuntu:opencode:home"
STREAM = "project/kalshi_superbot"
ROOM = "status"
TOPIC = "agent-heartbeat"


def broadcast_heartbeat(cycles: int, task: str, pid: int = None) -> bool:
    """Broadcast a heartbeat event to MemPalace logstream."""
    if pid is None:
        pid = os.getpid()

    body = f"alive cycles={cycles} pid={pid} task={task}"

    cmd = [
        "mempalace", "logstream", "append",
        "--type", "status",
        "--stream", STREAM,
        "--room", ROOM,
        "--topic", TOPIC,
        "--from-agent", AGENT_IDENTITY,
        "--to-agent", "*",
        "--body", body,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Heartbeat broadcast: {body}")
            return True
        else:
            print(f"❌ Heartbeat failed: {result.stderr}", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Heartbeat timeout", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Heartbeat error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Broadcast agent heartbeat")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles completed")
    parser.add_argument("--task", type=str, default="idle", help="Current task name")
    parser.add_argument("--pid", type=int, default=None, help="Process ID (default: current)")
    args = parser.parse_args()

    success = broadcast_heartbeat(args.cycles, args.task, args.pid)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()