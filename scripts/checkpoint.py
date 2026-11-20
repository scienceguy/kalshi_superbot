#!/usr/bin/env python3
"""
checkpoint.py — Manage checkpoint state for long-running tasks.

Usage:
  python3 scripts/checkpoint.py save --phase backtest_optimization --progress 45 --state '{"best_ev": 0.0123}'
  python3 scripts/checkpoint.py load --phase backtest_optimization
  python3 scripts/checkpoint.py list
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

AGENT_IDENTITY = "ubuntu:opencode:home"
STREAM = "project/kalshi_superbot"
ROOM = "checkpoints"


def save_checkpoint(phase: str, progress: int, state: dict, topic: str = None) -> bool:
    """Save a checkpoint to MemPalace logstream."""
    if topic is None:
        topic = phase

    body = {
        "phase": phase,
        "progress": progress,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": state,
    }

    body_str = json.dumps(body, separators=(",", ":"))

    cmd = [
        "mempalace", "logstream", "append",
        "--type", "status",
        "--stream", STREAM,
        "--room", ROOM,
        "--topic", topic,
        "--from-agent", AGENT_IDENTITY,
        "--to-agent", "*",
        "--body", body_str,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"💾 Checkpoint saved: {phase} ({progress}%)")
            return True
        else:
            print(f"❌ Checkpoint save failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ Checkpoint error: {e}", file=sys.stderr)
        return False


def load_latest_checkpoint(phase: str, topic: str = None) -> dict:
    """Load the latest checkpoint for a phase."""
    if topic is None:
        topic = phase

    cmd = [
        "mempalace", "logstream", "list",
        "--type", "status",
        "--room", ROOM,
        "--topic", topic,
        "--limit", "5",
        "--json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            events = json.loads(result.stdout)
            for event in events.get("events", []):
                body_str = event.get("body", "")
                try:
                    body = json.loads(body_str)
                    if body.get("phase") == phase:
                        print(f"📂 Loaded checkpoint: {phase} ({body.get('progress', 0)}%)")
                        return body
                except json.JSONDecodeError:
                    continue
            print(f"📂 No checkpoint found for phase: {phase}")
            return {}
        else:
            print(f"❌ Checkpoint load failed: {result.stderr}", file=sys.stderr)
            return {}
    except Exception as e:
        print(f"❌ Load error: {e}", file=sys.stderr)
        return {}


def list_checkpoints() -> list:
    """List all checkpoints."""
    cmd = [
        "mempalace", "logstream", "list",
        "--type", "status",
        "--room", ROOM,
        "--limit", "20",
        "--json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            events = json.loads(result.stdout)
            checkpoints = []
            for event in events.get("events", []):
                body_str = event.get("body", "")
                try:
                    body = json.loads(body_str)
                    if "phase" in body and "progress" in body:
                        checkpoints.append({
                            "phase": body["phase"],
                            "progress": body["progress"],
                            "timestamp": body.get("timestamp", ""),
                            "topic": event.get("topic", ""),
                            "event_id": event.get("id", ""),
                        })
                except json.JSONDecodeError:
                    continue
            return checkpoints
        else:
            print(f"❌ List failed: {result.stderr}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"❌ List error: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Manage task checkpoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Save
    save_parser = subparsers.add_parser("save", help="Save a checkpoint")
    save_parser.add_argument("--phase", required=True, help="Phase name")
    save_parser.add_argument("--progress", type=int, required=True, help="Progress percentage (0-100)")
    save_parser.add_argument("--state", type=str, default="{}", help="State JSON string")
    save_parser.add_argument("--topic", type=str, help="Topic/lane (default: phase name)")

    # Load
    load_parser = subparsers.add_parser("load", help="Load latest checkpoint")
    load_parser.add_argument("--phase", required=True, help="Phase name")
    load_parser.add_argument("--topic", type=str, help="Topic/lane (default: phase name)")

    # List
    subparsers.add_parser("list", help="List all checkpoints")

    args = parser.parse_args()

    if args.command == "save":
        try:
            state = json.loads(args.state)
        except json.JSONDecodeError:
            print("❌ Invalid JSON for --state", file=sys.stderr)
            sys.exit(1)
        success = save_checkpoint(args.phase, args.progress, state, args.topic)
        sys.exit(0 if success else 1)

    elif args.command == "load":
        checkpoint = load_latest_checkpoint(args.phase, args.topic)
        if checkpoint:
            print(json.dumps(checkpoint, indent=2))
        else:
            print("{}")

    elif args.command == "list":
        checkpoints = list_checkpoints()
        if checkpoints:
            print(f"{'Phase':<30} {'Progress':>8} {'Timestamp':<30} {'Topic':<20}")
            print("-" * 90)
            for cp in checkpoints:
                print(f"{cp['phase']:<30} {cp['progress']:>7}%  {cp['timestamp']:<30} {cp['topic']:<20}")
        else:
            print("No checkpoints found")


if __name__ == "__main__":
    main()