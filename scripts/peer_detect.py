#!/usr/bin/env python3
"""
peer_detect.py — Watch for peer agent heartbeats and wake dormant peers.

Usage: python3 scripts/peer_detect.py --watch
       python3 scripts/peer_detect.py --once
"""

import argparse
import json
import os
import subprocess
import sys
import time

AGENT_IDENTITY = "ubuntu:opencode:home"
STREAM = "project/kalshi_superbot"
ROOM = "status"
TOPIC = "agent-heartbeat"


def watch_heartbeats(once: bool = False) -> None:
    """Watch for peer heartbeats on the logstream."""
    cmd = [
        "mempalace", "logstream", "watch",
        "--agent", AGENT_IDENTITY,
        "--stream", STREAM,
        "--type", "status",
        "--room", ROOM,
        "--topic", TOPIC,
        "--json",
    ]

    print(f"[WATCH] Watching for peer heartbeats on {STREAM}/{ROOM}/{TOPIC}...")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Track our own PID to filter ourselves
        own_pid = str(os.getpid())
        
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                body = event.get("body", "")
                
                # Parse body to extract PID
                pid = None
                if "pid=" in body:
                    parts = body.split()
                    for part in parts:
                        if part.startswith("pid="):
                            pid = part[4:]
                            break
                
                # Skip our own PID
                if pid == own_pid:
                    continue

                # Found a peer heartbeat
                from_agent = event.get("from_agent", "")
                correlation_id = event.get("correlation_id")
                event_id = event.get("id", "")

                print(f"📡 Peer heartbeat from {from_agent} (pid={pid}): {body}")

                # Check if we share a correlation/topic with this peer
                # If so, and they're dormant, we could ping them
                # For now, just log the detection

            except json.JSONDecodeError:
                continue

            if once:
                break

    except KeyboardInterrupt:
        print("\n👋 Stopped watching")
    except Exception as e:
        print(f"❌ Watch error: {e}", file=sys.stderr)
    finally:
        if 'proc' in locals():
            proc.terminate()


def check_recent_heartbeats() -> list:
    """Check recent heartbeats from peers (one-time sweep)."""
    cmd = [
        "mempalace", "logstream", "list",
        "--stream", STREAM,
        "--type", "status",
        "--room", ROOM,
        "--topic", TOPIC,
        "--limit", "100",
        "--json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            events = json.loads(result.stdout)
            peers = []
            seen = set()  # Track unique (pid, task) pairs
            for event in events.get("events", []):
                from_agent = event.get("from_agent", "")
                body = event.get("body", "")
                
                # Parse body to extract PID and task
                # Format: "alive cycles=N pid=PID task=TASK"
                pid = None
                task = None
                if "pid=" in body and "task=" in body:
                    parts = body.split()
                    for part in parts:
                        if part.startswith("pid="):
                            pid = part[4:]
                        elif part.startswith("task="):
                            task = part[5:]
                
                # Create unique key for this peer instance
                peer_key = f"{pid}:{task}"
                
                # Include all unique PID/task combinations (including historical peers)
                if peer_key not in seen and pid is not None:
                    seen.add(peer_key)
                    peers.append({
                        "agent": from_agent,
                        "pid": pid,
                        "task": task,
                        "body": body,
                        "timestamp": event.get("created_at", ""),
                        "event_id": event.get("id", ""),
                    })
            return peers
        else:
            print(f"❌ List failed: {result.stderr}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"❌ Check error: {e}", file=sys.stderr)
        return []


def wake_peer(peer_agent: str, correlation_id: str, message: str) -> bool:
    """Send a wake-up ping to a dormant peer."""
    cmd = [
        "mempalace", "logstream", "append",
        "--type", "task.request",
        "--stream", STREAM,
        "--room", "delegation",
        "--topic", "peer-wake",
        "--correlation-id", correlation_id,
        "--from-agent", AGENT_IDENTITY,
        "--to-agent", peer_agent,
        "--body", message,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"🔔 Woke peer {peer_agent} with correlation {correlation_id}")
            return True
        else:
            print(f"❌ Wake failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ Wake error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Detect and wake peer agents")
    parser.add_argument("--watch", action="store_true", help="Continuously watch for heartbeats")
    parser.add_argument("--once", action="store_true", help="Check recent heartbeats once")
    parser.add_argument("--wake", type=str, help="Wake a specific peer agent")
    parser.add_argument("--correlation-id", type=str, help="Correlation ID for wake message")
    parser.add_argument("--message", type=str, default="Wake up - shared task needs attention", help="Wake message")
    args = parser.parse_args()

    if args.watch:
        watch_heartbeats(once=False)
    elif args.once:
        peers = check_recent_heartbeats()
        if peers:
            print(f"🔍 Found {len(peers)} recent peer heartbeats:")
            for p in peers:
                print(f"  - {p['agent']}: {p['body']} (at {p['timestamp']})")
        else:
            print("🔍 No recent peer heartbeats found")
    elif args.wake:
        if not args.correlation_id:
            print("❌ --correlation-id required for --wake", file=sys.stderr)
            sys.exit(1)
        wake_peer(args.wake, args.correlation_id, args.message)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()