#!/usr/bin/env python3
"""
inbox_sweep.py — Sweep and process MemPalace inbox events.

Usage:
  python3 scripts/inbox_sweep.py --once
  python3 scripts/inbox_sweep.py --watch
  python3 scripts/inbox_sweep.py --ack <event_id> --status applied
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

AGENT_IDENTITY = "ubuntu:opencode:home"
STREAM = "project/kalshi_superbot"

# Cursor file to track last processed event
CURSOR_FILE = "/home/ubuntu/kalshi_superbot/.inbox_cursor"


def get_cursor() -> str:
    """Get the last processed event ID cursor."""
    try:
        with open(CURSOR_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def set_cursor(event_id: str) -> None:
    """Save the last processed event ID cursor."""
    with open(CURSOR_FILE, "w") as f:
        f.write(event_id)


def sweep_inbox(once: bool = True, limit: int = 100) -> list:
    """Sweep inbox for events addressed to this agent."""
    cursor = get_cursor()

    cmd = [
        "mempalace", "logstream", "list",
        "--stream", STREAM,
        "--to-agent", AGENT_IDENTITY,
        "--limit", str(limit),
        "--json",
    ]

    if cursor:
        cmd.extend(["--since-event-id", cursor])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            events = json.loads(result.stdout)
            return events.get("events", [])
        else:
            print(f"❌ Inbox sweep failed: {result.stderr}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"❌ Sweep error: {e}", file=sys.stderr)
        return []


def apply_patch(artifact_id: str) -> bool:
    """Download and apply a patch from MemPalace artifact.
    
    Returns True if applied successfully, False otherwise.
    """
    try:
        # Get the artifact
        cmd = ["mempalace", "artifact", "get", artifact_id, "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"  ❌ Failed to get artifact {artifact_id}: {result.stderr}")
            return False
        
        artifact = json.loads(result.stdout)
        content = artifact.get("content", "")
        sha256 = artifact.get("sha256", "")
        
        # Verify sha256
        if sha256:
            computed = hashlib.sha256(content.encode()).hexdigest()
            if computed != sha256:
                print(f"  ❌ SHA256 mismatch: expected {sha256}, got {computed}")
                return False
        
        # Parse patch - expected format: JSON with file_path and diff
        try:
            patch_data = json.loads(content)
            file_path = patch_data.get("file_path", "")
            diff = patch_data.get("diff", "")
            
            if not file_path or not diff:
                print(f"  ❌ Invalid patch format: missing file_path or diff")
                return False
            
            # Apply patch using git apply
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(diff)
                patch_file = f.name
            
            try:
                # Check if we can apply cleanly
                check_cmd = ["git", "apply", "--check", patch_file]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True, cwd="/home/ubuntu/kalshi_superbot")
                if check_result.returncode != 0:
                    print(f"  ❌ Patch does not apply cleanly: {check_result.stderr}")
                    return False
                
                # Apply the patch
                apply_cmd = ["git", "apply", patch_file]
                apply_result = subprocess.run(apply_cmd, capture_output=True, text=True, cwd="/home/ubuntu/kalshi_superbot")
                if apply_result.returncode != 0:
                    print(f"  ❌ Patch apply failed: {apply_result.stderr}")
                    return False
                
                print(f"  ✅ Applied patch to {file_path}")
                return True
                
            finally:
                try:
                    os.unlink(patch_file)
                except:
                    pass
                    
        except json.JSONDecodeError:
            # Maybe it's a raw diff?
            print(f"  ❌ Patch content is not valid JSON")
            return False
            
    except Exception as e:
        print(f"  ❌ Patch application error: {e}")
        return False


def process_task_request(event: dict) -> bool:
    """Process a task.request event - accept and track the task."""
    event_id = event.get("id", "")
    body = event.get("body", "")
    correlation_id = event.get("correlation_id", "")
    from_agent = event.get("from_agent", "")
    
    print(f"  📋 Task request from {from_agent}: {body[:200]}...")
    
    # Try to parse task details from body
    # Expected format: "TASK: <goal>\nBRANCH: <branch>\nBASE_COMMIT: <sha>\nDOD: <definition of done>"
    goal = ""
    branch = ""
    base_commit = ""
    dod = ""
    
    for line in body.split("\n"):
        if line.startswith("TASK:"):
            goal = line[5:].strip()
        elif line.startswith("BRANCH:"):
            branch = line[7:].strip()
        elif line.startswith("BASE_COMMIT:"):
            base_commit = line[12:].strip()
        elif line.startswith("DOD:"):
            dod = line[4:].strip()
    
    if not goal:
        goal = body[:200]  # fallback
    
    # Log the task acceptance
    note = f"Accepted task: {goal[:100]}"
    if branch:
        note += f" | branch: {branch}"
    if correlation_id:
        note += f" | corr: {correlation_id}"
    
    # Ack as claimed
    return ack_event(event_id, "claimed", note)


def process_patch_ready(event: dict) -> bool:
    """Process a patch.ready event - download and apply the patch."""
    event_id = event.get("id", "")
    body = event.get("body", "")
    correlation_id = event.get("correlation_id", "")
    
    print(f"  📦 Patch ready: {body[:200]}...")
    
    # Extract artifact_id from body (expected format: "ARTIFACT: <id>")
    artifact_id = ""
    for line in body.split("\n"):
        if line.startswith("ARTIFACT:"):
            artifact_id = line[9:].strip()
            break
    
    if not artifact_id:
        # Maybe the body IS the artifact_id?
        if body.strip():
            artifact_id = body.strip()
    
    if not artifact_id:
        print(f"  ❌ No artifact_id found in patch.ready event")
        return ack_event(event_id, "failed", "No artifact_id in patch.ready")
    
    # Apply the patch
    if apply_patch(artifact_id):
        return ack_event(event_id, "applied", f"Applied patch from artifact {artifact_id}")
    else:
        return ack_event(event_id, "failed", f"Failed to apply patch from artifact {artifact_id}")


def process_event(event: dict) -> bool:
    """Process a single inbox event. Returns True if handled."""
    event_id = event.get("id", "")
    event_type = event.get("type", "")
    body = event.get("body", "")
    from_agent = event.get("from_agent", "")
    correlation_id = event.get("correlation_id", "")
    topic = event.get("topic", "")
    room = event.get("room", "")

    print(f"📨 Processing event {event_id[:8]}... type={event_type} from={from_agent} topic={topic} room={room}")

    # Handle different event types
    if event_type == "task.request":
        return process_task_request(event)

    elif event_type == "task.reply":
        print(f"  📬 Task reply: {body[:100]}...")
        return ack_event(event_id, "applied", f"Reply received")

    elif event_type == "patch.ready":
        return process_patch_ready(event)

    elif event_type == "status":
        if room == "checkpoints":
            print(f"  📍 Checkpoint: {body[:100]}...")
        elif topic == "agent-heartbeat":
            print(f"  💓 Peer heartbeat: {body[:100]}...")
        return True  # Status events don't need ack

    else:
        print(f"  ❓ Unknown event type: {event_type}")
        return ack_event(event_id, "applied", f"Unknown event type acknowledged")

    return True


def ack_event(event_id: str, status: str, note: str = "") -> bool:
    """Acknowledge an event."""
    cmd = [
        "mempalace", "logstream", "ack",
        event_id,
        "--status", status,
    ]
    if note:
        cmd.extend(["--note", note])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✅ Acked as {status}")
            return True
        else:
            print(f"  ❌ Ack failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"  ❌ Ack error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Sweep and process MemPalace inbox")
    parser.add_argument("--once", action="store_true", help="Sweep once and exit")
    parser.add_argument("--watch", action="store_true", help="Continuously watch inbox")
    parser.add_argument("--ack", type=str, help="Manually ack an event ID")
    parser.add_argument("--status", type=str, choices=["applied", "failed", "blocked", "claimed"], default="applied", help="Ack status")
    parser.add_argument("--note", type=str, default="", help="Ack note")
    parser.add_argument("--limit", type=int, default=50, help="Max events to fetch")

    args = parser.parse_args()

    if args.ack:
        success = ack_event(args.ack, args.status, args.note)
        sys.exit(0 if success else 1)

    if args.watch:
        print(f"👁️  Watching inbox for {AGENT_IDENTITY}...")
        try:
            while True:
                events = sweep_inbox(once=True, limit=args.limit)
                for event in events:
                    event_id = event.get("id", "")
                    if event_id:
                        process_event(event)
                        set_cursor(event_id)
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n👋 Stopped watching")
        except Exception as e:
            print(f"❌ Watch error: {e}", file=sys.stderr)
    else:
        # Default: sweep once
        events = sweep_inbox(once=True, limit=args.limit)
        if events:
            print(f"📥 Found {len(events)} inbox events")
            for event in events:
                event_id = event.get("id", "")
                if event_id:
                    process_event(event)
                    set_cursor(event_id)
        else:
            print("📭 Inbox empty")


if __name__ == "__main__":
    import time
    main()