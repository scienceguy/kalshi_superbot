#!/usr/bin/env python3
"""
delegate.py — Delegate tasks to peer agents via MemPalace logstream.

Usage:
  python3 scripts/delegate.py --goal "Run Polymarket backtest" --branch backtest-v2 --base-commit abc123 --dod "EV/trade > 0.01"
  python3 scripts/delegate.py --correlation-id task_abc123 --status claimed
"""

import argparse
import json
import os
import subprocess
import sys
import uuid

AGENT_IDENTITY = "ubuntu:opencode:home"
STREAM = "project/kalshi_superbot"
ROOM = "delegation"


def delegate_task(goal: str, branch: str, base_commit: str, dod: str, topic: str = None, correlation_id: str = None) -> str:
    """Delegate a task to peer agents."""
    if correlation_id is None:
        correlation_id = f"task_{uuid.uuid4().hex[:8]}"

    if topic is None:
        topic = "delegated-task"

    body = f"goal: {goal}\nbranch: {branch}\nbase_commit: {base_commit}\ndefinition_of_done: {dod}"

    cmd = [
        "mempalace", "logstream", "append",
        "--type", "task.request",
        "--stream", STREAM,
        "--room", ROOM,
        "--topic", topic,
        "--correlation-id", correlation_id,
        "--from-agent", AGENT_IDENTITY,
        "--to-agent", AGENT_IDENTITY,  # Broadcast to all agents in project
        "--body", body,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"📤 Delegated task {correlation_id}: {goal}")
            print(f"   Branch: {branch}")
            print(f"   Base: {base_commit}")
            print(f"   DoD: {dod}")
            return correlation_id
        else:
            print(f"❌ Delegate failed: {result.stderr}", file=sys.stderr)
            return ""
    except Exception as e:
        print(f"❌ Delegate error: {e}", file=sys.stderr)
        return ""


def claim_task(correlation_id: str) -> bool:
    """Claim a delegated task."""
    # First check if already claimed by us
    cmd_check = [
        "mempalace", "logstream", "list",
        "--type", "task.reply",
        "--correlation-id", correlation_id,
        "--limit", "5",
        "--json",
    ]

    try:
        result = subprocess.run(cmd_check, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            events = json.loads(result.stdout)
            for event in events.get("events", []):
                if event.get("from_agent") == AGENT_IDENTITY and event.get("status") == "claimed":
                    print(f"⚠️  Already claimed by us: {correlation_id}")
                    return False
    except Exception:
        pass

    # Claim it
    body = f"Claimed by {AGENT_IDENTITY}"

    cmd = [
        "mempalace", "logstream", "append",
        "--type", "task.reply",
        "--stream", STREAM,
        "--room", ROOM,
        "--topic", "claim",
        "--correlation-id", correlation_id,
        "--from-agent", AGENT_IDENTITY,
        "--to-agent", AGENT_IDENTITY,
        "--body", body,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Claimed task: {correlation_id}")
            return True
        else:
            print(f"❌ Claim failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ Claim error: {e}", file=sys.stderr)
        return False


def deliver_patch(correlation_id: str, artifact_id: str, sha256: str, notes: str = "") -> bool:
    """Deliver a code patch for a task."""
    body = f"Patch delivered\nartifact_id: {artifact_id}\nsha256: {sha256}"
    if notes:
        body += f"\nnotes: {notes}"

    cmd = [
        "mempalace", "logstream", "append",
        "--type", "patch.ready",
        "--stream", STREAM,
        "--room", ROOM,
        "--topic", "patch",
        "--correlation-id", correlation_id,
        "--from-agent", AGENT_IDENTITY,
        "--to-agent", AGENT_IDENTITY,
        "--body", body,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"📦 Delivered patch for {correlation_id}: {artifact_id}")
            return True
        else:
            print(f"❌ Patch delivery failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ Patch error: {e}", file=sys.stderr)
        return False


def reply_task(correlation_id: str, status: str, message: str = "") -> bool:
    """Reply to a task with result."""
    body = f"status: {status}"
    if message:
        body += f"\n{message}"

    cmd = [
        "mempalace", "logstream", "append",
        "--type", "task.reply",
        "--stream", STREAM,
        "--room", ROOM,
        "--topic", "reply",
        "--correlation-id", correlation_id,
        "--from-agent", AGENT_IDENTITY,
        "--to-agent", AGENT_IDENTITY,
        "--body", body,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"📬 Replied to {correlation_id}: {status}")
            return True
        else:
            print(f"❌ Reply failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ Reply error: {e}", file=sys.stderr)
        return False


def list_tasks(status_filter: str = None) -> list:
    """List active tasks."""
    cmd = [
        "mempalace", "logstream", "list",
        "--type", "task.request",
        "--room", ROOM,
        "--limit", "20",
        "--json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            events = json.loads(result.stdout)
            tasks = []
            for event in events.get("events", []):
                correlation_id = event.get("correlation_id", "")
                body = event.get("body", "")
                if status_filter and status_filter not in body:
                    continue
                tasks.append({
                    "correlation_id": correlation_id,
                    "body": body,
                    "topic": event.get("topic", ""),
                    "timestamp": event.get("created_at", ""),
                })
            return tasks
        else:
            print(f"❌ List failed: {result.stderr}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"❌ List error: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Delegate tasks via MemPalace")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Delegate
    delegate_parser = subparsers.add_parser("delegate", help="Delegate a new task")
    delegate_parser.add_argument("--goal", required=True, help="Task objective")
    delegate_parser.add_argument("--branch", required=True, help="Git branch name")
    delegate_parser.add_argument("--base-commit", required=True, help="Base commit SHA")
    delegate_parser.add_argument("--dod", required=True, help="Definition of done")
    delegate_parser.add_argument("--topic", help="Topic/lane")
    delegate_parser.add_argument("--correlation-id", help="Custom correlation ID")

    # Claim
    claim_parser = subparsers.add_parser("claim", help="Claim a task")
    claim_parser.add_argument("--correlation-id", required=True, help="Task correlation ID")

    # Deliver patch
    patch_parser = subparsers.add_parser("patch", help="Deliver a patch")
    patch_parser.add_argument("--correlation-id", required=True, help="Task correlation ID")
    patch_parser.add_argument("--artifact-id", required=True, help="MemPalace artifact ID")
    patch_parser.add_argument("--sha256", required=True, help="Patch SHA256")
    patch_parser.add_argument("--notes", default="", help="Additional notes")

    # Reply
    reply_parser = subparsers.add_parser("reply", help="Reply to a task")
    reply_parser.add_argument("--correlation-id", required=True, help="Task correlation ID")
    reply_parser.add_argument("--status", choices=["applied", "failed", "blocked"], required=True, help="Result status")
    reply_parser.add_argument("--message", default="", help="Reply message")

    # List
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", help="Filter by status")

    args = parser.parse_args()

    if args.command == "delegate":
        cid = delegate_task(args.goal, args.branch, args.base_commit, args.dod, args.topic, args.correlation_id)
        if cid:
            print(cid)
        else:
            sys.exit(1)

    elif args.command == "claim":
        success = claim_task(args.correlation_id)
        sys.exit(0 if success else 1)

    elif args.command == "patch":
        success = deliver_patch(args.correlation_id, args.artifact_id, args.sha256, args.notes)
        sys.exit(0 if success else 1)

    elif args.command == "reply":
        success = reply_task(args.correlation_id, args.status, args.message)
        sys.exit(0 if success else 1)

    elif args.command == "list":
        tasks = list_tasks(args.status)
        if tasks:
            print(f"{'Correlation ID':<20} {'Topic':<20} {'Timestamp':<30} Goal")
            print("-" * 100)
            for t in tasks:
                goal_line = t['body'].split('\n')[0].replace('goal: ', '') if 'goal: ' in t['body'] else t['body'][:50]
                print(f"{t['correlation_id']:<20} {t['topic']:<20} {t['timestamp']:<30} {goal_line}")
        else:
            print("No tasks found")


if __name__ == "__main__":
    main()