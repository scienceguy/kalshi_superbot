"""
mempalace_bridge.py — MemPalace bridge for single-owner service access (T13-42, T13-43).

Provides a clean API for agents to use MemPalace without directly opening
logstreams. The bridge handles connection management, retries, and serialization.

Single-owner service pattern: one systemd service owns the palace; every agent
uses this bridge. Replace direct Logstream open with bridge calls.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("mempalace_bridge")


@dataclass
class BridgeConfig:
    """Configuration for the MemPalace bridge."""
    palace_dir: Path = field(default_factory=lambda: Path(os.environ.get("MEMPALACE_HOME", "/home/ubuntu/.mempalace")))
    cli_path: str = "mempalace"  # or "python3 -m mempalace"
    default_agent: str = "ubuntu:opencode:home"
    default_stream: str = "project/home"
    default_room: str = "delegation"
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0


class MemPalaceBridge:
    """Bridge to MemPalace CLI for agent coordination."""

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self._lock = threading.Lock()
        self._agent_identity = self.config.default_agent

    def set_agent_identity(self, agent: str):
        """Set the agent identity for this bridge instance."""
        self._agent_identity = agent

    def _run_cmd(self, args: List[str], input_data: str = "") -> Dict[str, Any]:
        """Run a mempalace CLI command and return parsed JSON result."""
        cmd = [self.config.cli_path] + args
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.config.palace_dir),
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=self.config.request_timeout,
                )
                if result.returncode == 0:
                    try:
                        return json.loads(result.stdout) if result.stdout.strip() else {"success": True}
                    except json.JSONDecodeError:
                        return {"success": True, "raw": result.stdout}
                else:
                    last_error = result.stderr
                    log.warning("MemPalace CLI attempt %d failed: %s", attempt + 1, result.stderr[:200])
            except subprocess.TimeoutExpired:
                last_error = "timeout"
                log.warning("MemPalace CLI attempt %d timed out", attempt + 1)
            except Exception as e:
                last_error = str(e)
                log.warning("MemPalace CLI attempt %d error: %s", attempt + 1, e)

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_backoff * (attempt + 1))

        return {"success": False, "error": last_error or "unknown error"}

    # ─── Logstream operations ──────────────────────────────────────────────────

    def logstream_append(
        self,
        event_type: str,
        stream: str = None,
        room: str = None,
        topic: str = None,
        correlation_id: str = None,
        status: str = "open",
        body: str = "",
        to_agent: str = "*",
        from_agent: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Append an event to the logstream."""
        args = [
            "logstream", "append",
            "--type", event_type,
            "--stream", stream or self.config.default_stream,
            "--room", room or self.config.default_room,
            "--to-agent", to_agent,
            "--from-agent", from_agent or self._agent_identity,
            "--status", status,
            "--body", body,
        ]
        if topic:
            args.extend(["--topic", topic])
        if correlation_id:
            args.extend(["--correlation-id", correlation_id])
        if metadata:
            args.extend(["--metadata", json.dumps(metadata)])

        return self._run_cmd(args)

    def logstream_list(
        self,
        to_agent: str = None,
        since_event_id: str = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List events from the logstream (inbox sweep)."""
        args = [
            "logstream", "list",
            "--limit", str(limit),
        ]
        if to_agent:
            args.extend(["--to-agent", to_agent])
        if since_event_id:
            args.extend(["--since-event-id", since_event_id])

        return self._run_cmd(args)

    def logstream_ack(self, event_id: str, status: str = "applied") -> Dict[str, Any]:
        """Acknowledge an event."""
        args = [
            "logstream", "ack",
            "--event-id", event_id,
            "--status", status,
        ]
        return self._run_cmd(args)

    def logstream_wait(
        self,
        correlation_id: str,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Wait for an event with a specific correlation_id."""
        args = [
            "logstream", "wait",
            "--correlation-id", correlation_id,
            "--timeout", str(int(timeout)),
        ]
        return self._run_cmd(args)

    # ─── KG operations ──────────────────────────────────────────────────────────

    def kg_add(self, key: str, value: str) -> Dict[str, Any]:
        """Add a KG fact."""
        args = ["kg", "add", "--key", key, "--value", value]
        return self._run_cmd(args)

    def kg_query(self, key: str) -> Dict[str, Any]:
        """Query a KG fact."""
        args = ["kg", "query", "--key", key]
        return self._run_cmd(args)

    def kg_supersede(self, key: str, value: str) -> Dict[str, Any]:
        """Supersede a KG fact (single-valued fact change)."""
        args = ["kg", "supersede", "--key", key, "--value", value]
        return self._run_cmd(args)

    def kg_invalidate(self, key: str) -> Dict[str, Any]:
        """Invalidate a KG fact (fact ended without replacement)."""
        args = ["kg", "invalidate", "--key", key]
        return self._run_cmd(args)

    # ─── Artifact operations ────────────────────────────────────────────────────

    def artifact_put(
        self,
        kind: str,
        content: str,
        created_by: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Store an artifact (e.g., patch, checkpoint)."""
        args = ["artifact", "put", "--kind", kind, "--created-by", created_by or self._agent_identity, "--content", content]
        if metadata:
            args.extend(["--metadata", json.dumps(metadata)])
        return self._run_cmd(args)

    def artifact_get(self, artifact_id: str) -> Dict[str, Any]:
        """Retrieve an artifact."""
        args = ["artifact", "get", "--id", artifact_id]
        return self._run_cmd(args)

    # ─── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Search the palace."""
        args = ["search", "--query", query, "--limit", str(limit)]
        return self._run_cmd(args)

    # ─── Convenience methods for T13-43 deliberate usage ────────────────────────

    def checkpoint(
        self,
        checkpoint_id: str,
        payload: Dict[str, Any],
        forecast_id: str = None,
        model_version: str = None,
    ) -> Dict[str, Any]:
        """Store an operational checkpoint with forecast/model metadata."""
        meta = {
            "type": "checkpoint",
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if forecast_id:
            meta["forecast_id"] = forecast_id
        if model_version:
            meta["model_version"] = model_version

        return self.artifact_put(
            kind="json",
            content=json.dumps(payload),
            created_by=self._agent_identity,
            metadata=meta,
        )

    def drawer_put(
        self,
        wing: str,
        room: str,
        content: str,
        source_file: str = None,
        added_by: str = None,
    ) -> Dict[str, Any]:
        """File durable narrative in a drawer."""
        return self._run_cmd([
            "add_drawer",
            "--wing", wing,
            "--room", room,
            "--content", content,
            "--source-file", source_file or "",
            "--added-by", added_by or self._agent_identity,
        ])

    def heartbeat(
        self,
        role: str,
        task: str = "",
        cycles: int = 0,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Send a status heartbeat."""
        body = f"status=active | agent={self._agent_identity} | ts={datetime.now(timezone.utc).isoformat()} | role={role}"
        if task:
            body += f" | task={task}"
        if cycles:
            body += f" | cycles={cycles}"

        meta = metadata or {}
        meta.update({"heartbeat": True, "role": role})

        return self.logstream_append(
            event_type="status",
            stream="project/home",
            room="status",
            to_agent="*",
            body=body,
            metadata=meta,
        )

    def delegate_task(
        self,
        goal: str,
        branch: str,
        base_commit: str,
        dod: str,
        correlation_id: str = None,
    ) -> Dict[str, Any]:
        """Delegate a task to another agent."""
        if correlation_id is None:
            correlation_id = f"task_{int(time.time())}"

        body = f"goal={goal}\nbranch={branch}\nbase_commit={base_commit}\ndod={dod}"

        return self.logstream_append(
            event_type="task.request",
            stream="project/home",
            room="delegation",
            topic="kalshi-superbot",
            correlation_id=correlation_id,
            status="open",
            body=body,
            to_agent="*",
        )

    def file_incident(
        self,
        incident_id: str,
        description: str,
        severity: str = "medium",
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """File an incident to MemPalace."""
        payload = {
            "incident_id": incident_id,
            "description": description,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self._agent_identity,
        }
        if metadata:
            payload.update(metadata)

        # Log to KG for querying
        self.kg_add(f"incident_{incident_id}", json.dumps(payload))

        # Also send status event
        return self.logstream_append(
            event_type="status",
            stream="project/home",
            room="status",
            topic="incident",
            to_agent="*",
            body=f"INCIDENT {incident_id}: {description}",
            metadata={"incident": True, "severity": severity},
        )


# ─── Global bridge instance ───────────────────────────────────────────────────

_bridge: Optional[MemPalaceBridge] = None


def get_bridge(config: Optional[BridgeConfig] = None) -> MemPalaceBridge:
    """Get the global MemPalace bridge singleton."""
    global _bridge
    if _bridge is None:
        _bridge = MemPalaceBridge(config)
    return _bridge


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    bridge = get_bridge()

    # Demo: send a heartbeat
    result = bridge.heartbeat(role="demo", task="testing bridge", cycles=1)
    print(f"Heartbeat: {result}")

    # Demo: file a checkpoint
    result = bridge.checkpoint("test_001", {"balance": 1000.0, "positions": 5}, forecast_id="fcst_001", model_version="v1.2.3")
    print(f"Checkpoint: {result}")

    # Demo: delegate a task
    result = bridge.delegate_task(
        goal="Run backtest",
        branch="backtest-v2",
        base_commit="abc123",
        dod="EV/trade > 0.01",
    )
    print(f"Delegate: {result}")

    # Demo: search
    result = bridge.search("kalshi superbot")
    print(f"Search: {result}")