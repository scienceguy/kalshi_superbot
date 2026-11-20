"""
agent_protocol.py — Inter-agent collaboration protocol (T13-44).

Roles: data_feed, quant_forecaster, poly_trader, kalshi_trader,
risk_officer (binding veto), ops_n8n, opencode_dev (code changes via commits only).

Topics: signal.proposal, risk.veto, forecast.published, param.change.request,
incident, sync.health, agent.checkpoint.

Envelope: {id, correlation_id, from, to, topic, ts, ttl_ms, payload, ack_required}
Hot-path messages ack within 2s; heartbeat every 5s; missed heartbeat = incident.

Rules: nobody edits live config at runtime. Parameter change = proposal ->
risk_officer approval -> commit -> new config_hash -> restart/hot-reload.
Sub-second coordination stays in-process; logstream is durable mirror.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

log = logging.getLogger("agent_protocol")


class AgentRole(Enum):
    """Defined agent roles."""
    DATA_FEED = "data_feed"
    QUANT_FORECASTER = "quant_forecaster"
    POLY_TRADER = "poly_trader"
    KALSHI_TRADER = "kalshi_trader"
    RISK_OFFICER = "risk_officer"  # binding veto
    OPS_N8N = "ops_n8n"
    OPENCODE_DEV = "opencode_dev"  # code changes via commits only


class Topic(Enum):
    """Message topics."""
    AGENT_CHECKPOINT = "agent.checkpoint"
    SIGNAL_PROPOSAL = "signal.proposal"
    RISK_VETO = "risk.veto"
    FORECAST_PUBLISHED = "forecast.published"
    PARAM_CHANGE_REQUEST = "param.change.request"
    INCIDENT = "incident"
    SYNC_HEALTH = "sync.health"


class MessageStatus(Enum):
    """Message delivery status."""
    PENDING = "pending"
    SENT = "sent"
    ACKED = "acked"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Envelope:
    """Message envelope for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    from_agent: str = ""
    to_agent: str = ""  # "*" for broadcast
    topic: Topic = Topic.AGENT_CHECKPOINT
    ts: float = field(default_factory=time.time)
    ttl_ms: int = 30000  # 30 second default TTL
    payload: Dict[str, Any] = field(default_factory=dict)
    ack_required: bool = False
    status: MessageStatus = MessageStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "topic": self.topic.value,
            "ts": self.ts,
            "ttl_ms": self.ttl_ms,
            "payload": self.payload,
            "ack_required": self.ack_required,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Envelope":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            correlation_id=data.get("correlation_id", ""),
            from_agent=data.get("from", ""),
            to_agent=data.get("to", ""),
            topic=Topic(data.get("topic", "agent.checkpoint")),
            ts=data.get("ts", time.time()),
            ttl_ms=data.get("ttl_ms", 30000),
            payload=data.get("payload", {}),
            ack_required=data.get("ack_required", False),
            status=MessageStatus(data.get("status", "pending")),
        )

    def is_expired(self) -> bool:
        return (time.time() - self.ts) * 1000 > self.ttl_ms


# ─── Protocol handlers ────────────────────────────────────────────────────────

class ProtocolHandler:
    """Handles sending/receiving protocol messages via MemPalace bridge."""

    def __init__(self, bridge, agent_id: str, role: AgentRole):
        self.bridge = bridge
        self.agent_id = agent_id
        self.role = role
        self._handlers: Dict[Topic, List[Callable]] = {}
        self._pending_acks: Dict[str, Envelope] = {}
        self._lock = threading.Lock()

    def register_handler(self, topic: Topic, handler: Callable[[Envelope], None]):
        """Register a handler for a topic."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)

    def send(
        self,
        topic: Topic,
        payload: Dict[str, Any],
        to_agent: str = "*",
        correlation_id: str = None,
        ack_required: bool = False,
        ttl_ms: int = 30000,
    ) -> Envelope:
        """Send a protocol message."""
        envelope = Envelope(
            correlation_id=correlation_id or f"{topic.value}_{int(time.time())}",
            from_agent=self.agent_id,
            to_agent=to_agent,
            topic=topic,
            ts=time.time(),
            ttl_ms=ttl_ms,
            payload=payload,
            ack_required=ack_required,
        )

        # Send via MemPalace logstream
        self.bridge.logstream_append(
            event_type="protocol",
            stream="project/home",
            room="delegation",
            topic=topic.value,
            correlation_id=envelope.correlation_id,
            status="open",
            body=json.dumps(envelope.to_dict()),
            to_agent=to_agent,
            from_agent=self.agent_id,
            metadata={"protocol": True, "role": self.role.value},
        )

        if ack_required:
            with self._lock:
                self._pending_acks[envelope.id] = envelope

        log.debug("Sent %s to %s (correlation=%s)", topic.value, to_agent, envelope.correlation_id)
        return envelope

    def broadcast(self, topic: Topic, payload: Dict[str, Any], **kwargs) -> Envelope:
        """Broadcast to all agents."""
        return self.send(topic, payload, to_agent="*", **kwargs)

    def handle_inbox(self, events: List[Dict[str, Any]]):
        """Process inbox events."""
        for event in events:
            try:
                body = event.get("body", "")
                if isinstance(body, str):
                    data = json.loads(body)
                else:
                    data = body

                envelope = Envelope.from_dict(data)

                # Check if expired
                if envelope.is_expired():
                    log.debug("Message %s expired", envelope.id)
                    continue

                # Handle acks
                if envelope.topic == Topic.AGENT_CHECKPOINT and "ack" in envelope.payload:
                    self._handle_ack(envelope)
                    continue

                # Route to handlers
                handlers = self._handlers.get(envelope.topic, [])
                for handler in handlers:
                    try:
                        handler(envelope)
                    except Exception as e:
                        log.error("Handler error for %s: %s", envelope.topic.value, e)

                # Send ack if required
                if envelope.ack_required:
                    self._send_ack(envelope)

            except Exception as e:
                log.error("Failed to process inbox event: %s", e)

    def _handle_ack(self, envelope: Envelope):
        """Handle an acknowledgment."""
        ack_id = envelope.payload.get("ack")
        with self._lock:
            if ack_id in self._pending_acks:
                self._pending_acks[ack_id].status = MessageStatus.ACKED
                del self._pending_acks[ack_id]

    def _send_ack(self, envelope: Envelope):
        """Send acknowledgment for a received message."""
        ack_envelope = Envelope(
            correlation_id=envelope.correlation_id,
            from_agent=self.agent_id,
            to_agent=envelope.from_agent,
            topic=Topic.AGENT_CHECKPOINT,
            payload={"ack": envelope.id},
        )
        self.bridge.logstream_append(
            event_type="protocol",
            stream="project/home",
            room="delegation",
            topic=Topic.AGENT_CHECKPOINT.value,
            correlation_id=envelope.correlation_id,
            status="ack",
            body=json.dumps(ack_envelope.to_dict()),
            to_agent=envelope.from_agent,
            from_agent=self.agent_id,
        )

    def check_pending_acks(self, timeout_ms: int = 2000) -> List[Envelope]:
        """Check for pending acks that have timed out."""
        timed_out = []
        now = time.time()
        with self._lock:
            for msg_id, envelope in list(self._pending_acks.items()):
                if (now - envelope.ts) * 1000 > timeout_ms:
                    envelope.status = MessageStatus.EXPIRED
                    timed_out.append(envelope)
                    del self._pending_acks[msg_id]
        return timed_out


# ─── Heartbeat manager ────────────────────────────────────────────────────────

class HeartbeatManager:
    """Manages agent heartbeats and detects dead agents."""

    def __init__(self, bridge, agent_id: str, role: AgentRole, interval: float = 5.0):
        self.bridge = bridge
        self.agent_id = agent_id
        self.role = role
        self.interval = interval
        self._last_heartbeat: Dict[str, float] = {}
        self._dead_threshold = interval * 3  # 3 missed heartbeats = dead
        self._running = False
        self._thread = None

    def start(self):
        """Start heartbeat loop."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"heartbeat-{self.agent_id}")
        self._thread.start()
        log.info("Heartbeat manager started for %s (%s)", self.agent_id, self.role.value)

    def stop(self):
        """Stop heartbeat loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("Heartbeat manager stopped for %s", self.agent_id)

    def _run(self):
        while self._running:
            self._send_heartbeat()
            self._check_peers()
            time.sleep(self.interval)

    def _send_heartbeat(self):
        """Send heartbeat via bridge."""
        self.bridge.heartbeat(
            role=self.role.value,
            task="running",
            cycles=0,
            metadata={"agent_id": self.agent_id, "role": self.role.value},
        )

    def _check_peers(self):
        """Check for dead peers (would query MemPalace for last heartbeats)."""
        # In a full implementation, this would query MemPalace for other agents' heartbeats
        # For now, we just log our own status
        pass

    def record_peer_heartbeat(self, peer_id: str, timestamp: float = None):
        """Record a peer's heartbeat timestamp."""
        self._last_heartbeat[peer_id] = timestamp or time.time()

    def is_peer_alive(self, peer_id: str) -> bool:
        """Check if a peer is alive based on last heartbeat."""
        last = self._last_heartbeat.get(peer_id, 0)
        return (time.time() - last) < self._dead_threshold


# ─── Parameter change protocol ────────────────────────────────────────────────

class ParameterChangeManager:
    """Manages parameter change proposals with risk officer approval."""

    def __init__(self, bridge, agent_id: str):
        self.bridge = bridge
        self.agent_id = agent_id
        self._pending_changes: Dict[str, Dict] = {}

    def propose_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        rationale: str,
        correlation_id: str = None,
    ) -> str:
        """Propose a parameter change (requires risk officer approval)."""
        if correlation_id is None:
            correlation_id = f"param_change_{param_name}_{int(time.time())}"

        proposal = {
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "rationale": rationale,
            "proposer": self.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "proposed",
            "approvals": [],
        }

        self._pending_changes[correlation_id] = proposal

        # Send to risk officer
        self.bridge.logstream_append(
            event_type="task.request",
            stream="project/home",
            room="delegation",
            topic=Topic.PARAM_CHANGE_REQUEST.value,
            correlation_id=correlation_id,
            status="open",
            body=json.dumps(proposal),
            to_agent="risk_officer",
            from_agent=self.agent_id,
        )

        log.info("Parameter change proposed: %s = %s -> %s (%s)", param_name, old_value, new_value, correlation_id)
        return correlation_id

    def approve_change(self, correlation_id: str, approver: str) -> bool:
        """Approve a parameter change (risk officer only)."""
        proposal = self._pending_changes.get(correlation_id)
        if not proposal:
            return False

        proposal["approvals"].append({
            "approver": approver,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        proposal["status"] = "approved"

        # In a real implementation, this would trigger a commit with new config_hash
        log.info("Parameter change approved: %s by %s", correlation_id, approver)
        return True

    def reject_change(self, correlation_id: str, reason: str) -> bool:
        """Reject a parameter change."""
        proposal = self._pending_changes.get(correlation_id)
        if not proposal:
            return False

        proposal["status"] = "rejected"
        proposal["rejection_reason"] = reason

        log.warning("Parameter change rejected: %s - %s", correlation_id, reason)
        return True


# ─── Global instances ─────────────────────────────────────────────────────────

import threading

_protocol_handler: Optional[ProtocolHandler] = None
_heartbeat_manager: Optional[HeartbeatManager] = None
_param_change_manager: Optional[ParameterChangeManager] = None


def get_protocol_handler(bridge, agent_id: str, role: AgentRole) -> ProtocolHandler:
    global _protocol_handler
    if _protocol_handler is None:
        _protocol_handler = ProtocolHandler(bridge, agent_id, role)
    return _protocol_handler


def get_heartbeat_manager(bridge, agent_id: str, role: AgentRole, interval: float = 5.0) -> HeartbeatManager:
    global _heartbeat_manager
    if _heartbeat_manager is None:
        _heartbeat_manager = HeartbeatManager(bridge, agent_id, role, interval)
    return _heartbeat_manager


def get_param_change_manager(bridge, agent_id: str) -> ParameterChangeManager:
    global _param_change_manager
    if _param_change_manager is None:
        _param_change_manager = ParameterChangeManager(bridge, agent_id)
    return _param_change_manager


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mempalace_bridge import get_bridge

    bridge = get_bridge()
    agent_id = "ubuntu:opencode:home"
    role = AgentRole.KALSHI_TRADER

    handler = get_protocol_handler(bridge, agent_id, role)
    hb = get_heartbeat_manager(bridge, agent_id, role)

    # Demo: send a signal proposal
    handler.send(
        topic=Topic.SIGNAL_PROPOSAL,
        payload={"ticker": "KXBTC15M-26NOV090415-15", "side": "YES", "edge_cents": 3.5},
        to_agent="risk_officer",
        ack_required=True,
    )

    # Demo: send heartbeat
    hb.start()
    time.sleep(2)
    hb.stop()

    print("Protocol demo complete")