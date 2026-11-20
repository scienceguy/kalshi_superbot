"""
systemd/ — Systemd service unit files for running as services (T13-50).

Provides systemd units per role with:
- Restart=always
- WatchdogSec
- EnvironmentFile/OCI Vault for secrets
- Separate unprivileged user
- Firewall rules
- chrony for time sync
- Log rotation
- Litestream for SQLite replication
"""

from __future__ import annotations

from .generator import generate_all_configs, SERVICES, generate_service_units, generate_env_file

__all__ = ["generate_all_configs", "SERVICES", "generate_service_units", "generate_env_file"]