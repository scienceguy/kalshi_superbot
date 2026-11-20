#!/usr/bin/env python3
"""
import_n8n_workflows.py — Import n8n workflows from JSON files (T13-40).

Replaces inject_n8n_workflows.py DB injection with clean API/CLI imports.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("import_n8n_workflows")

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "n8n" / "workflows"


def import_via_cli(workflow_file: Path) -> bool:
    """Import a workflow using n8n CLI."""
    try:
        result = subprocess.run(
            ["n8n", "import:workflow", "--input", str(workflow_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            log.info("Imported %s via CLI", workflow_file.name)
            return True
        else:
            log.warning("CLI import failed for %s: %s", workflow_file.name, result.stderr)
            return False
    except FileNotFoundError:
        log.warning("n8n CLI not found, trying API")
        return False
    except Exception as e:
        log.warning("CLI import error for %s: %s", workflow_file.name, e)
        return False


def import_via_api(workflow_file: Path, api_key: str) -> bool:
    """Import a workflow using n8n REST API."""
    import requests

    try:
        with open(workflow_file) as f:
            workflow_data = json.load(f)

        # n8n API expects the workflow in a specific format
        payload = {
            "name": workflow_data.get("name"),
            "nodes": workflow_data.get("nodes", []),
            "connections": workflow_data.get("connections", {}),
            "settings": workflow_data.get("settings", {}),
            "active": workflow_data.get("active", True),
        }

        headers = {
            "X-N8N-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        # Try to create/update workflow
        response = requests.post(
            "http://127.0.0.1:5678/api/v1/workflows",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code in (200, 201):
            log.info("Imported %s via API", workflow_file.name)
            return True
        elif response.status_code == 409:
            # Workflow exists, try to update
            workflow_id = response.json().get("id") or workflow_data.get("id")
            if workflow_id:
                update_response = requests.put(
                    f"http://127.0.0.1:5678/api/v1/workflows/{workflow_id}",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                if update_response.status_code in (200, 201):
                    log.info("Updated %s via API", workflow_file.name)
                    return True
            log.warning("Failed to update existing workflow %s", workflow_file.name)
            return False
        else:
            log.warning("API import failed for %s: %s", workflow_file.name, response.text)
            return False
    except Exception as e:
        log.warning("API import error for %s: %s", workflow_file.name, e)
        return False


def load_api_key() -> str:
    """Load n8n API key from env or file."""
    api_key = os.environ.get("N8N_API_KEY", "").strip()
    if api_key:
        return api_key

    key_file = Path(os.environ.get("N8N_API_KEY_FILE", "/home/ubuntu/.secrets/n8n_api_key"))
    if key_file.exists():
        try:
            return key_file.read_text().strip()
        except Exception:
            pass

    return ""


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not WORKFLOWS_DIR.exists():
        log.error("Workflows directory not found: %s", WORKFLOWS_DIR)
        return 1

    workflow_files = sorted(WORKFLOWS_DIR.glob("*.json"))
    if not workflow_files:
        log.warning("No workflow JSON files found in %s", WORKFLOWS_DIR)
        return 0

    api_key = load_api_key()
    if not api_key:
        log.warning("No n8n API key found, CLI-only mode")

    success = 0
    failed = 0

    for wf_file in workflow_files:
        log.info("Importing %s...", wf_file.name)

        # Try CLI first, then API
        if import_via_cli(wf_file):
            success += 1
            continue

        if api_key and import_via_api(wf_file, api_key):
            success += 1
            continue

        failed += 1
        log.error("Failed to import %s", wf_file.name)

    log.info("Import complete: %d succeeded, %d failed", success, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())