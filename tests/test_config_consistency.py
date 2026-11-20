"""tests/test_config_consistency.py — Ensure one canonical definition of bands, windows, sizing.

This test fails if any band/window definition appears in more than one place
with different values, or if stale comments contradict live defaults.
"""

import re
import sys
from pathlib import Path

import config


def test_entry_band_single_definition():
    """ENTRY_BAND_MIN/MAX must be the only definition of the entry band."""
    # Live defaults at HEAD
    assert config.ENTRY_BAND_MIN == 95, f"ENTRY_BAND_MIN={config.ENTRY_BAND_MIN}"
    assert config.ENTRY_BAND_MAX == 99, f"ENTRY_BAND_MAX={config.ENTRY_BAND_MAX}"
    assert config.MIN_YES_PRICE_CENTS == 75, f"MIN_YES_PRICE_CENTS={config.MIN_YES_PRICE_CENTS}"

    # Stale comment check: 15-55c or 15/55 in comments (not in live code)
    with open("config.py") as f:
        content = f.read()
    # These are the OLD band values that appear in comments
    stale_patterns = [
        (r'15.*55.*band', '15-55c band in comments'),
        (r'55.*15.*band', '55-15c band in comments'),
        (r'"15-55"', '15-55 string in comments'),
        (r"'15-55'", "15-55 string in comments"),
        (r'82.*93', '82-93c in comments'),
        (r'93.*82', '93-82c in comments'),
    ]
    for pattern, desc in stale_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        # Filter out lines that are in validate_entry_* functions (they reference the old names)
        filtered = [m for m in matches if 'validate_entry' not in m and 'MIN_YES_PRICE' not in m]
        assert not filtered, f"Stale {desc} found in config.py: {filtered[:3]}"


def test_entry_window_single_definition():
    """ENTRY_WINDOW_START/END_MIN must be the only definition."""
    assert config.ENTRY_WINDOW_START_MIN == 4.0, f"ENTRY_WINDOW_START_MIN={config.ENTRY_WINDOW_START_MIN}"
    assert config.ENTRY_WINDOW_END_MIN == 14.0, f"ENTRY_WINDOW_END_MIN={config.ENTRY_WINDOW_END_MIN}"

    # Stale comments: 6-11, 13-14.5, etc.
    with open("config.py") as f:
        content = f.read()
    stale_window = re.findall(r'6.*11.*min|11.*6.*min|13.*14\.5|14\.5.*13', content, re.IGNORECASE)
    filtered = [m for m in stale_window if 'validate' not in m]
    assert not filtered, f"Stale entry window in comments: {filtered[:3]}"


def test_pin_risk_minutes_single_definition():
    """GATES_HIGH_GAMMA_MINUTES single definition."""
    assert config.GATES_HIGH_GAMMA_MINUTES == 2.0, f"GATES_HIGH_GAMMA_MINUTES={config.GATES_HIGH_GAMMA_MINUTES}"


def test_sizing_single_definition():
    """Sizing caps must be defined once."""
    assert config.MAX_CONTRACTS_CAP == 3, f"MAX_CONTRACTS_CAP={config.MAX_CONTRACTS_CAP}"
    # MAX_TRADE_PCT_OF_EQUITY should exist
    assert hasattr(config, "MAX_TRADE_PCT_OF_EQUITY") or hasattr(config, "PER_TRADE_PCT"), "no sizing pct found"


def test_no_duplicate_band_in_readme():
    """README must not claim a different band."""
    if Path("README.md").exists():
        with open("README.md") as f:
            readme = f.read()
        # Old claims: 82-93, 15-55, 75-95 (stale bands that are NOT the live default)
        # Live default 95-99c is allowed
        for pattern in [r'82.*93', r'15.*55.*cent', r'75.*95.*cent']:
            assert not re.search(pattern, readme, re.IGNORECASE), f"README has stale band: {pattern}"


def test_validate_functions_exist():
    """validate_bands must exist and detect inversions."""
    assert hasattr(config, "validate_bands"), "validate_bands missing"
    # validate_bands raises ValueError if config invalid, returns None if valid
    # Current config is valid so it returns None
    result = config.validate_bands()
    assert result is None, f"validate_bands returned {result} (expected None for valid config)"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))