"""Config consistency tests for T15-04.

Ensures:
- Entry window and band constants are internally consistent (start < end, min <= max).
- Values align with evidence from measure_edge.py on VPS data.
- No hardcoded defaults that contradict documented evidence.
"""

import pytest
import config


class TestEntryWindowConsistency:
    """Entry window min/max validation."""

    def test_start_before_end(self):
        assert config.ENTRY_WINDOW_START_MIN < config.ENTRY_WINDOW_END_MIN, \
            f"ENTRY_WINDOW_START_MIN ({config.ENTRY_WINDOW_START_MIN}) must be < ENTRY_WINDOW_END_MIN ({config.ENTRY_WINDOW_END_MIN})"

    def test_positive_minutes(self):
        assert config.ENTRY_WINDOW_START_MIN >= 0
        assert config.ENTRY_WINDOW_END_MIN <= 15, \
            "Window end cannot exceed 15-minute contract duration"


class TestEntryBandConsistency:
    """Entry price band min/max validation."""

    def test_min_le_max(self):
        assert config.ENTRY_BAND_MIN <= config.ENTRY_BAND_MAX, \
            f"ENTRY_BAND_MIN ({config.ENTRY_BAND_MIN}) must be <= ENTRY_BAND_MAX ({config.ENTRY_BAND_MAX})"

    def test_band_in_valid_range(self):
        assert 0 <= config.ENTRY_BAND_MIN <= 100
        assert 0 <= config.ENTRY_BAND_MAX <= 100


class TestEvidenceAlignment:
    """Config values must align with measure_edge.py evidence on VPS data.

    Evidence (504 resolved markets, 1507 quotes, 8-14 min window = 729 quotes):
    - 70-80c band: win=0.8333, price=0.7449, fee=0.0133, netEV=+0.0752, lo95=0.7000 (CLEARS)
    - 60-70c band: win=0.6452, price=0.6421, fee=0.0160, netEV=-0.0130, lo95=0.4767 (FAILS)
    - 80-90c band: n=22 (too few)
    - 95-99c band: n=11 (too few)
    """

    def test_entry_window_matches_evidence(self):
        """Entry window should be 8-14 min per evidence."""
        assert config.ENTRY_WINDOW_START_MIN == 8.0, \
            f"Evidence shows 8 min start is optimal (729 quotes), got {config.ENTRY_WINDOW_START_MIN}"
        assert config.ENTRY_WINDOW_END_MIN == 14.0, \
            f"Evidence shows 14 min end is optimal, got {config.ENTRY_WINDOW_END_MIN}"

    def test_entry_band_matches_evidence(self):
        """Entry band should be 70-80c per evidence (only band with +EV and lo95 > 50%)."""
        assert config.ENTRY_BAND_MIN == 70, \
            f"Evidence shows 70c min is optimal (+7.5c EV, 70% lower bound), got {config.ENTRY_BAND_MIN}"
        assert config.ENTRY_BAND_MAX == 80, \
            f"Evidence shows 80c max is optimal, got {config.ENTRY_BAND_MAX}"

    def test_band_excludes_negative_ev_ranges(self):
        """Band should not include 60-70c (negative EV per evidence)."""
        assert config.ENTRY_BAND_MIN >= 70, \
            "Band must not include 60-70c range (netEV=-0.0130, lo95=47.67%)"

    def test_band_excludes_sparse_ranges(self):
        """Band should not rely on 80-90c (n=22, too few) or 95-99c (n=11, too few)."""
        # The evidence explicitly says these are "too few to judge"
        # Our band ending at 80 avoids this
        assert config.ENTRY_BAND_MAX <= 80, \
            "Band must not extend into 80-90c (n=22 too few) or 95-99c (n=11 too few)"


class TestCrossConfigConsistency:
    """Cross-parameter consistency checks."""

    def test_yes_band_overlaps_entry_band(self):
        """MIN_YES_PRICE_CENTS..MAX_YES_PRICE_CENTS should overlap with ENTRY_BAND."""
        # YES leg uses MIN_YES_PRICE_CENTS..MAX_YES_PRICE_CENTS
        # Entry band is the price range we're willing to make markets in
        # They should overlap for the strategy to work
        assert config.MIN_YES_PRICE_CENTS <= config.ENTRY_BAND_MAX, \
            f"MIN_YES_PRICE_CENTS ({config.MIN_YES_PRICE_CENTS}) > ENTRY_BAND_MAX ({config.ENTRY_BAND_MAX}) means no overlap"
        assert config.MAX_YES_PRICE_CENTS >= config.ENTRY_BAND_MIN, \
            f"MAX_YES_PRICE_CENTS ({config.MAX_YES_PRICE_CENTS}) < ENTRY_BAND_MIN ({config.ENTRY_BAND_MIN}) means no overlap"

    def test_no_band_related_to_yes_band(self):
        """NO entry band (MIN_NO/MAX_NO) should be complementary to YES band."""
        # NO price = 100 - YES price
        # If YES band is 70-80c, NO band should be 20-30c
        # But MIN_NO_ENTRY_CENTS is a floor, not the full band
        # Just verify they're in reasonable relationship
        assert config.MIN_NO_ENTRY_CENTS <= 100 - config.ENTRY_BAND_MIN, \
            "NO floor should be <= complement of YES min"
        assert config.MAX_NO_ENTRY_CENTS >= 100 - config.ENTRY_BAND_MAX, \
            "NO ceiling should be >= complement of YES max"


class TestNoHardcodedContradictions:
    """Ensure no hardcoded defaults contradict documented evidence."""

    def test_agmd_optimal_params_not_used_as_defaults(self):
        """The AGENTS.md 'Optimal Parameters' (15-55c) are NEGATIVE on larger sample.
        
        Do not use them as config defaults. Our defaults are 70-80c per evidence.
        """
        # The old "optimal" band 15-55c was from 900 markets backtest
        # But measure_edge on 504 markets with 1507 quotes shows 15-55c is negative
        # Our config defaults must reflect the newer, larger-sample evidence
        assert config.ENTRY_BAND_MIN >= 70, "Default must not be the old 15c negative-EV band"
        assert config.ENTRY_BAND_MAX <= 80, "Default must not be the old 55c negative-EV band"

    def test_entry_window_not_old_backtest_values(self):
        """Old backtest used 6-11 min; evidence shows 8-14 min is better."""
        assert config.ENTRY_WINDOW_START_MIN >= 8, "Old 6 min start has sparse data (n<30 per band)"
        assert config.ENTRY_WINDOW_END_MIN == 14, "Full 15-min window includes negative-EV early minutes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])