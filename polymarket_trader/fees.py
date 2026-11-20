"""
polymarket_trader.fees — Polymarket fee calculations (T13-46).

Polymarket fee structure:
- Protocol fee: 2% of trade value (on Ethereum mainnet)
- On Polygon: gas fees only (typically < $0.01)
- For prediction markets: typically 0% protocol fee, only gas

This module calculates fees for both mainnet and Polygon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

log = logging.getLogger("polymarket_trader.fees")


class Network(Enum):
    """Supported networks."""
    MAINNET = "mainnet"
    POLYGON = "polygon"


@dataclass
class FeeResult:
    """Result of fee calculation."""
    protocol_fee: float
    gas_fee: float
    total_fee: float
    network: Network

    def to_dict(self) -> Dict:
        return {
            "protocol_fee": self.protocol_fee,
            "gas_fee": self.gas_fee,
            "total_fee": self.total_fee,
            "network": self.network.value,
        }


class PolymarketFeeCalculator:
    """Calculates Polymarket fees."""

    # Protocol fee rates (percentage of trade value)
    PROTOCOL_FEE_MAINNET = 0.02  # 2%
    PROTOCOL_FEE_POLYGON = 0.0   # 0% on Polygon

    # Gas estimates (in USD)
    GAS_MAINNET_USD = 5.0        # ~$5 on Ethereum mainnet
    GAS_POLYGON_USD = 0.01       # ~$0.01 on Polygon

    def __init__(self, network: Network = Network.POLYGON):
        self.network = network

    def calculate_trade_fee(
        self,
        size: float,
        price: float,
        network: Optional[Network] = None,
    ) -> FeeResult:
        """
        Calculate fee for a trade.

        Args:
            size: Number of shares
            price: Price per share (0-1)
            network: Network override

        Returns:
            FeeResult with protocol_fee, gas_fee, total_fee
        """
        net = network or self.network
        trade_value = size * price

        if net == Network.MAINNET:
            protocol_fee = trade_value * self.PROTOCOL_FEE_MAINNET
            gas_fee = self.GAS_MAINNET_USD
        else:
            protocol_fee = trade_value * self.PROTOCOL_FEE_POLYGON
            gas_fee = self.GAS_POLYGON_USD

        total_fee = protocol_fee + gas_fee

        return FeeResult(
            protocol_fee=protocol_fee,
            gas_fee=gas_fee,
            total_fee=total_fee,
            network=net,
        )

    def calculate_effective_price(
        self,
        price: float,
        size: float,
        side: str,  # "buy" or "sell"
        network: Optional[Network] = None,
    ) -> float:
        """
        Calculate effective price including fees.

        For buys: effective_price = price + fee_per_share
        For sells: effective_price = price - fee_per_share
        """
        fee_result = self.calculate_trade_fee(size, price, network)
        fee_per_share = fee_result.total_fee / size if size > 0 else 0

        if side == "buy":
            return price + fee_per_share
        else:
            return max(0.0, price - fee_per_share)

    def calculate_break_even_price(
        self,
        entry_price: float,
        size: float,
        side: str,  # "buy" or "sell"
        network: Optional[Network] = None,
    ) -> float:
        """
        Calculate the price needed to break even after fees.

        For long (buy): need settlement at 1.0, so break_even = entry_price + 2*fee_per_share
        For short (sell): need settlement at 0.0, so break_even = entry_price - 2*fee_per_share
        """
        fee_result = self.calculate_trade_fee(size, entry_price, network)
        fee_per_share = fee_result.total_fee / size if size > 0 else 0

        if side == "buy":
            return entry_price + 2 * fee_per_share
        else:
            return max(0.0, entry_price - 2 * fee_per_share)


# ─── Convenience functions ───────────────────────────────────────────────────

_DEFAULT_CALCULATOR: Optional[PolymarketFeeCalculator] = None


def get_fee_calculator(network: Network = Network.POLYGON) -> PolymarketFeeCalculator:
    """Get the default fee calculator."""
    global _DEFAULT_CALCULATOR
    if _DEFAULT_CALCULATOR is None:
        _DEFAULT_CALCULATOR = PolymarketFeeCalculator(network)
    return _DEFAULT_CALCULATOR


def calculate_fee(
    size: float,
    price: float,
    network: Network = Network.POLYGON,
) -> FeeResult:
    """Convenience function to calculate trade fee."""
    return get_fee_calculator(network).calculate_trade_fee(size, price, network)


def effective_price(
    price: float,
    size: float,
    side: str,
    network: Network = Network.POLYGON,
) -> float:
    """Convenience function to calculate effective price."""
    return get_fee_calculator(network).calculate_effective_price(price, size, side, network)


# ─── CLI / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    calc = PolymarketFeeCalculator(Network.POLYGON)

    print("Polymarket Fee Calculator Demo:")
    print("=" * 50)

    # Example: Buy 100 shares at $0.85 on Polygon
    size = 100
    price = 0.85
    for net in [Network.POLYGON, Network.MAINNET]:
        result = calc.calculate_trade_fee(size, price, net)
        eff_buy = calc.calculate_effective_price(price, size, "buy", net)
        be = calc.calculate_break_even_price(price, size, "buy", net)
        print(f"\n{net.value.upper()}:")
        print(f"  Trade value: ${size * price:.2f}")
        print(f"  Protocol fee: ${result.protocol_fee:.4f}")
        print(f"  Gas fee: ${result.gas_fee:.4f}")
        print(f"  Total fee: ${result.total_fee:.4f}")
        print(f"  Effective buy price: ${eff_buy:.4f}")
        print(f"  Break-even price: ${be:.4f}")