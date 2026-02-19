"""
Example: EMA Crossover strategy backtest (Long-only + Long & Short).

Demonstrates how to use the backtesting engine with an EMA 9/21 crossover
strategy on INDEX:BTCUSD 1D data. Runs both run_backtest (long-only) and
run_backtest_long_short (long + short) to verify both engine paths.

Chart data: INDEX:BTCUSD 1D (TradingView export)
Slippage: NOT simulated (set to 0) — requires tick-level data which is
          expensive to obtain. Set slippage to 0 in both engine and TV.

Settings (match these in TradingView for comparison):
- 2018-01-01 to 2069-12-31
- Initial capital: $1,000
- 100% of equity per trade
- 0.1% commission
- 0 slippage
- Fast EMA: 9, Slow EMA: 21
- Margin long/short: 0%

Expected results (long-only):
- Net Profit: $9,180 (917.95%)
- Total Trades: 60
- Win Rate: 33.33%
- Profit Factor: 1.813
- Max Drawdown (intrabar): -$3,420 (-61.01%)

Expected results (long + short, EMA 9/21 long + EMA 5/13 short, with reversals):
- Net Profit: $1,501 (150.11%)
- Total Trades: 155
- Profit Factor: 1.167
- Open P&L: ~$513 (unrealised, not included in Net Profit)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import load_tv_export
from engine import (
    BacktestConfig, calc_ema, detect_crossover, detect_crossunder,
    ema_cross_signals, run_backtest, run_backtest_long_short,
    print_kpis, print_trades,
)


def main():
    # Load TradingView-exported INDEX:BTCUSD 1D data
    # The file goes back to 2014, giving plenty of warmup before 2018-01-01.
    # The last bar is automatically dropped (unfinished candle).
    df = load_tv_export("INDEX_BTCUSD, 1D.csv")

    print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total bars: {len(df)}")

    warmup_bars = len(df[df.index < "2018-01-01"])
    print(f"Warmup bars (before 2018-01-01): {warmup_bars}")

    # Configure backtest to match TradingView settings
    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.1,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        pyramiding=1,
        start_date="2018-01-01",
        end_date="2069-12-31",
    )

    # ---- 1. Long-only backtest ------------------------------------------------
    # Use original EMA 9/21 signals (no reversal modifications)
    df_long = df.copy()
    df_long = ema_cross_signals(df_long, fast_len=9, slow_len=21)
    kpis = run_backtest(df_long, config)

    print("\n" + "=" * 60)
    print("  BACKTEST CONFIGURATION")
    print("=" * 60)
    print(f"  Chart Data:       INDEX:BTCUSD 1D (TradingView export)")
    print(f"  Date Range:       {kpis['actual_start_date']} to {kpis['actual_end_date']}")
    print(f"  Initial Capital:  ${config.initial_capital:,.0f}")
    print(f"  Order Size:       {config.qty_value:.0f}% of equity")
    print(f"  Commission:       {config.commission_pct}%")
    print(f"  Slippage:         {config.slippage_ticks} (NOT simulated — requires tick-level data)")
    print(f"  Margin Long/Short: 0%")
    print(f"  Strategy:         EMA Crossover (Fast: 9, Slow: 21)")
    print("=" * 60)

    print("\n--- LONG-ONLY (run_backtest) ---")
    print_kpis(kpis)
    print_trades(kpis["trades"], max_trades=10)

    # Print first few trades in detail for verification
    print("\n\nDETAILED FIRST 5 TRADES (for TradingView comparison):")
    print("=" * 80)
    for i, t in enumerate(kpis["trades"][:5], 1):
        print(f"\nTrade #{i}:")
        print(f"  Entry: {t.entry_date.date()} @ ${t.entry_price:,.2f}")
        print(f"  Qty:   {t.entry_qty:.8f} BTC")
        print(f"  Entry Commission: ${t.entry_commission:.4f}")
        print(f"  Exit:  {t.exit_date.date() if t.exit_date else 'OPEN'} @ ${t.exit_price:,.2f}" if t.exit_price else "  Exit: OPEN")
        print(f"  Exit Commission:  ${t.exit_commission:.4f}")
        print(f"  PnL:   ${t.pnl:,.2f} ({t.pnl_pct:.2f}%)" if t.pnl else "  PnL: N/A")

    # ---- 2. Long + Short backtest ---------------------------------------------
    # Build signals with reversal logic (separate copy to avoid affecting long-only)
    df_ls = df.copy()
    df_ls = ema_cross_signals(df_ls, fast_len=9, slow_len=21)

    # Add short signals using a separate EMA pair (5/13)
    df_ls["short_fast"] = calc_ema(df_ls["Close"], 5)
    df_ls["short_slow"] = calc_ema(df_ls["Close"], 13)
    short_cross_under = detect_crossunder(df_ls["short_fast"], df_ls["short_slow"])
    short_cross_over = detect_crossover(df_ls["short_fast"], df_ls["short_slow"])

    # In TradingView, strategy.entry("Short") reverses the position:
    # it closes the long AND opens a short on the same bar. To match,
    # a short_entry must also act as a long_exit, and vice versa.
    df_ls["short_entry"] = short_cross_under
    df_ls["short_exit"] = short_cross_over | df_ls["long_entry"]  # long entry also closes short
    df_ls["long_exit"] = df_ls["long_exit"] | short_cross_under   # short entry also closes long

    kpis_ls = run_backtest_long_short(df_ls, config)

    print("\n\n--- LONG + SHORT (run_backtest_long_short) ---")
    print_kpis(kpis_ls)
    print_trades(kpis_ls["trades"], max_trades=10)


if __name__ == "__main__":
    main()
