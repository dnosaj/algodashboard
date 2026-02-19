"""
Backtesting engine that matches TradingView's PineScript strategy behavior.

Usage:
    from engine import BacktestConfig, run_backtest, print_kpis
    from engine import load_tv_export, calc_ema, detect_crossover
"""

from engine.engine import (
    __version__,
    BacktestConfig,
    Trade,
    calc_ema,
    calc_smma,
    calc_wma,
    calc_hma,
    calc_ehma,
    calc_thma,
    calc_gaussian,
    detect_crossover,
    detect_crossunder,
    get_source,
    ema_cross_signals,
    run_backtest,
    run_backtest_long_short,
    compute_kpis,
    print_kpis,
    print_trades,
)

from engine.data import (
    load_tv_export,
    fetch_btc_daily,
)
