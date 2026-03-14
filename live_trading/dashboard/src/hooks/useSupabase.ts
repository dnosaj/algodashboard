import { useState, useEffect, useCallback, useRef } from 'react';
import { supabase } from '../lib/supabase';

export interface DriftStatus {
  strategy_id: string;
  source: string;
  total_trades: number;
  live_wr: number | null;
  live_pf: number | null;
  live_total_pnl: number | null;
  first_trade_date: string | null;
  last_trade_date: string | null;
  backtest_expected_wr: number | null;
  backtest_expected_pf: number | null;
  backtest_expected_sharpe: number | null;
  backtest_max_dd: number | null;
  wr_deviation: number | null;
  pf_deviation: number | null;
  wr_z_score: number | null;
  status: 'GREEN' | 'YELLOW' | 'RED' | 'INSUFFICIENT_DATA' | 'NO_BENCHMARK';
}

export interface RollingPoint {
  strategy_id: string;
  source: string;
  trade_num: number;
  cumulative_pnl: number;
  rolling_20_wr: number | null;
  rolling_20_pf: number | null;
  trade_date: string;
}

export interface DailyStat {
  strategy_id: string;
  trade_date: string;
  source: string;
  trade_count: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  total_pnl: number;
  profit_factor: number | null;
  tp_count: number;
  sl_count: number;
  be_time_count: number;
  eod_count: number;
}

export interface SupabaseData {
  drift: DriftStatus[];
  rolling: RollingPoint[];
  dailyStats: DailyStat[];
  loaded: boolean;
  error: string | null;
}

const POLL_INTERVAL = 120_000; // 2 minutes

export function useSupabase(): SupabaseData {
  const [drift, setDrift] = useState<DriftStatus[]>([]);
  const [rolling, setRolling] = useState<RollingPoint[]>([]);
  const [dailyStats, setDailyStats] = useState<DailyStat[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    if (!supabase) {
      setError('Supabase not configured');
      return;
    }

    try {
      // Fetch in parallel
      const [driftRes, rollingRes, dailyRes] = await Promise.all([
        supabase
          .from('live_vs_backtest')
          .select('*')
          .in('source', ['paper', 'live']),
        supabase
          .from('rolling_performance')
          .select('strategy_id,source,trade_num,cumulative_pnl,rolling_20_wr,rolling_20_pf,trade_date')
          .in('source', ['paper', 'live'])
          .order('trade_num', { ascending: true })
          .limit(2000),
        supabase
          .from('daily_stats')
          .select('*')
          .in('source', ['paper', 'live'])
          .order('trade_date', { ascending: false })
          .limit(50),
      ]);

      if (!mountedRef.current) return;

      if (driftRes.error) throw driftRes.error;
      if (rollingRes.error) throw rollingRes.error;
      if (dailyRes.error) throw dailyRes.error;

      // Only show active strategies (paper/live, exclude shelved)
      const activeDrift = (driftRes.data || []).filter(
        (d: DriftStatus) => d.source === 'paper' || d.source === 'live'
      );

      setDrift(activeDrift);
      setRolling(rollingRes.data || []);
      setDailyStats(dailyRes.data || []);
      setLoaded(true);
      setError(null);
    } catch (err: unknown) {
      if (!mountedRef.current) return;
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      console.error('Supabase fetch error:', err);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetchData();
    const interval = setInterval(fetchData, POLL_INTERVAL);
    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [fetchData]);

  return { drift, rolling, dailyStats, loaded, error };
}
