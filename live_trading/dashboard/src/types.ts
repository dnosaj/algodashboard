export interface Position {
  instrument: string;
  strategy_id?: string;
  side: 'FLAT' | 'LONG' | 'SHORT';
  entry_price: number | null;
  unrealized_pnl: number;
}

export interface InstrumentData {
  instrument: string;
  strategy_id: string;
  last_price: number;
  sm_value: number;
  rsi_value: number;
  cooldown_remaining: number;
  cooldown_total: number;
  max_loss_pts: number;
  bars_held: number;
  exit_mode: string;
  tp_pts: number;
  long_used: boolean;
  short_used: boolean;
}

export interface Trade {
  instrument: string;
  strategy_id?: string;
  side: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price: number;
  entry_time: string;
  exit_time: string;
  entry_time_et_epoch?: number;
  exit_time_et_epoch?: number;
  pts: number;
  pnl: number;
  exit_reason: string;
  bars_held: number;
  mfe?: number;
  mae?: number;
}

export interface BarData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StatusData {
  connected: boolean;
  trading_active: boolean;
  paper_mode: boolean;
  positions: Position[];
  instruments: Record<string, InstrumentData>;
  daily_pnl: number;
  uptime_seconds: number;
  trade_count_today: number;
  consecutive_losses: number;
  broker: string;
  account: string;
  safety?: SafetyStatusData;
}

export interface DailyPnLEntry {
  date: string;
  pnl: number;
  cumulative: number;
}

export interface SignalEvent {
  type: string;
  instrument: string;
  reason: string;
  sm_value: number;
  rsi_value: number;
  ts: string;
}

export interface WSMessage {
  type: 'bar' | 'signal' | 'trade' | 'trade_update' | 'status' | 'safety_status' | 'fill' | 'error';
  data: Record<string, unknown>;
  ts: string;
}

export interface SessionInfo {
  filename: string;
  date: string;
  saved_at: string;
  bars: number;
  trades: number;
}

export interface SessionData {
  date: string;
  saved_at: string;
  timezone?: string;  // "ET" for new sessions, absent for old UTC sessions
  bars: Record<string, BarData[]>;
  trades: Trade[];
}

export interface SafetyStrategyStatus {
  strategy_id: string;
  instrument: string;
  paused: boolean;
  pause_reason: string;
  manual_override: boolean;
  qty_override: number | null;
  sl_count_today: number;
  trade_count_today: number;
  daily_pnl: number;
}

export interface SafetyStatusData {
  halted: boolean;
  halt_reason: string;
  daily_pnl: number;
  consecutive_losses: number;
  trade_count_today: number;
  data_feed_healthy: boolean;
  seconds_since_last_bar: number;
  drawdown_enabled: boolean;
  drawdown_mode: 'NORMAL' | 'REDUCED' | 'EXTENDED_PAUSE';
  extended_pause: boolean;
  extended_pause_reason: string;
  rolling_sl_5d: number;
  strategies: Record<string, SafetyStrategyStatus>;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting';
