export interface Position {
  instrument: string;
  side: 'FLAT' | 'LONG' | 'SHORT';
  entry_price: number | null;
  unrealized_pnl: number;
}

export interface InstrumentData {
  last_price: number;
  sm_value: number;
  rsi_value: number;
  cooldown_remaining: number;
  cooldown_total: number;
  max_loss_pts: number;
  bars_held: number;
  version: string;
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
  type: 'bar' | 'signal' | 'trade' | 'status' | 'fill' | 'error';
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
  bars: Record<string, BarData[]>;
  trades: Trade[];
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting';
