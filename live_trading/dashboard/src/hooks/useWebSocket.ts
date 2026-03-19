import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  BarData,
  BlockedSignal,
  ConnectionStatus,
  SafetyStatusData,
  StatusData,
  Trade,
  DailyPnLEntry,
  SignalEvent,
  WSMessage,
} from '../types';
import { supabase } from '../lib/supabase';

interface UseWebSocketReturn {
  status: StatusData | null;
  trades: Trade[];
  dailyPnl: DailyPnLEntry[];
  signals: SignalEvent[];
  blockedSignals: BlockedSignal[];
  bars: Record<string, BarData[]>;
  connected: ConnectionStatus;
  sendCommand: (command: string, payload?: Record<string, unknown>) => void;
}

const INITIAL_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;
const MAX_SIGNALS = 50;

export function useWebSocket(url: string): UseWebSocketReturn {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>('disconnected');
  const [status, setStatus] = useState<StatusData | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [dailyPnl, setDailyPnl] = useState<DailyPnLEntry[]>([]);
  const [signals, setSignals] = useState<SignalEvent[]>([]);
  const [blockedSignals, setBlockedSignals] = useState<BlockedSignal[]>([]);
  const [bars, setBars] = useState<Record<string, BarData[]>>({});

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const fetchInitialData = useCallback(async () => {
    try {
      const [statusRes, tradesRes, mnqBarsRes, mesBarsRes] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/trades'),
        fetch('/api/bars/MNQ'),
        fetch('/api/bars/MES'),
      ]);

      if (statusRes.ok) {
        const statusData: StatusData = await statusRes.json();
        if (mountedRef.current) setStatus(statusData);
      }

      if (tradesRes.ok) {
        let tradesData: Trade[] = (await tradesRes.json()) || [];

        // Also load today's trades from Supabase (survives engine restarts)
        if (supabase) {
          try {
            const today = new Date();
            const etOffset = -5; // ET offset (approximate, DST-aware would need more logic)
            const etDate = new Date(today.getTime() + etOffset * 3600000);
            const dateStr = etDate.toISOString().slice(0, 10);

            const { data: sbTrades } = await supabase
              .from('trades')
              .select('strategy_id,instrument,side,entry_price,exit_price,entry_time,exit_time,pts,pnl_net,exit_reason,bars_held,qty,is_partial')
              .in('source', ['paper', 'live'])
              .gte('trade_date', dateStr)
              .order('exit_time', { ascending: false });

            if (sbTrades && sbTrades.length > 0) {
              // Convert Supabase rows to Trade format
              const supabaseTrades: Trade[] = sbTrades.map((t: Record<string, unknown>) => ({
                instrument: t.instrument as string,
                strategy_id: t.strategy_id as string,
                side: ((t.side as string) || '').toUpperCase() as 'LONG' | 'SHORT',
                entry_price: t.entry_price as number,
                exit_price: t.exit_price as number | null,
                entry_time: t.entry_time as string,
                exit_time: t.exit_time as string | null,
                pts: t.pts as number,
                pnl: t.pnl_net as number,
                exit_reason: t.exit_reason as string,
                bars_held: t.bars_held as number,
                qty: t.qty as number,
                is_partial: t.is_partial as boolean,
              }));

              // Merge: use engine trades as primary, fill in any Supabase trades
              // that aren't already in the engine's list (from before restart)
              const engineKeys = new Set(
                tradesData.map((t: Trade) => `${t.strategy_id}_${t.entry_time}_${t.is_partial}`)
              );
              const missingTrades = supabaseTrades.filter(
                (t) => !engineKeys.has(`${t.strategy_id}_${t.entry_time}_${t.is_partial}`)
              );
              if (missingTrades.length > 0) {
                tradesData = [...tradesData, ...missingTrades];
                console.log(`[Supabase] Loaded ${missingTrades.length} trades from before engine restart`);
              }
            }
          } catch (err) {
            console.warn('[Supabase] Failed to load today trades:', err);
          }
        }

        if (mountedRef.current) setTrades(tradesData);
      }

      if (mountedRef.current) {
        const newBars: Record<string, BarData[]> = {};
        if (mnqBarsRes.ok) {
          const data = await mnqBarsRes.json();
          if (data && data.length > 0) newBars['MNQ'] = data;
        }
        if (mesBarsRes.ok) {
          const data = await mesBarsRes.json();
          if (data && data.length > 0) newBars['MES'] = data;
        }
        if (Object.keys(newBars).length > 0) {
          setBars(prev => ({ ...prev, ...newBars }));
        }
      }
    } catch {
      // API not available yet
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setConnectionStatus('connected');
      reconnectDelayRef.current = INITIAL_RECONNECT_DELAY;
      fetchInitialData();
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!mountedRef.current) return;

      try {
        const message: WSMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'status':
            setStatus(message.data as unknown as StatusData);
            break;

          case 'safety_status':
            setStatus((prev) => prev ? { ...prev, safety: message.data as unknown as SafetyStatusData } : prev);
            break;

          case 'trade': {
            const trade = message.data as unknown as Trade;
            setTrades((prev) => [trade, ...prev]);
            break;
          }

          case 'trade_update': {
            const updated = message.data as unknown as Trade;
            setTrades((prev) =>
              prev.map((t) =>
                t.entry_time === updated.entry_time && t.strategy_id === updated.strategy_id
                  ? updated
                  : t
              )
            );
            break;
          }

          case 'signal_blocked': {
            const blocked = message.data as unknown as BlockedSignal;
            setBlockedSignals((prev) => [blocked, ...prev].slice(0, 200));
            // Also add to signal feed so blocked entries show in Signal Activity
            const blockedSig: SignalEvent = {
              type: blocked.side === 'BUY' ? 'BUY' : 'SELL',
              instrument: blocked.instrument,
              reason: blocked.reason || '',
              sm_value: (message.data.sm_value as number) || 0,
              rsi_value: (message.data.rsi_value as number) || 0,
              ts: message.ts,
              blocked: true,
              block_reason: blocked.reason || '',
              strategy_id: blocked.strategy_id,
            };
            setSignals((prev) => [blockedSig, ...prev].slice(0, MAX_SIGNALS));
            break;
          }

          case 'signal': {
            const sig: SignalEvent = {
              type: (message.data.type as string) || '',
              instrument: (message.data.instrument as string) || '',
              reason: (message.data.reason as string) || '',
              sm_value: (message.data.sm_value as number) || 0,
              rsi_value: (message.data.rsi_value as number) || 0,
              ts: message.ts,
            };
            setSignals((prev) => [sig, ...prev].slice(0, MAX_SIGNALS));
            break;
          }

          case 'bar': {
            const d = message.data;
            const inst = d.instrument as string;
            // Use server's ET epoch (d.time) if available, else fallback to parsing timestamp
            const time = (typeof d.time === 'number' && (d.time as number) > 0)
              ? Math.floor((d.time as number) / 60) * 60  // snap to minute
              : Math.floor(new Date(d.timestamp as string).getTime() / 1000);
            const newBar: BarData = {
              time,
              open: d.open as number,
              high: d.high as number,
              low: d.low as number,
              close: d.close as number,
              volume: d.volume as number,
            };
            setBars((prev) => {
              const existing = prev[inst] || [];
              // Update last bar if same time, otherwise append
              if (existing.length > 0 && existing[existing.length - 1].time === time) {
                const updated = [...existing];
                updated[updated.length - 1] = newBar;
                return { ...prev, [inst]: updated };
              }
              return { ...prev, [inst]: [...existing, newBar] };
            });
            break;
          }
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnectionStatus('reconnecting');
      scheduleReconnect();
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      ws.close();
    };
  }, [url, fetchInitialData]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }

    reconnectTimerRef.current = setTimeout(() => {
      if (!mountedRef.current) return;
      reconnectDelayRef.current = Math.min(
        reconnectDelayRef.current * 2,
        MAX_RECONNECT_DELAY
      );
      connect();
    }, reconnectDelayRef.current);
  }, [connect]);

  const sendCommand = useCallback(
    (command: string, payload?: Record<string, unknown>) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command, ...payload }));
      }
    },
    []
  );

  useEffect(() => {
    mountedRef.current = true;
    connect();

    // Poll /api/status every 5 seconds
    const pollInterval = setInterval(async () => {
      if (!mountedRef.current) return;
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const data: StatusData = await res.json();
          if (mountedRef.current) setStatus(data);
        }
      } catch {
        // Ignore
      }
    }, 5000);

    // Poll /api/trades every 10 seconds
    const tradesPollInterval = setInterval(async () => {
      if (!mountedRef.current) return;
      try {
        const res = await fetch('/api/trades');
        if (res.ok) {
          const data = await res.json();
          if (mountedRef.current) {
            setTrades((prev) => {
              const next = data || [];
              // Daily reset: trades shrank → clear stale blocked signals
              if (next.length < prev.length) {
                setBlockedSignals([]);
              }
              return next;
            });
          }
        }
      } catch {
        // Ignore
      }
    }, 10000);

    // Poll daily P&L every 60 seconds
    const pnlPollInterval = setInterval(async () => {
      if (!mountedRef.current) return;
      try {
        const res = await fetch('/api/daily_pnl');
        if (res.ok) {
          const data = await res.json();
          if (mountedRef.current) setDailyPnl(data || []);
        }
      } catch {
        // Ignore
      }
    }, 60000);

    // Initial daily P&L fetch
    (async () => {
      try {
        const res = await fetch('/api/daily_pnl');
        if (res.ok) {
          const data = await res.json();
          if (mountedRef.current) setDailyPnl(data || []);
        }
      } catch {
        // Ignore
      }
    })();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      clearInterval(pollInterval);
      clearInterval(tradesPollInterval);
      clearInterval(pnlPollInterval);
      wsRef.current?.close();
    };
  }, [connect]);

  return {
    status,
    trades,
    dailyPnl,
    signals,
    blockedSignals,
    bars,
    connected: connectionStatus,
    sendCommand,
  };
}
