import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  BarData,
  ConnectionStatus,
  StatusData,
  Trade,
  DailyPnLEntry,
  SignalEvent,
  WSMessage,
} from '../types';

interface UseWebSocketReturn {
  status: StatusData | null;
  trades: Trade[];
  dailyPnl: DailyPnLEntry[];
  signals: SignalEvent[];
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
        const tradesData = await tradesRes.json();
        if (mountedRef.current) setTrades(tradesData || []);
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

          case 'trade': {
            const trade = message.data as unknown as Trade;
            setTrades((prev) => [trade, ...prev]);
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
            const time = Math.floor(
              new Date(d.timestamp as string).getTime() / 1000
            );
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
          if (mountedRef.current) setTrades(data || []);
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
    bars,
    connected: connectionStatus,
    sendCommand,
  };
}
