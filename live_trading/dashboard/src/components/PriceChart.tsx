import { useEffect, useRef, useMemo, useState, useCallback } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type {
  IChartApi,
  ISeriesApi,
  SeriesMarker,
  Time,
} from 'lightweight-charts';
import type { BarData, Trade, SessionInfo, SessionData } from '../types';
import { SessionTradeList } from './SessionTradeList';

interface PriceChartProps {
  bars: BarData[];
  trades: Trade[];
  instrument: string;
}

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

const btnStyle = (active?: boolean): React.CSSProperties => ({
  padding: '4px 10px',
  fontSize: 10,
  fontFamily: FONT,
  fontWeight: 600,
  border: '1px solid #2a2a4a',
  borderRadius: 4,
  cursor: 'pointer',
  backgroundColor: active ? '#8888cc22' : '#1a1a2e',
  color: active ? '#8888cc' : '#666',
  letterSpacing: 0.5,
});

export function PriceChart({ bars, trades, instrument }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const prevBarsLenRef = useRef(0);

  // Session state
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [replayMode, setReplayMode] = useState(false);
  const [replayBars, setReplayBars] = useState<BarData[]>([]);
  const [replayTrades, setReplayTrades] = useState<Trade[]>([]);
  const [replayLabel, setReplayLabel] = useState('');
  const [saving, setSaving] = useState(false);

  const activeBars = replayMode ? replayBars : bars;
  const activeTrades = replayMode ? replayTrades : trades;

  // Fetch session list
  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch('/api/sessions');
      if (res.ok) {
        const data = await res.json();
        setSessions(data || []);
      }
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Save session
  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const res = await fetch('/api/session/save', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        console.log('Session saved:', data);
        fetchSessions();
      }
    } catch (e) {
      console.error('Save failed:', e);
    }
    setSaving(false);
  }, [fetchSessions]);

  // Load session
  const handleLoad = useCallback(async (filename: string) => {
    try {
      const res = await fetch(`/api/session/${filename}`);
      if (res.ok) {
        const data: SessionData = await res.json();
        const instBars = data.bars?.[instrument] || [];
        const instTrades = (data.trades || []).filter(
          (t) => t.instrument === instrument
        );
        setReplayBars(instBars);
        setReplayTrades(instTrades);
        setReplayLabel(data.date || filename);
        setReplayMode(true);
        prevBarsLenRef.current = 0; // force full setData
      }
    } catch (e) {
      console.error('Load failed:', e);
    }
  }, [instrument]);

  // Back to live
  const handleLive = useCallback(() => {
    setReplayMode(false);
    setReplayBars([]);
    setReplayTrades([]);
    setReplayLabel('');
    prevBarsLenRef.current = 0; // force full setData
  }, []);

  // Create chart on mount
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: '#16213e' },
        textColor: '#888',
        fontFamily: FONT,
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1e2a45' },
        horzLines: { color: '#1e2a45' },
      },
      timeScale: {
        borderColor: '#2a2a4a',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#2a2a4a',
      },
      crosshair: {
        horzLine: { color: '#8888cc44' },
        vertLine: { color: '#8888cc44' },
      },
    });

    const series = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff4444',
      borderUpColor: '#00ff88',
      borderDownColor: '#ff4444',
      wickUpColor: '#00ff8888',
      wickDownColor: '#ff444488',
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      prevBarsLenRef.current = 0;
    };
  }, []);

  // Update bar data
  useEffect(() => {
    if (!seriesRef.current || activeBars.length === 0) return;

    if (prevBarsLenRef.current === 0 || activeBars.length < prevBarsLenRef.current) {
      // Initial load, data reset, or mode switch
      seriesRef.current.setData(activeBars as unknown as { time: Time; open: number; high: number; low: number; close: number }[]);
      chartRef.current?.timeScale().scrollToRealTime();
    } else {
      // Incremental: update/append last bar
      seriesRef.current.update(activeBars[activeBars.length - 1] as unknown as { time: Time; open: number; high: number; low: number; close: number });
    }

    prevBarsLenRef.current = activeBars.length;
  }, [activeBars]);

  // Compute trade markers
  const markers = useMemo((): SeriesMarker<Time>[] => {
    if (!activeTrades || activeTrades.length === 0) return [];

    const mkrs: SeriesMarker<Time>[] = [];

    for (const trade of activeTrades) {
      if (!trade.entry_time) continue;

      const isV15 = (trade.strategy_id || '').toLowerCase().includes('v15');
      const entryTime = (Math.floor(
        new Date(trade.entry_time).getTime() / 1000 / 60
      ) * 60) as unknown as Time;

      // Entry marker
      if (trade.side === 'LONG') {
        mkrs.push({
          time: entryTime,
          position: 'belowBar',
          color: isV15 ? '#00aaff' : '#00ff88',
          shape: 'arrowUp',
          text: isV15 ? 'v15 L' : 'L',
        });
      } else {
        mkrs.push({
          time: entryTime,
          position: 'aboveBar',
          color: isV15 ? '#ff8800' : '#ff4444',
          shape: 'arrowDown',
          text: isV15 ? 'v15 S' : 'S',
        });
      }

      // Exit marker
      if (trade.exit_time) {
        const exitTime = (Math.floor(
          new Date(trade.exit_time).getTime() / 1000 / 60
        ) * 60) as unknown as Time;
        const win = (trade.pnl || 0) >= 0;

        mkrs.push({
          time: exitTime,
          position: trade.side === 'LONG' ? 'aboveBar' : 'belowBar',
          color: win ? '#00ff88' : '#ff4444',
          shape: 'circle',
          text: `${win ? '+' : ''}${(trade.pnl || 0).toFixed(0)}`,
        });
      }
    }

    // Lightweight Charts requires markers sorted by time
    mkrs.sort((a, b) => (a.time as number) - (b.time as number));
    return mkrs;
  }, [activeTrades]);

  // Apply markers to series
  useEffect(() => {
    if (!seriesRef.current) return;
    seriesRef.current.setMarkers(markers);
  }, [markers]);

  return (
    <div
      style={{
        backgroundColor: '#16213e',
        borderRadius: 8,
        padding: 16,
        border: '1px solid #2a2a4a',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <h3
            style={{
              fontSize: 16,
              fontWeight: 600,
              color: '#e0e0e0',
              margin: 0,
              fontFamily: FONT,
            }}
          >
            {instrument} 1m Chart
          </h3>
          {replayMode ? (
            <span
              style={{
                fontSize: 10,
                padding: '2px 8px',
                borderRadius: 3,
                backgroundColor: '#ff880022',
                color: '#ff8800',
                fontFamily: FONT,
                fontWeight: 600,
              }}
            >
              REPLAY {replayLabel}
            </span>
          ) : (
            <span
              style={{
                fontSize: 10,
                padding: '2px 8px',
                borderRadius: 3,
                backgroundColor: '#00ff8822',
                color: '#00ff88',
                fontFamily: FONT,
                fontWeight: 600,
              }}
            >
              LIVE
            </span>
          )}
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#555', fontFamily: FONT }}>
            {activeBars.length} bars
            {activeTrades.length > 0 && ` / ${activeTrades.length} trades`}
          </span>

          {replayMode && (
            <button style={btnStyle(true)} onClick={handleLive}>
              LIVE
            </button>
          )}

          <button
            style={btnStyle()}
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? 'SAVING...' : 'SAVE'}
          </button>

          {sessions.length > 0 && (
            <select
              style={{
                ...btnStyle(),
                appearance: 'auto' as const,
                paddingRight: 4,
              }}
              value=""
              onChange={(e) => {
                if (e.target.value) handleLoad(e.target.value);
              }}
            >
              <option value="">LOAD</option>
              {sessions.map((s) => (
                <option key={s.filename} value={s.filename}>
                  {s.date} ({s.bars} bars, {s.trades} trades)
                </option>
              ))}
            </select>
          )}
        </div>
      </div>
      <div ref={containerRef} />

      {/* Trade readout (replay or live trades) */}
      {activeTrades.length > 0 && (
        <SessionTradeList trades={activeTrades} />
      )}
    </div>
  );
}
