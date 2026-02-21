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

/** Compute ET offset in seconds for a given UTC epoch, DST-aware. */
function getEtOffset(utcEpoch: number): number {
  const dt = new Date(utcEpoch * 1000);
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: 'numeric',
    minute: 'numeric',
    hourCycle: 'h23',
  });
  const parts = formatter.formatToParts(dt);
  const etYear = Number(parts.find((p) => p.type === 'year')?.value ?? 0);
  const etMonth = Number(parts.find((p) => p.type === 'month')?.value ?? 1) - 1;
  const etDay = Number(parts.find((p) => p.type === 'day')?.value ?? 1);
  const etHour = Number(parts.find((p) => p.type === 'hour')?.value ?? 0);
  const etMinute = Number(parts.find((p) => p.type === 'minute')?.value ?? 0);
  const etAsUtcMs = Date.UTC(etYear, etMonth, etDay, etHour, etMinute);
  const utcMs = Date.UTC(
    dt.getUTCFullYear(), dt.getUTCMonth(), dt.getUTCDate(),
    dt.getUTCHours(), dt.getUTCMinutes(),
  );
  return Math.round((etAsUtcMs - utcMs) / 1000);
}

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

  // Deduplicate bars by time (keep last occurrence) — lightweight-charts
  // requires strictly ascending, unique timestamps.
  const activeBarsRaw = replayMode ? replayBars : bars;
  const activeBars = useMemo(() => {
    const seen = new Map<number, BarData>();
    for (const b of activeBarsRaw) {
      seen.set(b.time, b);
    }
    return Array.from(seen.values()).sort((a, b) => a.time - b.time);
  }, [activeBarsRaw]);
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
        let instBars = data.bars?.[instrument] || [];

        // Backward compat: old sessions saved with UTC epochs need conversion
        // New sessions have timezone="ET" and bar times are already ET-shifted
        if (!data.timezone || data.timezone !== 'ET') {
          // Convert UTC epochs to ET using DST-aware offset
          instBars = instBars.map((b) => ({
            ...b,
            time: b.time + getEtOffset(b.time),
          }));
        }

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
    const barTimes = activeBars.map((b) => b.time);

    // Snap a raw timestamp to the nearest bar time (fixes MES marker alignment)
    const snapToBar = (rawTime: number): number => {
      if (barTimes.length === 0) return rawTime;
      let closest = barTimes[0];
      let minDist = Math.abs(barTimes[0] - rawTime);
      for (const bt of barTimes) {
        const dist = Math.abs(bt - rawTime);
        if (dist < minDist) {
          minDist = dist;
          closest = bt;
        }
      }
      return closest;
    };

    // Get ET epoch time for a trade timestamp, falling back to JS Date parsing
    const getEtEpoch = (isoTime: string, etEpoch?: number): number => {
      if (etEpoch != null) {
        // Server already shifted to ET — snap to nearest minute
        return Math.floor(etEpoch / 60) * 60;
      }
      // Legacy: parse ISO -> UTC, then shift to ET
      const utcEpoch = Math.floor(new Date(isoTime).getTime() / 1000);
      return Math.floor((utcEpoch + getEtOffset(utcEpoch)) / 60) * 60;
    };

    for (const trade of activeTrades) {
      if (!trade.entry_time) continue;

      const isV15 = (trade.strategy_id || '').toLowerCase().includes('v15');
      const isMES = (trade.strategy_id || '').toLowerCase().includes('mes');
      const rawEntryTime = getEtEpoch(trade.entry_time, trade.entry_time_et_epoch);
      const entryTime = snapToBar(rawEntryTime) as unknown as Time;

      // Strategy-specific colors
      let entryColor: string;
      let entryText: string;
      if (isMES) {
        entryColor = trade.side === 'LONG' ? '#ffdd00' : '#ff66ff';
        entryText = trade.side === 'LONG' ? 'MES L' : 'MES S';
      } else if (isV15) {
        entryColor = trade.side === 'LONG' ? '#00aaff' : '#ff8800';
        entryText = trade.side === 'LONG' ? 'v15 L' : 'v15 S';
      } else {
        entryColor = trade.side === 'LONG' ? '#00ff88' : '#ff4444';
        entryText = trade.side === 'LONG' ? 'L' : 'S';
      }

      // Entry marker
      mkrs.push({
        time: entryTime,
        position: trade.side === 'LONG' ? 'belowBar' : 'aboveBar',
        color: entryColor,
        shape: trade.side === 'LONG' ? 'arrowUp' : 'arrowDown',
        text: entryText,
      });

      // Exit marker
      if (trade.exit_time) {
        const rawExitTime = getEtEpoch(trade.exit_time, trade.exit_time_et_epoch);
        const exitTime = snapToBar(rawExitTime) as unknown as Time;
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
  }, [activeTrades, activeBars]);

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
            {instrument} 1m Chart (ET)
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
