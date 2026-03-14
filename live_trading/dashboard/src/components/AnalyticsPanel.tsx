import { useMemo, useState } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  ReferenceLine,
} from 'recharts';
import type { DriftStatus, RollingPoint, DailyStat } from '../hooks/useSupabase';

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

// Strategy display colors
const STRATEGY_COLORS: Record<string, string> = {
  MNQ_V15: '#00aaff',
  MNQ_VSCALPB: '#00ff88',
  MNQ_VSCALPC: '#bb77ff',
  MES_V2: '#ffdd00',
};

const STATUS_COLORS: Record<string, string> = {
  GREEN: '#00ff88',
  YELLOW: '#ffaa00',
  RED: '#ff4444',
  INSUFFICIENT_DATA: '#666',
  NO_BENCHMARK: '#555',
};

interface Props {
  drift: DriftStatus[];
  rolling: RollingPoint[];
  dailyStats: DailyStat[];
  loaded: boolean;
  error: string | null;
}

// --- Drift Status Cards ---

function DriftCard({ d }: { d: DriftStatus }) {
  const statusColor = STATUS_COLORS[d.status] || '#666';
  const hasBenchmark = d.backtest_expected_wr != null;
  const enough = d.total_trades >= 20;

  return (
    <div style={{
      flex: '1 1 0', minWidth: 180, padding: '12px 14px',
      backgroundColor: '#1e2a45', borderRadius: 6,
      border: `1px solid ${statusColor}33`,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{
          fontSize: 12, fontWeight: 700, fontFamily: FONT,
          color: STRATEGY_COLORS[d.strategy_id] || '#8888cc',
        }}>
          {d.strategy_id}
        </span>
        <span style={{
          fontSize: 10, fontWeight: 700, fontFamily: FONT,
          padding: '2px 8px', borderRadius: 3,
          backgroundColor: `${statusColor}22`,
          color: statusColor,
          letterSpacing: 1,
        }}>
          {d.status === 'INSUFFICIENT_DATA' ? 'LOW N' : d.status === 'NO_BENCHMARK' ? 'NO BM' : d.status}
        </span>
      </div>

      {/* Metrics grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', fontSize: 11, fontFamily: FONT }}>
        <MetricRow label="WR" live={d.live_wr} backtest={d.backtest_expected_wr} suffix="%" />
        <MetricRow label="PF" live={d.live_pf} backtest={d.backtest_expected_pf} />
        {enough && hasBenchmark && d.wr_z_score != null && (
          <>
            <span style={{ color: '#888' }}>Z</span>
            <span style={{ color: Math.abs(d.wr_z_score) > 2 ? '#ff4444' : Math.abs(d.wr_z_score) > 1 ? '#ffaa00' : '#aaa' }}>
              {d.wr_z_score > 0 ? '+' : ''}{d.wr_z_score.toFixed(2)}
            </span>
          </>
        )}
        <span style={{ color: '#888' }}>P&L</span>
        <span style={{ color: (d.live_total_pnl ?? 0) >= 0 ? '#00ff88' : '#ff4444' }}>
          ${d.live_total_pnl?.toFixed(0) ?? '—'}
        </span>
        <span style={{ color: '#888' }}>N</span>
        <span style={{ color: '#aaa' }}>{d.total_trades}</span>
      </div>
    </div>
  );
}

function MetricRow({ label, live, backtest, suffix = '' }: {
  label: string; live: number | null; backtest: number | null; suffix?: string;
}) {
  const deviation = (live != null && backtest != null) ? live - backtest : null;
  const devColor = deviation == null ? '#888'
    : deviation >= 0 ? '#00ff88'
    : deviation < -10 ? '#ff4444'
    : '#ffaa00';

  return (
    <>
      <span style={{ color: '#888' }}>{label}</span>
      <span>
        <span style={{ color: '#ccc' }}>{live?.toFixed(1) ?? '—'}{suffix}</span>
        {backtest != null && (
          <span style={{ color: '#555', fontSize: 10 }}> / {backtest.toFixed(1)}{suffix}</span>
        )}
        {deviation != null && (
          <span style={{ color: devColor, fontSize: 10, marginLeft: 4 }}>
            ({deviation > 0 ? '+' : ''}{deviation.toFixed(1)})
          </span>
        )}
      </span>
    </>
  );
}

// --- Equity Curve Chart ---

function EquityCurve({ rolling }: { rolling: RollingPoint[] }) {
  const strategies = useMemo(() => {
    const map = new Map<string, RollingPoint[]>();
    for (const r of rolling) {
      const key = r.strategy_id;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(r);
    }
    return map;
  }, [rolling]);

  // Merge into unified data array keyed by trade_num
  const chartData = useMemo(() => {
    if (rolling.length === 0) return [];

    // Find max trade_num across strategies
    const maxN = Math.max(...rolling.map(r => r.trade_num));
    const data: Record<string, number | string>[] = [];

    for (let n = 1; n <= maxN; n++) {
      const point: Record<string, number | string> = { n };
      for (const [sid, points] of strategies) {
        const p = points.find(p => p.trade_num === n);
        if (p) point[sid] = p.cumulative_pnl;
      }
      // Only add points that have at least one strategy value
      if (Object.keys(point).length > 1) data.push(point);
    }
    return data;
  }, [rolling, strategies]);

  if (chartData.length === 0) return null;

  const strategyIds = Array.from(strategies.keys());

  return (
    <div style={{ width: '100%', height: 200 }}>
      <ResponsiveContainer>
        <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
          <XAxis
            dataKey="n" tick={{ fill: '#666', fontSize: 10, fontFamily: FONT }}
            axisLine={{ stroke: '#2a2a4a' }}
          />
          <YAxis
            tick={{ fill: '#666', fontSize: 10, fontFamily: FONT }}
            axisLine={{ stroke: '#2a2a4a' }}
            tickFormatter={(v: number) => `$${v}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#16213e', border: '1px solid #2a2a4a',
              fontFamily: FONT, fontSize: 11, borderRadius: 4,
            }}
            labelStyle={{ color: '#888' }}
            formatter={(value: number, name: string) => [`$${value.toFixed(2)}`, name]}
            labelFormatter={(label: number) => `Trade #${label}`}
          />
          <ReferenceLine y={0} stroke="#555" strokeDasharray="3 3" />
          {strategyIds.map(sid => (
            <Line
              key={sid}
              type="monotone"
              dataKey={sid}
              stroke={STRATEGY_COLORS[sid] || '#8888cc'}
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
          ))}
          <Legend
            wrapperStyle={{ fontFamily: FONT, fontSize: 10 }}
            formatter={(value: string) => <span style={{ color: STRATEGY_COLORS[value] || '#8888cc' }}>{value}</span>}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Recent Daily Stats Table ---

function RecentDays({ dailyStats }: { dailyStats: DailyStat[] }) {
  // Group by date, aggregate across strategies
  const byDate = useMemo(() => {
    const map = new Map<string, { date: string; strategies: DailyStat[]; total: number }>();
    for (const s of dailyStats) {
      if (!map.has(s.trade_date)) {
        map.set(s.trade_date, { date: s.trade_date, strategies: [], total: 0 });
      }
      const entry = map.get(s.trade_date)!;
      entry.strategies.push(s);
      entry.total += s.total_pnl;
    }
    // Sort descending, take last 7 days
    return Array.from(map.values())
      .sort((a, b) => b.date.localeCompare(a.date))
      .slice(0, 7);
  }, [dailyStats]);

  if (byDate.length === 0) return null;

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: FONT }}>
        <thead>
          <tr>
            <th style={thStyle}>Date</th>
            <th style={thStyle}>Trades</th>
            <th style={thStyle}>W/L</th>
            <th style={thStyle}>WR</th>
            <th style={thStyle}>P&L</th>
            <th style={thStyle}>PF</th>
            <th style={thStyle}>TP</th>
            <th style={thStyle}>SL</th>
          </tr>
        </thead>
        <tbody>
          {byDate.map(day => {
            const totalTrades = day.strategies.reduce((s, d) => s + d.trade_count, 0);
            const totalWins = day.strategies.reduce((s, d) => s + d.wins, 0);
            const totalLosses = day.strategies.reduce((s, d) => s + d.losses, 0);
            const wr = totalTrades > 0 ? (totalWins / totalTrades * 100) : 0;
            const totalTP = day.strategies.reduce((s, d) => s + d.tp_count, 0);
            const totalSL = day.strategies.reduce((s, d) => s + d.sl_count, 0);
            const grossProfit = day.strategies.reduce((s, d) => {
              return s + (d.total_pnl > 0 ? d.total_pnl : 0);
            }, 0);
            const grossLoss = Math.abs(day.strategies.reduce((s, d) => {
              return s + (d.total_pnl < 0 ? d.total_pnl : 0);
            }, 0));
            const pf = grossLoss > 0 ? grossProfit / grossLoss : null;
            const dayLabel = new Date(day.date + 'T12:00:00').toLocaleDateString('en-US', {
              weekday: 'short', month: 'short', day: 'numeric',
            });

            return (
              <tr key={day.date} style={{ borderBottom: '1px solid #2a2a4a' }}>
                <td style={tdStyle}>{dayLabel}</td>
                <td style={tdStyle}>{totalTrades}</td>
                <td style={tdStyle}>{totalWins}/{totalLosses}</td>
                <td style={{ ...tdStyle, color: wr >= 60 ? '#00ff88' : wr >= 40 ? '#ffaa00' : '#ff4444' }}>
                  {wr.toFixed(0)}%
                </td>
                <td style={{ ...tdStyle, color: day.total >= 0 ? '#00ff88' : '#ff4444', fontWeight: 600 }}>
                  ${day.total.toFixed(2)}
                </td>
                <td style={{ ...tdStyle, color: pf != null && pf >= 1 ? '#00ff88' : '#ff4444' }}>
                  {pf?.toFixed(2) ?? '—'}
                </td>
                <td style={{ ...tdStyle, color: '#00ff88' }}>{totalTP}</td>
                <td style={{ ...tdStyle, color: '#ff4444' }}>{totalSL}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: 'left', padding: '6px 8px', color: '#888', fontWeight: 600,
  borderBottom: '1px solid #2a2a4a', whiteSpace: 'nowrap',
};

const tdStyle: React.CSSProperties = {
  padding: '5px 8px', color: '#ccc', whiteSpace: 'nowrap',
};

// --- Main Panel ---

type Tab = 'drift' | 'equity' | 'daily';

export function AnalyticsPanel({ drift, rolling, dailyStats, loaded, error }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>('drift');

  if (!loaded && !error) {
    return (
      <div style={containerStyle}>
        <span style={{ color: '#666', fontSize: 11, fontFamily: FONT }}>Loading analytics...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={containerStyle}>
        <span style={{ color: '#ff4444', fontSize: 11, fontFamily: FONT }}>Analytics: {error}</span>
      </div>
    );
  }

  // Filter to strategies with meaningful data (exclude shelved/no-benchmark)
  const activeDrift = drift.filter(d =>
    d.status !== 'NO_BENCHMARK'
  );

  // Overall portfolio status (worst of all strategies)
  const worstStatus = activeDrift.reduce((worst, d) => {
    const rank: Record<string, number> = { RED: 3, YELLOW: 2, GREEN: 1, INSUFFICIENT_DATA: 0, NO_BENCHMARK: 0 };
    return (rank[d.status] || 0) > (rank[worst] || 0) ? d.status : worst;
  }, 'GREEN' as string);

  const portfolioPnl = activeDrift.reduce((s, d) => s + (d.live_total_pnl ?? 0), 0);
  const portfolioTrades = activeDrift.reduce((s, d) => s + d.total_trades, 0);

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#e0e0e0', fontFamily: FONT }}>
            Analytics
          </span>
          <span style={{
            fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 3,
            backgroundColor: `${STATUS_COLORS[worstStatus] || '#666'}22`,
            color: STATUS_COLORS[worstStatus] || '#666',
            fontFamily: FONT, letterSpacing: 1,
          }}>
            {worstStatus}
          </span>
          <span style={{ fontSize: 10, color: '#666', fontFamily: FONT }}>
            {portfolioTrades} trades | ${portfolioPnl.toFixed(0)} net
          </span>
        </div>

        {/* Tab buttons */}
        <div style={{ display: 'flex', gap: 2 }}>
          {(['drift', 'equity', 'daily'] as Tab[]).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '4px 12px', fontSize: 10, fontWeight: 600, fontFamily: FONT,
                border: 'none', borderRadius: 3, cursor: 'pointer',
                backgroundColor: activeTab === tab ? '#2a3a5a' : 'transparent',
                color: activeTab === tab ? '#e0e0e0' : '#666',
                transition: 'all 0.15s ease',
              }}
            >
              {tab === 'drift' ? 'Drift' : tab === 'equity' ? 'Equity' : 'Daily'}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      {activeTab === 'drift' && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {activeDrift.map(d => <DriftCard key={d.strategy_id} d={d} />)}
        </div>
      )}

      {activeTab === 'equity' && (
        <EquityCurve rolling={rolling} />
      )}

      {activeTab === 'daily' && (
        <RecentDays dailyStats={dailyStats} />
      )}
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  padding: '14px 16px',
  backgroundColor: '#16213e',
  borderRadius: 6,
  border: '1px solid #2a2a4a',
};
