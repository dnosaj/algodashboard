import {
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
  Cell,
} from 'recharts';
import type { DailyPnLEntry } from '../types';

interface DailyPnLProps {
  data: DailyPnLEntry[];
}

const styles = {
  container: {
    backgroundColor: '#16213e',
    borderRadius: '8px',
    padding: '20px',
    width: '100%',
  } as React.CSSProperties,

  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  } as React.CSSProperties,

  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#e0e0e0',
    margin: 0,
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,

  subtitle: {
    fontSize: '11px',
    color: '#666',
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,

  emptyState: {
    textAlign: 'center' as const,
    color: '#555',
    padding: '60px 0',
    fontSize: '14px',
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,
};

const tooltipStyle = {
  backgroundColor: '#1a1a2e',
  border: '1px solid #2a2a4a',
  borderRadius: '6px',
  padding: '10px 14px',
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  fontSize: '12px',
};

interface TooltipPayloadItem {
  dataKey: string;
  value: number;
  color: string;
  payload: DailyPnLEntry;
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;

  const pnl = payload.find((p) => p.dataKey === 'pnl');
  const cumulative = payload.find((p) => p.dataKey === 'cumulative');

  return (
    <div style={tooltipStyle}>
      <div style={{ color: '#888', marginBottom: '6px' }}>{label}</div>
      {pnl && (
        <div
          style={{
            color: pnl.value >= 0 ? '#00ff88' : '#ff4444',
            fontWeight: 700,
          }}
        >
          Day: {pnl.value >= 0 ? '+' : ''}${pnl.value.toFixed(2)}
        </div>
      )}
      {cumulative && (
        <div
          style={{
            color: cumulative.value >= 0 ? '#00cc66' : '#cc3333',
            marginTop: '4px',
          }}
        >
          Cumulative: {cumulative.value >= 0 ? '+' : ''}$
          {cumulative.value.toFixed(2)}
        </div>
      )}
    </div>
  );
}

function formatDate(dateStr: string): string {
  try {
    const parts = dateStr.split('-');
    return `${parts[1]}/${parts[2]}`;
  } catch {
    return dateStr;
  }
}

export function DailyPnL({ data }: DailyPnLProps) {
  // If we have data but no cumulative field computed, compute it
  const chartData = data.map((entry, i) => ({
    ...entry,
    dateLabel: formatDate(entry.date),
    cumulative:
      entry.cumulative !== undefined
        ? entry.cumulative
        : data.slice(0, i + 1).reduce((sum, e) => sum + e.pnl, 0),
  }));

  // Take last 30 days
  const displayData = chartData.slice(-30);

  const totalPnl = displayData.reduce((sum, d) => sum + d.pnl, 0);
  const winDays = displayData.filter((d) => d.pnl > 0).length;
  const loseDays = displayData.filter((d) => d.pnl < 0).length;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Daily P&L</h3>
        <span style={styles.subtitle}>
          {displayData.length} days | {winDays}W {loseDays}L |{' '}
          <span style={{ color: totalPnl >= 0 ? '#00ff88' : '#ff4444' }}>
            {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
          </span>
        </span>
      </div>

      {displayData.length === 0 ? (
        <div style={styles.emptyState}>No daily P&L data available</div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart
            data={displayData}
            margin={{ top: 10, right: 20, bottom: 10, left: 10 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#2a2a4a"
              vertical={false}
            />
            <XAxis
              dataKey="dateLabel"
              tick={{ fill: '#666', fontSize: 10 }}
              axisLine={{ stroke: '#2a2a4a' }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: '#666', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `$${v}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="pnl" radius={[3, 3, 0, 0]} maxBarSize={24}>
              {displayData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.pnl >= 0 ? '#00ff88' : '#ff4444'}
                  fillOpacity={0.7}
                />
              ))}
            </Bar>
            <Line
              type="monotone"
              dataKey="cumulative"
              stroke="#8888cc"
              strokeWidth={2}
              dot={false}
              strokeDasharray="4 2"
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
