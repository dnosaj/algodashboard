import type { Trade } from '../types';

interface TradeLogProps {
  trades: Trade[];
}

const styles = {
  container: {
    backgroundColor: '#16213e',
    borderRadius: '8px',
    padding: '20px',
    width: '100%',
    maxHeight: '420px',
    display: 'flex',
    flexDirection: 'column' as const,
  } as React.CSSProperties,

  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '14px',
  } as React.CSSProperties,

  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#e0e0e0',
    margin: 0,
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,

  summary: {
    fontSize: '12px',
    color: '#888',
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,

  tableWrapper: {
    overflow: 'auto',
    flex: 1,
  } as React.CSSProperties,

  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
    fontSize: '12px',
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,

  th: {
    position: 'sticky' as const,
    top: 0,
    backgroundColor: '#16213e',
    color: '#666',
    textAlign: 'left' as const,
    padding: '8px 10px',
    borderBottom: '1px solid #2a2a4a',
    fontSize: '10px',
    textTransform: 'uppercase' as const,
    letterSpacing: '1px',
    whiteSpace: 'nowrap' as const,
  } as React.CSSProperties,

  thRight: {
    textAlign: 'right' as const,
  } as React.CSSProperties,

  td: {
    padding: '7px 10px',
    borderBottom: '1px solid #1a1a2e',
    whiteSpace: 'nowrap' as const,
  } as React.CSSProperties,

  tdRight: {
    textAlign: 'right' as const,
  } as React.CSSProperties,

  row: (pnl: number | null) =>
    ({
      backgroundColor:
        pnl === null
          ? 'transparent'
          : pnl > 0
            ? 'rgba(0,255,136,0.04)'
            : pnl < 0
              ? 'rgba(255,68,68,0.04)'
              : 'transparent',
      transition: 'background-color 0.2s',
    }) as React.CSSProperties,

  sideBadge: (side: string) =>
    ({
      display: 'inline-block',
      padding: '2px 6px',
      borderRadius: '3px',
      fontSize: '10px',
      fontWeight: 700,
      backgroundColor:
        side === 'LONG' ? 'rgba(0,255,136,0.15)' : 'rgba(255,68,68,0.15)',
      color: side === 'LONG' ? '#00ff88' : '#ff4444',
    }) as React.CSSProperties,

  pnlValue: (value: number | null) =>
    ({
      fontWeight: 700,
      color:
        value === null
          ? '#888'
          : value > 0
            ? '#00ff88'
            : value < 0
              ? '#ff4444'
              : '#888',
    }) as React.CSSProperties,

  emptyState: {
    textAlign: 'center' as const,
    color: '#555',
    padding: '40px 0',
    fontSize: '14px',
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,

  summaryBar: {
    display: 'flex',
    gap: '24px',
    marginTop: '14px',
    paddingTop: '14px',
    borderTop: '1px solid #2a2a4a',
    fontSize: '11px',
    color: '#888',
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  } as React.CSSProperties,

  summaryValue: (color?: string) =>
    ({
      color: color || '#ccc',
      fontWeight: 600,
      marginLeft: '6px',
    }) as React.CSSProperties,
};

function formatTime(timestamp: string): string {
  try {
    const d = new Date(timestamp);
    return d.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return timestamp;
  }
}

function formatPrice(price: number | null): string {
  if (price === null) return '--';
  return price.toFixed(2);
}

function formatPnl(value: number | null): string {
  if (value === null) return '--';
  const sign = value >= 0 ? '+' : '';
  return `${sign}$${value.toFixed(2)}`;
}

function formatPts(value: number | null): string {
  if (value === null) return '--';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}`;
}

export function TradeLog({ trades }: TradeLogProps) {
  const closedTrades = trades;
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const winners = closedTrades.filter((t) => (t.pnl || 0) > 0).length;
  const losers = closedTrades.filter((t) => (t.pnl || 0) < 0).length;
  const winRate =
    closedTrades.length > 0
      ? ((winners / closedTrades.length) * 100).toFixed(1)
      : '0.0';

  // Per-strategy P&L breakdown
  const stratPnl: Record<string, number> = {};
  for (const t of closedTrades) {
    const key = t.strategy_id || t.instrument;
    stratPnl[key] = (stratPnl[key] || 0) + (t.pnl || 0);
  }

  // Sort trades: most recent first
  const sortedTrades = [...trades].sort(
    (a, b) => new Date(b.exit_time).getTime() - new Date(a.exit_time).getTime()
  );

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Trade Log</h3>
        <span style={styles.summary}>
          {trades.length} trades today
        </span>
      </div>

      <div style={styles.tableWrapper}>
        {sortedTrades.length === 0 ? (
          <div style={styles.emptyState}>No trades yet today</div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Time</th>
                <th style={styles.th}>Strategy</th>
                <th style={styles.th}>Side</th>
                <th style={{ ...styles.th, ...styles.thRight }}>Entry</th>
                <th style={{ ...styles.th, ...styles.thRight }}>Exit</th>
                <th style={{ ...styles.th, ...styles.thRight }}>Pts</th>
                <th style={{ ...styles.th, ...styles.thRight }}>P&L</th>
                <th style={styles.th}>Exit</th>
              </tr>
            </thead>
            <tbody>
              {sortedTrades.map((trade) => (
                <tr key={trade.entry_time + (trade.strategy_id || trade.instrument)} style={styles.row(trade.pnl)}>
                  <td style={styles.td}>{formatTime(trade.exit_time)}</td>
                  <td style={{ ...styles.td, color: '#8888cc', fontSize: 10 }}>
                    {trade.strategy_id || trade.instrument}
                  </td>
                  <td style={styles.td}>
                    <span style={styles.sideBadge(trade.side)}>
                      {trade.side}
                    </span>
                  </td>
                  <td style={{ ...styles.td, ...styles.tdRight, color: '#ccc' }}>
                    {formatPrice(trade.entry_price)}
                  </td>
                  <td style={{ ...styles.td, ...styles.tdRight, color: '#ccc' }}>
                    {formatPrice(trade.exit_price)}
                  </td>
                  <td
                    style={{
                      ...styles.td,
                      ...styles.tdRight,
                      ...styles.pnlValue(trade.pts),
                    }}
                  >
                    {formatPts(trade.pts)}
                  </td>
                  <td
                    style={{
                      ...styles.td,
                      ...styles.tdRight,
                      ...styles.pnlValue(trade.pnl),
                    }}
                  >
                    {formatPnl(trade.pnl)}
                  </td>
                  <td style={{ ...styles.td, color: '#666', fontSize: 10 }}>
                    {trade.exit_reason || '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {closedTrades.length > 0 && (
        <div style={styles.summaryBar}>
          <span>
            W/L:
            <span style={styles.summaryValue('#00ff88')}>{winners}</span>
            <span style={{ color: '#555' }}> / </span>
            <span style={styles.summaryValue('#ff4444')}>{losers}</span>
          </span>
          <span>
            WR:
            <span style={styles.summaryValue()}>{winRate}%</span>
          </span>
          <span style={{ color: '#555' }}>|</span>
          {Object.entries(stratPnl).map(([sid, pnl]) => (
            <span key={sid}>
              {sid}:
              <span style={styles.summaryValue(pnl >= 0 ? '#00ff88' : '#ff4444')}>
                {formatPnl(pnl)}
              </span>
            </span>
          ))}
          <span style={{ color: '#555' }}>|</span>
          <span style={{ fontWeight: 700 }}>
            Portfolio:
            <span
              style={styles.summaryValue(totalPnl >= 0 ? '#00ff88' : '#ff4444')}
            >
              {formatPnl(totalPnl)}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}
