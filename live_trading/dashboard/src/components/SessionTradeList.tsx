import { useMemo } from 'react';
import type { Trade } from '../types';

interface SessionTradeListProps {
  trades: Trade[];
}

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
    timeZone: 'America/New_York',
  });
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', {
    month: '2-digit',
    day: '2-digit',
    year: '2-digit',
    timeZone: 'America/New_York',
  });
}

function strategyLabel(id?: string): string {
  if (!id) return '?';
  if (id.includes('V15')) return 'v15';
  if (id.includes('V11')) return 'v11';
  return id;
}

function exitLabel(reason: string): string {
  const map: Record<string, string> = {
    SM_FLIP: 'SM Flip',
    SL: 'Max Loss',
    TP: 'Take Profit',
    EOD: 'EOD Close',
    TRAIL: 'Trail Stop',
  };
  return map[reason] || reason;
}

export function SessionTradeList({ trades }: SessionTradeListProps) {
  const { rows, totals } = useMemo(() => {
    let cumPnl = 0;
    const rows = trades.map((t, idx) => {
      cumPnl += t.pnl;
      return { ...t, tradeNum: idx + 1, cumPnl };
    });

    const totalPnl = trades.reduce((s, t) => s + t.pnl, 0);
    const winners = trades.filter((t) => t.pnl > 0).length;
    const losers = trades.filter((t) => t.pnl <= 0).length;
    const avgMfe = trades.length > 0
      ? trades.reduce((s, t) => s + (t.mfe || 0), 0) / trades.length
      : 0;
    const avgMae = trades.length > 0
      ? trades.reduce((s, t) => s + (t.mae || 0), 0) / trades.length
      : 0;

    return {
      rows,
      totals: { totalPnl, winners, losers, avgMfe, avgMae },
    };
  }, [trades]);

  if (trades.length === 0) return null;

  const cellStyle: React.CSSProperties = {
    padding: '5px 8px',
    borderBottom: '1px solid #1e2a45',
    fontSize: 10,
    whiteSpace: 'nowrap',
  };

  const headerStyle: React.CSSProperties = {
    ...cellStyle,
    color: '#666',
    fontWeight: 600,
    borderBottom: '1px solid #2a2a4a',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    position: 'sticky',
    top: 0,
    backgroundColor: '#16213e',
    zIndex: 1,
  };

  return (
    <div
      style={{
        marginTop: 12,
        backgroundColor: '#16213e',
        borderRadius: 8,
        border: '1px solid #2a2a4a',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px 12px 8px',
          borderBottom: '1px solid #2a2a4a',
        }}
      >
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: '#e0e0e0',
            fontFamily: FONT,
          }}
        >
          List of Trades
        </span>
        <div style={{ display: 'flex', gap: 16, fontSize: 10, fontFamily: FONT }}>
          <span style={{ color: '#888' }}>
            {trades.length} trades
          </span>
          <span style={{ color: '#00ff88' }}>
            W: {totals.winners}
          </span>
          <span style={{ color: '#ff4444' }}>
            L: {totals.losers}
          </span>
          <span style={{ color: totals.totalPnl >= 0 ? '#00ff88' : '#ff4444' }}>
            Net: ${totals.totalPnl.toFixed(2)}
          </span>
          <span style={{ color: '#888' }}>
            Avg MFE: {totals.avgMfe.toFixed(1)}pt
          </span>
          <span style={{ color: '#888' }}>
            Avg MAE: {totals.avgMae.toFixed(1)}pt
          </span>
        </div>
      </div>

      {/* Table */}
      <div style={{ maxHeight: 400, overflowY: 'auto' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontFamily: FONT,
          }}
        >
          <thead>
            <tr>
              <th style={{ ...headerStyle, textAlign: 'center' }}>#</th>
              <th style={{ ...headerStyle, textAlign: 'center' }}>Strategy</th>
              <th style={{ ...headerStyle, textAlign: 'center' }}>Side</th>
              <th style={{ ...headerStyle, textAlign: 'left' }}>Entry</th>
              <th style={{ ...headerStyle, textAlign: 'left' }}>Exit</th>
              <th style={{ ...headerStyle, textAlign: 'center' }}>Signal</th>
              <th style={{ ...headerStyle, textAlign: 'right' }}>Pts</th>
              <th style={{ ...headerStyle, textAlign: 'right' }}>P&L</th>
              <th style={{ ...headerStyle, textAlign: 'right' }}>MFE</th>
              <th style={{ ...headerStyle, textAlign: 'right' }}>MAE</th>
              <th style={{ ...headerStyle, textAlign: 'right' }}>Cum P&L</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isWin = r.pnl > 0;
              const isV15 = (r.strategy_id || '').includes('V15');
              const stratColor = isV15 ? '#00aaff' : '#8888cc';
              const sideColor = r.side === 'LONG' ? '#00ff88' : '#ff4444';

              return (
                <tr
                  key={`${r.strategy_id}-${r.tradeNum}`}
                  style={{
                    backgroundColor: r.tradeNum % 2 === 0 ? '#1a1e35' : 'transparent',
                  }}
                >
                  {/* # */}
                  <td style={{ ...cellStyle, textAlign: 'center', color: '#555' }}>
                    {r.tradeNum}
                  </td>

                  {/* Strategy */}
                  <td style={{ ...cellStyle, textAlign: 'center' }}>
                    <span
                      style={{
                        padding: '1px 6px',
                        borderRadius: 3,
                        backgroundColor: isV15 ? '#00aaff15' : '#8888cc15',
                        color: stratColor,
                        fontWeight: 600,
                        fontSize: 9,
                      }}
                    >
                      {strategyLabel(r.strategy_id)}
                    </span>
                  </td>

                  {/* Side */}
                  <td style={{ ...cellStyle, textAlign: 'center', color: sideColor, fontWeight: 600 }}>
                    {r.side}
                  </td>

                  {/* Entry */}
                  <td style={{ ...cellStyle, textAlign: 'left' }}>
                    <span style={{ color: '#aaa' }}>{formatDate(r.entry_time)}</span>
                    {' '}
                    <span style={{ color: '#ccc' }}>{formatTime(r.entry_time)}</span>
                    {' '}
                    <span style={{ color: '#e0e0e0', fontWeight: 600 }}>
                      {r.entry_price.toFixed(2)}
                    </span>
                  </td>

                  {/* Exit */}
                  <td style={{ ...cellStyle, textAlign: 'left' }}>
                    <span style={{ color: '#aaa' }}>{formatDate(r.exit_time)}</span>
                    {' '}
                    <span style={{ color: '#ccc' }}>{formatTime(r.exit_time)}</span>
                    {' '}
                    <span style={{ color: '#e0e0e0', fontWeight: 600 }}>
                      {r.exit_price.toFixed(2)}
                    </span>
                  </td>

                  {/* Signal (exit reason) */}
                  <td style={{ ...cellStyle, textAlign: 'center' }}>
                    <span
                      style={{
                        padding: '1px 6px',
                        borderRadius: 3,
                        fontSize: 9,
                        fontWeight: 600,
                        backgroundColor:
                          r.exit_reason === 'TP' ? '#00ff8815' :
                          r.exit_reason === 'SL' ? '#ff444415' :
                          r.exit_reason === 'SM_FLIP' ? '#8888cc15' :
                          '#88888815',
                        color:
                          r.exit_reason === 'TP' ? '#00ff88' :
                          r.exit_reason === 'SL' ? '#ff4444' :
                          r.exit_reason === 'SM_FLIP' ? '#8888cc' :
                          '#888',
                      }}
                    >
                      {exitLabel(r.exit_reason)}
                    </span>
                  </td>

                  {/* Pts */}
                  <td
                    style={{
                      ...cellStyle,
                      textAlign: 'right',
                      color: isWin ? '#00ff88' : '#ff4444',
                    }}
                  >
                    {r.pts >= 0 ? '+' : ''}{r.pts.toFixed(2)}
                  </td>

                  {/* P&L */}
                  <td
                    style={{
                      ...cellStyle,
                      textAlign: 'right',
                      color: isWin ? '#00ff88' : '#ff4444',
                      fontWeight: 600,
                    }}
                  >
                    ${r.pnl >= 0 ? '+' : ''}{r.pnl.toFixed(2)}
                  </td>

                  {/* MFE */}
                  <td style={{ ...cellStyle, textAlign: 'right', color: '#00cc66' }}>
                    {(r.mfe || 0).toFixed(1)}
                  </td>

                  {/* MAE */}
                  <td style={{ ...cellStyle, textAlign: 'right', color: '#cc6644' }}>
                    {(r.mae || 0).toFixed(1)}
                  </td>

                  {/* Cumulative P&L */}
                  <td
                    style={{
                      ...cellStyle,
                      textAlign: 'right',
                      color: r.cumPnl >= 0 ? '#00ff88' : '#ff4444',
                      fontWeight: 600,
                    }}
                  >
                    ${r.cumPnl >= 0 ? '+' : ''}{r.cumPnl.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
