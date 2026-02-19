import type { SignalEvent } from '../types';

interface SignalFeedProps {
  signals: SignalEvent[];
}

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  } catch {
    return '--:--';
  }
}

function getSignalColor(type: string): string {
  switch (type) {
    case 'BUY': return '#00ff88';
    case 'SELL': return '#ff4444';
    case 'CLOSE_LONG': return '#ffaa00';
    case 'CLOSE_SHORT': return '#ffaa00';
    default: return '#888';
  }
}

function getSignalLabel(type: string): string {
  switch (type) {
    case 'BUY': return 'LONG ENTRY';
    case 'SELL': return 'SHORT ENTRY';
    case 'CLOSE_LONG': return 'CLOSE LONG';
    case 'CLOSE_SHORT': return 'CLOSE SHORT';
    default: return type;
  }
}

export function SignalFeed({ signals }: SignalFeedProps) {
  return (
    <div style={{
      backgroundColor: '#16213e',
      borderRadius: 8,
      padding: 16,
      minHeight: 120,
      maxHeight: 220,
      display: 'flex',
      flexDirection: 'column',
    }}>
      <h3 style={{
        fontSize: 13, fontWeight: 600, color: '#e0e0e0', margin: '0 0 10px 0',
        fontFamily: FONT, letterSpacing: 1, textTransform: 'uppercase',
      }}>
        Signal Activity
      </h3>

      <div style={{ overflow: 'auto', flex: 1 }}>
        {signals.length === 0 ? (
          <div style={{ color: '#444', fontSize: 12, fontFamily: FONT, padding: '16px 0', textAlign: 'center' }}>
            Waiting for signals...
          </div>
        ) : (
          signals.map((sig, i) => {
            const color = getSignalColor(sig.type);
            return (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '5px 0',
                borderBottom: i < signals.length - 1 ? '1px solid #1a1a2e' : 'none',
                fontSize: 11, fontFamily: FONT,
                opacity: i === 0 ? 1 : 0.7 + (0.3 * Math.max(0, 1 - i / 10)),
              }}>
                <span style={{ color: '#555', minWidth: 60 }}>{formatTime(sig.ts)}</span>
                <span style={{ color: '#8888cc', minWidth: 32, fontWeight: 600 }}>{sig.instrument}</span>
                <span style={{
                  color, fontWeight: 700, minWidth: 90,
                  padding: '1px 5px', borderRadius: 2,
                  backgroundColor: `${color}10`,
                }}>
                  {getSignalLabel(sig.type)}
                </span>
                <span style={{ color: '#666' }}>
                  SM={sig.sm_value >= 0 ? '+' : ''}{sig.sm_value.toFixed(3)}
                </span>
                <span style={{ color: '#666' }}>
                  RSI={sig.rsi_value.toFixed(1)}
                </span>
                {sig.reason && (
                  <span style={{ color: '#444', marginLeft: 'auto' }}>{sig.reason}</span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
