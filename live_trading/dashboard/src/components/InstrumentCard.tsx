import type { Position, InstrumentData } from '../types';

interface InstrumentCardProps {
  position: Position;
  data: InstrumentData | null;
}

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

function formatPrice(price: number): string {
  return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function SmGauge({ value }: { value: number }) {
  // SM ranges roughly -1 to +1. Clamp for display.
  const clamped = Math.max(-1, Math.min(1, value));
  const pct = ((clamped + 1) / 2) * 100; // 0-100 where 50 is neutral
  const isBull = value > 0;
  const color = isBull ? '#00ff88' : '#ff4444';

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: '#666', fontFamily: FONT, letterSpacing: 1, textTransform: 'uppercase' }}>
          Smart Money
        </span>
        <span style={{ fontSize: 12, fontFamily: FONT, fontWeight: 700, color }}>
          {value >= 0 ? '+' : ''}{value.toFixed(4)} {isBull ? '\u25B2' : '\u25BC'}
        </span>
      </div>
      <div style={{
        height: 6, borderRadius: 3, backgroundColor: '#1a1a2e',
        position: 'relative', overflow: 'hidden',
      }}>
        {/* Center line */}
        <div style={{
          position: 'absolute', left: '50%', top: 0, bottom: 0,
          width: 1, backgroundColor: '#2a2a4a',
        }} />
        {/* Fill bar */}
        <div style={{
          position: 'absolute',
          top: 0, bottom: 0,
          left: isBull ? '50%' : `${pct}%`,
          width: `${Math.abs(pct - 50)}%`,
          backgroundColor: color,
          borderRadius: 3,
          boxShadow: `0 0 8px ${color}40`,
          transition: 'all 0.3s ease',
        }} />
      </div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: 9, color: '#444', fontFamily: FONT, marginTop: 2,
      }}>
        <span>BEAR</span>
        <span>BULL</span>
      </div>
    </div>
  );
}

function RsiBar({ value, buyLevel, sellLevel }: { value: number; buyLevel: number; sellLevel: number }) {
  const pct = Math.max(0, Math.min(100, value));
  const isOverbought = value >= buyLevel;
  const isOversold = value <= sellLevel;
  const color = isOverbought ? '#00ff88' : isOversold ? '#ff4444' : '#8888cc';

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: '#666', fontFamily: FONT, letterSpacing: 1, textTransform: 'uppercase' }}>
          RSI (5m)
        </span>
        <span style={{ fontSize: 12, fontFamily: FONT, fontWeight: 700, color }}>
          {value.toFixed(1)}
        </span>
      </div>
      <div style={{
        height: 6, borderRadius: 3, backgroundColor: '#1a1a2e',
        position: 'relative', overflow: 'hidden',
      }}>
        {/* Sell level marker */}
        <div style={{
          position: 'absolute', left: `${sellLevel}%`, top: 0, bottom: 0,
          width: 1, backgroundColor: '#ff444466',
        }} />
        {/* Buy level marker */}
        <div style={{
          position: 'absolute', left: `${buyLevel}%`, top: 0, bottom: 0,
          width: 1, backgroundColor: '#00ff8866',
        }} />
        {/* RSI fill */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0, left: 0,
          width: `${pct}%`,
          backgroundColor: color,
          borderRadius: 3,
          opacity: 0.7,
          transition: 'all 0.3s ease',
        }} />
      </div>
    </div>
  );
}

function CooldownBar({ remaining, total }: { remaining: number; total: number }) {
  const pct = total > 0 ? ((total - remaining) / total) * 100 : 100;
  const ready = remaining === 0;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
      <span style={{ fontSize: 10, color: '#666', fontFamily: FONT, letterSpacing: 1, textTransform: 'uppercase', minWidth: 56 }}>
        Cooldown
      </span>
      <div style={{
        flex: 1, height: 4, borderRadius: 2, backgroundColor: '#1a1a2e',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', borderRadius: 2,
          width: `${pct}%`,
          backgroundColor: ready ? '#00ff8866' : '#ffaa00',
          transition: 'width 0.3s ease',
        }} />
      </div>
      <span style={{
        fontSize: 10, fontFamily: FONT, fontWeight: 600, minWidth: 36, textAlign: 'right',
        color: ready ? '#00ff88' : '#ffaa00',
      }}>
        {ready ? 'READY' : `${remaining}`}
      </span>
    </div>
  );
}

export function InstrumentCard({ position, data }: InstrumentCardProps) {
  const inst = position.instrument;
  const side = position.side;
  const hasTrade = side !== 'FLAT';
  const sideColor = side === 'LONG' ? '#00ff88' : side === 'SHORT' ? '#ff4444' : '#555';

  // Derive RSI buy/sell levels from version
  const buyLevel = data?.version === 'v11' ? 60 : 55;
  const sellLevel = data?.version === 'v11' ? 40 : 45;

  return (
    <div style={{
      backgroundColor: '#16213e',
      borderRadius: 8,
      padding: 16,
      border: hasTrade ? `1px solid ${sideColor}33` : '1px solid #2a2a4a',
      flex: 1,
      minWidth: 320,
    }}>
      {/* Header: instrument + version + price */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 18, fontWeight: 700, color: '#8888cc', fontFamily: FONT }}>
            {inst}
          </span>
          <span style={{
            fontSize: 10, padding: '2px 6px', borderRadius: 3,
            backgroundColor: 'rgba(136,136,204,0.12)', color: '#8888cc',
            fontFamily: FONT, fontWeight: 600,
          }}>
            {data?.version || '--'}
          </span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#e0e0e0', fontFamily: FONT }}>
            {data?.last_price ? formatPrice(data.last_price) : '--'}
          </div>
        </div>
      </div>

      {/* SM Gauge */}
      <SmGauge value={data?.sm_value ?? 0} />

      {/* RSI Bar */}
      <RsiBar value={data?.rsi_value ?? 50} buyLevel={buyLevel} sellLevel={sellLevel} />

      {/* Cooldown */}
      <CooldownBar
        remaining={data?.cooldown_remaining ?? 0}
        total={data?.cooldown_total ?? 20}
      />

      {/* Divider */}
      <div style={{ borderTop: '1px solid #2a2a4a', margin: '10px 0' }} />

      {/* Position */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{
            fontSize: 14, fontWeight: 700, color: sideColor, fontFamily: FONT,
            display: 'inline-block', padding: '2px 8px', borderRadius: 3,
            backgroundColor: hasTrade ? `${sideColor}15` : 'transparent',
          }}>
            {side}
          </span>
          {hasTrade && position.entry_price && (
            <span style={{ fontSize: 11, color: '#888', fontFamily: FONT, marginLeft: 8 }}>
              @ {formatPrice(position.entry_price)}
            </span>
          )}
        </div>
        <div style={{ textAlign: 'right' }}>
          {hasTrade ? (
            <>
              <div style={{
                fontSize: 14, fontWeight: 700, fontFamily: FONT,
                color: position.unrealized_pnl >= 0 ? '#00ff88' : '#ff4444',
              }}>
                {position.unrealized_pnl >= 0 ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
              </div>
              <div style={{ fontSize: 10, color: '#666', fontFamily: FONT }}>
                {data?.bars_held ?? 0} bars held
                {data?.max_loss_pts ? ` | SL ${data.max_loss_pts}pt` : ''}
              </div>
            </>
          ) : (
            <div style={{ fontSize: 10, color: '#555', fontFamily: FONT }}>
              {data?.long_used && data?.short_used ? 'Both sides used' :
               data?.long_used ? 'Long used' :
               data?.short_used ? 'Short used' : 'Waiting for signal'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
