import type { StatusData, ConnectionStatus } from '../types';

interface StatusPanelProps {
  status: StatusData | null;
  connected: ConnectionStatus;
}

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function formatCurrency(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}$${value.toFixed(2)}`;
}

function ConnectionDot({ status }: { status: ConnectionStatus }) {
  const color = status === 'connected' ? '#00ff88' : status === 'reconnecting' ? '#ffaa00' : '#ff4444';
  return (
    <span style={{
      width: 8, height: 8, borderRadius: '50%',
      backgroundColor: color, boxShadow: `0 0 6px ${color}`,
      display: 'inline-block',
    }} />
  );
}

function Badge({ label, active, color }: { label: string; active?: boolean; color?: string }) {
  const bg = color || (active ? '#00ff88' : '#ff4444');
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 3,
      fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase',
      backgroundColor: `${bg}18`, color: bg, border: `1px solid ${bg}33`,
      fontFamily: FONT,
    }}>
      {label}
    </span>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 9, color: '#555', textTransform: 'uppercase', letterSpacing: 1, fontFamily: FONT, marginBottom: 2 }}>
        {label}
      </div>
      <div style={{ fontSize: 14, fontWeight: 700, color: color || '#e0e0e0', fontFamily: FONT }}>
        {value}
      </div>
    </div>
  );
}

export function StatusPanel({ status, connected }: StatusPanelProps) {
  const dailyPnl = status?.daily_pnl ?? 0;
  const pnlColor = dailyPnl >= 0 ? '#00ff88' : '#ff4444';

  return (
    <div style={{
      backgroundColor: '#16213e', borderRadius: 8, padding: '12px 20px',
      display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
    }}>
      {/* Connection + Broker */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <ConnectionDot status={connected} />
        <span style={{ fontSize: 11, color: connected === 'connected' ? '#00ff88' : '#ff4444', fontFamily: FONT, textTransform: 'uppercase', letterSpacing: 1 }}>
          {connected}
        </span>
      </div>

      {status?.broker && (
        <span style={{ fontSize: 10, color: '#555', fontFamily: FONT }}>
          {status.broker}{status.account ? `:${status.account.slice(-4)}` : ''}
        </span>
      )}

      {/* Badges */}
      {status && (
        <div style={{ display: 'flex', gap: 6 }}>
          <Badge label={status.trading_active ? 'ACTIVE' : 'PAUSED'} active={status.trading_active} />
          <Badge label={status.paper_mode ? 'PAPER' : 'LIVE'} color={status.paper_mode ? '#ffaa00' : '#ff4444'} />
        </div>
      )}

      {/* Separator */}
      <div style={{ width: 1, height: 28, backgroundColor: '#2a2a4a' }} />

      {/* Stats */}
      <Stat label="Daily P&L" value={status ? formatCurrency(dailyPnl) : '--'} color={pnlColor} />
      <Stat label="Trades" value={status ? `${status.trade_count_today}` : '--'} color="#8888cc" />
      {(status?.consecutive_losses ?? 0) > 0 && (
        <Stat label="Consec Loss" value={`${status?.consecutive_losses}`} color="#ff4444" />
      )}

      {/* Right-aligned uptime */}
      <div style={{ marginLeft: 'auto' }}>
        <Stat label="Uptime" value={status ? formatUptime(status.uptime_seconds) : '--:--:--'} color="#555" />
      </div>
    </div>
  );
}
