import type { SafetyStatusData } from '../types';

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

interface ContextPanelProps {
  safety: SafetyStatusData;
}

function pnlColor(value: number): string {
  return value >= 0 ? '#00ff88' : '#ff4444';
}

function formatPnl(value: number): string {
  return `${value >= 0 ? '+' : ''}$${value.toFixed(0)}`;
}

function drawdownColor(pct: number): string {
  if (pct > 66) return '#ff4444';
  if (pct > 33) return '#ffaa00';
  return '#00ff88';
}

function drawdownBg(pct: number): string {
  if (pct > 66) return 'rgba(255,68,68,0.25)';
  if (pct > 33) return 'rgba(255,170,0,0.25)';
  return 'rgba(0,255,136,0.25)';
}

function getStatusMessage(
  ctx: NonNullable<SafetyStatusData['portfolio_context']>,
  bench: NonNullable<SafetyStatusData['backtest_benchmarks']>,
  halted: boolean,
  dailyLimitHit: boolean,
): { message: string; color: string } {
  // Daily limit special message
  if (dailyLimitHit) {
    const expectedLossDays = Math.round(22 * bench.expected_loss_day_rate);
    return {
      message: `Daily limit reached. Month: ${formatPnl(ctx.monthly_pnl)}. This is loss day ${ctx.loss_days_this_month + 1} of ~${expectedLossDays} expected.`,
      color: '#ffaa00',
    };
  }

  // System paused/halted
  if (halted) {
    const ddPct = bench.max_drawdown > 0
      ? (ctx.current_drawdown / bench.max_drawdown) * 100
      : 0;
    return {
      message: `System paused. Drawdown: $${ctx.current_drawdown.toFixed(0)} (${ddPct.toFixed(0)}% of max). Month: ${formatPnl(ctx.monthly_pnl)}.`,
      color: '#ffaa00',
    };
  }

  const ddPct = bench.max_drawdown > 0
    ? (ctx.current_drawdown / bench.max_drawdown) * 100
    : 0;
  const streak = ctx.consecutive_losses;

  // RED
  if (ddPct > 66 || streak >= 5) {
    return {
      message: 'Outside historical parameters. Review settings.',
      color: '#ff4444',
    };
  }

  // YELLOW
  if (ddPct > 33 || streak >= 3) {
    return {
      message: 'Unusual but seen in backtesting. Monitor, don\'t intervene.',
      color: '#ffaa00',
    };
  }

  // GREEN
  return {
    message: 'Within normal variance. System is working as designed.',
    color: '#00ff88',
  };
}

export function ContextPanel({ safety }: ContextPanelProps) {
  const ctx = safety.portfolio_context;
  const bench = safety.backtest_benchmarks;

  // Don't render if data hasn't arrived yet
  if (!ctx || !bench) return null;

  const ddPct = bench.max_drawdown > 0
    ? (ctx.current_drawdown / bench.max_drawdown) * 100
    : 0;
  const ddPctClamped = Math.min(ddPct, 100);

  const expectedLossDays = Math.round(22 * bench.expected_loss_day_rate);

  // Detect if daily limit was just hit (halted + daily P&L is negative)
  const dailyLimitHit = safety.halted && ctx.daily_pnl < 0;

  const { message, color: msgColor } = getStatusMessage(
    ctx, bench, safety.halted || safety.extended_pause, dailyLimitHit,
  );

  return (
    <div style={{
      backgroundColor: '#16213e',
      borderRadius: 8,
      padding: '10px 16px',
      border: '1px solid #2a2a4a',
    }}>
      {/* Row 1: P&L Context */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        marginBottom: 8,
      }}>
        <span style={{ fontSize: 10, color: '#666', fontFamily: FONT, textTransform: 'uppercase', letterSpacing: 1 }}>
          Context
        </span>
        <span style={{ color: '#2a2a4a', fontSize: 10 }}>|</span>
        <span style={{ fontSize: 11, fontFamily: FONT, color: '#888' }}>Today:</span>
        <span style={{ fontSize: 11, fontFamily: FONT, fontWeight: 600, color: pnlColor(ctx.daily_pnl) }}>
          {formatPnl(ctx.daily_pnl)}
        </span>
        <span style={{ color: '#2a2a4a', fontSize: 10 }}>|</span>
        <span style={{ fontSize: 11, fontFamily: FONT, color: '#888' }}>Week:</span>
        <span style={{ fontSize: 11, fontFamily: FONT, fontWeight: 600, color: pnlColor(ctx.weekly_pnl) }}>
          {formatPnl(ctx.weekly_pnl)}
        </span>
        <span style={{ color: '#2a2a4a', fontSize: 10 }}>|</span>
        <span style={{ fontSize: 11, fontFamily: FONT, color: '#888' }}>Month:</span>
        <span style={{ fontSize: 11, fontFamily: FONT, fontWeight: 600, color: pnlColor(ctx.monthly_pnl) }}>
          {formatPnl(ctx.monthly_pnl)}
        </span>

        {/* Streak + Loss Days — right side */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 10, fontFamily: FONT, color: '#888' }}>
            Consec losses:
          </span>
          <span style={{
            fontSize: 10, fontFamily: FONT, fontWeight: 600,
            color: ctx.consecutive_losses >= 5 ? '#ff4444'
              : ctx.consecutive_losses >= 3 ? '#ffaa00' : '#aaa',
          }}>
            {ctx.consecutive_losses} / {bench.worst_streak} max
          </span>
          <span style={{ color: '#2a2a4a', fontSize: 10 }}>|</span>
          <span style={{ fontSize: 10, fontFamily: FONT, color: '#888' }}>
            Loss days:
          </span>
          <span style={{ fontSize: 10, fontFamily: FONT, fontWeight: 600, color: '#aaa' }}>
            {ctx.loss_days_this_month} / ~{expectedLossDays} expected
          </span>
        </div>
      </div>

      {/* Row 2: Drawdown bar + status message */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {/* Drawdown bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 280 }}>
          <span style={{ fontSize: 10, fontFamily: FONT, color: '#888', whiteSpace: 'nowrap' }}>
            DD: ${ctx.current_drawdown.toFixed(0)} / ${bench.max_drawdown}
          </span>
          <div style={{
            flex: 1, height: 8, backgroundColor: '#1a1a2e',
            borderRadius: 4, overflow: 'hidden', minWidth: 100,
          }}>
            <div style={{
              width: `${ddPctClamped}%`,
              height: '100%',
              backgroundColor: drawdownBg(ddPct),
              borderRadius: 4,
              transition: 'width 0.3s ease',
            }} />
          </div>
          <span style={{
            fontSize: 10, fontFamily: FONT, fontWeight: 600,
            color: drawdownColor(ddPct), whiteSpace: 'nowrap',
          }}>
            {ddPct.toFixed(0)}%
          </span>
        </div>

        {/* Status message */}
        <span style={{
          fontSize: 11, fontFamily: FONT, fontWeight: 600,
          color: msgColor, flex: 1,
        }}>
          {message}
        </span>
      </div>
    </div>
  );
}
