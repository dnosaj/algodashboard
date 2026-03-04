import { useState, useRef, useEffect } from 'react';
import type { SafetyStatusData, Position } from '../types';

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

interface SafetyPanelProps {
  safety: SafetyStatusData;
  positions: Position[];
  sendCommand: (command: string, payload?: Record<string, unknown>) => void;
}

function DrawdownBadge({ mode }: { mode: string }) {
  const colors: Record<string, { bg: string; text: string }> = {
    NORMAL: { bg: 'rgba(0,255,136,0.12)', text: '#00ff88' },
    REDUCED: { bg: 'rgba(255,170,0,0.12)', text: '#ffaa00' },
    EXTENDED_PAUSE: { bg: 'rgba(255,68,68,0.12)', text: '#ff4444' },
  };
  const c = colors[mode] || colors.NORMAL;
  const label = mode === 'EXTENDED_PAUSE' ? 'EXTENDED PAUSE' : mode;

  return (
    <span style={{
      fontSize: 11, fontWeight: 700, fontFamily: FONT,
      padding: '3px 10px', borderRadius: 4,
      backgroundColor: c.bg, color: c.text,
      letterSpacing: 1,
    }}>
      {label}
    </span>
  );
}

function PausedBadge({ paused }: { paused: boolean }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, fontFamily: FONT,
      padding: '2px 6px', borderRadius: 3,
      backgroundColor: paused ? 'rgba(255,68,68,0.12)' : 'rgba(0,255,136,0.12)',
      color: paused ? '#ff4444' : '#00ff88',
    }}>
      {paused ? 'PAUSED' : 'ACTIVE'}
    </span>
  );
}

const sizingBtnStyle: React.CSSProperties = {
  fontSize: 10, fontFamily: FONT, fontWeight: 700,
  width: 18, height: 20, borderRadius: 3,
  border: '1px solid #2a2a4a', cursor: 'pointer',
  backgroundColor: '#16213e', color: '#8888cc',
  padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
};

function SizingInput({ label, value, min = 1, max = 99, onChange }: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
      <span style={{ fontSize: 10, color: '#666', fontFamily: FONT }}>{label}:</span>
      <button
        onClick={() => { if (value > min) onChange(value - 1); }}
        disabled={value <= min}
        style={{ ...sizingBtnStyle, opacity: value <= min ? 0.3 : 1 }}
      >
        -
      </button>
      <span style={{
        fontSize: 11, fontFamily: FONT, fontWeight: 700,
        color: '#8888cc', minWidth: 16, textAlign: 'center',
      }}>
        {value}
      </span>
      <button
        onClick={() => { if (value < max) onChange(value + 1); }}
        disabled={value >= max}
        style={{ ...sizingBtnStyle, opacity: value >= max ? 0.3 : 1 }}
      >
        +
      </button>
    </div>
  );
}

export function SafetyPanel({ safety, positions, sendCommand }: SafetyPanelProps) {
  const strategies = Object.values(safety.strategies || {});
  const isHaltedOrPaused = safety.halted || safety.extended_pause;
  const [closingStrategy, setClosingStrategy] = useState<string | null>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  const getPositionSide = (strategyId: string): string => {
    const pos = positions.find((p) => p.strategy_id === strategyId);
    return pos?.side || 'FLAT';
  };

  const handleClose = (strategyId: string) => {
    setClosingStrategy(strategyId);
    sendCommand('strategy_close', { strategy_id: strategyId });
    // Re-enable after 3s (status update will confirm FLAT sooner)
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
    }
    closeTimerRef.current = setTimeout(() => setClosingStrategy(null), 3000);
  };

  return (
    <div style={{
      backgroundColor: '#16213e',
      borderRadius: 8,
      padding: 16,
      border: '1px solid #2a2a4a',
    }}>
      {/* Header row */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{
            fontSize: 12, fontWeight: 700, color: '#888',
            fontFamily: FONT, letterSpacing: 1, textTransform: 'uppercase',
          }}>
            Safety
          </span>
          <DrawdownBadge mode={safety.drawdown_mode} />
          {safety.halted && (
            <span style={{
              fontSize: 10, fontWeight: 700, fontFamily: FONT,
              padding: '2px 8px', borderRadius: 3,
              backgroundColor: 'rgba(255,68,68,0.2)', color: '#ff4444',
            }}>
              HALTED: {safety.halt_reason}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 10, color: '#666', fontFamily: FONT }}>
            VIX: {safety.vix_close?.toFixed(1) ?? '—'}
          </span>
          <span style={{ fontSize: 10, color: '#666', fontFamily: FONT }}>
            5d SL: {safety.rolling_sl_5d}
          </span>
          <button
            onClick={() => sendCommand('drawdown_toggle', { enabled: !safety.drawdown_enabled })}
            style={{
              fontSize: 10, fontFamily: FONT, fontWeight: 600,
              padding: '4px 10px', borderRadius: 4,
              border: '1px solid #2a2a4a', cursor: 'pointer',
              backgroundColor: safety.drawdown_enabled ? 'rgba(0,255,136,0.08)' : 'rgba(255,68,68,0.08)',
              color: safety.drawdown_enabled ? '#00ff88' : '#ff4444',
            }}
          >
            DD {safety.drawdown_enabled ? 'ON' : 'OFF'}
          </button>
          {isHaltedOrPaused && (
            <button
              onClick={() => { sendCommand('force_resume'); sendCommand('resume'); }}
              style={{
                fontSize: 10, fontFamily: FONT, fontWeight: 700,
                padding: '4px 10px', borderRadius: 4,
                border: '1px solid #ff444466', cursor: 'pointer',
                backgroundColor: 'rgba(255,68,68,0.12)', color: '#ff4444',
              }}
            >
              FORCE RESUME ALL
            </button>
          )}
        </div>
      </div>

      {/* Per-strategy rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {strategies.map((s) => (
          <div
            key={s.strategy_id}
            style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '8px 12px', borderRadius: 6,
              backgroundColor: '#1a1a2e',
            }}
          >
            {/* Strategy ID */}
            <span style={{
              fontSize: 12, fontWeight: 700, fontFamily: FONT,
              color: '#8888cc', minWidth: 90,
            }}>
              {s.strategy_id}
            </span>

            {/* Status badge: PAUSED > VIX GATE > ACTIVE */}
            {s.paused ? (
              <PausedBadge paused={true} />
            ) : s.vix_gated ? (
              <span style={{
                fontSize: 10, fontWeight: 600, fontFamily: FONT,
                padding: '2px 6px', borderRadius: 3,
                backgroundColor: 'rgba(255,170,0,0.12)',
                color: '#ffaa00',
              }}>
                VIX GATE
              </span>
            ) : (
              <PausedBadge paused={false} />
            )}

            {/* Pause/Resume button */}
            <button
              onClick={() => sendCommand(
                s.paused ? 'strategy_resume' : 'strategy_pause',
                { strategy_id: s.strategy_id },
              )}
              style={{
                fontSize: 10, fontFamily: FONT, fontWeight: 600,
                padding: '3px 8px', borderRadius: 3,
                border: '1px solid #2a2a4a', cursor: 'pointer',
                backgroundColor: '#16213e', color: '#aaa',
              }}
            >
              {s.paused ? 'Resume' : 'Pause'}
            </button>

            {/* Sizing controls */}
            {s.config_partial_tp_pts > 0 ? (
              /* Strategies WITH partial exit: Entry + TP1 controls */
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <SizingInput
                  label="Entry"
                  value={s.qty_override ?? s.config_entry_qty}
                  min={2}
                  onChange={(v) => sendCommand('strategy_sizing', {
                    strategy_id: s.strategy_id,
                    entry_qty: v,
                  })}
                />
                <SizingInput
                  label="TP1"
                  value={s.partial_qty_override ?? s.config_partial_qty}
                  min={1}
                  max={(s.qty_override ?? s.config_entry_qty) - 1}
                  onChange={(v) => sendCommand('strategy_sizing', {
                    strategy_id: s.strategy_id,
                    partial_qty: v,
                  })}
                />
              </div>
            ) : (
              /* Strategies WITHOUT partial exit: single Qty control */
              <SizingInput
                label="Qty"
                value={s.qty_override ?? s.config_entry_qty}
                min={1}
                onChange={(v) => sendCommand('strategy_sizing', {
                  strategy_id: s.strategy_id,
                  entry_qty: v,
                })}
              />
            )}

            {/* SL today + rolling 5d */}
            <span style={{
              fontSize: 10, fontFamily: FONT,
              color: s.sl_count_today > 0 ? '#ff4444' : '#555',
              minWidth: 150,
            }}>
              SL: {s.sl_count_today} today | {s.sl_rolling_5d ?? 0} (5d)
            </span>

            {/* Daily P&L */}
            <span style={{
              fontSize: 11, fontWeight: 600, fontFamily: FONT,
              color: s.daily_pnl >= 0 ? '#00ff88' : '#ff4444',
              minWidth: 70, textAlign: 'right',
            }}>
              {s.daily_pnl >= 0 ? '+' : ''}${s.daily_pnl.toFixed(2)}
            </span>

            {/* EXIT button — visible only when position is non-FLAT */}
            {getPositionSide(s.strategy_id) !== 'FLAT' && (
              <button
                onClick={() => handleClose(s.strategy_id)}
                disabled={closingStrategy === s.strategy_id}
                style={{
                  fontSize: 10, fontFamily: FONT, fontWeight: 700,
                  padding: '3px 10px', borderRadius: 3,
                  border: '1px solid #ff444466', cursor: 'pointer',
                  backgroundColor: closingStrategy === s.strategy_id
                    ? 'rgba(255,68,68,0.05)' : 'rgba(255,68,68,0.15)',
                  color: closingStrategy === s.strategy_id ? '#ff444466' : '#ff4444',
                  marginLeft: 'auto',
                }}
              >
                {closingStrategy === s.strategy_id ? 'CLOSING...' : 'EXIT'}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
