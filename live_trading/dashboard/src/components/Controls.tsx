import { useState, useCallback } from 'react';
import type { StatusData } from '../types';

interface ControlsProps {
  status: StatusData | null;
  sendCommand: (command: string, payload?: Record<string, unknown>) => void;
}

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

export function Controls({ status, sendCommand }: ControlsProps) {
  const [showKillConfirm, setShowKillConfirm] = useState(false);

  const isActive = status?.trading_active ?? false;

  const handleTogglePause = useCallback(() => {
    sendCommand(isActive ? 'pause' : 'resume');
  }, [isActive, sendCommand]);

  const handleKillSwitch = useCallback(() => {
    setShowKillConfirm(true);
  }, []);

  const handleConfirmKill = useCallback(() => {
    sendCommand('kill');
    setShowKillConfirm(false);
  }, [sendCommand]);

  const handleCancelKill = useCallback(() => {
    setShowKillConfirm(false);
  }, []);

  const btnBase: React.CSSProperties = {
    padding: '8px 18px', borderRadius: 5, border: 'none',
    fontSize: 11, fontWeight: 700, cursor: 'pointer', fontFamily: FONT,
    textTransform: 'uppercase', letterSpacing: 1, transition: 'all 0.15s ease',
  };

  return (
    <>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button
          style={{
            ...btnBase,
            backgroundColor: isActive ? '#ffaa00' : '#00ff88',
            color: '#1a1a2e',
            opacity: status ? 1 : 0.5,
          }}
          onClick={handleTogglePause}
          disabled={!status}
        >
          {isActive ? 'Pause' : 'Resume'}
        </button>
        <button
          style={{
            ...btnBase,
            backgroundColor: '#ff4444',
            color: '#fff',
            opacity: status ? 1 : 0.5,
          }}
          onClick={handleKillSwitch}
          disabled={!status}
        >
          Kill Switch
        </button>
      </div>

      {showKillConfirm && (
        <div
          style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.7)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
          }}
          onClick={handleCancelKill}
        >
          <div
            style={{
              backgroundColor: '#16213e', border: '2px solid #ff4444',
              borderRadius: 12, padding: 28, maxWidth: 420, textAlign: 'center',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: 18, fontWeight: 700, color: '#ff4444', marginBottom: 12, fontFamily: FONT }}>
              KILL SWITCH
            </div>
            <div style={{ fontSize: 13, color: '#ccc', marginBottom: 24, lineHeight: 1.6, fontFamily: FONT }}>
              Are you sure? This will flatten all positions and halt all trading activity immediately.
            </div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
              <button
                style={{
                  ...btnBase, padding: '10px 24px',
                  backgroundColor: '#1a1a2e', color: '#ccc', border: '1px solid #2a2a4a',
                }}
                onClick={handleCancelKill}
              >
                Cancel
              </button>
              <button
                style={{
                  ...btnBase, padding: '10px 24px',
                  backgroundColor: '#ff4444', color: '#fff',
                }}
                onClick={handleConfirmKill}
              >
                CONFIRM KILL
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
