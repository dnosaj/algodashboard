import { useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { StatusPanel } from './components/StatusPanel';
import { Controls } from './components/Controls';
import { InstrumentCard } from './components/InstrumentCard';
import { PriceChart } from './components/PriceChart';
import { SignalFeed } from './components/SignalFeed';
import { TradeLog } from './components/TradeLog';
import { DailyPnL } from './components/DailyPnL';
import { SafetyPanel } from './components/SafetyPanel';

const WS_URL = `ws://${window.location.hostname}:${window.location.port || '8000'}/ws`;
const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

function App() {
  const { status, trades, dailyPnl, signals, bars, connected, sendCommand } =
    useWebSocket(WS_URL);

  const [selectedInstrument, setSelectedInstrument] = useState<string>('MNQ');

  const chartBars = bars[selectedInstrument] || [];
  const chartTrades = trades.filter((t) => t.instrument === selectedInstrument);

  const positions = status?.positions || [];
  // Prefer the non-FLAT MNQ position (could be V11 or V15)
  const mnqPos = positions.find((p) => p.instrument === 'MNQ' && p.side !== 'FLAT')
    || positions.find((p) => p.instrument === 'MNQ')
    || { instrument: 'MNQ', side: 'FLAT' as const, entry_price: null, unrealized_pnl: 0 };
  // Pair indicator data to the strategy that owns the shown position
  const mnqStrategyId = mnqPos.strategy_id || 'MNQ_V11';
  const mesPos = positions.find((p) => p.instrument === 'MES' && p.side !== 'FLAT')
    || positions.find((p) => p.instrument === 'MES')
    || { instrument: 'MES', side: 'FLAT' as const, entry_price: null, unrealized_pnl: 0 };
  const mesStrategyId = mesPos.strategy_id || 'MES_V94';

  const safetyStatus = status?.safety ?? null;

  return (
    <div style={{
      minHeight: '100vh', backgroundColor: '#1a1a2e', color: '#e0e0e0',
      fontFamily: FONT, padding: 20,
    }}>
      {/* Header row: title + controls */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid #2a2a4a',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#e0e0e0', margin: 0, letterSpacing: 1 }}>
            NQ TRADING
          </h1>
          <span style={{ fontSize: 11, color: '#555', letterSpacing: 2, textTransform: 'uppercase' }}>
            Live Dashboard
          </span>
        </div>
        <Controls status={status} sendCommand={sendCommand} />
      </div>

      {/* Status bar */}
      <div style={{ marginBottom: 16 }}>
        <StatusPanel status={status} connected={connected} />
      </div>

      {/* Instrument cards row */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <InstrumentCard
          position={mnqPos}
          data={status?.instruments?.[mnqStrategyId] ?? null}
          selected={selectedInstrument === 'MNQ'}
          onSelect={() => setSelectedInstrument('MNQ')}
        />
        <InstrumentCard
          position={mesPos}
          data={status?.instruments?.[mesStrategyId] ?? null}
          selected={selectedInstrument === 'MES'}
          onSelect={() => setSelectedInstrument('MES')}
        />
      </div>

      {/* Safety controls panel */}
      {safetyStatus && (
        <div style={{ marginBottom: 16 }}>
          <SafetyPanel safety={safetyStatus} positions={positions} sendCommand={sendCommand} />
        </div>
      )}

      {/* Price Chart (key forces remount on instrument switch) */}
      {chartBars.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <PriceChart
            key={selectedInstrument}
            bars={chartBars}
            trades={chartTrades}
            instrument={selectedInstrument}
          />
        </div>
      )}

      {/* Signal feed */}
      <div style={{ marginBottom: 16 }}>
        <SignalFeed signals={signals} />
      </div>

      {/* Trade log */}
      <div style={{ marginBottom: 16 }}>
        <TradeLog trades={trades} />
      </div>

      {/* Daily P&L chart */}
      <div style={{ marginBottom: 16 }}>
        <DailyPnL data={dailyPnl} />
      </div>
    </div>
  );
}

export default App;
