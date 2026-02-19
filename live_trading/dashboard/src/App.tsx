import { useWebSocket } from './hooks/useWebSocket';
import { StatusPanel } from './components/StatusPanel';
import { Controls } from './components/Controls';
import { InstrumentCard } from './components/InstrumentCard';
import { PriceChart } from './components/PriceChart';
import { SignalFeed } from './components/SignalFeed';
import { TradeLog } from './components/TradeLog';
import { DailyPnL } from './components/DailyPnL';

const WS_URL = `ws://${window.location.hostname}:${window.location.port || '8000'}/ws`;
const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";

function App() {
  const { status, trades, dailyPnl, signals, bars, connected, sendCommand } =
    useWebSocket(WS_URL);

  const mnqBars = bars['MNQ'] || [];
  const mnqTrades = trades.filter((t) => t.instrument === 'MNQ');

  const positions = status?.positions || [];
  const mnqPos = positions.find((p) => p.instrument === 'MNQ') || {
    instrument: 'MNQ', side: 'FLAT' as const, entry_price: null, unrealized_pnl: 0,
  };
  const mesPos = positions.find((p) => p.instrument === 'MES') || {
    instrument: 'MES', side: 'FLAT' as const, entry_price: null, unrealized_pnl: 0,
  };

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
        <InstrumentCard position={mnqPos} data={status?.instruments?.['MNQ'] ?? null} />
        <InstrumentCard position={mesPos} data={status?.instruments?.['MES'] ?? null} />
      </div>

      {/* MNQ Price Chart */}
      {mnqBars.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <PriceChart bars={mnqBars} trades={mnqTrades} instrument="MNQ" />
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
