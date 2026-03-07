import { getTodayEvents } from '../lib/newsCalendar';

export function NewsAlert() {
  const events = getTodayEvents();
  if (events.length === 0) return null;

  return (
    <div style={{
      marginBottom: 16,
      padding: '10px 14px',
      background: 'rgba(255,170,0,0.06)',
      border: '1px solid rgba(255,170,0,0.2)',
      borderRadius: 6,
    }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
        <span style={{
          fontSize: 10,
          fontWeight: 700,
          color: '#ffaa00',
          letterSpacing: 2,
          lineHeight: '20px',
          whiteSpace: 'nowrap',
        }}>
          NEWS TODAY
        </span>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {events.map((ev) => (
            <span key={ev.event} style={{ fontSize: 12, color: '#ccc', lineHeight: '20px' }}>
              <strong style={{ color: '#ffaa00' }}>{ev.event}</strong> at {ev.time} ET
              {' — '}{ev.warning} MES v2 unaffected.
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
