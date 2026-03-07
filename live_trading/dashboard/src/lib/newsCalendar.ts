export interface NewsEvent {
  date: string;       // "YYYY-MM-DD"
  event: string;      // "FOMC Rate Decision" | "CPI" | "NFP" | "Retail Sales"
  time: string;       // "08:30" or "14:00" (ET)
  warning: string;    // strategy-specific warning
}

const CALENDAR: NewsEvent[] = [
  // FOMC Rate Decision — 14:00 ET (statement release on 2nd day)
  ...[
    '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29',
    '2026-09-16', '2026-10-28', '2026-12-09',
  ].map((date): NewsEvent => ({
    date,
    event: 'FOMC Rate Decision',
    time: '14:00',
    warning: 'V15 lost -$414 on FOMC days. Consider pausing V15.',
  })),

  // NFP / Employment Situation — 08:30 ET
  ...[
    '2026-04-03', '2026-05-08', '2026-06-05', '2026-07-02',
    '2026-08-07', '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04',
  ].map((date): NewsEvent => ({
    date,
    event: 'NFP',
    time: '08:30',
    warning: 'V15 lost -$326, vScalpB -$77 on NFP. Consider pausing both MNQ strategies.',
  })),

  // CPI — 08:30 ET
  ...[
    '2026-03-11', '2026-04-10', '2026-05-12', '2026-06-10',
    '2026-07-14', '2026-08-12', '2026-09-11', '2026-10-14',
    '2026-11-10', '2026-12-10',
  ].map((date): NewsEvent => ({
    date,
    event: 'CPI',
    time: '08:30',
    warning: 'V15 lost -$114 on CPI days. Consider pausing V15.',
  })),

  // Retail Sales — 08:30 ET
  ...[
    '2026-03-17', '2026-04-16', '2026-05-14', '2026-06-17',
    '2026-07-16', '2026-08-14', '2026-09-16', '2026-10-15',
  ].map((date): NewsEvent => ({
    date,
    event: 'Retail Sales',
    time: '08:30',
    warning: 'vScalpB lost -$388 on Retail Sales. Consider pausing vScalpB.',
  })),
];

/** Get today's date string in ET (DST-safe) */
function getTodayET(): string {
  const fmt = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  return fmt.format(new Date()); // "YYYY-MM-DD"
}

/** Return all news events scheduled for today (ET). Empty array on quiet days. */
export function getTodayEvents(): NewsEvent[] {
  const today = getTodayET();
  return CALENDAR.filter((e) => e.date === today);
}
