import { useState, useEffect, useCallback, useRef } from 'react';
import { supabase } from '../lib/supabase';

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace";
const POLL_INTERVAL = 300_000; // 5 min — digests only change twice/day

interface Digest {
  id: string;
  digest_date: string;
  digest_type: 'eod' | 'morning';
  markdown: string;
  content: Record<string, unknown>;
  model: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  cost_usd: number | null;
  created_at: string;
}

// ── Simple markdown renderer (no dependency needed) ──

function renderMarkdown(md: string): JSX.Element[] {
  const lines = md.split('\n');
  const elements: JSX.Element[] = [];
  let inList = false;
  let listItems: string[] = [];

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`list-${elements.length}`} style={{ margin: '4px 0 8px 0', paddingLeft: 20 }}>
          {listItems.map((item, i) => (
            <li key={i} style={{ marginBottom: 2 }}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Headers
    if (line.startsWith('# ')) {
      flushList();
      elements.push(
        <h2 key={i} style={{ fontSize: 15, fontWeight: 700, color: '#e0e0e0', margin: '16px 0 8px 0', borderBottom: '1px solid #2a2a4a', paddingBottom: 4 }}>
          {renderInline(line.slice(2))}
        </h2>
      );
    } else if (line.startsWith('## ')) {
      flushList();
      elements.push(
        <h3 key={i} style={{ fontSize: 13, fontWeight: 700, color: '#ccc', margin: '12px 0 6px 0' }}>
          {renderInline(line.slice(3))}
        </h3>
      );
    } else if (line.startsWith('### ')) {
      flushList();
      elements.push(
        <h4 key={i} style={{ fontSize: 12, fontWeight: 700, color: '#aaa', margin: '10px 0 4px 0' }}>
          {renderInline(line.slice(4))}
        </h4>
      );
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      inList = true;
      listItems.push(line.slice(2));
    } else if (line.trim() === '') {
      flushList();
      // Skip blank lines
    } else {
      flushList();
      elements.push(
        <p key={i} style={{ margin: '4px 0', lineHeight: 1.5 }}>
          {renderInline(line)}
        </p>
      );
    }
  }
  flushList();
  return elements;
}

function renderInline(text: string): (string | JSX.Element)[] {
  // Handle **bold**, *italic*, `code`, and $values
  const parts: (string | JSX.Element)[] = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|(\$-?[\d,.]+(?:\.\d+)?))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      // **bold**
      parts.push(<strong key={match.index} style={{ fontWeight: 700, color: '#e0e0e0' }}>{match[2]}</strong>);
    } else if (match[3]) {
      // *italic*
      parts.push(<em key={match.index} style={{ color: '#aaa' }}>{match[3]}</em>);
    } else if (match[4]) {
      // `code`
      parts.push(
        <code key={match.index} style={{
          backgroundColor: '#1a1a2e', padding: '1px 5px', borderRadius: 3,
          fontSize: '0.9em', color: '#bb77ff',
        }}>
          {match[4]}
        </code>
      );
    } else if (match[5]) {
      // $value — color green if positive, red if negative
      const val = parseFloat(match[5].replace(/[$,]/g, ''));
      const color = isNaN(val) ? '#ccc' : val >= 0 ? '#00ff88' : '#ff4444';
      parts.push(<span key={match.index} style={{ color, fontWeight: 600 }}>{match[5]}</span>);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts.length > 0 ? parts : [text];
}

// ── Digest card ──

function DigestCard({ digest, expanded, onToggle }: {
  digest: Digest; expanded: boolean; onToggle: () => void;
}) {
  const isEod = digest.digest_type === 'eod';
  const dateObj = new Date(digest.digest_date + 'T12:00:00');
  const dayLabel = dateObj.toLocaleDateString('en-US', {
    weekday: 'short', month: 'short', day: 'numeric',
  });

  // Extract headline P&L from content if available
  const portfolioSummary = digest.content?.portfolio_summary as Record<string, unknown> | undefined;
  const totalPnl = portfolioSummary?.total_pnl as number | undefined;

  return (
    <div style={{
      backgroundColor: '#1e2a45',
      borderRadius: 6,
      border: '1px solid #2a2a4a',
      overflow: 'hidden',
      marginBottom: 8,
    }}>
      {/* Header — always visible, clickable */}
      <div
        onClick={onToggle}
        style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '10px 14px', cursor: 'pointer',
          borderBottom: expanded ? '1px solid #2a2a4a' : 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, fontFamily: FONT,
            padding: '2px 8px', borderRadius: 3, letterSpacing: 1,
            backgroundColor: isEod ? '#00aaff22' : '#ffaa0022',
            color: isEod ? '#00aaff' : '#ffaa00',
          }}>
            {isEod ? 'EOD' : 'AM'}
          </span>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#e0e0e0', fontFamily: FONT }}>
            {dayLabel}
          </span>
          {totalPnl !== undefined && (
            <span style={{
              fontSize: 12, fontWeight: 700, fontFamily: FONT,
              color: totalPnl >= 0 ? '#00ff88' : '#ff4444',
            }}>
              {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {digest.cost_usd != null && (
            <span style={{ fontSize: 9, color: '#555', fontFamily: FONT }}>
              ${digest.cost_usd.toFixed(2)}
            </span>
          )}
          <span style={{ fontSize: 12, color: '#555', transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'none' }}>
            ▼
          </span>
        </div>
      </div>

      {/* Body — expanded */}
      {expanded && (
        <div style={{
          padding: '12px 16px',
          fontSize: 12, fontFamily: FONT, color: '#bbb',
          lineHeight: 1.6, maxHeight: 600, overflowY: 'auto',
        }}>
          {renderMarkdown(digest.markdown)}
        </div>
      )}
    </div>
  );
}

// ── Main panel ──

export function IntelPanel() {
  const [digests, setDigests] = useState<Digest[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [autoExpanded, setAutoExpanded] = useState(false);
  const mountedRef = useRef(true);

  const fetchDigests = useCallback(async () => {
    if (!supabase) {
      setError('Supabase not configured');
      return;
    }
    try {
      const { data, error: err } = await supabase
        .from('digests')
        .select('id,digest_date,digest_type,markdown,content,model,tokens_in,tokens_out,cost_usd,created_at')
        .order('digest_date', { ascending: false })
        .order('created_at', { ascending: false })
        .limit(20);

      if (!mountedRef.current) return;
      if (err) throw err;

      setDigests(data || []);
      setLoaded(true);
      setError(null);

      // Auto-expand the most recent digest on first load
      if (!autoExpanded && data && data.length > 0) {
        setExpandedId(data[0].id);
        setAutoExpanded(true);
      }
    } catch (err: unknown) {
      if (!mountedRef.current) return;
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    }
  }, [autoExpanded]);

  useEffect(() => {
    mountedRef.current = true;
    fetchDigests();
    const interval = setInterval(fetchDigests, POLL_INTERVAL);
    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [fetchDigests]);

  if (!loaded && !error) {
    return (
      <div style={containerStyle}>
        <span style={{ color: '#666', fontSize: 11, fontFamily: FONT }}>Loading intel...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={containerStyle}>
        <span style={{ color: '#ff4444', fontSize: 11, fontFamily: FONT }}>Intel: {error}</span>
      </div>
    );
  }

  if (digests.length === 0) {
    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#e0e0e0', fontFamily: FONT }}>Intel</span>
        </div>
        <span style={{ color: '#666', fontSize: 11, fontFamily: FONT }}>
          No digests yet. Run: <code style={{ color: '#bb77ff' }}>python -m agents.digest.cli --mode eod</code>
        </span>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#e0e0e0', fontFamily: FONT }}>
            Intel
          </span>
          <span style={{ fontSize: 10, color: '#555', fontFamily: FONT }}>
            {digests.length} digests
          </span>
        </div>
      </div>

      {/* Digest feed */}
      {digests.map(d => (
        <DigestCard
          key={d.id}
          digest={d}
          expanded={expandedId === d.id}
          onToggle={() => setExpandedId(expandedId === d.id ? null : d.id)}
        />
      ))}
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  padding: '14px 16px',
  backgroundColor: '#16213e',
  borderRadius: 6,
  border: '1px solid #2a2a4a',
};
