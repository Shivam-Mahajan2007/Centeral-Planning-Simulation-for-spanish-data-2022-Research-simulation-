import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Activity } from 'lucide-react';
import { checkRunStatus, startRun, createRunEventSource, getChartUrl } from '../api';

function Lightbox({ src, onClose }) {
  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <img src={src} className="lightbox-img" onClick={e => e.stopPropagation()} alt="Expanded Chart" />
    </div>
  );
}

export default function RunPanel({ config }) {
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState({ current: 0, total: 20 });
  const [logs, setLogs] = useState([]);
  const [charts, setCharts] = useState([]);
  const [activeImage, setActiveImage] = useState(null);
  const logEndRef = useRef(null);

  useEffect(() => {
    // Check initial status
    checkRunStatus().then(st => {
      setStatus(st.status);
      if (st.total > 0) setProgress({ current: st.progress, total: st.total });
      if (st.charts) setCharts(st.charts);
    }).catch(console.error);

    // Setup SSE
    const es = createRunEventSource();
    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'ping') return;
      if (data.type === 'log') {
        setLogs(prev => [...prev, data.msg].slice(-200)); // keep last 200
      } else if (data.type === 'progress') {
        setProgress({ current: data.progress, total: data.total });
      } else if (data.type === 'status') {
        setStatus(data.status);
      } else if (data.type === 'done') {
        setStatus('done');
        if (data.charts) setCharts(data.charts);
      } else if (data.type === 'error') {
        setStatus('error');
        setLogs(prev => [...prev, `[ERROR] ${data.msg}`]);
      }
    };
    return () => es.close();
  }, []);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const handleStart = async () => {
    if (!config) return;
    setLogs([]);
    setCharts([]);
    setStatus('starting');
    setProgress({ current: 0, total: config.n_quarters || 20 });
    try {
      await startRun(config);
    } catch (e) {
      console.error(e);
      setStatus('error');
      setLogs([`Failed to start: ${e.message}`]);
    }
  };

  const progressPct = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Single Run</h2>
          <span className={`status-badge status-${status}`}>{status.toUpperCase()}</span>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            className="btn btn-primary" 
            disabled={status === 'running' || status === 'starting'}
            onClick={handleStart}
          >
            <Play size={16} /> Run Simulation
          </button>
        </div>
      </div>

      <div className="progress-container">
        <div className="progress-bar" style={{ width: `${progressPct}%` }} />
      </div>

      <div className="log-window">
        {logs.length === 0 && <span style={{ color: 'var(--text-muted)' }}>Ready...</span>}
        {logs.map((L, i) => <div key={i} className="log-entry">{L}</div>)}
        <div ref={logEndRef} />
      </div>

      {charts.length > 0 && (
        <div className="charts-grid">
          {charts.map(chart => (
            <div key={chart} className="chart-card">
              <div className="chart-header">{chart.replace(/_/g, ' ').toUpperCase()}</div>
              <div className="chart-body">
                <img 
                  src={getChartUrl(chart)} 
                  alt={chart} 
                  className="chart-img"
                  onClick={() => setActiveImage(getChartUrl(chart))}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {activeImage && <Lightbox src={activeImage} onClose={() => setActiveImage(null)} />}
    </div>
  );
}
