const API_BASE = 'http://localhost:8000';

export async function fetchConfig() {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error('Failed to load config');
  return res.json();
}

export async function saveConfig(config) {
  const res = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  if (!res.ok) throw new Error('Failed to save config');
  return res.json();
}

export async function startRun(config) {
  const res = await fetch(`${API_BASE}/run/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  if (!res.ok) throw new Error('Failed to start run');
  return res.json();
}

export async function checkRunStatus() {
  const res = await fetch(`${API_BASE}/run/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function startMonteCarlo(config) {
  const res = await fetch(`${API_BASE}/montecarlo/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  if (!res.ok) throw new Error('Failed to start Monte Carlo');
  return res.json();
}

export async function checkMonteCarloStatus() {
  const res = await fetch(`${API_BASE}/montecarlo/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export function createRunEventSource() {
  return new EventSource(`${API_BASE}/run/stream`);
}

export function createMonteCarloEventSource() {
  return new EventSource(`${API_BASE}/montecarlo/stream`);
}

export function getChartUrl(chartKey) {
  return `${API_BASE}/charts/${chartKey}`;
}

export function getMonteCarloChartUrl(chartKey) {
  return `${API_BASE}/montecarlo/charts/${chartKey}`;
}
