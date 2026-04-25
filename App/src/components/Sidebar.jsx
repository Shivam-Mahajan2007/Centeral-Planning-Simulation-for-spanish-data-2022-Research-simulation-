import React, { useEffect } from 'react';
import { fetchConfig, saveConfig } from '../api';
import { Settings, Save } from 'lucide-react';

const CONFIG_SCHEMA = [
  { key: 'n_quarters', label: 'Quarters (T)', type: 'number', step: 1 },
  { key: 'delta', label: 'Depreciation (δ)', type: 'number', step: 0.005 },
  { key: 'neumann_k', label: 'Neumann Depth (k)', type: 'number', step: 1 },
  { key: 'kappa_factor', label: 'Capital Multiplier (κ)', type: 'number', step: 0.5 },
  { key: 'L_total', label: 'Total Labor (L)', type: 'number', step: 1e9 },
  { key: 'wage_rate', label: 'Wage Rate (w)', type: 'number', step: 0.5 },
  { key: 'primal_tol', label: 'Primal Tol', type: 'number', step: 1e-4 },
  { key: 'dual_tol', label: 'Dual Tol', type: 'number', step: 1e-4 },
  { key: 'eta_K', label: 'Capital Margin (η_K)', type: 'number', step: 0.05 },
  { key: 'eta_L', label: 'Labor Margin (η_L)', type: 'number', step: 0.05 },
  { key: 'g_step', label: 'Govt. Policy (g)', type: 'number', step: 0.01 },
  { key: 'habit_persistence', label: 'Habit Weight', type: 'number', step: 0.1 },
  { key: 'theta_drift', label: 'Theta Drift', type: 'number', step: 0.01 },
  { key: 'pref_drift_sigma', label: 'Pref Volatility', type: 'number', step: 0.01 },
  { key: 'max_iter', label: 'Max Iterations', type: 'number', step: 500 },
  { key: 'rng_seed', label: 'RNG Seed', type: 'number', step: 1 },
];

export default function Sidebar({ config, setConfig }) {
  useEffect(() => {
    fetchConfig().then(setConfig).catch(console.error);
  }, [setConfig]);

  const handleChange = (k, v) => {
    setConfig(prev => ({ ...prev, [k]: Number(v) }));
  };

  const handleSave = async () => {
    try {
      await saveConfig(config);
      // Brief visual feedback could go here
    } catch (e) {
      console.error(e);
    }
  };

  if (!config) return <aside className="sidebar"><div className="sidebar-content">Loading...</div></aside>;

  return (
    <aside className="sidebar">
      <header className="sidebar-header">
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.25rem' }}>
          <Settings size={20} />
          Configuration
        </h2>
      </header>
      <div className="sidebar-content">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {CONFIG_SCHEMA.map(({ key, label, step }) => (
            <div key={key} className="form-group">
              <label className="form-label">{label}</label>
              <input
                type="number"
                step={step}
                className="form-input"
                value={config[key] ?? ""}
                onChange={(e) => handleChange(key, e.target.value)}
              />
            </div>
          ))}
        </div>
      </div>
      <div style={{ padding: '1.5rem', borderTop: '1px solid var(--border-color)' }}>
        <button className="btn btn-primary" style={{ width: '100%' }} onClick={handleSave}>
          <Save size={16} /> Save Config
        </button>
      </div>
    </aside>
  );
}
