import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import RunPanel from './components/RunPanel';
import MonteCarloPanel from './components/MonteCarloPanel';

function App() {
  const [activeTab, setActiveTab] = useState('run');
  const [config, setConfig] = useState(null);

  return (
    <div className="app-container">
      <Sidebar config={config} setConfig={setConfig} />
      
      <main className="main-content">
        <header className="top-nav">
          <nav role="tablist">
            <button
              role="tab"
              className={`nav-tab ${activeTab === 'run' ? 'active' : ''}`}
              onClick={() => setActiveTab('run')}
            >
              Single Run
            </button>
            <button
              role="tab"
              className={`nav-tab ${activeTab === 'mc' ? 'active' : ''}`}
              onClick={() => setActiveTab('mc')}
            >
              Monte Carlo
            </button>
          </nav>
        </header>

        <section className="content-area">
          {activeTab === 'run' && <RunPanel config={config} />}
          {activeTab === 'mc' && <MonteCarloPanel config={config} />}
        </section>
      </main>
    </div>
  );
}

export default App;
