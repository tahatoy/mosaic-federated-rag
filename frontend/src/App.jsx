import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import TopologyModal from './components/TopologyModal';
import FederatedModal from './components/FederatedModal';
import { fetchConfig, fetchHealth } from './api/client';

export default function App() {
  const [config, setConfig] = useState(null);
  const [health, setHealth] = useState(null);
  const [activeBranch, setActiveBranch] = useState(0);
  const [isTopologyOpen, setIsTopologyOpen] = useState(false);
  const [isFederatedOpen, setIsFederatedOpen] = useState(false);

  useEffect(() => {
    async function init() {
      const cfg = await fetchConfig();
      setConfig(cfg);
      if (cfg.env === 'branch') {
        const branchNum = cfg.branch_id?.includes('2') ? 2 : 1;
        setActiveBranch(branchNum);
      }

      const h = await fetchHealth();
      setHealth(h);
    }
    init();

    const interval = setInterval(async () => {
      const h = await fetchHealth();
      setHealth(h);
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-screen w-screen flex flex-col bg-garanti-bg text-garanti-textDark font-sans overflow-hidden">
      {/* 1. MosaicBank Kurumsal Header */}
      <Header
        config={config}
        health={health}
        activeBranch={activeBranch}
        onBranchChange={setActiveBranch}
      />

      {/* 2. Ana Çalışma Alanı: Sol Panel (Belgeler & FL) + Canlı Sanal Danışman */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sol Panel: Doküman Yükleme & Federe Öğrenme */}
        <Sidebar
          activeBranch={activeBranch}
          onOpenTopology={() => setIsTopologyOpen(true)}
          onOpenFederated={() => setIsFederatedOpen(true)}
        />

        {/* Ana Panel: MosaicBank Kurumsal Finansal Danışman */}
        <ChatArea activeBranch={activeBranch} />
      </div>

      {/* 3. Dağıtık Düğüm Topolojisi Modalı */}
      <TopologyModal
        isOpen={isTopologyOpen}
        onClose={() => setIsTopologyOpen(false)}
      />

      {/* 4. Federe Öğrenme & KVKK Kanıt Merkezi Modalı */}
      <FederatedModal
        isOpen={isFederatedOpen}
        onClose={() => setIsFederatedOpen(false)}
      />
    </div>
  );
}
