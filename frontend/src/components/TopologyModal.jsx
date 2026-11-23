import React, { useState, useEffect } from 'react';
import { X, Network, Radio, ShieldCheck, RefreshCw, CheckCircle2, AlertTriangle, AlertCircle } from 'lucide-react';
import { fetchNetworkStatus } from '../api/client';

export default function TopologyModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const [pinging, setPinging] = useState(false);
  const [nodes, setNodes] = useState([
    {
      id: 'hq',
      name: 'Düğüm 1: HQ (Genel Merkez)',
      ip: '10.0.0.10 (Blackwell RTX 5060 Ti)',
      role: 'Orkestratör & Qwen3-8B Expert',
      status: 'starting',
      latency: 'Ölçülüyor...'
    },
    {
      id: 'vm1',
      name: 'Düğüm 2: Şube-1 VM (Bireysel / Kredi)',
      ip: '10.0.0.11:8001 (2 vCPU, 10 GB RAM)',
      role: 'Branch-1 ChromaDB & Yerel UI',
      status: 'starting',
      latency: 'Ölçülüyor...'
    },
    {
      id: 'vm2',
      name: 'Düğüm 3: Şube-2 VM (Kurumsal / Yatırım)',
      ip: '10.0.0.12:8002 (2 vCPU, 10 GB RAM)',
      role: 'Branch-2 ChromaDB & Yerel UI',
      status: 'starting',
      latency: 'Ölçülüyor...'
    }
  ]);

  const handlePing = async () => {
    setPinging(true);
    try {
      const data = await fetchNetworkStatus();
      if (data && data.nodes) {
        setNodes(data.nodes);
      }
    } catch (e) {
      console.warn('Ping error:', e);
    } finally {
      setPinging(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      handlePing();
    }
  }, [isOpen]);

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4 select-none">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full border border-garanti-border overflow-hidden">
        {/* Header */}
        <div className="bg-[#004481] text-white px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5 font-bold text-sm">
            <Network className="w-5 h-5 text-garanti-green" />
            <span>Dağıtık Şube Altyapısı ve Ağ Sağlığı</span>
          </div>
          <button onClick={onClose} className="text-white/80 hover:text-white p-1 rounded hover:bg-white/10 transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          <div className="flex items-center justify-between text-xs text-garanti-textMuted">
            <span>3 Düğümlü Canlı Topoloji (Canlı HTTP Gecikme Ölçümü)</span>
            <button
              onClick={handlePing}
              disabled={pinging}
              className="flex items-center gap-1.5 font-bold text-[#004481] hover:text-garanti-green transition"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${pinging ? 'animate-spin' : ''}`} />
              <span>{pinging ? 'Ölçülüyor...' : 'Canlı Gecikmeyi Test Et'}</span>
            </button>
          </div>

          <div className="space-y-3">
            {nodes.map((n) => {
              const isUp = n.status === 'up';
              return (
                <div
                  key={n.id}
                  className={`p-3.5 rounded-lg border flex items-center justify-between text-xs transition ${
                    isUp ? 'border-garanti-border bg-white' : 'border-amber-200 bg-amber-50/40'
                  }`}
                >
                  <div className="space-y-0.5">
                    <div className="font-bold text-garanti-navy flex items-center gap-2">
                      <span>{n.name}</span>
                      {isUp ? (
                        <span className="text-[10px] font-semibold bg-garanti-greenSoft text-garanti-green px-1.5 py-0.2 rounded border border-emerald-200">
                          Çevrimiçi
                        </span>
                      ) : (
                        <span className="text-[10px] font-semibold bg-amber-100 text-amber-800 px-1.5 py-0.2 rounded border border-amber-300">
                          Beklemede / Kapalı
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-garanti-textMuted">{n.ip} • {n.role}</div>
                  </div>

                  <span
                    className={`font-mono font-bold text-xs px-2.5 py-1 rounded border flex items-center gap-1.5 shrink-0 ${
                      isUp
                        ? 'text-garanti-green bg-garanti-greenSoft border-emerald-200'
                        : 'text-amber-800 bg-amber-100/70 border-amber-200'
                    }`}
                  >
                    <Radio className={`w-3 h-3 ${isUp ? 'text-garanti-green animate-pulse' : 'text-amber-600'}`} />
                    {n.latency}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="pt-2 text-[11px] text-garanti-textMuted flex items-center justify-between border-t border-garanti-border">
            <div className="flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-garanti-green" />
              <span>Soket Gecikmesi: Gerçek zamanlı HTTP GET ping yanıt süresidir.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
