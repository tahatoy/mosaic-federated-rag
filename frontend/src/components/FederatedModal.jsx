import React, { useState, useEffect } from 'react';
import { X, Lock, ShieldCheck } from 'lucide-react';
import { fetchFLAudit } from '../api/client';

// Federe Öğrenme ve KVKK Denetim Özeti penceresi (UI açık tema ile uyumlu).
const fmt = (v, nd = 4) => (v === null || v === undefined ? '—' : typeof v === 'number' ? v.toFixed(nd) : String(v));

export default function FederatedModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const [audit, setAudit] = useState(null);

  useEffect(() => {
    if (isOpen) {
      fetchFLAudit().then(setAudit);
    }
  }, [isOpen]);

  const r = audit?.router_fl;
  const ab = audit?.router_ablation;

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4 select-none">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full border border-garanti-border overflow-hidden flex flex-col max-h-[85vh]">
        {/* Başlık — TopologyModal ile aynı açık kurumsal üst bant */}
        <div className="bg-[#004481] text-white px-6 py-4 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="w-5 h-5 text-garanti-green shrink-0" />
            <div>
              <h1 className="text-sm font-bold tracking-tight">Federe Öğrenme ve KVKK Denetim Özeti</h1>
              <p className="text-[11px] text-blue-200">Tezde ölçülen federe eğitim ve gizlilik bulguları</p>
            </div>
          </div>
          <button onClick={onClose} className="text-white/80 hover:text-white p-1 rounded hover:bg-white/10 transition cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* İçerik */}
        <div className="flex-1 overflow-y-auto p-6 bg-garanti-bg space-y-4 text-xs">
          {!audit ? (
            <div className="text-garanti-textMuted py-8 text-center">Denetim verisi yükleniyor...</div>
          ) : (
            <>
              {/* 1. Yönlendirici Federasyonu: FedAvg vs FedProx */}
              <div className="bg-white border border-garanti-border rounded-lg p-4 space-y-2.5 shadow-2xs">
                <div className="font-bold text-garanti-navy text-xs">
                  {r.title || 'BERT Yönlendirici — FedAvg / FedProx'}
                </div>
                <table className="w-full text-left text-xs">
                  <thead className="text-garanti-textMuted border-b border-garanti-border text-[11px]">
                    <tr>
                      <th className="py-1.5 font-semibold">Strateji</th>
                      <th className="py-1.5 font-semibold">Doğruluk</th>
                      <th className="py-1.5 font-semibold text-right">Macro F1</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-garanti-borderLight text-garanti-textDark">
                    {['FedAvg', 'FedProx'].map((s) => (
                      <tr key={s} className={s === 'FedProx' ? 'font-bold text-garanti-green bg-garanti-greenSoft/40' : ''}>
                        <td className="py-2 pl-1">{s} {s === 'FedProx' ? '(μ = 0.01)' : ''}</td>
                        <td className="py-2">{fmt(r[s]?.accuracy)}</td>
                        <td className="py-2 pr-1 text-right">{fmt(r[s]?.f1_macro)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* 2. Gizlilik Bedeli: Federe vs Merkezi */}
              <div className="bg-white border border-garanti-border rounded-lg p-4 space-y-2.5 shadow-2xs">
                <div className="font-bold text-garanti-navy text-xs">
                  {ab.title || 'Merkezi vs Federe Başarım Karşılaştırması'}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-garanti-greenSoft/50 border border-emerald-200 rounded-lg p-3">
                    <div className="text-[11px] text-garanti-textMuted">Federe FedProx Yönlendirici</div>
                    <div className="text-lg font-bold text-garanti-green mt-0.5">{fmt(ab.federated_fedprox_f1)}</div>
                  </div>
                  <div className="bg-garanti-bg border border-garanti-border rounded-lg p-3">
                    <div className="text-[11px] text-garanti-textMuted">Merkezi BERT Yönlendirici</div>
                    <div className="text-lg font-bold text-garanti-navy mt-0.5">{fmt(ab.central_bert_f1)}</div>
                  </div>
                </div>
                <p className="text-[11px] text-garanti-textMuted">{ab.note}</p>
              </div>
            </>
          )}
        </div>

        {/* Alt bilgi */}
        <div className="bg-white px-6 py-3 border-t border-garanti-border flex items-center justify-between text-[11px] text-garanti-textMuted shrink-0">
          <div className="flex items-center gap-1.5">
            <Lock className="w-3.5 h-3.5 text-garanti-green shrink-0" />
            <span>Şubeden metin çıkmaz; yalnızca maskelenmiş parçalar veya model güncellemeleri paylaşılır.</span>
          </div>
          <button
            onClick={onClose}
            className="bg-[#004481] hover:bg-[#043263] text-white font-semibold py-1.5 px-4 rounded-lg text-xs transition cursor-pointer shrink-0 ml-3"
          >
            Kapat
          </button>
        </div>
      </div>
    </div>
  );
}
