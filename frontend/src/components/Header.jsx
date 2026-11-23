import React from 'react';
import { ShieldCheck, Globe, Lock, CheckCircle2 } from 'lucide-react';

export default function Header({ config, health, activeBranch, onBranchChange }) {
  const isBranch = config?.env === 'branch';
  const branchName = activeBranch === 2
    ? "Şube-2 (Kurumsal / Yatırım)"
    : "Şube-1 (Bireysel / Kredi)";

  return (
    <header className="bg-[#004481] text-white px-6 py-3 flex items-center justify-between shadow-sm shrink-0">
      {/* Sol: MosaicBank Logosu ve Asistan Başlığı */}
      <div className="flex items-center space-x-3.5">
        <div className="w-9 h-9 rounded-full bg-white flex items-center justify-center text-[#00853F] shadow-xs shrink-0">
          <svg className="w-5 h-5 fill-current" viewBox="0 0 24 24">
            <circle cx="8.5" cy="8.5" r="4.5" fill="#00853F" />
            <circle cx="15.5" cy="8.5" r="4.5" fill="#00853F" />
            <circle cx="8.5" cy="15.5" r="4.5" fill="#00853F" />
            <circle cx="15.5" cy="15.5" r="4.5" fill="#00853F" />
            <path d="M12 12v6" stroke="#00853F" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-base font-bold tracking-tight text-white font-sans">
              MosaicBank
            </span>
            <span className="text-xs bg-white/15 text-white px-2 py-0.5 rounded font-medium">
              {isBranch ? `${branchName} Asistanı` : "Kurumsal Finansal Asistan"}
            </span>
            <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-medium text-emerald-100 bg-[#00853F] px-2 py-0.5 rounded">
              <ShieldCheck className="w-3 h-3" />
              BDDK / KVKK Korumalı
            </span>
          </div>
        </div>
      </div>

      {/* Sağ: Şube Kapsamı Seçimi & Sistem Durumu */}
      <div className="flex items-center space-x-4">
        {isBranch ? (
          /* Şube VM'lerinde Seçim Kilitlidir (Dropdown Yoktur, Sabit İzole Rozet Gösterilir) */
          <div className="flex items-center bg-[#043263] border border-emerald-400/40 rounded px-3 py-1.5 text-xs text-white shadow-xs">
            <Lock className="w-3.5 h-3.5 text-emerald-400 mr-2 shrink-0" />
            <span className="text-blue-200 mr-1.5 font-medium">Aktif Düğüm:</span>
            <span className="font-bold text-emerald-300 tracking-wide">{branchName}</span>
            <span className="ml-2 text-[10px] bg-emerald-500/20 text-emerald-200 px-1.5 py-0.5 rounded border border-emerald-500/30">
              İzole VM
            </span>
          </div>
        ) : (
          /* HQ Genel Merkezde Tüm Şubeler Arasında Geçiş Yapılabilir */
          <div className="flex items-center bg-[#043263] border border-white/20 rounded px-3 py-1.5 text-xs text-white">
            <Globe className="w-3.5 h-3.5 text-[#00853F] mr-2" />
            <span className="text-blue-200 mr-2 font-medium">Şube:</span>
            <select
              value={activeBranch}
              onChange={(e) => onBranchChange(Number(e.target.value))}
              className="bg-transparent text-white font-semibold focus:outline-none cursor-pointer"
            >
              <option value={0} className="bg-[#004481] text-white">Genel Merkez (Tüm Şubeler)</option>
              <option value={1} className="bg-[#004481] text-white">Şube-1 (Bireysel / Kredi)</option>
              <option value={2} className="bg-[#004481] text-white">Şube-2 (Kurumsal / Yatırım)</option>
            </select>
          </div>
        )}

        <div className="hidden md:flex items-center gap-2 text-xs text-blue-100 bg-[#043263]/80 border border-white/15 px-3 py-1.5 rounded">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>Sistem Çevrimiçi</span>
        </div>
      </div>
    </header>
  );
}
