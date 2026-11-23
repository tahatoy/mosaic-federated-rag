import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, RefreshCw, Lock, Server, ChevronRight } from 'lucide-react';
import { uploadDocument } from '../api/client';

export default function Sidebar({ activeBranch, onStatsUpdate, onOpenTopology, onOpenFederated }) {
  // Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  // Şubelere dağıtılmış korpuslar (Tez Bölüm 3.1.4, Çizelge 3.4; non-IID, %6 ortak havuz)
  const recentDocs = [
    { name: 'Şube-1 korpusu — bireysel / kredi', branch: 'Şube-1 VM (10.0.0.11)', chunks: '7.854 parça' },
    { name: 'Şube-2 korpusu — kurumsal / yatırım', branch: 'Şube-2 VM (10.0.0.12)', chunks: '14.767 parça' },
    { name: 'Genel merkez — yalnızca ortak bilgi', branch: 'HQ (şubeye özgü veri şubede kalır)', chunks: '429 parça' }
  ];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setUploadStatus(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || isUploading) return;
    setIsUploading(true);
    setUploadStatus(null);

    try {
      const res = await uploadDocument(selectedFile, '', activeBranch);
      setUploadStatus({
        success: true,
        message: `${selectedFile.name} başarıyla indekslendi ve anında sorgulanabilir hale geldi.`
      });
      setSelectedFile(null);
      if (onStatsUpdate) onStatsUpdate();
    } catch (err) {
      setUploadStatus({
        success: false,
        message: err.message || 'Yükleme başarısız oldu'
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <aside className="w-80 md:w-96 bg-white border-r border-garanti-border flex flex-col h-[calc(100vh-57px)] overflow-y-auto shrink-0 select-none">
      {/* 1. Belge ve Mevzuat Yükleme Alanı */}
      <div className="p-5 border-b border-garanti-border space-y-3.5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-garanti-navy flex items-center gap-2">
            <FileText className="w-4 h-4 text-garanti-green" />
            Doküman ve Mevzuat Yükle
          </h2>
          <span className="text-[11px] font-semibold text-garanti-textMuted bg-garanti-bg px-2 py-0.5 rounded border border-garanti-border">
            PDF / TXT
          </span>
        </div>

        <p className="text-xs text-garanti-textMuted leading-relaxed">
          Şubenize ait kredi kılavuzlarını veya resmi tebliğleri yükleyin. Veriler yerel ChromaDB'ye işlenir, ham dosya şubeden dışarı çıkmaz.
        </p>

        {/* Dosya Yükleme Kutusu */}
        <div className="border border-dashed border-garanti-border hover:border-garanti-green rounded-lg p-4 bg-garanti-bg text-center transition">
          <input
            type="file"
            id="sidebar-doc"
            accept=".pdf,.txt,.doc"
            onChange={handleFileChange}
            className="hidden"
          />
          <label htmlFor="sidebar-doc" className="cursor-pointer flex flex-col items-center gap-2">
            <UploadCloud className="w-7 h-7 text-garanti-navy" />
            <span className="text-xs font-semibold text-garanti-navy">
              {selectedFile ? selectedFile.name : 'Dosya Seç veya Sürükle'}
            </span>
            <span className="text-[10px] text-garanti-textMuted">
              {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : 'PDF kılavuz, tebliğ (max 50 MB)'}
            </span>
          </label>
        </div>

        {selectedFile && (
          <button
            onClick={handleUpload}
            disabled={isUploading}
            className="w-full bg-garanti-green hover:bg-garanti-greenHover text-white font-semibold py-2 rounded text-xs flex items-center justify-center gap-2 shadow-xs transition"
          >
            {isUploading ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>İndeksleniyor...</span>
              </>
            ) : (
              <span>Dokümanı Şubeye Ekle</span>
            )}
          </button>
        )}

        {/* Yükleme Geri Bildirimi */}
        {uploadStatus && (
          <div
            className={`p-2.5 rounded text-xs flex items-start gap-2 ${
              uploadStatus.success
                ? 'bg-garanti-greenSoft border border-emerald-200 text-garanti-green'
                : 'bg-red-50 border border-red-200 text-red-700'
            }`}
          >
            {uploadStatus.success ? (
              <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            )}
            <span className="text-[11px] leading-tight font-medium">{uploadStatus.message}</span>
          </div>
        )}

        {/* Son Yüklü Belgeler */}
        <div className="pt-2">
          <span className="text-[11px] font-bold text-garanti-navy uppercase tracking-wider block mb-2">
            İndeksli Şube Belgeleri
          </span>
          <div className="space-y-1.5">
            {recentDocs.map((doc, idx) => (
              <div
                key={idx}
                className="p-2 rounded bg-garanti-bg border border-garanti-border text-xs flex items-center justify-between"
              >
                <div className="truncate pr-2">
                  <div className="font-semibold text-garanti-textDark truncate">{doc.name}</div>
                  <div className="text-[10px] text-garanti-textMuted">{doc.branch}</div>
                </div>
                <span className="text-[10px] font-mono text-garanti-green font-semibold shrink-0">
                  {doc.chunks}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 2. Federe Kanıt & Ağ Durumu */}
      <div className="p-5 mt-auto border-t border-garanti-border space-y-2.5 bg-slate-50/40">
        <button
          onClick={onOpenFederated}
          className="w-full bg-white hover:bg-garanti-greenSoft/50 border border-garanti-border text-garanti-navy font-semibold py-2.5 px-3 rounded-lg text-xs flex items-center justify-between transition shadow-2xs cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <Lock className="w-3.5 h-3.5 text-garanti-green" />
            <span>Federe Kanıt & KVKK Denetimi</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 text-garanti-textMuted" />
        </button>

        <button
          onClick={onOpenTopology}
          className="w-full bg-garanti-bg hover:bg-slate-100 border border-garanti-border text-garanti-navy font-semibold py-2.5 px-3 rounded-lg text-xs flex items-center justify-between transition cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <Server className="w-3.5 h-3.5 text-garanti-navy" />
            <span>Dağıtık Şube Ağ Durumu</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 text-garanti-textMuted" />
        </button>
      </div>
    </aside>
  );
}
