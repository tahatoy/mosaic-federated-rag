import React, { useState, useRef, useEffect } from 'react';
import { Send, FileText, Bot, ShieldCheck, ChevronDown, ChevronRight, Zap, RefreshCw, Database } from 'lucide-react';
import { streamQuery } from '../api/client';
import { marked } from 'marked';
import katex from 'katex';
import 'katex/dist/katex.min.css';

// Configure marked for banking markdown & tables
// MosaicBank bankacılık tabloları ve markdown kuralları
marked.setOptions({
  breaks: true,
  gfm: true,
});

// GFM Tablo ve Başlık Düzeltici: LLM çıktılarındaki malformed delimiter satırlarını,
// eksik başlık boşluklarını (###Başlık) ve tablo öncesi/sonrası boş satırları onarır.
function normalizeMarkdown(text) {
  if (!text) return '';

  const lines = text.split('\n');
  const result = [];
  let inCodeFence = false;

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      inCodeFence = !inCodeFence;
      result.push(line);
      continue;
    }

    if (inCodeFence) {
      result.push(line);
      continue;
    }

    // ###Başlık -> ### Başlık
    if (/^#{1,6}[^\s#]/.test(trimmed)) {
      line = line.replace(/^(\s*#{1,6})([^\s#])/, '$1 $2');
    }

    // Tablo sınır satırını tespit et: örn. | :--- | :--- : | :--- |
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      const innerCells = trimmed.split('|').slice(1, -1);
      const isDelimiter = innerCells.length > 0 && innerCells.every(c => /^[\s\-:]+$/.test(c) && c.includes('-'));

      if (isDelimiter) {
        const fixedCells = innerCells.map(c => {
          const s = c.replace(/\s+/g, '');
          const hasLeft = s.startsWith(':');
          const hasRight = s.endsWith(':');
          const dashes = s.replace(/:/g, '');
          const cleanDashes = dashes.length >= 3 ? dashes : '---';
          return ` ${hasLeft ? ':' : ''}${cleanDashes}${hasRight ? ':' : ''} `;
        });

        // Tablo başlığı öncesinde boşluk satırı yoksa ekle
        if (result.length >= 2) {
          const beforeHeader = result[result.length - 2];
          if (beforeHeader.trim() !== '' && !beforeHeader.trim().startsWith('|')) {
            result.splice(result.length - 1, 0, '');
          }
        }

        result.push('|' + fixedCells.join('|') + '|');
        continue;
      }
    } else if (
      result.length > 0 &&
      result[result.length - 1].trim().startsWith('|') &&
      result[result.length - 1].trim().endsWith('|') &&
      trimmed !== ''
    ) {
      // Tablodan hemen sonra boş satır olmadan metin/başlık geliyorsa ayır
      result.push('');
    }

    result.push(line);
  }

  return result.join('\n');
}

function renderKatexExpr(expr, displayMode) {
  const safeExpr = expr.trim().replace(/(^|[^\\])%/g, '$1\\%');
  return katex.renderToString(safeExpr, {
    displayMode,
    throwOnError: false,
    strict: 'ignore',
  });
}

function renderMarkdownWithMath(rawText) {
  if (!rawText) return '';

  const codeSegments = [];
  const mathItems = [];

  // 1. Kod bloklarını ve satır içi kodu geçici olarak koru
  let protectedText = rawText.replace(/```[\s\S]*?```|`[^`\n]+`/g, (match) => {
    const idx = codeSegments.length;
    codeSegments.push(match);
    return `MOSAICCODE${idx}END`;
  });

  const stashMath = (expr, displayMode) => {
    const idx = mathItems.length;
    mathItems.push({ expr, displayMode });
    return displayMode ? `\n\nMOSAICMATHD${idx}END\n\n` : `MOSAICMATHI${idx}END`;
  };

  // 2. Blok formülleri ($$ ... $$ ve \[ ... \]) çıkar
  protectedText = protectedText.replace(/\$\$([\s\S]*?)\$\$/g, (_, expr) => stashMath(expr, true));
  protectedText = protectedText.replace(/\\\[([\s\S]*?)\\\]/g, (_, expr) => stashMath(expr, true));

  // 3. Satır içi formülleri (\(...\) ve matematik içeren $...$) çıkar
  protectedText = protectedText.replace(/\\\(([\s\S]*?)\\\)/g, (_, expr) => stashMath(expr, false));
  protectedText = protectedText.replace(
    /(?<!\\)\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)/g,
    (full, expr) => {
      // Para birimi gibi yalnızca sayı/noktalama içeren ifadeleri ($100 vb.) matematik sanma
      if (!/[\\=^_{}*]|\\times|\\frac/.test(expr)) {
        return full;
      }
      return stashMath(expr, false);
    }
  );

  // 4. Korunan kod bloklarını geri koy ve Markdown -> HTML dönüştür
  const restoredForMarkdown = protectedText.replace(/MOSAICCODE(\d+)END/g, (_, i) => codeSegments[Number(i)]);
  let html = marked.parse(normalizeMarkdown(restoredForMarkdown));

  // 5. Matematik yer tutucularını KaTeX çıktısıyla değiştir
  html = html.replace(/<p>\s*MOSAICMATHD(\d+)END\s*<\/p>/g, (_, i) => {
    const item = mathItems[Number(i)];
    return `<div class="mosaic-math-block">${renderKatexExpr(item.expr, true)}</div>`;
  });
  html = html.replace(/MOSAICMATHD(\d+)END/g, (_, i) => {
    const item = mathItems[Number(i)];
    return `<div class="mosaic-math-block">${renderKatexExpr(item.expr, true)}</div>`;
  });
  html = html.replace(/MOSAICMATHI(\d+)END/g, (_, i) => {
    const item = mathItems[Number(i)];
    return `<span class="mosaic-math-inline">${renderKatexExpr(item.expr, false)}</span>`;
  });

  return html;
}

// Şubeye göre karşılama metni (Şube-1: bireysel / kredi, Şube-2: kurumsal / yatırım, HQ: genel merkez)
const welcomeText = (b) => b === 1
  ? 'Merhaba, ben MosaicBank Şube-1 Bireysel ve Kredi Bankacılığı asistanıyım. Konut, taşıt ve ihtiyaç kredileri, kredi kartları, mevduat ürünleri ve şube kılavuzları hakkında sorularınızı yanıtlayabilirim. Diğer şubelerin belgelerine ihtiyaç duyulursa sorgunuz genel merkez üzerinden güvenli biçimde iletilir.'
  : b === 2
  ? 'Merhaba, ben MosaicBank Şube-2 Kurumsal ve Yatırım Bankacılığı asistanıyım. Şirket finansmanı, çalışma sermayesi, piyasa ve şirket analizleri hakkında sorularınızı yanıtlayabilirim. Diğer şubelerin belgelerine ihtiyaç duyulursa sorgunuz genel merkez üzerinden güvenli biçimde iletilir.'
  : 'Merhaba, ben MosaicBank Genel Merkez finansal asistanıyım. Şube kılavuzları, BDDK/TCMB düzenlemeleri ve şirket analizleri hakkında tüm şubelerin belgeleri üzerinden size yardımcı olabilirim.';

const branchBadge = (b) => {
  const s = String(b || '').toLowerCase();
  if (s === 'b1' || s.includes('01') || s === 'branch1') {
    return { text: 'b1 · Şube-1', cls: 'bg-blue-50 text-blue-800 border-blue-200' };
  }
  if (s === 'b2' || s.includes('02') || s === 'branch2') {
    return { text: 'b2 · Şube-2', cls: 'bg-emerald-50 text-emerald-800 border-emerald-300' };
  }
  if (s === 'hq' || s === 'global' || s === 'hq_ortak') {
    return { text: 'hq · Genel Merkez', cls: 'bg-slate-100 text-slate-700 border-slate-200' };
  }
  return b ? { text: String(b), cls: 'bg-slate-100 text-slate-700 border-slate-200' } : null;
};

export default function ChatArea({ activeBranch }) {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: '',
      metadata: {
        model: 'MOSAIC Danışman v2',
        route: 'Hazır',
        sources: []
      }
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const messagesEndRef = useRef(null);

  // Hazır sorular: şube korpuslarında yanıtı bulunan sorular ve tezdeki örnek diyaloglar (Bölüm 4.9)
  const presets = activeBranch === 1 ? [
    { label: 'Konut Peşinatı (PDF yüklendikten sonra)', query: 'Şube 1 konut kredisi kılavuzuna göre 3.500.000 TL değerindeki bir konut için minimum peşinat ne kadardır?' },
    { label: 'Kredi Kartı Yapılandırma', query: 'Kredi kartı borcumu yapılandırmak kredi notumu etkiler mi?' },
    { label: 'Taşıt Kredisi & Kasko', query: 'Taşıt kredisi alırken kasko yaptırmak zorunlu mu?' },
    { label: 'Çapraz Şube: Pfizer AR-GE', query: 'Pfizer Inc. geçen yıl araştırma ve geliştirmeye yaklaşık kaç milyar dolar yatırım yaptı?' }
  ] : activeBranch === 2 ? [
    { label: 'Çalışma Sermayesi', query: 'Çalışma sermayesi yönetimi hangi varlıkları içerir ve nasıl yönetilir?' },
    { label: 'Kurumsal Yatırımcı', query: 'Kurumsal yatırımcı nedir?' },
    { label: 'Çapraz Şube: Dow Jones', query: 'Açılışta Dow Jones endeksi kaç puana geriledi?' },
    { label: 'Çapraz Şube: Şube-1 Konut Peşinatı', query: 'Şube 1 konut kredisi kılavuzuna göre 3.500.000 TL değerindeki bir konut için minimum peşinat ne kadardır?' }
  ] : [
    { label: 'Selamlaşma (Günlük Konuşma)', query: 'Merhaba, bana hangi konularda yardımcı olabilirsin?' },
    { label: 'BSMV ve KKDF', query: 'BSMV ve KKDF nedir?' },
    { label: 'Çapraz Şube: Pfizer AR-GE', query: 'Pfizer Inc. geçen yıl araştırma ve geliştirmeye yaklaşık kaç milyar dolar yatırım yaptı?' },
    { label: 'Çapraz Şube: Dow Jones', query: 'Açılışta Dow Jones endeksi kaç puana geriledi?' }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  const handleSend = async (textToSend) => {
    const query = (textToSend || inputQuery).trim();
    if (!query || isStreaming) return;

    setInputQuery('');
    const userMsgId = Date.now().toString();
    const assistantMsgId = (Date.now() + 1).toString();

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', content: query }
    ]);

    setMessages((prev) => [
      ...prev,
      {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        metadata: {
          route: 'Analiz Ediliyor...',
          model: 'Bekleniyor...',
          sources: [],
          latency: null
        }
      }
    ]);

    setIsStreaming(true);
    const startTime = performance.now();

    await streamQuery({
      query,
      branchId: activeBranch,
      history: messages.filter(m => m.id !== 'welcome').map(m => ({ role: m.role, content: m.content })),
      onChunk: (chunk) => {
        setMessages((prev) => {
          return prev.map((msg) => {
            if (msg.id !== assistantMsgId) return msg;

            const updated = { ...msg };

            if (chunk.step === 'route' || chunk.route) {
              updated.metadata.route = (chunk.route || 'Analitik').toUpperCase();
              updated.metadata.model = chunk.model || (chunk.route === 'complex' ? 'Qwen3-8B Expert' : 'Gemma-3-4B Finance');
            }

            if (chunk.step === 'sources' || chunk.sources) {
              updated.metadata.sources = chunk.sources || [];
            }

            if (chunk.token) {
              updated.content += chunk.token;
            } else if (typeof chunk === 'string') {
              updated.content += chunk;
            }

            return updated;
          });
        });
      },
      onDone: () => {
        const totalDuration = ((performance.now() - startTime) / 1000).toFixed(2);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, metadata: { ...msg.metadata, latency: `${totalDuration}s` } }
              : msg
          )
        );
        setIsStreaming(false);
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  content: msg.content + `\n\n[Sistem Hatası: Bağlantı kurulamadı (${err.message}). Lütfen servisleri kontrol ediniz.]`,
                  metadata: { ...msg.metadata, route: 'Hata' }
                }
              : msg
          )
        );
        setIsStreaming(false);
      }
    });
  };

  const toggleSource = (msgId) => {
    setExpandedSources((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-57px)] bg-garanti-bg overflow-hidden">
      {/* Mesaj Akış Alanı */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((msg) => {
            const isUser = msg.role === 'user';

            return (
              <div
                key={msg.id}
                className={`flex gap-3.5 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="w-9 h-9 rounded-full bg-garanti-navy text-white flex items-center justify-center shrink-0 mt-1 shadow-xs">
                    <Bot className="w-5 h-5 text-garanti-green" />
                  </div>
                )}

                <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-3xl w-full`}>
                  {/* Başlık ve Rozetler */}
                  {!isUser && (
                    <div className="flex items-center gap-2 mb-1.5 text-xs text-garanti-textMuted">
                      <span className="font-bold text-garanti-navy">MosaicBank Danışman</span>
                      {msg.metadata?.model && (
                        <span className="bg-garanti-greenSoft text-garanti-green font-semibold text-[11px] px-2 py-0.5 rounded border border-emerald-200">
                          {msg.metadata.model}
                        </span>
                      )}
                      {msg.metadata?.route && msg.metadata.route !== 'Hazır' && (
                        <span className="bg-white text-garanti-navy font-semibold text-[10px] px-2 py-0.5 rounded border border-garanti-border">
                          {msg.metadata.route}
                        </span>
                      )}
                      {msg.metadata?.latency && (
                        <span className="text-slate-400 font-mono text-[10px]">
                          ({msg.metadata.latency})
                        </span>
                      )}
                    </div>
                  )}

                  {/* Mesaj Kartı */}
                  <div
                    className={`p-5 rounded-xl text-sm leading-relaxed ${
                      isUser
                        ? 'bg-garanti-navy text-white rounded-tr-none shadow-sm ml-auto max-w-xl'
                        : 'bg-white border border-garanti-border text-garanti-textDark rounded-tl-none shadow-xs w-full'
                    }`}
                  >
                    {isUser ? (
                      <div className="whitespace-pre-wrap font-sans text-sm font-medium">{msg.content}</div>
                    ) : (
                      <div
                        className="prose-garanti max-w-none"
                        dangerouslySetInnerHTML={{
                          __html: renderMarkdownWithMath((msg.id === 'welcome' ? welcomeText(activeBranch) : msg.content) || (isStreaming ? '*Analiz üretiliyor...*' : ''))
                        }}
                      />
                    )}

                    {/* Kaynakça Çekmecesi */}
                    {!isUser && msg.metadata?.sources && msg.metadata.sources.length > 0 && (
                      <div className="mt-4 pt-3.5 border-t border-garanti-border">
                        <button
                          onClick={() => toggleSource(msg.id)}
                          className="flex items-center gap-2 text-xs text-garanti-navy hover:text-garanti-green font-bold transition"
                        >
                          <Database className="w-4 h-4 text-garanti-green" />
                          <span>Doğrulanan Yasal Kaynaklar ({msg.metadata.sources.length} Doküman Alıntısı)</span>
                          {expandedSources[msg.id] ? (
                            <ChevronDown className="w-3.5 h-3.5" />
                          ) : (
                            <ChevronRight className="w-3.5 h-3.5" />
                          )}
                        </button>

                        {expandedSources[msg.id] && (
                          <div className="mt-3 space-y-2.5">
                            {msg.metadata.sources.map((src, sIdx) => (
                              <div
                                key={sIdx}
                                className="bg-garanti-bg border border-garanti-border p-3.5 rounded-lg text-xs"
                              >
                                <div className="flex items-center justify-between text-garanti-textMuted mb-1.5">
                                  <span className="font-bold text-garanti-navy flex items-center gap-1.5 text-xs">
                                    <FileText className="w-3.5 h-3.5 text-garanti-green shrink-0" />
                                    {(() => { const badge = branchBadge(src.branch); return badge ? (<span className={`font-mono text-[10px] px-1.5 py-0.5 rounded border font-bold ${badge.cls}`}>{badge.text}</span>) : null; })()}
                                    {src.source || `Kılavuz #${sIdx + 1}`}
                                  </span>
                                  {src.score !== undefined && (
                                    <span className="font-mono text-[11px] bg-garanti-greenSoft text-garanti-green px-2 py-0.5 rounded font-bold border border-emerald-200">
                                      Eşleşme: %{(src.score * 100).toFixed(0)}
                                    </span>
                                  )}
                                </div>
                                <p className="text-garanti-textDark text-[12px] bg-white p-3 rounded border border-garanti-borderLight leading-relaxed font-sans">
                                  "{src.text || src.content || src.chunk || 'Metin alıntısı'}"
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Alt Bölüm: Hızlı Öneriler & Mesaj Giriş Barı */}
      <div className="bg-white border-t border-garanti-border p-4 shrink-0 shadow-xs">
        <div className="max-w-4xl mx-auto space-y-2.5">
          {/* Hızlı Öneri Çipleri */}
          <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-none">
            <span className="text-[11px] font-bold text-garanti-navy uppercase whitespace-nowrap flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 text-garanti-green" />
              Öneriler:
            </span>
            {presets.map((p, idx) => (
              <button
                key={idx}
                disabled={isStreaming}
                onClick={() => handleSend(p.query)}
                className="text-xs bg-garanti-bg hover:bg-slate-100 border border-garanti-border hover:border-garanti-navy text-garanti-textDark px-3.5 py-1.5 rounded-full transition font-medium whitespace-nowrap shadow-xs"
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Giriş Formu */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center gap-2.5"
          >
            <input
              type="text"
              value={inputQuery}
              disabled={isStreaming}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Finansal mevzuat, kredi şartları veya bilanço rasyosu hakkında danışın..."
              className="flex-1 bg-garanti-bg border border-garanti-border focus:border-garanti-navy rounded-lg px-4 py-3 text-sm text-garanti-textDark placeholder-garanti-textMuted focus:outline-none focus:ring-1 focus:ring-garanti-navy shadow-xs"
            />
            <button
              type="submit"
              disabled={isStreaming || !inputQuery.trim()}
              className="bg-garanti-green hover:bg-garanti-greenHover disabled:bg-slate-300 disabled:text-slate-500 text-white font-bold px-7 py-3 rounded-lg text-sm flex items-center gap-2 transition shadow-sm"
            >
              {isStreaming ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Yanıtlanıyor...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Gönder</span>
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
