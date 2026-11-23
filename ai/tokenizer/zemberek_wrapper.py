"""
Faz 2.1 — Zemberek Python Wrapper
===================================
Zemberek Java kütüphanesinin Python üzerinden kullanımı için ince sarmalayıcı.
zemberek-python (pip) veya JPype bridge üzerinden çalışır.

Kullanım:
  from tokenizer.zemberek_wrapper import ZemberekAnalyzer
  analyzer = ZemberekAnalyzer()
  morphemes = analyzer.analyze("bankacılığın")
  # -> ["banka", "cılık", "ın"]
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from loguru import logger


class ZemberekAnalyzer:
    """
    Zemberek morfolojik analizci wrapper.
    Önce zemberek-python pip paketi denenir, başarısız olursa
    JPype bridge'e düşer.
    """

    def __init__(self, use_cache: bool = True):
        self._morphology = None
        self._cache: dict[str, list[str]] = {} if use_cache else None  # type: ignore
        self._available = False
        self._init_zemberek()

    def _init_zemberek(self) -> None:
        """Zemberek başlatma — önce pip paketi, sonra JPype."""
        # --- Seçenek A: zemberek-python pip paketi ---
        try:
            import zemberek  # type: ignore
            from zemberek import TurkishMorphology  # type: ignore
            self._morphology = TurkishMorphology.create_with_defaults()
            self._mode = "pip"
            self._available = True
            logger.info("Zemberek (pip) başarıyla başlatıldı.")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"zemberek-python başlatma hatası: {e}")

        # --- Seçenek B: JPype bridge ---
        try:
            import jpype  # type: ignore
            import jpype.imports  # type: ignore

            jar_candidates = [
                Path.home() / ".zemberek" / "zemberek-full.jar",
                Path(__file__).parent / "zemberek-full.jar",
            ]
            jar_path = next((p for p in jar_candidates if p.exists()), None)
            if jar_path is None:
                logger.warning(
                    "Zemberek JAR bulunamadı. İndirin:\n"
                    "  https://github.com/ahmetaa/zemberek-nlp/releases\n"
                    f"  Dosyayı şuraya koyun: {jar_candidates[0]}"
                )
                return

            if not jpype.isJVMStarted():
                jpype.startJVM(classpath=[str(jar_path)])

            from zemberek.morphology import TurkishMorphology  # type: ignore
            self._morphology = TurkishMorphology.createWithDefaults()
            self._mode = "jpype"
            self._available = True
            logger.info("Zemberek (JPype) başarıyla başlatıldı.")
        except ImportError:
            logger.warning("JPype kurulu değil. pip install jpype1")
        except Exception as e:
            logger.warning(f"Zemberek JPype başlatma hatası: {e}")

        if not self._available:
            logger.warning(
                "Zemberek kullanılamıyor — kelime bölme yalnızca boşluk tabanlı olacak."
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def _analyze_pip(self, word: str) -> list[str]:
        """zemberek-python ile morfolojik analiz."""
        try:
            results = self._morphology.analyze(word)
            if not results:
                return [word]
            # En olası analizi seç (ilk sonuç)
            best = results[0]
            morphemes = [m.surface for m in best.morphemes if m.surface]
            return morphemes if morphemes else [word]
        except Exception:
            return [word]

    def _analyze_jpype(self, word: str) -> list[str]:
        """JPype bridge ile morfolojik analiz."""
        try:
            word_analysis = self._morphology.analyzeAndDisambiguate(word)
            if not word_analysis or word_analysis.isEmpty():
                return [word]
            best = word_analysis.get(0)
            morphemes = [str(m.getSurface()) for m in best.getMorphemes() if str(m.getSurface())]
            return morphemes if morphemes else [word]
        except Exception:
            return [word]

    def analyze(self, word: str) -> list[str]:
        """
        Bir kelimeyi morfemlerine ayırır.

        Args:
            word: Türkçe kelime

        Returns:
            Morfem listesi. Örn: "bankacılığın" -> ["banka", "cılık", "ın"]
            Zemberek kullanılamıyorsa [word] döner.
        """
        word = word.strip()
        if not word:
            return []

        # Cache kontrolü
        if self._cache is not None and word in self._cache:
            return self._cache[word]

        if not self._available:
            result = [word]
        elif self._mode == "pip":
            result = self._analyze_pip(word)
        else:
            result = self._analyze_jpype(word)

        # Cache'e ekle
        if self._cache is not None:
            self._cache[word] = result

        return result

    def analyze_batch(self, words: list[str]) -> list[list[str]]:
        """Kelime listesini toplu analiz eder (gecikme optimizasyonu için)."""
        return [self.analyze(w) for w in words]

    def tokenize_sentence(self, sentence: str) -> list[str]:
        """
        Bir cümleyi morfem token'larına ayırır.

        Args:
            sentence: Türkçe cümle

        Returns:
            Morfem token listesi
        """
        import re
        # Türkçe'de kelimeleri böl (noktalama ayır)
        words = re.findall(r"[a-zA-ZğüşıöçĞÜŞİÖÇ0-9]+|[^\w\s]", sentence)
        tokens = []
        for word in words:
            if re.match(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]{2,}", word):
                tokens.extend(self.analyze(word))
            else:
                tokens.append(word)
        return tokens

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()


# ─── GLOBAL SİNGLETON ─────────────────────────────────────────────────────────
_analyzer_instance: Optional[ZemberekAnalyzer] = None


def get_analyzer() -> ZemberekAnalyzer:
    """Thread-safe singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ZemberekAnalyzer()
    return _analyzer_instance


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    az = ZemberekAnalyzer()
    test_words = ["bankacılığın", "yatırımcılar", "faiz", "sigortacılık", "kredi"]
    print(f"Zemberek mevcut: {az.is_available}")
    for w in test_words:
        print(f"  {w!r} -> {az.analyze(w)}")
