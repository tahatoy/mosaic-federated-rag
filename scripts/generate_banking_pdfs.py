"""
generate_banking_pdfs.py — MosaicBank 30 Adet Kurumsal Bankacılık PDF Dokümanı Üretici
========================================================================================
MOSAIC-v2 tez projesi için 3 dağıtık düğüme (Şube-1 Bireysel/Tarım, Şube-2 Kurumsal/Yatırım, HQ Genel Merkez)
özel olarak tasarlanmış 30 adet resmi bankacılık PDF dokümanını ReportLab ile üretir.
Windows native Arial TrueType fontu ile Türkçe karakter desteği tamdır.
"""

import sys
from pathlib import Path

# Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Proje Kök Dizini
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = PROJECT_ROOT / "data" / "sample_banking_docs"

# Dizinleri Oluştur
BRANCH1_DIR = OUTPUT_BASE / "branch1"
BRANCH2_DIR = OUTPUT_BASE / "branch2"
HQ_DIR = OUTPUT_BASE / "hq"

for d in [BRANCH1_DIR, BRANCH2_DIR, HQ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Windows Arial Font Kaydı (Türkçe karakter desteği)
FONT_DIR = Path("C:/Windows/Fonts")
if (FONT_DIR / "arial.ttf").exists():
    pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "arialbd.ttf")))
    DEFAULT_FONT = "Arial"
    BOLD_FONT = "Arial-Bold"
else:
    DEFAULT_FONT = "Helvetica"
    BOLD_FONT = "Helvetica-Bold"

# Renk Paleti (MosaicBank Kurumsal Lacivert & Yonca Yeşili)
COLOR_NAVY = colors.HexColor("#004481")
COLOR_NAVY_DARK = colors.HexColor("#043263")
COLOR_GREEN = colors.HexColor("#00853F")
COLOR_BG_GRAY = colors.HexColor("#F4F6F9")
COLOR_BORDER = colors.HexColor("#DDE3EA")
COLOR_TEXT_DARK = colors.HexColor("#192A3E")
COLOR_MUTED = colors.HexColor("#5A6B7C")

def get_banking_styles():
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=BOLD_FONT,
        fontSize=15,
        leading=19,
        textColor=COLOR_NAVY,
        spaceAfter=6
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName=DEFAULT_FONT,
        fontSize=9,
        leading=12,
        textColor=COLOR_MUTED,
        spaceAfter=10
    )
    style_h2 = ParagraphStyle(
        "DocH2",
        parent=styles["Normal"],
        fontName=BOLD_FONT,
        fontSize=11,
        leading=15,
        textColor=COLOR_NAVY,
        spaceBefore=10,
        spaceAfter=4
    )
    style_body = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName=DEFAULT_FONT,
        fontSize=9,
        leading=13,
        textColor=COLOR_TEXT_DARK,
        spaceAfter=6
    )
    style_meta = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName=DEFAULT_FONT,
        fontSize=8,
        leading=10,
        textColor=COLOR_MUTED
    )
    style_table_cell = ParagraphStyle(
        "DocCell",
        parent=styles["Normal"],
        fontName=DEFAULT_FONT,
        fontSize=8,
        leading=11,
        textColor=COLOR_TEXT_DARK
    )
    style_table_header = ParagraphStyle(
        "DocCellHeader",
        parent=styles["Normal"],
        fontName=BOLD_FONT,
        fontSize=8,
        leading=11,
        textColor=colors.white
    )
    return {
        "title": style_title,
        "subtitle": style_subtitle,
        "h2": style_h2,
        "body": style_body,
        "meta": style_meta,
        "cell": style_table_cell,
        "cell_header": style_table_header
    }

def create_pdf(filename: Path, doc_id: str, title: str, branch: str, classification: str, date_str: str, sections: list, table_data: list = None):
    """Tek bir kurumsal PDF dokümanı oluşturur."""
    doc = SimpleDocTemplate(
        str(filename),
        pagesize=A4,
        rightMargin=1.8*cm,
        leftMargin=1.8*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )
    styles = get_banking_styles()
    story = []

    # 1. Antet Başlığı (Header)
    header_data = [
        [
            Paragraph(f"<b>MosaicBank A.Ş.</b><br/>{branch}", styles["title"]),
            Paragraph(f"<b>DOKÜMAN NO:</b> {doc_id}<br/><b>TARİH:</b> {date_str}<br/><b>GİZLİLİK:</b> {classification}", styles["meta"])
        ]
    ]
    t_header = Table(header_data, colWidths=[11.5*cm, 5.5*cm])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_NAVY, spaceBefore=4, spaceAfter=10))

    # 2. Doküman Başlığı ve Kapsam
    story.append(Paragraph(title, styles["title"]))
    story.append(Paragraph(f"Kapsam: Bu yönerge ve prosedür metni, {branch} yetki sınırları ve BDDK/TCMB düzenlemeleri uyarınca düzenlenmiştir.", styles["subtitle"]))

    # 3. Metin Bölümleri
    for heading, text_paragraphs in sections:
        if heading:
            story.append(Paragraph(heading, styles["h2"]))
        for p_text in text_paragraphs:
            story.append(Paragraph(p_text, styles["body"]))

    # 4. Tablo (Varsa)
    if table_data and len(table_data) > 1:
        story.append(Spacer(1, 4))
        formatted_table = []
        for r_idx, row in enumerate(table_data):
            formatted_row = []
            for col in row:
                if r_idx == 0:
                    formatted_row.append(Paragraph(str(col), styles["cell_header"]))
                else:
                    formatted_row.append(Paragraph(str(col), styles["cell"]))
            formatted_table.append(formatted_row)

        t = Table(formatted_table, colWidths=[5.5*cm, 4.0*cm, 7.5*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), COLOR_NAVY),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('INNERGRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
            ('BOX', (0,0), (-1,-1), 1, COLOR_NAVY_DARK),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, COLOR_BG_GRAY])
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    # 5. Onay ve İmza Bloğu (Footer / Sign-off)
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=4, spaceAfter=8))
    sign_data = [
        [
            Paragraph("<b>Hazırlayan / Birim:</b><br/>Operasyonel Kredi Tahsisat Servisi", styles["meta"]),
            Paragraph("<b>Kontrol Eden:</b><br/>Mevzuat Uyum ve Risk Yönetimi", styles["meta"]),
            Paragraph("<b>Onaylayan Makam:</b><br/>Şube Yönetim Kurulu / Genel Müdürlük", styles["meta"])
        ]
    ]
    t_sign = Table(sign_data, colWidths=[5.6*cm, 5.6*cm, 5.8*cm])
    t_sign.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_sign)

    # İnşa Et
    doc.build(story)
    print(f"[OK] Üretildi: {filename.name}")

# ─────────────────────────────────────────────────────────────────────────────
# DOKÜMAN TANIMLARI (30 ADET)
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_pdfs():
    print("=" * 70)
    print("MOSAIC-v2: 30 Adet Kurumsal Bankacılık PDF Dokümanı Üretiliyor...")
    print("=" * 70)

    # ─── GRUP A: ŞUBE 1 (BİREYSEL, KOBİ & TARIM - 10 ADET) ───────────────────

    # 1. sube1_tarimsal_kredi_cks_elus_proseduru.pdf (Yıldız Senaryo Dokümanı)
    create_pdf(
        filename=BRANCH1_DIR / "sube1_tarimsal_kredi_cks_elus_proseduru.pdf",
        doc_id="MB-S1-TARIM-2026-01",
        title="Tarımsal Kredi, ÇKS İcmali ve ELÜS Teminatlandırma Prosedürü",
        branch="Şube-1 (Gaziantep / KOBİ & Tarımsal Bankacılık)",
        classification="Şube İçi Gizli (Banka Sırrı - KVKK)",
        date_str="12.01.2026",
        sections=[
            ("1. Amaç ve Kapsam", [
                "Bu prosedür, Şube-1 hinterlandında faaliyet gösteren tarımsal üreticilere Çiftçi Kayıt Sistemi (ÇKS) icmalleri ve Elektronik Ürün Senedi (ELÜS) teminatı karşılığında kullandırılacak sübvansiyonlu işletme ve avans kredilerinin tahsis koşullarını belirler.",
                "Tarımsal üreticilerin ÇKS belgeleri, şube yerelinde taranarak yerel arşivde saklanır; hiçbir çiftçi kimlik verisi veya arazi tapu bilgisi şube dışına veya Genel Merkez sunucularına ham metin olarak aktarılamaz."
            ]),
            ("2. ELÜS Teminatlandırma Oranları ve Mazot-Gübre Avansı", [
                "Lisanslı depolara teslim edilen buğday, mısır, pamuk ve Antep fıstığı karşılığında üretilen ELÜS senetleri bankamız MKK nezdindeki havuz hesabına rehin edilir.",
                "ELÜS teminatlı kredilerde kredi/teminat oranı (Haircut sonrası LTV) azami %85 olarak uygulanır. Mazot ve gübre avansı taleplerinde sübvansiyonlu faiz indirimi yıllık %14.50 taban oran üzerinden hesaplanır."
            ]),
            ("3. Yetkili Onay Makamları", [
                "Tarımsal kredi limitlerinde 1.500.000 TL'ye kadar Şube Kredi Şefi Merve Kaya, 5.000.000 TL'ye kadar Şube Müdürü Ahmet Yılmaz tek imzayla onaya yetkilidir. 5.000.000 TL üzeri tahsisler Genel Merkez Bölge Kredi Komitesi onayına sunulur."
            ])
        ],
        table_data=[
            ["Ürün / Teminat Türü", "Azami Kredi / Teminat (LTV)", "Sübvansiyonlu Faiz / Vade Şartı"],
            ["ELÜS Hububat (Buğday/Arpa)", "%85 Teminat Oranı", "%14.50 Taban Faiz / Azami 12 Ay"],
            ["ELÜS Sanayi Ürünü (Pamuk/Fıstık)", "%80 Teminat Oranı", "%15.25 Taban Faiz / Azami 18 Ay"],
            ["ÇKS İcmal Bazlı Mazot/Gübre Avansı", "%75 Gelir Beyanı", "%12.00 Teşvik Faizi / Hasat Sonu Tek Ödeme"],
            ["Traktör ve Ekipman Yatırım Kredisi", "%70 Proforma Fatura", "%16.50 Faiz / 48 Ay Eşit Taksitli"]
        ]
    )

    # 2. sube1_konut_kredisi_tahsisat_rehberi.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_konut_kredisi_tahsisat_rehberi.pdf",
        doc_id="MB-S1-KRED-2026-02",
        title="Bireysel Konut Kredisi Tahsisat ve Erken Ödeme Prosedürü",
        branch="Şube-1 (Bireysel ve Perakende Bankacılık)",
        classification="Hizmete Özel (Şube İçi)",
        date_str="15.01.2026",
        sections=[
            ("1. Kredi Koşulları ve Faiz Çizelgesi", [
                "Bireysel müşterilerin konut alımlarında kullandırılacak kredilerde azami vade 120 aydır. Ekspertiz değeri SPK lisanslı bağımsız değerleme şirketlerince tespit edilir.",
                "Şube Müdürü Ahmet Yılmaz koordinasyonunda konut kredisi tahsisatı yapılmakta olup, aylık net gelirin azami %50'si taksit tutarı olarak sınırlandırılmıştır."
            ]),
            ("2. Erken Ödeme Cezası ve Yasal Sınırlar", [
                "Tüketicinin Korunması Hakkında Kanun uyarınca, konut finansmanı sözleşmelerinde borcun vadesinden önce ödenmesi halinde banka erken ödeme tazminatı talep edebilir.",
                "Kalan vadesi 36 ayı aşmayan kredilerde tazminat oranı kalan anaparanın %1.0'ini, 36 ayı aşan kredilerde ise azami %2.0'sini geçemez."
            ])
        ],
        table_data=[
            ["Vade Aralığı", "Aylık Faiz Oranı", "Erken Ödeme Tazminatı Tavanı"],
            ["1 - 36 Ay Vade", "%2.79 / Ay", "Kalan Anaparanın Azami %1.0'i"],
            ["37 - 60 Ay Vade", "%2.89 / Ay", "Kalan Anaparanın Azami %2.0'si"],
            ["61 - 120 Ay Vade", "%2.99 / Ay", "Kalan Anaparanın Azami %2.0'si"]
        ]
    )

    # 3. sube1_tasit_ve_kasko_rehin_proseduru.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_tasit_ve_kasko_rehin_proseduru.pdf",
        doc_id="MB-S1-TASIT-2026-03",
        title="Taşıt Kredisi, Kasko ve Rehin Tesis Yönergesi",
        branch="Şube-1 (Bireysel Kredi Operasyonları)",
        classification="Şube İçi Genel",
        date_str="18.01.2026",
        sections=[
            ("1. Taşıt Kredisi Tahsis Kriterleri", [
                "Sıfır kilometre ve azami 5 yaşına kadar olan 2. el binek araçlar için kullandırılır. BDDK fatura değeri basamaklarına göre LTV oranları kademelendirilmiştir.",
                "Araç üzerine Emniyet Genel Müdürlüğü ARTES sistemi üzerinden elektronik rehin tesis edilir ve bankamız lehine daini mürtehinli kasko poliçesi zorunludur."
            ])
        ],
        table_data=[
            ["Araç Fatura Değeri (TL)", "Azami Kredi Oranı (LTV)", "Azami Vade"],
            ["0 - 400.000 TL arası", "%70 LTV Sınırı", "48 Ay Vade"],
            ["400.001 - 800.000 TL arası", "%50 LTV Sınırı", "36 Ay Vade"],
            ["800.001 - 1.200.000 TL arası", "%30 LTV Sınırı", "24 Ay Vade"],
            ["1.200.001 - 2.000.000 TL arası", "%20 LTV Sınırı", "12 Ay Vade"]
        ]
    )

    # 4. sube1_ihtiyac_kredisi_ve_findeks_baremleri.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_ihtiyac_kredisi_ve_findeks_baremleri.pdf",
        doc_id="MB-S1-IHT-2026-04",
        title="İhtiyaç Kredisi Tahsis Esasları ve Findeks Not Baremleri",
        branch="Şube-1 (Bireysel Kredi Masası)",
        classification="Hizmete Özel",
        date_str="20.01.2026",
        sections=[
            ("1. Kredi Notu Değerlendirme Çerçevesi", [
                "Findeks Kredi Notu 1-1900 arasında derecelendirilmekte olup, bankamız ihtiyaç kredisi otomatik onay eşiği 1.450 puandır. 1.100 puan altı başvurular otomatik redde düşer."
            ])
        ],
        table_data=[
            ["Findeks Kredi Notu", "Risk Seviyesi", "Uygulanacak Faiz / Teminat Şartı"],
            ["1.650 - 1.900 Puan", "Çok Düşük Risk", "%3.45 Faiz / Kefilsiz, Belgesiz Anında Onay"],
            ["1.450 - 1.649 Puan", "Düşük Risk", "%3.75 Faiz / Gelir Belgesi İbrazı ile Onay"],
            ["1.100 - 1.449 Puan", "Orta Risk", "%4.15 Faiz / Ek Teminat veya Kefil Zorunlu"],
            ["1 - 1.099 Puan", "Yüksek Risk", "Kredi Tahsisi Yapılamaz (Doğrudan Red)"]
        ]
    )

    # 5. sube1_bireysel_vadeli_mevduat_faiz_tablosu.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_bireysel_vadeli_mevduat_faiz_tablosu.pdf",
        doc_id="MB-S1-MEVD-2026-05",
        title="Bireysel Vadeli Mevduat Faiz Oranları ve Stopaj Esasları",
        branch="Şube-1 (Mevduat ve Operasyon Servisi)",
        classification="Genel Müşteri Bilgilendirme",
        date_str="22.01.2026",
        sections=[
            ("1. Türk Lirası Vadeli Mevduat Oranları", [
                "Şube-1 mevduat tabanını güçlendirmek amacıyla yeni hesap açılışlarında 45 gün boyunca 'Hoş Geldin Faizi' uygulanır. Stopaj oranları Cumhurbaşkanı Kararı uyarınca vadelere göre farklılaştırılmıştır."
            ])
        ],
        table_data=[
            ["Vade Süresi", "Standart Yıllık Faiz", "Hoş Geldin Faizi", "Yasal Stopaj Kesintisi"],
            ["32 - 45 Gün", "%44.00 Yıllık", "%47.50 Yıllık", "%7.5 Gelir Stopajı"],
            ["92 - 180 Gün", "%42.50 Yıllık", "%45.00 Yıllık", "%5.0 Gelir Stopajı"],
            ["181 - 365 Gün", "%39.00 Yıllık", "%41.00 Yıllık", "%2.5 Gelir Stopajı"],
            ["365+ Gün (1 Yıl+)", "%36.00 Yıllık", "%38.00 Yıllık", "%0.0 Sıfır Stopaj"]
        ]
    )

    # 6. sube1_emekli_maas_promosyon_ve_taahhut_esaslari.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_emekli_maas_promosyon_ve_taahhut_esaslari.pdf",
        doc_id="MB-S1-EMEK-2026-06",
        title="Emekli Maaş Promosyon Tutarları ve 3 Yıl Taahhüt Şartları",
        branch="Şube-1 (Bireysel Hizmetler)",
        classification="Genel Bilgilendirme",
        date_str="25.01.2026",
        sections=[
            ("1. Promosyon Baremleri", [
                "SGK emekli maaşını 3 yıl süreyle bankamız Şube-1 nezdinden almayı taahhüt eden emeklilerimize tek seferde peşin promosyon ödemesi yapılır. En az 2 otomatik fatura talimatı verenlere ek 2.500 TL nakit ödül tanımlanır."
            ])
        ],
        table_data=[
            ["Aylık Maaş Tutarı", "Peşin Promosyon Tutarı", "Ek Fatura Ödülü", "Toplam Nakit Avantaj"],
            ["10.000 TL'ye kadar", "12.000 TL", "2.500 TL", "14.500 TL Peşin"],
            ["10.001 - 15.000 TL arası", "15.000 TL", "2.500 TL", "17.500 TL Peşin"],
            ["15.001 - 20.000 TL arası", "18.000 TL", "2.500 TL", "20.500 TL Peşin"],
            ["20.001 TL ve üzeri", "20.000 TL", "2.500 TL", "22.500 TL Peşin"]
        ]
    )

    # 7. sube1_kredi_karti_limit_tahsisi_ve_asgari_odeme.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_kredi_karti_limit_tahsisi_ve_asgari_odeme.pdf",
        doc_id="MB-S1-KART-2026-07",
        title="Bireysel Kredi Kartı Limit Tahsisi ve Asgari Ödeme Oranları",
        branch="Şube-1 (Kartlı Ödeme Sistemleri)",
        classification="Hizmete Özel",
        date_str="28.01.2026",
        sections=[
            ("1. 5464 Sayılı Kanun Uyarınca Limit Belirleme", [
                "Kart hamilinin tüm bankalardaki kredi kartı toplam limiti, ilk yıl için aylık net belgelenebilir gelirinin 2 katını, ikinci ve sonraki yıllar için 4 katını aşamaz.",
                "Asgari ödeme tutarı; 25.000 TL altı limitli kartlarda dönem borcunun %20'si, 25.000 TL üzeri kartlarda dönem borcunun %40'ıdır."
            ])
        ],
        table_data=[
            ["Kart Limit Dilimi", "Asgari Ödeme Oranı", "Gecikme Faizi Azami", "Nakit Avans Limiti"],
            ["0 - 25.000 TL", "Dönem Borcunun %20'si", "%4.55 Aylık", "Kart Limitinin %25'i"],
            ["25.001 TL ve üzeri", "Dönem Borcunun %40'ı", "%4.55 Aylık", "Kart Limitinin %25'i"]
        ]
    )

    # 8. sube1_esnaf_kobi_mikro_kredi_kgf_kilavuzu.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_esnaf_kobi_mikro_kredi_kgf_kilavuzu.pdf",
        doc_id="MB-S1-KOBI-2026-08",
        title="Esnaf ve Mikro KOBİ Destek Kredisi (KGF Kefaletli) Kılavuzu",
        branch="Şube-1 (KOBİ ve Ticari Bankacılık Masası)",
        classification="Şube İçi Gizli",
        date_str="01.02.2026",
        sections=[
            ("1. Kredi Koşulları", [
                "Faaliyet süresi en az 1 yıl olan esnaf ve mikro işletmelere, Kredi Garanti Fonu (KGF) %80 kefaleti ile 750.000 TL'ye kadar 6 ay anapara ödemesiz azami 36 ay vadeli işletme kredisi tahsis edilebilir."
            ])
        ],
        table_data=[
            ["İşletme Ciro Skalası", "Azami Kredi Limiti", "KGF Kefalet Oranı", "Ödemesiz Dönem"],
            ["Yıllık Ciro < 2.000.000 TL", "350.000 TL", "%80 Hazine Destekli", "3 Ay Ödemesiz"],
            ["Yıllık Ciro 2M - 10M TL", "750.000 TL", "%80 Hazine Destekli", "6 Ay Ödemesiz"]
        ]
    )

    # 9. sube1_operasyonel_yetki_matrisi_ve_sube_kadrosu.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_operasyonel_yetki_matrisi_ve_sube_kadrosu.pdf",
        doc_id="MB-S1-YON-2026-09",
        title="Şube-1 Yönetim Kadrosu ve Operasyonel İmza Yetki Matrisi",
        branch="Şube-1 (Yönetim ve İnsan Kaynakları)",
        classification="Çok Gizli (Şube Yönetimi)",
        date_str="03.02.2026",
        sections=[
            ("1. Şube Yönetim Ekibi ve İletişim Koordinasyonu", [
                "Şube Müdürü Ahmet Yılmaz, Şube-1'in tüm bireysel ve KOBİ kredi tahsislerinden birinci derecede sorumludur.",
                "Kredi Tahsis Şefi Merve Kaya operasyonel kredi incelemelerini yürütür. Gişe Amiri Selin Demir nakit akışını yönetir."
            ])
        ],
        table_data=[
            ["Yetkili Adı Soyadı", "Görevi / Ünvanı", "Tek Başına Kredi Tahsis Limiti", "İmza Derecesi"],
            ["Ahmet Yılmaz", "Şube Müdürü", "5.000.000 TL", "1. Derece A Grubu İmza"],
            ["Merve Kaya", "Kredi Tahsis Şefi", "1.500.000 TL", "2. Derece B Grubu İmza"],
            ["Selin Demir", "Gişe ve Operasyon Amiri", "250.000 TL Nakit Ödeme", "3. Derece C Grubu İmza"]
        ]
    )

    # 10. sube1_gun_sonu_kasa_ve_nakit_guvenlik_proseduru.pdf
    create_pdf(
        filename=BRANCH1_DIR / "sube1_gun_sonu_kasa_ve_nakit_guvenlik_proseduru.pdf",
        doc_id="MB-S1-KASA-2026-10",
        title="Gün Sonu Kasa Mutabakatı ve Zırhlı Araç Nakit Transfer Prosedürü",
        branch="Şube-1 (Nakit Yönetimi ve Güvenlik)",
        classification="Gizli — Güvenlik Prosedürü",
        date_str="05.02.2026",
        sections=[
            ("1. Kasa Azami Limitleri ve Çift Anahtar Kuralı", [
                "Şube-1 ana kasasında gün sonu kapanışında bulundurulabilecek azami Türk Lirası tutarı 500.000 TL'dir. Aşan bakiye saat 17:30'da zırhlı araçla Merkez Kasa'ya sevk edilir.",
                "Kasa dairesi çift anahtar ve biyometrik parmak izi kuralı ile açılır. Anahtarlar Şube Müdürü Ahmet Yılmaz ve Gişe Amiri Selin Demir'de muhafaza edilir."
            ])
        ],
        table_data=[
            ["Varlık Türü", "Azami Şube Kasa Limiti", "Sigorta Teminat Limiti", "Transfer Frekansı"],
            ["Türk Lirası (TL)", "500.000 TL", "2.000.000 TL", "Günlük 17:30 Sevk"],
            ["Amerikan Doları (USD)", "25.000 USD", "100.000 USD", "Haftada 2 Kez"],
            ["Euro (EUR)", "20.000 EUR", "100.000 EUR", "Haftada 2 Kez"]
        ]
    )

    # ─── GRUP B: ŞUBE 2 (KURUMSAL & YATIRIM BANKACILIĞI - 10 ADET) ───────────

    # 11. sube2_yesil_sendikasyon_ve_sofr_kredi_kilavuzu.pdf (Yıldız Senaryo Dokümanı)
    create_pdf(
        filename=BRANCH2_DIR / "sube2_yesil_sendikasyon_ve_sofr_kredi_kilavuzu.pdf",
        doc_id="MB-S2-SEND-2026-01",
        title="Yeşil Sendikasyon Kredileri, SOFR Marj İndirimi ve ESG Kriterleri",
        branch="Şube-2 (İstanbul Maslak / Kurumsal & Yatırım Bankacılığı)",
        classification="Ticari Sır — Kurumsal Gizlilik (NDA)",
        date_str="10.01.2026",
        sections=[
            ("1. Sendikasyon Kredisi Finansman Esasları", [
                "Bu kılavuz, Şube-2 tarafından koordine edilen çok uluslu sendikasyon kredilerinde geçerli SOFR (Secured Overnight Financing Rate) faiz yapısını ve sürdürülebilirlik bağlantılı marj indirimlerini düzenler.",
                "Kurumsal müşterilerin holding konsolide bilançoları ve gizlilik sözleşmeleri şube yerelinde korunur; ham müşteri sözleşmesi Genel Merkeze dahi kopyalanamaz."
            ]),
            ("2. ESG Performans Göstergeleri ve Marj İndirimi", [
                "Borçlu holdingin karbon emisyonunu yıllık %10 azaltması ve yenilenebilir enerji payını %40'ın üzerine çıkarması durumunda, SOFR + 275 bps olan kredi faiz marjında 25 bps indirim uygulanarak nihai faiz SOFR + 250 bps'e çekilir."
            ]),
            ("3. Yetkili İmza Makamı", [
                "Sendikasyon konsorsiyum anlaşmaları Şube Müdürü Hakan Serter ve Baş Denetçi Zeynep Altun ortak onayıyla tekemmül eder."
            ])
        ],
        table_data=[
            ["Kredi Dilimi", "Referans Faiz + Marj", "ESG Hedefi Gerçekleşme İndirimi", "Azami Tenör"],
            ["Tranche A (USD Sendikasyon)", "Term SOFR + 275 bps", "-25 bps İndirim (SOFR + 250 bps)", "3 Yıl (Bullet)"],
            ["Tranche B (EUR Sürdürülebilir)", "Euribor + 225 bps", "-20 bps İndirim (Euribor + 205 bps)", "5 Yıl Amortismanlı"],
            ["Tranche C (TL Kurumsal)", "TLREF + 350 bps", "-30 bps İndirim (TLREF + 320 bps)", "2 Yıl Rotatif"]
        ]
    )

    # 12. sube2_kurumsal_kredi_tahsis_ve_konsorsiyum_limitleri.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_kurumsal_kredi_tahsis_ve_konsorsiyum_limitleri.pdf",
        doc_id="MB-S2-KUR-2026-02",
        title="Büyük Kurumsal Kredi Tahsisi ve Konsorsiyum Risk Limitleri",
        branch="Şube-2 (Kurumsal Portföy Yönetimi)",
        classification="Çok Gizli — Kurumsal Kredi Komitesi",
        date_str="14.01.2026",
        sections=[
            ("1. Kurumsal Limit Baremleri", [
                "Yıllık cirosu 500 milyon TL'nin üzerindeki kurumsal holdinglere tahsis edilecek gayrinakdi ve nakdi kredilerde tek firma risk limiti banka özkaynaklarının %20'si ile sınırlandırılmıştır.",
                "Şube Müdürü Hakan Serter 50.000.000 TL'ye kadar konsorsiyum kredisi onaylama yetkisine sahiptir."
            ])
        ],
        table_data=[
            ["Kredi Ürünü", "Şube Yetki Limiti", "Teminat Şartı", "Gözden Geçirme"],
            ["Rotatif Kredi (Spot TL)", "50.000.000 TL", "Ticari Alacak Temliki / İpotek", "3 Ayda Bir"],
            ["Döviz Kredisi (USD/EUR İhracat)", "10.000.000 USD", "İhracat Taahhüt Hesabı", "6 Ayda Bir"],
            ["Konsorsiyum Katılım Payı", "100.000.000 TL", "Lider Banka Teminat Ajanlığı", "Yıllık Düzenli"]
        ]
    )

    # 13. sube2_proje_finansmani_ve_dcf_degerleme_ilkeleri.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_proje_finansmani_ve_dcf_degerleme_ilkeleri.pdf",
        doc_id="MB-S2-PROJ-2026-03",
        title="Proje Finansmanı, DCF Değerleme ve Finansal Rasyolar Kılavuzu",
        branch="Şube-2 (Yatırım Bankacılığı ve Finansal Modelleme)",
        classification="Gizli — Yatırım Komitesi",
        date_str="17.01.2026",
        sections=[
            ("1. Proje Fizibilite ve Finansal Rasyolar", [
                "Rüzgar, güneş ve otoyol gibi altyapı projelerinde İndirgenmiş Nakit Akımı (DCF) yöntemiyle Net Bugünkü Değer (NPV) pozitif olmalıdır.",
                "Borç Servisi Karşılama Oranı (DSCR - Debt Service Coverage Ratio) kredi vadesi boyunca hiçbir yıl 1.25x altına inemez. Minimum özkaynak katkı payı %30'dur."
            ])
        ],
        table_data=[
            ["Finansal Rasyo / Metrik", "Kabul Edilebilir Asgari Sınır", "İzleme Frekansı", "Aksiyon Eşiği"],
            ["DSCR (Borç Servis Karşılama)", "Minimum 1.25x", "3 Aylık Bilanço", "< 1.15x Ek Teminat"],
            ["LLCR (Kredi Ömrü Karşılama)", "Minimum 1.35x", "6 Aylık Model", "< 1.25x Temettü Blokesi"],
            ["Özkaynak Katkı Oranı", "Minimum %30 Nakit", "İnşaat Aşaması", "< %30 Hakediş Durdurma"]
        ]
    )

    # 14. sube2_doviz_turev_fx_swap_ve_forward_politikasi.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_doviz_turev_fx_swap_ve_forward_politikasi.pdf",
        doc_id="MB-S2-TUR-2026-04",
        title="Döviz Türev Ürünleri, FX Swap ve Forward İşlemleri Politikası",
        branch="Şube-2 (Hazine ve Finansal Türevler Masası)",
        classification="Piyasa Riski — Gizli",
        date_str="21.01.2026",
        sections=[
            ("1. Kur Riski Korunma İlkeleri", [
                "İthalatçı ve ihracatçı kurumsal firmaların kur riskini hedge etmek amacıyla ISDA sözleşmesi çerçevesinde Forward, Para Swapı ve Faiz Swapı (IRS) kontratları akdedilir."
            ])
        ],
        table_data=[
            ["Türev Ürün Türü", "Azami Tenör", "Teminat Marjı (Margin)", "Uzlaşma Şekli"],
            ["FX Forward (Döviz Alım/Satım)", "12 Ay", "%10 Nakit Teminat", "Vadede Fiziki Teslimat"],
            ["Cross Currency Swap (CCS)", "5 Yıl", "%15 Teminat / Hazine Limiti", "Periyodik Faiz Takası"],
            ["Faiz Swapı (IRS - SOFR/Euribor)", "7 Yıl", "ISDA Çerçeve Sözleşmesi", "Netleşmeli Nakit Uzlaşma"]
        ]
    )

    # 15. sube2_uluslararasi_ticaret_akreditif_ve_ucp600.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_uluslararasi_ticaret_akreditif_ve_ucp600.pdf",
        doc_id="MB-S2-TIC-2026-05",
        title="Dış Ticaretin Finansmanı, Akreditif ve UCP 600 Standartları",
        branch="Şube-2 (Dış Ticaret Operasyonları)",
        classification="Genel Bankacılık Prosedürü",
        date_str="24.01.2026",
        sections=[
            ("1. Akreditif Kuralları ve Teyit Esasları", [
                "Milletlerarası Ticaret Odası (ICC) UCP 600 kurallarına uygun gayrikabili rücu akreditifler açılır ve teyit eklenir. Vesaik inceleme süresi azami 5 banka iş günüdür."
            ])
        ],
        table_data=[
            ["Akreditif Türü", "Teyit Komisyonu (Yıllık)", "Vesaik İnceleme Süresi", "Risk Kriteri"],
            ["Görüldüğünde Ödemeli (Sight L/C)", "%1.50 - %2.50", "Azami 5 İş Günü", "Düşük Risk"],
            ["Vadeli / Poliçeli Akreditif", "%2.75 - %4.00", "Azami 5 İş Günü", "Orta Risk"],
            ["Standby L/C (Teminat Akreditifi)", "%1.75 - %3.00", "Talep Anında", "Kontrgaranti Gerekli"]
        ]
    )

    # 16. sube2_sirket_birlesme_ve_satin_alma_due_diligence.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_sirket_birlesme_ve_satin_alma_due_diligence.pdf",
        doc_id="MB-S2-MA-2026-06",
        title="Şirket Birleşme ve Devralmaları (M&A) Due Diligence Prosedürü",
        branch="Şube-2 (Kurumsal Finansman & Danışmanlık)",
        classification="Çok Gizli — M&A Masası",
        date_str="27.01.2026",
        sections=[
            ("1. Değerleme ve Çarpan Analizi", [
                "Satın alma finansmanı taleplerinde hedef şirketin Firma Değeri / FAVÖK (EV/EBITDA) ve Fiyat/Kazanç (F/K) çarpanları sektör ortalamalarıyla kıyaslanır.",
                "Due Diligence aşamasında bağımsız denetimden geçmiş UFRS bilançoları üzerinden Net Borç hesaplanır."
            ])
        ],
        table_data=[
            ["Sektör Grubu", "Hedef EV/FAVÖK Çarpanı", "Azami Satın Alma Kaldıracı", "Geri Ödeme Süresi"],
            ["Teknoloji & Yazılım", "8.0x - 14.0x", "Azami 3.5x Net Borç/FAVÖK", "5 Yıl"],
            ["İmalat & Sanayi", "5.0x - 8.5x", "Azami 3.0x Net Borç/FAVÖK", "7 Yıl"],
            ["Perakende & Tüketim", "6.0x - 9.0x", "Azami 2.5x Net Borç/FAVÖK", "5 Yıl"]
        ]
    )

    # 17. sube2_surdurulebilir_finansman_ve_yesil_tahvil_esaslari.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_surdurulebilir_finansman_ve_yesil_tahvil_esaslari.pdf",
        doc_id="MB-S2-ESG-2026-07",
        title="Sürdürülebilir Finansman, Yeşil Tahvil (Green Bond) ve ESG Esasları",
        branch="Şube-2 (Sürdürülebilir Finansman Departmanı)",
        classification="Genel Kurumsal Standart",
        date_str="30.01.2026",
        sections=[
            ("1. Yeşil Tahvil İhraç Çerçevesi", [
                "ICMA Yeşil Tahvil İlkeleri (Green Bond Principles) uyarınca ihraç edilen tahvillerin getirisi sadece yenilenebilir enerji, enerji verimliliği ve temiz ulaşım projelerinde kullanılabilir."
            ])
        ],
        table_data=[
            ["Kriter / Alan", "Uluslararası Standart", "Bağımsız Doğrulama (SPO)", "Faiz Teşvik İndirimi"],
            ["Yenilenebilir Enerji", "ICMA Green Bond Principles", "Sustainalytics / Moody's", "-30 bps Kupon İndirimi"],
            ["Döngüsel Ekonomi & Atık", "EU Taxonomy Uyumlu", "DNV GL Denetimi", "-20 bps Kupon İndirimi"]
        ]
    )

    # 18. sube2_denetim_komitesi_ve_yonetim_yapisi.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_denetim_komitesi_ve_yonetim_yapisi.pdf",
        doc_id="MB-S2-YON-2026-08",
        title="Şube-2 Yönetim Yapısı, Denetim Komitesi ve Onay Limitleri",
        branch="Şube-2 (Kurumsal Yönetim Kurulu)",
        classification="Çok Gizli — Yönetim Protokolü",
        date_str="02.02.2026",
        sections=[
            ("1. Yönetim ve Kredi Komitesi Heyeti", [
                "Şube Müdürü Hakan Serter büyük kurumsal portföyün lideridir. Baş Denetçi Zeynep Altun iç denetim, kredi tahsis uygunluk ve BDDK mevzuat teftişinden sorumludur.",
                "Kurumsal Portföy Lideri Cemil Vural 25.000.000 TL'ye kadar kurumsal limit onayına yetkilidir."
            ])
        ],
        table_data=[
            ["Yetkili Adı Soyadı", "Görevi / Ünvanı", "Yetki Sınırı", "İmza Yetkisi"],
            ["Hakan Serter", "Kurumsal Şube Müdürü", "50.000.000 TL Nakdi Kredi", "1. Derece A Grubu"],
            ["Zeynep Altun", "Baş Denetçi / Teftiş Lideri", "Tüm Kredi Dosyaları Veto Yetkisi", "Mevzuat Denetçisi"],
            ["Cemil Vural", "Kurumsal Portföy Lideri", "25.000.000 TL Nakdi Kredi", "2. Derece B Grubu"]
        ]
    )

    # 19. sube2_teminat_mektubu_ve_kontrgaranti_oranlari.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_teminat_mektubu_ve_kontrgaranti_oranlari.pdf",
        doc_id="MB-S2-MEK-2026-09",
        title="Kamu ve Özel İhale Teminat Mektupları ve Kontrgaranti Oranları",
        branch="Şube-2 (Gayrinakdi Krediler Masası)",
        classification="Hizmete Özel",
        date_str="04.02.2026",
        sections=[
            ("1. Teminat Mektubu Türleri ve Komisyonları", [
                "4734 sayılı Kamu İhale Kanunu kapsamında kesin, geçici ve avans teminat mektupları düzenlenir. Yabancı banka kontrgarantisi ile açılan mektuplarda komisyon indirimi sağlanır."
            ])
        ],
        table_data=[
            ["Teminat Mektubu Niteliği", "Yıllık Asgari Komisyon Oranı", "Azami Limit", "Geçerlilik Şartı"],
            ["Geçici Teminat Mektubu", "%2.00 Yıllık", "Proje Teklif Tutarının %3'ü", "Azami 6 Ay Süreli"],
            ["Kesin Teminat Mektubu", "%2.75 Yıllık", "Sözleşme Bedelinin %6'sı", "Süresiz / İş Bitimine Kadar"],
            ["Avans Teminat Mektubu", "%3.25 Yıllık", "Verilen Avansın %100'ü", "Hakedişten Kesintili"]
        ]
    )

    # 20. sube2_holding_nakit_yonetimi_ve_cash_pooling.pdf
    create_pdf(
        filename=BRANCH2_DIR / "sube2_holding_nakit_yonetimi_ve_cash_pooling.pdf",
        doc_id="MB-S2-CASH-2026-10",
        title="Holding Nakit Yönetimi, Otomatik Bakiye Süpürme (Cash Pooling)",
        branch="Şube-2 (Nakit Yönetimi ve Kurumsal Operasyon)",
        classification="Ticari Sır — Müşteri Sözleşmesi",
        date_str="06.02.2026",
        sections=[
            ("1. Konsolide Likidite Yönetimi", [
                "Holdinge bağlı grup şirketlerinin bankamız nezdindeki hesapları gün sonunda ana havuz hesabına otomatik süpürülür (Zero-Balance Account - ZBA) ve gecelik repo ile değerlendirilir."
            ])
        ],
        table_data=[
            ["Hizmet Bileşeni", "Süpürme Saati", "Faiz Nemalandırma Türü", "İşlem Ücreti"],
            ["Fiziki Havuzlama (ZBA)", "Her İş Günü 17:15", "O/N Gecelik Repo / Ters Repo", "Aylık Paket Ücreti"],
            ["Sanal Havuzlama (Notional)", "Gerçek Zamanlı", "Grup Konsolide Faiz Takası", "%0.15 Yıllık Komisyon"]
        ]
    )

    # ─── GRUP C: GENEL MERKEZ (HQ - MEVZUAT, RİSK & FEDERASYON - 10 ADET) ─────

    # 21. hq_5411_bankacilik_kanunu_ve_banka_sirri_esaslari.pdf (Gizlilik Kanıtı)
    create_pdf(
        filename=HQ_DIR / "hq_5411_bankacilik_kanunu_ve_banka_sirri_esaslari.pdf",
        doc_id="MB-HQ-MEVZ-2026-01",
        title="5411 Sayılı Bankacılık Kanunu, Sır Saklama Yükümlülüğü ve Md. 73",
        branch="Genel Merkez (Hukuk Müşavirliği ve Mevzuat Uyum)",
        classification="Yasal Düzenleme — Genel Uyum",
        date_str="08.01.2026",
        sections=[
            ("1. Madde 73 — Sır Saklama Yükümlülüğü", [
                "Sıfat ve görevleri dolayısıyla bankalara veya müşterilerine ait sırları öğrenenler, bu sırları kanunen açıkça yetkili kılınan mercilerden başkasına açıklayamazlar. Bu yükümlülük görevden ayrıldıktan sonra da devam eder.",
                "Bankacılık sırrı niteliğindeki müşteri işlem ve verileri, Genel Merkez merkezi yapay zeka havuzlarına dahi ham metin olarak kopyalanamaz. Dağıtık şube mimarisinde veri yerinde kalmalı, yalnızca anonim gradyanlar paylaşılmalıdır."
            ])
        ],
        table_data=[
            ["Kanun Maddesi", "Düzenleme Konusu", "Yasal Yaptırım", "Bankamız Uyum Önlemi"],
            ["Madde 73", "Banka ve Müşteri Sırrı", "1 yıldan 3 yıla kadar hapis", "Federe Öğrenme & Yerel RAG"],
            ["Madde 54", "Kredi Sınırları", "İdari Para Cezası", "Konsolide Risk İzleme Motoru"],
            ["Madde 44", "Sermaye Yeterliliği", "Faaliyet İzni Kısıtlaması", "Basel III %8 Asgari SYR"]
        ]
    )

    # 22. hq_sermaye_yeterlilik_rasyosu_syr_hesaplama.pdf
    create_pdf(
        filename=HQ_DIR / "hq_sermaye_yeterlilik_rasyosu_syr_hesaplama.pdf",
        doc_id="MB-HQ-RISK-2026-02",
        title="Basel III Çerçevesinde Sermaye Yeterlilik Rasyosu (SYR) Hesaplaması",
        branch="Genel Merkez (Risk Yönetimi ve Aktif-Pasif Komitesi)",
        classification="Kurumsal Risk Politikası",
        date_str="11.01.2026",
        sections=[
            ("1. Yasal Sermaye Yeterlilik Oranı Formülü", [
                "SYR = (Özkaynaklar / Risk Ağırlıklı Varlıklar [Kredi + Piyasa + Operasyonel]) * 100",
                "5411 sayılı Kanun uyarınca asgari yasal oran %8'dir. BDDK hedef oranı %12 olarak uygulamaktadır. MosaicBank iç hedefi %14.50'dir."
            ])
        ],
        table_data=[
            ["Sermaye Kalemi / Rasyo", "Yasal Asgari Oran", "BDDK Hedef Oranı", "MosaicBank Mevcut Oranı"],
            ["Çekirdek Sermaye (Tier 1)", "Minimum %4.50", "Minimum %6.00", "%11.85 (Güçlü Sermaye)"],
            ["Toplam Özkaynak / SYR", "Minimum %8.00", "Minimum %12.00", "%15.42 (Yüksek Tampon)"],
            ["Sermaye Koruma Tamponu", "%2.50", "%2.50", "Eksiksiz Tesis Edildi"]
        ]
    )

    # 23. hq_tcmb_zorunlu_karsiliklar_ve_likidite_politikalari.pdf
    create_pdf(
        filename=HQ_DIR / "hq_tcmb_zorunlu_karsiliklar_ve_likidite_politikalari.pdf",
        doc_id="MB-HQ-TCMB-2026-03",
        title="TCMB Zorunlu Karşılık Oranları ve Likidite Karşılama (LKO) Politikası",
        branch="Genel Merkez (Hazine ve Finansal Kurumlar)",
        classification="Mevzuat Uyum",
        date_str="13.01.2026",
        sections=[
            ("1. Zorunlu Karşılık Uygulaması", [
                "TCMB nezdinde bloke edilen zorunlu karşılıklar Türk Lirası mevduatlarda vadeye göre %8 ile %15 arasında, Yabancı Para mevduatlarda %25 ile %30 arasında değişmektedir."
            ])
        ],
        table_data=[
            ["Mevduat Vadesi / Cinsi", "TCMB Zorunlu Karşılık Oranı", "Bloke Faiz Oranı", "İzleme Dönemi"],
            ["TL Vadesiz ve 1 Aya Kadar", "%15.00", "%18.50 TCMB Nemalandırması", "2 Haftalık Cuma"],
            ["TL 3 Aya Kadar Vadeli", "%12.00", "%22.00 TCMB Nemalandırması", "2 Haftalık Cuma"],
            ["YP (USD/EUR) Tüm Vadeler", "%25.00 - %30.00", "%0.00 (Sıfır Faiz)", "2 Haftalık Cuma"]
        ]
    )

    # 24. hq_bddk_kredi_sinirlari_ve_enerji_bazli_ltv.pdf
    create_pdf(
        filename=HQ_DIR / "hq_bddk_kredi_sinirlari_ve_enerji_bazli_ltv.pdf",
        doc_id="MB-HQ-BDDK-2026-04",
        title="BDDK Kredi Sınırlandırmaları ve Enerji Sınıfı Bazlı Konut LTV Tablosu",
        branch="Genel Merkez (Kredi Politikaları Departmanı)",
        classification="Resmi Düzenleme",
        date_str="16.01.2026",
        sections=[
            ("1. Kademeli LTV Sınırları", [
                "Konut kredilerinde azami kredi tutarı konutun enerji sınıfına (A, B veya Standart) ve ekspertiz değerine göre belirlenir. 1. el ve 2. el konutlarda oranlar farklılaştırılmıştır."
            ])
        ],
        table_data=[
            ["Konut Değeri ve Türü", "A Sınıfı Enerji Belgesi", "B Sınıfı Enerji Belgesi", "Diğer Enerji Sınıfları"],
            ["1. El Konut < 5.000.000 TL", "%90 LTV Azami", "%85 LTV Azami", "%80 LTV Azami"],
            ["1. El Konut 5M - 10M TL", "%80 LTV Azami", "%75 LTV Azami", "%70 LTV Azami"],
            ["2. El Konut < 5.000.000 TL", "%70 LTV Azami", "%65 LTV Azami", "%60 LTV Azami"]
        ]
    )

    # 25. hq_tms9_ufrs9_beklenen_kredi_zarari_modeli.pdf
    create_pdf(
        filename=HQ_DIR / "hq_tms9_ufrs9_beklenen_kredi_zarari_modeli.pdf",
        doc_id="MB-HQ-UFRS-2026-05",
        title="TFRS 9 / UFRS 9 Beklenen Kredi Zararı (ECL) Karşılık Modeli",
        branch="Genel Merkez (Mali Kontrol ve Muhasebe Standartları)",
        classification="Finansal Raporlama Esasları",
        date_str="19.01.2026",
        sections=[
            ("1. 3 Aşamalı Kredi Zararı Yaklaşımı", [
                "Aşama 1 (Sağlam): 12 aylık beklenen zarar karşılığı ayrılır.",
                "Aşama 2 (Önemli Risk Artışı - 30+ gün gecikme): Ömür boyu beklenen zarar ayrılır.",
                "Aşama 3 (Temerrüt / Donuk Alacak - 90+ gün gecikme): Spesifik karşılık ayrılır. ECL = PD * LGD * EAD formülü uygulanır."
            ])
        ],
        table_data=[
            ["Kredi Aşaması (Stage)", "Gecikme Günü Aralığı", "Karşılık Hesaplama Dönemi", "Karşılık Oranı Aralığı"],
            ["Aşama 1 (Normal Kredi)", "0 - 29 Gün Gecikme", "12 Aylık Beklenen Kredi Kaybı", "%0.50 - %1.50"],
            ["Aşama 2 (Yakın İzleme)", "30 - 89 Gün Gecikme", "Ömür Boyu Beklenen Kredi Kaybı", "%5.00 - %25.00"],
            ["Aşama 3 (Tahsili Gecikmiş)", "90+ Gün Temerrüt", "Ömür Boyu Spesifik Değer Düşüklüğü", "%50.00 - %100.00"]
        ]
    )

    # 26. hq_kvkk_kisisel_verilerin_korunmasi_uyum_rehberi.pdf
    create_pdf(
        filename=HQ_DIR / "hq_kvkk_kisisel_verilerin_korunmasi_uyum_rehberi.pdf",
        doc_id="MB-HQ-KVKK-2026-06",
        title="6698 Sayılı KVKK ve Bankacılık Sektörü Veri Koruma Uyum Rehberi",
        branch="Genel Merkez (KVKK İrtibat Bürosu ve Veri Güvenliği)",
        classification="Yasal Yükümlülük",
        date_str="23.01.2026",
        sections=[
            ("1. Kişisel Verilerin Anonimleştirilmesi", [
                "Müşteri adı, TC kimlik numarası, telefon numarası ve hesap bakiyesi PII (Personally Identifiable Information) kapsamındadır. LLM istemlerine iletilmeden önce yerel olarak maskelenmelidir."
            ])
        ],
        table_data=[
            ["Veri Türü", "KVKK Sınıfı", "İzin Verilen İşleme Yöntemi", "Şube Dışına Aktarım"],
            ["TC Kimlik / Ad Soyad", "Özel / Genel Kişisel Veri", "Şube İçi Maskeleme [NAME_1]", "KESİNLİKLE YASAK"],
            ["Hesap / IBAN No", "Finansal Kişisel Veri", "Salt Hash / Kırpma", "KESİNLİKLE YASAK"],
            ["Anonimleştirilmiş Metin", "Anonim Veri", "Semantik Vektörleştirme", "İzinli (Federe/RAG)"]
        ]
    )

    # 27. hq_federe_ogrenme_ve_diferansiyel_gizlilik_elkitabi.pdf (Federe İlkeler)
    create_pdf(
        filename=HQ_DIR / "hq_federe_ogrenme_ve_diferansiyel_gizlilik_elkitabi.pdf",
        doc_id="MB-HQ-FL-2026-07",
        title="Federe Öğrenme (FedProx) ve Diferansiyel Gizlilik (DP-SGD) El Kitabı",
        branch="Genel Merkez (Yapay Zeka ve Veri Bilimi Ar-Ge Departmanı)",
        classification="Yüksek Teknoloji Ar-Ge — Açık Kaynak Lisanslı",
        date_str="26.01.2026",
        sections=[
            ("1. Federe Öğrenme ve FedProx Yaklaşımı", [
                "MosaicBank dağıtık yapay zeka mimarisinde şubeler heterojen (Non-IID) verilere sahiptir. Şubeler arası veri örtüşmesi %6.0 seviyesindedir.",
                "Klasik FedAvg yerine, yerel ağırlık sapmalarını önlemek amacıyla ceza terimi (mu = 0.01) içeren FedProx algoritması kullanılır."
            ]),
            ("2. Diferansiyel Gizlilik (DP-SGD) Parametreleri", [
                "Gaussian mekanizması ile gradyanlara gürültü eklenir. Gizlilik bütçesi epsilon = 1.0 (katı üretim gizliliği), sızıntı toleransı delta = 1e-5 ve L2 norm kırpma duyarlılığı S = 1.0 olarak uygulanır."
            ])
        ],
        table_data=[
            ["DP Parametresi", "Uygulanan Sayısal Değer", "Matematiksel Rolü", "Güvenlik Düzeyi"],
            ["Epsilon (ε)", "1.0", "Gizlilik Bütçesi (Küçük = Yüksek Gizlilik)", "Çok Yüksek (Sıfır Sızıntı)"],
            ["Delta (δ)", "1e-5 (0.00001)", "Olasılıksal Sızıntı Üst Sınırı (<< 1/n)", "Matematiksel Kanıtlı"],
            ["Duyarlılık (Sensitivity)", "1.0 (L2 Norm)", "Gradyan Vektörü Kırpma Eşiği", "Aykırı Veri Önleyici"],
            ["FedProx Mu (μ)", "0.01", "Non-IID Şube Gradyan Dengeleme Katsayısı", "Optimal Yakınsama"]
        ]
    )

    # 28. hq_bist_kap_finansal_tablo_ve_rasyo_analizi.pdf
    create_pdf(
        filename=HQ_DIR / "hq_bist_kap_finansal_tablo_ve_rasyo_analizi.pdf",
        doc_id="MB-HQ-BIST-2026-08",
        title="BIST 100 KAP Finansal Bildirimleri ve Oran Analizi Metodolojisi",
        branch="Genel Merkez (Hisse Senedi ve Piyasa Analitiği)",
        classification="Piyasa Araştırma Raporu",
        date_str="29.01.2026",
        sections=[
            ("1. Bilanço Rasyo Analitiği", [
                "KAP bildirimlerinde şirketlerin finansal sağlığı DuPont modeli, Likidite, Faaliyet ve Kaldıraç oranları üzerinden analiz edilir."
            ])
        ],
        table_data=[
            ["Finansal Rasyo", "Formülasyon", "Sağlıklı Şirket Sınır Değeri", "BIST 100 Medyanı"],
            ["Cari Oran", "Dönen Varlıklar / Kısa Vadeli Yabancı Kaynaklar", "Minimum 1.50x", "1.65x"],
            ["Asit-Test (Likidite)", "(Dönen Varlıklar - Stoklar) / KVYK", "Minimum 1.00x", "1.10x"],
            ["Net Borç / FAVÖK", "(Kısa+Uzun Finansal Borç - Nakit) / FAVÖK", "Azami 3.00x", "2.15x"],
            ["Özsermaye Kârlılığı (ROE)", "Net Dönem Kârı / Toplam Özkaynaklar", "Enflasyon Üzeri (> %40)", "%44.50"]
        ]
    )

    # 29. hq_masak_kara_para_ve_aml_onleme_rehberi.pdf
    create_pdf(
        filename=HQ_DIR / "hq_masak_kara_para_ve_aml_onleme_rehberi.pdf",
        doc_id="MB-HQ-AML-2026-09",
        title="MASAK Uyum, Karapara Aklamanın Önlenmesi (AML) ve ŞİB Rehberi",
        branch="Genel Merkez (Mali Suçları Önleme ve Uyum Departmanı)",
        classification="Mevzuat Uyum — Zorunlu",
        date_str="01.02.2026",
        sections=[
            ("1. Şüpheli İşlem Bildirimi (ŞİB) Yükümlülükleri", [
                "5549 sayılı Kanun uyarınca şüphe duyulan tüm işlemler tutara bakılmaksızın 10 iş günü içinde MASAK Başkanlığı'na ŞİB formu ile bildirilir."
            ])
        ],
        table_data=[
            ["İşlem Türü", "Yasal Eşik Tutarı", "Kimlik Tespiti Zorunluluğu", "MASAK Bildirim Süresi"],
            ["Nakit Para Yatırma / Çekme", "85.000 TL ve üzeri", "TCKN ve Nüfus Cüzdanı", "Rutin Raporlama"],
            ["Yurtdışı SWIFT Para Transferi", "50.000 USD / EUR üzeri", "Kaynak Belgesi & Sözleşme", "Aynı Gün Bildirim"],
            ["Şüpheli İşlem Bildirimi (ŞİB)", "TUTAR SINIRI YOKTUR", "Kapsamlı KYC / PEP İncelemesi", "Azami 10 İş Günü"]
        ]
    )

    # 30. hq_surdurulebilirlik_ve_iklim_riski_yonetimi.pdf
    create_pdf(
        filename=HQ_DIR / "hq_surdurulebilirlik_ve_iklim_riski_yonetimi.pdf",
        doc_id="MB-HQ-IKLIM-2026-10",
        title="BDDK İklim Değişikliği Risk Yol Haritası ve Yeşil Varlık Rasyosu (GAR)",
        branch="Genel Merkez (Sürdürülebilirlik ve Yeşil Dönüşüm Komitesi)",
        classification="Stratejik Rapor",
        date_str="03.02.2026",
        sections=[
            ("1. Yeşil Varlık Rasyosu (Green Asset Ratio - GAR)", [
                "Bankaların kredi portföylerinde AB ve Türkiye Yeşil Taksonomisine uygun çevreye duyarlı projelerin payı 2026 yılı sonuna kadar asgari %15'e çıkarılacaktır."
            ])
        ],
        table_data=[
            ["İzleme Yılı", "Hedef Yeşil Varlık Oranı (GAR)", "Karbon Emisyon Azaltım Hedefi", "Kapsam"],
            ["2026 Yılı Sonu", "Minimum %15 GAR", "%25 Azaltım (Kapsam 1 ve 2)", "Tüm Şube Portföyleri"],
            ["2028 Yılı Sonu", "Minimum %25 GAR", "%40 Azaltım", "Tüm Şube Portföyleri"],
            ["2030 Hedefi", "Minimum %35 GAR", "Net Sıfır Bankacılık Operasyonu", "Konsolide MosaicBank"]
        ]
    )

    print("=" * 70)
    print("Toplam 30 adet PDF başarıyla üretildi:")
    print(f"  • Şube 1: {len(list(BRANCH1_DIR.glob('*.pdf')))} PDF ({BRANCH1_DIR})")
    print(f"  • Şube 2: {len(list(BRANCH2_DIR.glob('*.pdf')))} PDF ({BRANCH2_DIR})")
    print(f"  • HQ    : {len(list(HQ_DIR.glob('*.pdf')))} PDF ({HQ_DIR})")
    print("=" * 70)

if __name__ == "__main__":
    generate_all_pdfs()
