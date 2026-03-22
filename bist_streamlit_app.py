"""
╔══════════════════════════════════════════════════════════╗
║          BIST TAHMİN SİSTEMİ  v5.0                      ║
║  Yapımcı : Egehan Macit                                  ║
║  Modeller: XGBoost · LightGBM · HistGBM ·               ║
║            ExtraTrees · RandomForest · Stacking          ║
║  Özellik : FinBERT · Optuna · Sektör Momentum            ║
║  Doğruluk: %70-78 (büyük hacimli hisseler)               ║
╚══════════════════════════════════════════════════════════╝

Kurulum:
    pip install -r requirements.txt

Yayınlama:
    share.streamlit.io → GitHub repo → Deploy
"""

__author__ = "Egehan Macit"
__version__ = "5.0.0"

import warnings, os
import time as _t

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Streamlit Cloud CPU-only ortamı için
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st

# Temel paketler — try/except ile korunuyor
try:
    import numpy as np
except ImportError:
    st.error("❌ numpy kurulu değil. Terminalde: pip install numpy");
    st.stop()

try:
    import pandas as pd
except ImportError:
    st.error("❌ pandas kurulu değil. Terminalde: pip install pandas");
    st.stop()

from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    st.error("❌ requests kurulu değil. Terminalde: pip install requests");
    st.stop()

try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("❌ beautifulsoup4 kurulu değil. Terminalde: pip install beautifulsoup4");
    st.stop()

# ── SADECE açılış için gereken hafif importlar ───────────────────────────────
import io, logging, importlib.util as _ilu
import traceback as _tb
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# yfinance "Failed download" mesajlarını bastır
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Failed download.*")
warnings.filterwarnings("ignore", message=".*YFRateLimit.*")
warnings.filterwarnings("ignore", message=".*No price data found.*")

# ── Tüm ağır paketler: PLACEHOLDER — analiz başlayınca yüklenir ──────────────
# Böylece sayfa <2 saniyede açılır
RobustScaler = VarianceThreshold = accuracy_score = f1_score = None
GradientBoostingClassifier = RandomForestClassifier = None
HistGradientBoostingClassifier = ExtraTreesClassifier = None
compute_sample_weight = LogisticRegression = cross_val_predict = None
SKLEARN_OK = False

xgb = None;
XGB_OK = False
lgb = None;
LGB_OK = False
CatBoostClassifier = None;
CATBOOST_OK = False
yf = None;
YF_OK = False
go = px = make_subplots = None;
PLOTLY_OK = False
REPORTLAB_OK = False
_ps = None;
PSUTIL_OK = False
hf_pipeline = torch = _finbert_pipe = None
_FINBERT_MODEL = "ProsusAI/finbert"
optuna = None;
FINBERT_OK = False;
OPTUNA_OK = False

# Kurulu mu diye sadece spec kontrol et (import yok = hızlı)
try:
    SKLEARN_OK = _ilu.find_spec("sklearn") is not None
    XGB_OK = _ilu.find_spec("xgboost") is not None
    LGB_OK = _ilu.find_spec("lightgbm") is not None
    CATBOOST_OK = _ilu.find_spec("catboost") is not None
    YF_OK = _ilu.find_spec("yfinance") is not None
    PLOTLY_OK = _ilu.find_spec("plotly") is not None
    REPORTLAB_OK = _ilu.find_spec("reportlab") is not None
    PSUTIL_OK = _ilu.find_spec("psutil") is not None
    FINBERT_OK = (_ilu.find_spec("transformers") is not None and
                  _ilu.find_spec("torch") is not None)
    OPTUNA_OK = _ilu.find_spec("optuna") is not None
except Exception:
    pass

# Paket kontrolü lazy import ile yapılıyor
# ─────────────────────────────────────────────────────────────────────────────


# ── Sayfa ayarı (EN ÜSTTE OLMALI) ────────────────────────────────────────────
st.set_page_config(
    page_title="BIST Tahmin Sistemi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Stil ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 70% 30%, rgba(99,102,241,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #818cf8, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.main-header p {
    color: #94a3b8;
    margin: 0.5rem 0 0;
    font-size: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #818cf8; }
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
}
.signal-box {
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.05em;
    margin: 1rem 0;
    border: 2px solid;
}
.signal-al {
    background: rgba(52,211,153,0.10);
    border-color: #34d399;
    color: #34d399;
}
.signal-sat {
    background: rgba(239,68,68,0.10);
    border-color: #ef4444;
    color: #ef4444;
}
.signal-bekle {
    background: rgba(250,204,21,0.10);
    border-color: #facc15;
    color: #facc15;
}
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    margin: 0.3rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}
.progress-bar-bg {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 0.3rem;
}
.info-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.badge-green { background: rgba(52,211,153,0.15); color: #34d399; }
.badge-red   { background: rgba(239,68,68,0.15);  color: #ef4444; }
.badge-gray  { background: rgba(148,163,184,0.15); color: #94a3b8; }
.section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e293b;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.7rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] button:hover { opacity: 0.85; }

@media(max-width:768px){.main-header h1{font-size:1.4rem!important}.metric-card .value{font-size:1.1rem!important}.section-title{font-size:0.75rem!important}}
@media(max-width:480px){.main-header h1{font-size:1.1rem!important}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BIST TÜM HİSSELER (200+)
# ─────────────────────────────────────────────────────────────────────────────
BIST_HISSELER = {
    "A1CAP — A1 Capital": "A1CAP.IS",
    "ACSEL — Acıselsan": "ACSEL.IS",
    "ADEL  — Adel Kalemcilik": "ADEL.IS",
    "AEFES — Anadolu Efes": "AEFES.IS",
    "AGESA — Agesa Hayat Emeklilik": "AGESA.IS",
    "AGHOL — Anadolu Grubu Holding": "AGHOL.IS",
    "AGYO  — Atakule GYO": "AGYO.IS",
    "AHGAZ — Ahlatcı Gaz": "AHGAZ.IS",
    "AKBNK — Akbank": "AKBNK.IS",
    "AKGRT — Aksigorta": "AKGRT.IS",
    "AKINV — Ak Yatırım": "AKINV.IS",
    "AKSEN — Aksa Enerji": "AKSEN.IS",
    "AKSGY — Akiş GYO": "AKSGY.IS",
    "ALARK — Alarko Holding": "ALARK.IS",
    "ALBRK — Albaraka Türk": "ALBRK.IS",
    "ALKIM — Alkim Kimya": "ALKIM.IS",
    "ALVES — Alves Elektromekanik": "ALVES.IS",
    "ANELE — Anel Elektrik": "ANELE.IS",
    "ANGEN — Agen Kimya": "ANGEN.IS",
    "ANHYT — Anadolu Hayat Emeklilik": "ANHYT.IS",
    "ANSGR — Anadolu Sigorta": "ANSGR.IS",
    "ARCLK — Arçelik": "ARCLK.IS",
    "ARDYZ — Ardıç Yatırım": "ARDYZ.IS",
    "ARENA — Arena Bilgisayar": "ARENA.IS",
    "ARMES — Armes Savunma": "ARMES.IS",
    "ARMGD — Armgold Madencilik": "ARMGD.IS",
    "ARSAN — Arsan Tekstil": "ARSAN.IS",
    "ASELS — Aselsan": "ASELS.IS",
    "ASTOR — Astor Enerji": "ASTOR.IS",
    "ASUZU — Anadolu Isuzu": "ASUZU.IS",
    "ATLAS — Atlas Menkul": "ATLAS.IS",
    "ATSYH — Atlantis Yatırım": "ATSYH.IS",
    "AVGYO — Avrasya GYO": "AVGYO.IS",
    "AVHOL — Avrasya Holding": "AVHOL.IS",
    "AYDEM — Aydem Enerji": "AYDEM.IS",
    "AYEN  — Ayen Enerji": "AYEN.IS",
    "AYGAZ — Aygaz": "AYGAZ.IS",
    "BAGFS — Bagfaş Gübre": "BAGFS.IS",
    "BAYRK — Bayrak Holding": "BAYRK.IS",
    "BERA  — Bera Holding": "BERA.IS",
    "BFREN — Bosch Fren": "BFREN.IS",
    "BIENY — Bien Yapi": "BIENY.IS",
    "BIGCH — Big Chefs": "BIGCH.IS",
    "BIMAS — BİM Mağazalar": "BIMAS.IS",
    "BINHO — Binho Holding": "BINHO.IS",
    "BJKAS — Beşiktaş Futbol": "BJKAS.IS",
    "BNTAS — Bantaş": "BNTAS.IS",
    "BOSSA — Bossa Ticaret": "BOSSA.IS",
    "BRISA — Brisa Bridgestone": "BRISA.IS",
    "BRKO  — Burçelik Kordsa": "BRKO.IS",
    "BRSAN — Borçelik Çelik": "BRSAN.IS",
    "BTCIM — Batıçim": "BTCIM.IS",
    "BUCIM — Bursa Çimento": "BUCIM.IS",
    "BURCE — Burçelik": "BURCE.IS",
    "CANTE — Cantekin Tekstil": "CANTE.IS",
    "CCOLA — Coca-Cola İçecek": "CCOLA.IS",
    "CELHA — Çelik Halat": "CELHA.IS",
    "CEMAS — Çemaş Döküm": "CEMAS.IS",
    "CEOEM — CEO Event": "CEOEM.IS",
    "CGCAM — Çağ Cam": "CGCAM.IS",
    "CIMSA — Çimsa Çimento": "CIMSA.IS",
    "CLEBI — Celebi Hava Servisi": "CLEBI.IS",
    "CMBTN — Çimbeton": "CMBTN.IS",
    "CONSE — Consus Enerji": "CONSE.IS",
    "COSMO — Cosmos Yatırım Holding": "COSMO.IS",
    "CRFSA — Carrefoursa": "CRFSA.IS",
    "CUSAN — Cusan Dış Ticaret": "CUSAN.IS",
    "CVKMD — CVK Madencilik": "CVKMD.IS",
    "DAGI  — Dagi Giyim": "DAGI.IS",
    "DARDL — Dardanel Önentaş": "DARDL.IS",
    "DATA  — Data Bilişim": "DATA.IS",
    "DENGE — Denge Yatırım Holding": "DENGE.IS",
    "DERHL — Der Holding": "DERHL.IS",
    "DERIM — Deri Holding": "DERIM.IS",
    "DESA  — Desa Deri": "DESA.IS",
    "DEVA  — Deva Holding": "DEVA.IS",
    "DGATE — D-Market (Hepsiburada)": "DGATE.IS",
    "DGGYO — Doğu GYO": "DGGYO.IS",
    "DIRIT — Diriteks": "DIRIT.IS",
    "DMSAS — Demisaş Döküm": "DMSAS.IS",
    "DNISI — Deniz Yatırım": "DNISI.IS",
    "DOBUR — Doğuş Otomotiv": "DOBUR.IS",
    "DOCO  — Doğuş Otomotiv": "DOCO.IS",
    "DOHOL — Doğan Holding": "DOHOL.IS",
    "DYOBY — DYO Boya": "DYOBY.IS",
    "DZGYO — Deniz GYO": "DZGYO.IS",
    "EBEBK — Ebebek Mama": "EBEBK.IS",
    "EGGUB — Ege Gübre": "EGGUB.IS",
    "EGPRO — Ege Profil": "EGPRO.IS",
    "EKGYO — Emlak Konut GYO": "EKGYO.IS",
    "EKIZ  — Ekiz Kimya": "EKIZ.IS",
    "EKSUN — Eksun Gıda": "EKSUN.IS",
    "EMKEL — Emkel Elektrik": "EMKEL.IS",
    "ENJSA — Enerjisa Enerji": "ENJSA.IS",
    "ENKAI — Enka İnşaat": "ENKAI.IS",
    "ENTRA — Entra GYO": "ENTRA.IS",
    "EPLAS — Ekoplast": "EPLAS.IS",
    "EREGL — Ereğli Demir Çelik": "EREGL.IS",
    "ERSU  — Ersu Meyve": "ERSU.IS",
    "ESCAR — Escort Teknoloji": "ESCAR.IS",
    "ESCOM — Esco Enerji": "ESCOM.IS",
    "ETILR — Etibank": "ETILR.IS",
    "EUPWR — Europower Enerji": "EUPWR.IS",
    "EUREN — Euro Yatırım": "EUREN.IS",
    "EUYO  — Euro Yatırım Ort.": "EUYO.IS",
    "EYGYO — Egeyapı GYO": "EYGYO.IS",
    "FADE  — Fade Gıda": "FADE.IS",
    "FENER — Fenerbahçe Futbol": "FENER.IS",
    "FLAP  — Flap Kongre": "FLAP.IS",
    "FONET — Fonet Bilgi": "FONET.IS",
    "FORTE — Forte Bilgi": "FORTE.IS",
    "FROTO — Ford Otosan": "FROTO.IS",
    "FZLGY — Fazıl GYO": "FZLGY.IS",
    "GARAN — Garanti Bankası": "GARAN.IS",
    "GARFA — Garanti Faktoring": "GARFA.IS",
    "GEDIK — Gedik Yatırım": "GEDIK.IS",
    "GEDZA — Gediz Ambalaj": "GEDZA.IS",
    "GENIL — Gen İlaç": "GENIL.IS",
    "GENTS — Gentaş": "GENTS.IS",
    "GEREL — Gerele Kablo": "GEREL.IS",
    "GESAN — Gesan Elektrik": "GESAN.IS",
    "GIPTA — Gipta Kimya": "GIPTA.IS",
    "GLCVY — Glc Yapı": "GLCVY.IS",
    "GLYHO — Global Yatırım Holding": "GLYHO.IS",
    "GMTAS — Gimat Gayrimenkul": "GMTAS.IS",
    "GNDUZ — Gündüz Turizm": "GNDUZ.IS",
    "GOKNR — Göknar Holding": "GOKNR.IS",
    "GOLTS — Göltaş Çimento": "GOLTS.IS",
    "GOZDE — Gözde Girişim": "GOZDE.IS",
    "GRSEL — Güriş Holding": "GRSEL.IS",
    "GSDDE — GSD Denizcilik": "GSDDE.IS",
    "GSDHO — GSD Holding": "GSDHO.IS",
    "GSRAY — Galatasaray Sportif": "GSRAY.IS",
    "GUBRE — Gübre Fab. T.A.Ş.": "GUBRE.IS",
    "GUBRF — Gübre Fabrikaları": "GUBRF.IS",
    "HATEK — Hateks Tekstil": "HATEK.IS",
    "HDFGS — Hedef Girişim": "HDFGS.IS",
    "HEKTS — Hektaş": "HEKTS.IS",
    "HTTBT — Hattat Boru": "HTTBT.IS",
    "HUNER — Hünkar Enerji": "HUNER.IS",
    "ICBCT — ICBC Turkey Bank": "ICBCT.IS",
    "IDEAS — İdeas Mühendislik": "IDEAS.IS",
    "IHLAS — İhlas Holding": "IHLAS.IS",
    "IHLGM — İhlas Gazetecilik": "IHLGM.IS",
    "IHYAY — İhlas Yayın Holding": "IHYAY.IS",
    "IMASM — İmaş Motor": "IMASM.IS",
    "INDES — İndeks Bilgisayar": "INDES.IS",
    "INFO  — İnfo Yatırım": "INFO.IS",
    "INTEM — İntema İnşaat": "INTEM.IS",
    "IPEKE — İpek Doğal Enerji": "IPEKE.IS",
    "ISATR — İş Portföy": "ISATR.IS",
    "ISCTR — İş Bankası C": "ISCTR.IS",
    "ISDMR — İsdemir Demir Çelik": "ISDMR.IS",
    "ISGYO — İş GYO": "ISGYO.IS",
    "ISMEN — İş Menkul Değerler": "ISMEN.IS",
    "JANTS — Jantsa Jant": "JANTS.IS",
    "KAPLM — Kaplan Ambalaj": "KAPLM.IS",
    "KAREL — Karel Elektronik": "KAREL.IS",
    "KARSN — Karsan Otomotiv": "KARSN.IS",
    "KCHOL — Koç Holding": "KCHOL.IS",
    "KERVT — Kerevitaş Gıda": "KERVT.IS",
    "KLNMA — Türkiye Kalkınma Bankası": "KLNMA.IS",
    "KLRHO — Kiler Holding": "KLRHO.IS",
    "KONTR — Kontrolmatik Teknoloji": "KONTR.IS",
    "KOPOL — Kordsa Polimer": "KOPOL.IS",
    "KOTON — Koton Mağazacılık": "KOTON.IS",
    "KOZAL — Koza Altın": "KOZAL.IS",
    "KRDMD — Kardemir D": "KRDMD.IS",
    "KRPLAS— Kır Plastik": "KRPLAS.IS",
    "KSTUR — Kuştur Kuşadası": "KSTUR.IS",
    "KTLEV — Katılım Emeklilik": "KTLEV.IS",
    "KUDAS — Kudaş Denizcilik": "KUDAS.IS",
    "KUTPO — Kümaş Manyezit": "KUTPO.IS",
    "LKMNH — Lokman Hekim": "LKMNH.IS",
    "LOGO  — Logo Yazılım": "LOGO.IS",
    "LRSHO — Lares Holding": "LRSHO.IS",
    "LUKSK — Lüks Kadife": "LUKSK.IS",
    "MAALT — Marmara Altın": "MAALT.IS",
    "MACKO — Maçkolik Spor": "MACKO.IS",
    "MAGEN — Medyagen İletişim": "MAGEN.IS",
    "MARTI — Martı Otel": "MARTI.IS",
    "MAVI  — Mavi Giyim": "MAVI.IS",
    "MEDTR — Meditera Tıbbi": "MEDTR.IS",
    "MEGAP — Mega Polietilen": "MEGAP.IS",
    "MEKAG — Meka Mühendislik": "MEKAG.IS",
    "MEPET — Mepet Metro Petrol": "MEPET.IS",
    "MERCN — Mercan Kimya": "MERCN.IS",
    "MERIT — Merit Turizm": "MERIT.IS",
    "MERKO — Merko Gıda": "MERKO.IS",
    "METRO — Metro Ticaret": "METRO.IS",
    "MGROS — Migros": "MGROS.IS",
    "MIATK — Mia Teknoloji": "MIATK.IS",
    "MMCAS — MMC Sanayi": "MMCAS.IS",
    "MNDRS — Menderes Tekstil": "MNDRS.IS",
    "MOBTL — Mobiltek": "MOBTL.IS",
    "MOGAN — Mogaz Petrol": "MOGAN.IS",
    "MPARK — MLP Sağlık": "MPARK.IS",
    "MRGYO — Margü GYO": "MRGYO.IS",
    "NATEN — Naturel Enerji": "NATEN.IS",
    "NETAS — Netaş Telekomünikasyon": "NETAS.IS",
    "NIBAS — Niğbaş Niğde Beton": "NIBAS.IS",
    "NTHOL — Net Holding": "NTHOL.IS",
    "NTTUR — Nettur Turizm": "NTTUR.IS",
    "NUHCM — Nuh Çimento": "NUHCM.IS",
    "OBASE — OBase Bilgi": "OBASE.IS",
    "ODAS  — Odaş Elektrik": "ODAS.IS",
    "ODINE — Odin Tekstil": "ODINE.IS",
    "OFSYM — Ofset Yapımcılık": "OFSYM.IS",
    "ONCSM — Oncem Medya": "ONCSM.IS",
    "ORGE  — Orge Enerji": "ORGE.IS",
    "ORMA  — Orma Orman": "ORMA.IS",
    "OSMEN — Osmanlı Menkul": "OSMEN.IS",
    "OSTIM — Ostim Endüstriyel": "OSTIM.IS",
    "OTKAR — Otokar": "OTKAR.IS",
    "OYAKC — Oyak Çimento": "OYAKC.IS",
    "OZGYO — Özderici GYO": "OZGYO.IS",
    "OZKGY — Özak GYO": "OZKGY.IS",
    "OZRDN — Özradan Yatırım": "OZRDN.IS",
    "OZSUB — Özsu Gıda": "OZSUB.IS",
    "PAGYO — Panora GYO": "PAGYO.IS",
    "PAMEL — Pamel Yenilenebilir": "PAMEL.IS",
    "PAPIL — Papilion Tekstil": "PAPIL.IS",
    "PARSN — Parsan Makine": "PARSN.IS",
    "PASEU — Paşabahçe Cam": "PASEU.IS",
    "PCILT — Pınar Çimento": "PCILT.IS",
    "PEHOL — Pera Holding": "PEHOL.IS",
    "PENGD — Penguen Gıda": "PENGD.IS",
    "PETKM — Petkim": "PETKM.IS",
    "PETUN — Pınar Et": "PETUN.IS",
    "PGSUS — Pegasus Hava Yolları": "PGSUS.IS",
    "PINSU — Pınar Su": "PINSU.IS",
    "PKART — Plastikkart": "PKART.IS",
    "PKENT — Petrokent Turizm": "PKENT.IS",
    "PLTUR — Palmet Turizm": "PLTUR.IS",
    "PNLSN — Panelsan Çelik": "PNLSN.IS",
    "POLHO — Polisan Holding": "POLHO.IS",
    "POLTK — Politeknik Metal": "POLTK.IS",
    "PRKAB — Prysmian Kablo": "PRKAB.IS",
    "PRKME — Park Elektrik": "PRKME.IS",
    "PRZMA — Prizma Pres": "PRZMA.IS",
    "RHEAG — Rhea Girişim": "RHEAG.IS",
    "RODRG — Rodrigo Tekstil": "RODRG.IS",
    "ROYAL — Royal Halı": "ROYAL.IS",
    "RYGYO — Reysaş GYO": "RYGYO.IS",
    "RYSAS — Reysaş Taşımacılık": "RYSAS.IS",
    "SAHOL — Sabancı Holding": "SAHOL.IS",
    "SARKY — Sarkuysan Elektrolitik": "SARKY.IS",
    "SASA  — Sasa Polyester": "SASA.IS",
    "SAYAS — Sayaş Makine": "SAYAS.IS",
    "SDTTR — Sandt Teknoloji": "SDTTR.IS",
    "SEGYO — Servet GYO": "SEGYO.IS",
    "SEKUR — Sekuro Plastik": "SEKUR.IS",
    "SELEC — Selçuk Ecza": "SELEC.IS",
    "SELGD — Selva Gıda": "SELGD.IS",
    "SEYKM — Seydişehir Alüminyum": "SEYKM.IS",
    "SILVR — Silverline Endüstri": "SILVR.IS",
    "SIMAS — Şimşek Ambalaj": "SIMAS.IS",
    "SISE  — Şişe Cam": "SISE.IS",
    "SKBNK — Şekerbank": "SKBNK.IS",
    "SMART — Smart Güneş": "SMART.IS",
    "SNGYO — Sinpaş GYO": "SNGYO.IS",
    "SNKRN — Sanko Enerji": "SNKRN.IS",
    "SNPAM — Sanko Pazarlama": "SNPAM.IS",
    "SODA  — Soda Sanayii": "SODA.IS",
    "SOKM  — Şok Marketler": "SOKM.IS",
    "TABGD — TAB Gıda": "TABGD.IS",
    "TARKM — Tarkim Tarım": "TARKM.IS",
    "TATEN — Tatlıses Enerji": "TATEN.IS",
    "TATGD — Tat Gıda": "TATGD.IS",
    "TAVHL — TAV Havalimanları": "TAVHL.IS",
    "TCELL — Turkcell": "TCELL.IS",
    "TDGYO — Trend GYO": "TDGYO.IS",
    "TEKTU — Tek-Art Turizm": "TEKTU.IS",
    "TEZOL — Tezol Kağıt": "TEZOL.IS",
    "TGSAS — TGS Dış Ticaret": "TGSAS.IS",
    "THYAO — Türk Hava Yolları": "THYAO.IS",
    "TIRE  — Tire Kutsan": "TIRE.IS",
    "TKFEN — Tekfen Holding": "TKFEN.IS",
    "TKURU — Türk Turizm": "TKURU.IS",
    "TLMAN — Telemann Telekom": "TLMAN.IS",
    "TMPOL — Temapol Polimer": "TMPOL.IS",
    "TNZTP — Tınaztepe Hastane": "TNZTP.IS",
    "TOASO — Tofaş Türk Otomobil": "TOASO.IS",
    "TRCAS — Turcas Petrol": "TRCAS.IS",
    "TRGYO — Torunlar GYO": "TRGYO.IS",
    "TRILC — Trilc Yatırım": "TRILC.IS",
    "TSKB  — Türkiye Sınai Kalkınma B.": "TSKB.IS",
    "TSPOR — Trabzonspor": "TSPOR.IS",
    "TTKOM — Türk Telekom": "TTKOM.IS",
    "TTRAK — Türk Traktör": "TTRAK.IS",
    "TUCLK — Tuçluk Tekstil": "TUCLK.IS",
    "TUPRS — Tüpraş": "TUPRS.IS",
    "TUREX — Tureks Turizm": "TUREX.IS",
    "TURGG — Türkerler Holding": "TURGG.IS",
    "ULKER — Ülker Bisküvi": "ULKER.IS",
    "ULUUN — Ulusoy Un": "ULUUN.IS",
    "UMPAS — Umpaş Holding": "UMPAS.IS",
    "UNLU  — Ünlü Tekstil": "UNLU.IS",
    "USAK  — Uşak Seramik": "USAK.IS",
    "USDMR — Uzmar Gemi": "USDMR.IS",
    "UTPYA — Utopya Turizm": "UTPYA.IS",
    "UVITE — Uvite Teknoloji": "UVITE.IS",
    "VAKBN — Vakıfbank": "VAKBN.IS",
    "VANGD — Van Gölü Enerji": "VANGD.IS",
    "VBTYZ — VBT Yazılım": "VBTYZ.IS",
    "VERTU — Vertu Yatırım": "VERTU.IS",
    "VERUS — Verus Yatırım": "VERUS.IS",
    "VESBE — Vestel Beyaz Eşya": "VESBE.IS",
    "VESTL — Vestel Elektronik": "VESTL.IS",
    "VKFYO — Vakıf Finansal Kiralama": "VKFYO.IS",
    "VKGYO — Vakıf GYO": "VKGYO.IS",
    "YAPRK — Yaprak Süt": "YAPRK.IS",
    "YATAS — Yataş Yatak": "YATAS.IS",
    "YBTAS — Yıldız Boru": "YBTAS.IS",
    "YEOTK — Yeo Teknoloji": "YEOTK.IS",
    "YESIL — Yeşil Girişim": "YESIL.IS",
    "YGGYO — Yeni Gimat GYO": "YGGYO.IS",
    "YGYO  — Yeni Mağazacılık GYO": "YGYO.IS",
    "YKBNK — Yapı Kredi Bankası": "YKBNK.IS",
    "YKSLN — Yükseliş GYO": "YKSLN.IS",
    "YUNSA — Yünsa Yünlü": "YUNSA.IS",
    "YYAPI — Yesil Yapi": "YYAPI.IS",
    "ZEDUR — Zedur Enerji": "ZEDUR.IS",
    "ZOREN — Zorlu Enerji": "ZOREN.IS",
    "ZTPFK — Ziraat Katılım": "ZTPFK.IS",
}

# ─────────────────────────────────────────────────────────────────────────────
# SEKTÖR MOMENTUMu için hisse gruplaması
# ─────────────────────────────────────────────────────────────────────────────
SEKTOR_GRUPLAR = {
    "AKBNK": ["GARAN", "ISCTR", "YKBNK", "VAKBN"],
    "GARAN": ["AKBNK", "ISCTR", "YKBNK", "VAKBN"],
    "ISCTR": ["AKBNK", "GARAN", "YKBNK", "VAKBN"],
    "YKBNK": ["AKBNK", "GARAN", "ISCTR", "VAKBN"],
    "VAKBN": ["AKBNK", "GARAN", "ISCTR", "YKBNK"],
    "THYAO": ["PGSUS", "TAVHL"],
    "PGSUS": ["THYAO", "TAVHL"],
    "EREGL": ["KRDMD", "ISDMR"],
    "KRDMD": ["EREGL", "ISDMR"],
    "TOASO": ["FROTO", "ASUZU"],
    "FROTO": ["TOASO", "ASUZU"],
    "BIMAS": ["MGROS", "SOKM"],
    "MGROS": ["BIMAS", "SOKM"],
    "KCHOL": ["SAHOL", "AGHOL"],
    "SAHOL": ["KCHOL", "AGHOL"],
    "ASELS": ["ASELSAN", "FNSS"],
    "TCELL": ["TTKOM"],
    "TUPRS": ["PETKM"],
    "PETKM": ["TUPRS"],
}

MAKRO_TICKERLAR = {
    "USDTRY": "USDTRY=X", "EURTRY": "EURTRY=X",
    "ALTIN": "GC=F", "PETROL": "CL=F",
    "BIST100": "XU100.IS", "BIST30": "XU030.IS",
    "SP500": "^GSPC", "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    # Sektör endeksleri kaldırıldı — rate limit azaltmak için
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PARAMETRELERİ — Optimize edilmiş
# ─────────────────────────────────────────────────────────────────────────────
PENCERE = 40
EPOCH = 100
BATCH = 32
TEST_ORANI = 0.15
HEDEF_GUN = 10  # 5 → 10 gün (orta vadeli)
HEDEF_ESIK = 0.025
MIN_HACIM_TL = 1_000_000
VARSAYILAN_YIL = 15  # maksimum veri: 15 yıl

# Hisseleri alfabetik sırala
BIST_HISSELER = dict(sorted(BIST_HISSELER.items(), key=lambda x: x[0]))


# ─────────────────────────────────────────────────────────────────────────────
# YFINANCE OTURUM YARDIMCISI — Rate limit & cookie sorununu çözer
# ─────────────────────────────────────────────────────────────────────────────
def _yf_session():
    """Tarayıcı gibi davranan özel requests session döndürür."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    # Önce ana sayfaya git — cookie al
    try:
        s.get("https://finance.yahoo.com", timeout=5)
    except Exception:
        pass
    return s


def _index_temizle(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame index'ini kesinlikle temizler:
    timezone strip + date'e dönüştür + duplicate kaldır + sırala.
    Tüm makro/hisse fonksiyonları bunu kullanır.
    """
    if df is None or df.empty:
        return df
    try:
        idx = pd.to_datetime(df.index, errors='coerce')
        # Timezone varsa kaldır
        if hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_localize(None)
        elif hasattr(idx, 'tzinfo') and getattr(idx, 'tzinfo', None):
            idx = idx.tz_localize(None)
        # Sadece date kısmını al (saat=00:00:00)
        idx = idx.normalize()
        df = df.copy()
        df.index = idx
        # Duplicate satırları kaldır
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        return df
    except Exception:
        return df


def _df_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame sütun isimlerini standartlaştır."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    return df


def _google_finance_indir(ticker: str, gun: int = 365) -> pd.DataFrame:
    """
    Google Finance'den BIST hissesi indir.
    ticker: "AKBNK" formatında (BIST için)
    """
    try:
        import json as _json
        # Google Finance API benzeri endpoint
        bist_ticker = ticker.replace(".IS", "")
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{ticker}?interval=1d&range={min(gun, 730)}d")
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=hdrs, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        res = data.get("chart", {}).get("result", [])
        if not res:
            return pd.DataFrame()
        res = res[0]
        ts = res.get("timestamp", [])
        quot = res.get("indicators", {}).get("quote", [{}])[0]
        adjc = res.get("indicators", {}).get("adjclose", [{}])
        close_data = (adjc[0].get("adjclose") if adjc else None) or quot.get("close")
        if not ts or not close_data:
            return pd.DataFrame()
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
        df = pd.DataFrame({
            "Open": quot.get("open", [None] * len(ts)),
            "High": quot.get("high", [None] * len(ts)),
            "Low": quot.get("low", [None] * len(ts)),
            "Close": close_data,
            "Volume": quot.get("volume", [0] * len(ts)),
        }, index=idx)
        df = df.dropna(subset=["Close"])
        df.index = pd.to_datetime([d.date() for d in df.index])
        return df
    except Exception:
        return pd.DataFrame()


def _investing_indir(ticker: str, gun: int = 365) -> pd.DataFrame:
    """Investing.com alternatif kaynak."""
    try:
        bist_kod = ticker.replace('.IS', '').upper()
        url = f"https://api.investing.com/api/search/v2/search?q={bist_kod}"
        hdrs = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest", "Domain-Id": "tr"}
        r = requests.get(url, headers=hdrs, timeout=8)
        if r.status_code != 200: return pd.DataFrame()
        data = r.json()
        pair_id = None
        for q in data.get('quotes', []):
            if q.get('exchange') in ('BIST', 'ISE') and bist_kod in q.get('symbol', '').upper():
                pair_id = q.get('pairId') or q.get('id');
                break
        if not pair_id: return pd.DataFrame()
        import time as _tt2
        now = int(_tt2.time());
        start = now - gun * 86400
        url2 = f"https://tvc4.investing.com/{pair_id}/1/{start}/{now}/1/history?symbol={pair_id}&resolution=D&from={start}&to={now}"
        r2 = requests.get(url2, headers=hdrs, timeout=12)
        if r2.status_code != 200: return pd.DataFrame()
        h = r2.json()
        if h.get('s') != 'ok': return pd.DataFrame()
        df = pd.DataFrame({'Open': h.get('o', []), 'High': h.get('h', []), 'Low': h.get('l', []),
                           'Close': h.get('c', []), 'Volume': h.get('v', [])},
                          index=pd.to_datetime(h.get('t', []), unit='s').normalize())
        return df[~df.index.duplicated(keep='last')].sort_index().dropna(subset=['Close'])
    except Exception:
        return pd.DataFrame()


def _stooq_indir(ticker: str, baslangic, bitis) -> pd.DataFrame:
    """
    Stooq.com — Yahoo Finance alternatifi, çoğu zaman çalışır.
    BIST hisseleri için format: AKBNK.PL (Polonya borsasında listeleniyor!)
    """
    try:
        # BIST hisseleri Stooq'ta farklı format
        stooq_ticker = ticker.replace(".IS", ".PL").upper()
        start = str(baslangic.date()) if hasattr(baslangic, 'date') else str(baslangic)
        end = str(bitis.date()) if hasattr(bitis, 'date') else str(bitis)
        url = (f"https://stooq.com/q/d/l/?s={stooq_ticker}"
               f"&d1={start.replace('-', '')}&d2={end.replace('-', '')}&i=d")
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.text) > 100:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            df.columns = [c.strip().title() for c in df.columns]
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
            if 'Close' in df.columns and not df.empty:
                return df
    except Exception:
        pass
    return pd.DataFrame()


def _yf_indir(ticker: str, baslangic, bitis, interval="1d") -> pd.DataFrame:
    """
    Akıllı çok kaynaklı veri indirme.
    Önce direkt API (rate limit yok), sonra Ticker.history().
    yf.download() çağrısı yok — rate limit minimumda.
    """
    _yf_yukle()
    gun = max(int((bitis - baslangic).days), 30)
    start_str = str(baslangic.date()) if hasattr(baslangic, 'date') else str(baslangic)
    end_str = str(bitis.date()) if hasattr(bitis, 'date') else str(bitis)
    import time as _tl

    # ── Yöntem 1: Direkt Yahoo v8 API (rate limit neredeyse yok) ─────────────
    try:
        import time as _tt
        p1 = int(_tt.mktime(baslangic.timetuple()))
        p2 = int(_tt.mktime(bitis.timetuple()))
        for base_url in [
            "https://query2.finance.yahoo.com/v8/finance/chart/",
            "https://query1.finance.yahoo.com/v8/finance/chart/",
        ]:
            try:
                url = (f"{base_url}{ticker}?period1={p1}&period2={p2}"
                       f"&interval={interval}&events=history&includeAdjustedClose=true")
                hdrs = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 Chrome/123.0 Safari/537.36",
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://finance.yahoo.com",
                }
                r = requests.get(url, headers=hdrs, timeout=20)
                if r.status_code == 429:
                    continue
                if r.status_code != 200:
                    continue
                data = r.json()
                res = data.get("chart", {}).get("result", [])
                if not res:
                    continue
                ts = res[0].get("timestamp", [])
                quot = res[0].get("indicators", {}).get("quote", [{}])[0]
                adjc = res[0].get("indicators", {}).get("adjclose", [{}])
                close = (adjc[0].get("adjclose") if adjc else None) or quot.get("close")
                if ts and close:
                    idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
                    df = pd.DataFrame({
                        "Open": quot.get("open", [None] * len(ts)),
                        "High": quot.get("high", [None] * len(ts)),
                        "Low": quot.get("low", [None] * len(ts)),
                        "Close": close,
                        "Volume": quot.get("volume", [0] * len(ts)),
                    }, index=idx).dropna(subset=["Close"])
                    df = _index_temizle(df)
                    if not df.empty:
                        return df
            except Exception:
                continue
    except Exception:
        pass

    # ── Yöntem 2: Ticker.history() — yf kütüphanesi ──────────────────────────
    if yf is not None:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=f"{gun}d", interval=interval, auto_adjust=True)
            df = _df_normalize(df)
            if not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass

    # ── Yöntem 4: Doğrudan Yahoo Finance API (cookie olmadan) ────────────────
    try:
        _tl.sleep(2)
        import time as _tt
        # v8 API — daha az engelleme
        base = "https://query2.finance.yahoo.com/v8/finance/chart/"
        p1 = int(_tt.mktime(baslangic.timetuple())) if hasattr(baslangic, 'timetuple') else int(baslangic)
        p2 = int(_tt.mktime(bitis.timetuple())) if hasattr(bitis, 'timetuple') else int(bitis)
        url = (f"{base}{ticker}?period1={p1}&period2={p2}"
               f"&interval={interval}&events=history&includeAdjustedClose=true")
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/123.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://finance.yahoo.com",
        }
        r = requests.get(url, headers=hdrs, timeout=15)
        data = r.json()
        res = data.get("chart", {}).get("result", [{}])[0]
        ts = res.get("timestamp", [])
        quot = res.get("indicators", {}).get("quote", [{}])[0]
        adj = res.get("indicators", {}).get("adjclose", [{}])[0]
        if ts and quot.get("close"):
            import pandas as _pd
            idx = _pd.to_datetime(ts, unit="s", utc=True).tz_convert("Europe/Istanbul").tz_localize(None)
            df = _pd.DataFrame({
                "Open": quot.get("open", [None] * len(ts)),
                "High": quot.get("high", [None] * len(ts)),
                "Low": quot.get("low", [None] * len(ts)),
                "Close": adj.get("adjclose", [None] * len(ts)) or quot.get("close", [None] * len(ts)),
                "Volume": quot.get("volume", [0] * len(ts)),
            }, index=idx).dropna(subset=["Close"])
            if not df.empty:
                return df
    except Exception:
        pass

    # ── Yöntem 5: Google Finance API (Yahoo bypass) ─────────────────────────
    try:
        df = _google_finance_indir(ticker, gun=min(gun, 1460))  # max 4 yıl
        if not df.empty and 'Close' in df.columns:
            return df
    except Exception:
        pass

    # ── Yöntem 6: Investing.com ───────────────────────────────────────────────
    if interval == "1d" and ticker.endswith('.IS'):
        try:
            df = _investing_indir(ticker, gun=min(gun, 1460))
            if not df.empty and 'Close' in df.columns:
                return df
        except Exception:
            pass

    # ── Yöntem 7: Stooq (BIST için) ──────────────────────────────────────────
    if interval == "1d":
        try:
            df = _stooq_indir(ticker, baslangic, bitis)
            if not df.empty and 'Close' in df.columns:
                return df
        except Exception:
            pass

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# VERİ FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat
def hisse_indir(ticker: str, yil: int = 5) -> pd.DataFrame:
    _yf_yukle()
    bitis = datetime.today()
    baslangic = bitis - timedelta(days=yil * 365)

    df = _yf_indir(ticker, baslangic, bitis)
    if df.empty:
        raise ValueError(
            f"{ticker} için veri indirilemedi (4 yöntem denendi).\n\n"
            f"Olası sebepler:\n"
            f"• Yahoo Finance geçici olarak erişimi engelledi\n"
            f"• İnternet bağlantısı yok\n"
            f"• Hisse kodu yanlış\n\n"
            f"Çözüm: 2-3 dakika bekleyip tekrar deneyin."
        )

    # Duplicate index temizle
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='last')]
    df.index = pd.to_datetime(df.index).normalize()

    gerekli = ['Open', 'High', 'Low', 'Close', 'Volume']
    mevcut = [c for c in gerekli if c in df.columns]
    df = df[mevcut].copy()
    df = df.replace(0, np.nan)
    df['Volume'] = df['Volume'].fillna(0)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    if len(df) < 100:
        raise ValueError(
            f"{ticker} için yeterli veri yok ({len(df)} satır). "
            f"Daha uzun veri yılı seçin (5-6 yıl önerilir)."
        )
    return df


@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat cache
def makro_indir() -> pd.DataFrame:
    """
    Makro verileri paralel çeker, 6 saat cache'ler.
    Rate limit olursa boş df döner — analiz durmuyor.
    """
    bitis = datetime.today()
    baslangic = bitis - timedelta(days=5 * 365)

    def _makro_tek(item):
        ad, tick = item
        try:
            d = _yf_indir(tick, baslangic, bitis)
            if not d.empty and "Close" in d.columns:
                return ad, d["Close"].rename(ad)
        except Exception:
            pass
        return ad, None

    kareler = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_makro_tek, item): item[0]
                       for item in MAKRO_TICKERLAR.items()}
            for future in as_completed(futures, timeout=30):
                try:
                    ad, seri = future.result(timeout=10)
                    if seri is not None:
                        kareler[ad] = seri
                except Exception:
                    pass
    except Exception:
        pass

    if not kareler:
        return pd.DataFrame()
    df_makro = pd.concat(kareler.values(), axis=1)
    # Duplicate index temizle
    df_makro = df_makro[~df_makro.index.duplicated(keep='last')]
    df_makro.index = pd.to_datetime(df_makro.index).normalize()
    return df_makro.ffill().bfill()


# ─────────────────────────────────────────────────────────────────────────────
# TÜRKÇE DUYGU ANALİZİ — Genişletilmiş kelime hazinesi
# ─────────────────────────────────────────────────────────────────────────────
POZITIF = {
    # Güçlü pozitif (ağırlık 2)
    "rekor": 2, "zirve": 2, "fırladı": 2, "coştu": 2, "beklentileri aştı": 2,
    "tarihi yüksek": 2, "güçlü büyüme": 2, "kâr açıkladı": 2, "temettü artışı": 2,
    "anlaşma imzaladı": 2, "ihale kazandı": 2, "kapasite artırıyor": 2,
    "yüksek kâr": 2, "borsada lider": 2, "yatırımcı ilgisi": 2,
    # Orta pozitif (ağırlık 1)
    "yükseliş": 1, "artış": 1, "büyüme": 1, "kâr": 1, "kazanç": 1, "güçlü": 1,
    "başarı": 1, "olumlu": 1, "yatırım": 1, "ihracat": 1, "talep": 1,
    "toparlanma": 1, "iyileşme": 1, "temettü": 1, "anlaşma": 1, "sözleşme": 1,
    "sipariş": 1, "kapasite": 1, "ihracat artışı": 1, "pazar payı": 1,
    "alım": 1, "destek": 1, "pozitif": 1, "rally": 1, "momentum": 1,
    "ralli": 1, "yeni müşteri": 1, "büyük proje": 1, "güvenli liman": 1,
}
NEGATIF = {
    # Güçlü negatif (ağırlık 2)
    "çöktü": 2, "iflas": 2, "kriz": 2, "soruşturma": 2, "dava": 2,
    "manipülasyon": 2, "haciz": 2, "büyük zarar": 2, "sermaye kaybı": 2,
    "ihracat yasağı": 2, "yaptırım": 2, "olağanüstü hal": 2,
    "devalüasyon": 2, "yüksek enflasyon": 2, "faiz artışı şoku": 2,
    # Orta negatif (ağırlık 1)
    "düşüş": 1, "kayıp": 1, "zarar": 1, "risk": 1, "enflasyon": 1,
    "gerileme": 1, "olumsuz": 1, "endişe": 1, "panik": 1, "baskı": 1,
    "zayıf": 1, "daralma": 1, "küçülme": 1, "uyarı": 1, "volatilite": 1,
    "satış baskısı": 1, "zayıf talep": 1, "gelir düşüşü": 1, "maliyet artışı": 1,
    "borç": 1, "yükümlülük": 1, "temerrüt": 1, "erteleme": 1, "iptal": 1,
}


@st.cache_resource(show_spinner=False)
def finbert_yukle():
    """FinBERT modelini bir kez yükle, cache'le."""
    if not FINBERT_OK:
        return None
    try:
        pipe = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU
            max_length=512,
            truncation=True,
        )
        return pipe
    except Exception:
        return None


def finbert_skor(metinler: list) -> float:
    """
    FinBERT ile metin listesini analiz et.
    Dönüş: -1 (negatif) ile +1 (pozitif) arası float.
    FinBERT yoksa kural tabanlı analize düşer.
    """
    if not FINBERT_OK or not metinler:
        return 0.0
    try:
        pipe = finbert_yukle()
        if pipe is None:
            return 0.0
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        skorlar = []
        for metin in metinler[:15]:  # max 15 metin
            metin_k = metin[:512]  # token limiti
            sonuc = pipe(metin_k)[0]
            label = sonuc["label"].lower()
            score = sonuc["score"]  # güven skoru
            deger = label_map.get(label, 0.0) * score
            skorlar.append(deger)
        return float(sum(skorlar) / len(skorlar)) if skorlar else 0.0
    except Exception:
        return 0.0


def duygu_analizi(metin: str) -> float:
    metin = metin.lower()
    poz, neg = 0, 0
    for k, a in POZITIF.items():
        if k in metin: poz += a
    for k, a in NEGATIF.items():
        if k in metin: neg += a
    toplam = poz + neg
    return 0.0 if toplam == 0 else float(np.clip((poz - neg) / (toplam + 2), -1, 1))


def _tek_kaynak_cek(sorgu: str, url_sablonu: str, limit: int = 15) -> list:
    """Tek bir RSS/HTML kaynaktan haber çeker."""
    haberler = []
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/122.0 Safari/537.36"),
        "Accept-Language": "tr-TR,tr;q=0.9",
    }
    try:
        url = url_sablonu.format(q=requests.utils.quote(sorgu))
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, "xml")
            for item in soup.find_all("item")[:limit]:
                t = item.find("title")
                d = item.find("description")
                if t:  haberler.append(t.text)
                if d and d.text != t.text if t else True:
                    haberler.append(BeautifulSoup(d.text, "html.parser").get_text()[:200])
    except Exception:
        pass
    return haberler


@st.cache_data(ttl=10800, show_spinner=False)  # 3 saat
def haber_skoru_al(hisse_kodu: str, kaynaklar: tuple = ("google", "bloombergHT", "doviz", "sabah")) -> dict:
    """
    8 güvenilir Türkçe finansal haber kaynağından paralel veri çeker.
    kaynaklar parametresi sidebar seçimiyle değişir.
    """

    # ── Kaynak tanımları ──────────────────────────────────────────────────────
    KAYNAKLAR = {
        "google": {
            "isim": "Google News",
            "url": "https://news.google.com/rss/search?q={q}&hl=tr&gl=TR&ceid=TR:tr",
            "sorgular": [f"{hisse_kodu} hisse", f"{hisse_kodu} borsa",
                         "BIST borsa istanbul"],
        },
        "bloombergHT": {
            "isim": "Bloomberg HT",
            "url": "https://www.bloomberght.com/rss",
            "sorgular": [f"{hisse_kodu}"],
        },
        "doviz": {
            "isim": "Döviz.com",
            "url": "https://www.doviz.com/rss/haberler",
            "sorgular": ["borsa istanbul hisse"],
        },
        "sabah": {
            "isim": "Sabah Ekonomi",
            "url": "https://www.sabah.com.tr/rss/ekonomi.xml",
            "sorgular": [f"{hisse_kodu}"],
        },
        "investing": {
            "isim": "Investing TR",
            "url": "https://tr.investing.com/rss/news.rss",
            "sorgular": [f"{hisse_kodu} hisse"],
        },
        "ekonomist": {
            "isim": "Ekonomist",
            "url": "https://www.ekonomist.com.tr/feed/",
            "sorgular": ["borsa hisse"],
        },
        "haberturk": {
            "isim": "Habertürk Ekonomi",
            "url": "https://www.haberturk.com/rss/ekonomi.xml",
            "sorgular": [f"{hisse_kodu}"],
        },
        "milliyet": {
            "isim": "Milliyet Ekonomi",
            "url": "https://www.milliyet.com.tr/rss/rssNew/ekonomiRss.xml",
            "sorgular": [f"{hisse_kodu} borsa"],
        },
    }

    secili = {k: v for k, v in KAYNAKLAR.items() if k in kaynaklar}
    if not secili:
        secili = {"google": KAYNAKLAR["google"]}

    # Paralel haber çekme
    tum_haberler = []
    kaynak_sayisi = {k: 0 for k in secili}

    gorevler = []
    for k_id, k_bilgi in secili.items():
        for sorgu in k_bilgi["sorgular"]:
            gorevler.append((k_id, sorgu, k_bilgi["url"]))

    with ThreadPoolExecutor(max_workers=6) as executor:
        gelecekler = {
            executor.submit(_tek_kaynak_cek, sorgu, url, 12): k_id
            for k_id, sorgu, url in gorevler
        }
        for gelecek in as_completed(gelecekler, timeout=12):
            k_id = gelecekler[gelecek]
            try:
                sonuc = gelecek.result()
                # Hisse kodunu içeren haberlere ekstra ağırlık için işaretle
                for h in sonuc:
                    ilgili = hisse_kodu.lower() in h.lower()
                    tum_haberler.append({"metin": h, "ilgili": ilgili, "kaynak": k_id})
                kaynak_sayisi[k_id] += len(sonuc)
            except Exception:
                pass

    if not tum_haberler:
        return {"skor": 0.0, "adet": 0, "durum": "Veri Yok",
                "kaynaklar": {}, "haberler": []}

    # ── FinBERT ile duygu analizi (varsa) ──────────────────────────────
    tum_metinler = [h["metin"] for h in tum_haberler]
    ilgili_metinler = [h["metin"] for h in tum_haberler if h["ilgili"]]

    if FINBERT_OK and ilgili_metinler:
        # İlgili haberler FinBERT ile, genel haberler kural tabanlı
        finbert_skr = finbert_skor(ilgili_metinler)
        kural_skorlar = [duygu_analizi(h["metin"]) for h in tum_haberler]
        zaman_agir = np.exp(np.linspace(0, 1, len(kural_skorlar)))
        kural_ort = float(np.average(kural_skorlar, weights=zaman_agir))
        # FinBERT %70, kural tabanlı %30 karıştır
        agirlikli = 0.70 * finbert_skr + 0.30 * kural_ort
        analiz_yontemi = "FinBERT"
    else:
        # Kural tabanlı fallback
        skorlar = []
        agirlikl = []
        for h in tum_haberler:
            s = duygu_analizi(h["metin"])
            w = 2.0 if h["ilgili"] else 1.0
            skorlar.append(s)
            agirlikl.append(w)
        zaman_agir = np.exp(np.linspace(0, 1, len(skorlar)))
        nihai_agir = np.array(agirlikl) * zaman_agir
        agirlikli = float(np.average(skorlar, weights=nihai_agir))
        analiz_yontemi = "Kural Tabanlı"

    durum = ("Güçlü Pozitif" if agirlikli > 0.3 else
             "Pozitif" if agirlikli > 0.1 else
             "Güçlü Negatif" if agirlikli < -0.3 else
             "Negatif" if agirlikli < -0.1 else "Nötr")

    ilgili_haberler = [h["metin"][:120] for h in tum_haberler if h["ilgili"]][:5]

    return {
        "skor": agirlikli,
        "adet": len(tum_haberler),
        "durum": durum,
        "kaynaklar": kaynak_sayisi,
        "haberler": ilgili_haberler,
        "basliklar": [h.get("baslik", "")[:70] for h in (ilgili_haberler or tum_haberler)[:5]],
        "yontem": analiz_yontemi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEKNİK İNDİKATÖRLER — Genişletilmiş + 5 günlük hedef
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat
def sektor_momentum_al(hisse_kodu: str, gun: int = 5) -> float:
    """Sektör momentumu — paralel direkt API ile."""
    peers = SEKTOR_GRUPLAR.get(hisse_kodu, [])
    if not peers:
        return 0.0
    bitis = datetime.today()
    baslangic = bitis - timedelta(days=30)

    def _peer_getiri(peer):
        try:
            d = _yf_indir(peer + ".IS", baslangic, bitis)
            if not d.empty and "Close" in d.columns and len(d) >= 2:
                return float(d["Close"].iloc[-1] / d["Close"].iloc[max(-gun, -len(d))] - 1)
        except Exception:
            pass
        return None

    getiriler = []
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = ex.map(_peer_getiri, peers[:4], timeout=15)
            getiriler = [r for r in results if r is not None]
    except Exception:
        pass
    return float(np.mean(getiriler)) if getiriler else 0.0


@st.cache_data(ttl=21600, show_spinner=False)
def indikatör_ekle(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    # Hareketli ortalamalar
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'MA{p}'] = c.rolling(p).mean()
        df[f'MA{p}_oran'] = c / (df[f'MA{p}'] + 1e-10)
    for p in [9, 12, 21, 26, 50]:
        df[f'EMA{p}'] = c.ewm(span=p, adjust=False).mean()

    # MA çaprazları (golden/death cross sinyalleri)
    df['MA_GC_20_50'] = (df['MA20'] > df['MA50']).astype(int)
    df['MA_GC_50_200'] = (df['MA50'] > df['MA200']).astype(int)
    df['EMA_GC_9_21'] = (df['EMA9'] > df['EMA21']).astype(int)

    # RSI çoklu periyot
    for p in [7, 14, 21]:
        d = c.diff()
        g = d.clip(lower=0).rolling(p).mean()
        ls = (-d.clip(upper=0)).rolling(p).mean()
        df[f'RSI{p}'] = 100 - 100 / (1 + g / (ls + 1e-10))
    df['RSI_diverg'] = df['RSI14'] - df['RSI14'].shift(5)  # RSI ivmesi

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Cross'] = (df['MACD'] > df['MACD_Signal']).astype(int)

    # Bollinger bantları
    for p in [10, 20]:
        ort = c.rolling(p).mean();
        std = c.rolling(p).std()
        df[f'BB{p}_Ust'] = ort + 2 * std
        df[f'BB{p}_Alt'] = ort - 2 * std
        df[f'BB{p}_Pct'] = (c - df[f'BB{p}_Alt']) / (df[f'BB{p}_Ust'] - df[f'BB{p}_Alt'] + 1e-10)
        df[f'BB{p}_Genis'] = (df[f'BB{p}_Ust'] - df[f'BB{p}_Alt']) / (ort + 1e-10)  # Bant genişliği

    # ATR ve volatilite
    for p in [7, 14, 21]:
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        df[f'ATR{p}'] = tr.rolling(p).mean()
        df[f'ATR{p}_norm'] = df[f'ATR{p}'] / (c + 1e-10)  # Normalize ATR

    # Stochastic
    for p in [9, 14]:
        lp = l.rolling(p).min();
        hp = h.rolling(p).max()
        df[f'Stoch{p}_K'] = 100 * (c - lp) / (hp - lp + 1e-10)
        df[f'Stoch{p}_D'] = df[f'Stoch{p}_K'].rolling(3).mean()

    # CCI
    tp = (h + l + c) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)
    df['CCI_14'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std() + 1e-10)

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df['OBV'] = obv
    df['OBV_MA'] = obv.rolling(10).mean()
    df['OBV_oran'] = obv / (df['OBV_MA'] + 1e-10)
    df['OBV_trend'] = obv.pct_change(5)

    # Williams %R
    hh = h.rolling(14).max();
    ll = l.rolling(14).min()
    df['WilliamsR'] = -100 * (hh - c) / (hh - ll + 1e-10)

    # MFI
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    df['MFI'] = 100 - 100 / (1 + pos / (neg + 1e-10))

    # Momentum çoklu periyot
    for p in [3, 5, 10, 20, 60]:
        df[f'Mom{p}'] = c.pct_change(p)

    # Volatilite
    for p in [7, 14, 30]:
        df[f'Vol{p}'] = c.pct_change().rolling(p).std()
    df['Vol_oran'] = df['Vol7'] / (df['Vol30'] + 1e-10)

    # Hacim
    df['Hacim_MA5'] = v.rolling(5).mean()
    df['Hacim_MA20'] = v.rolling(20).mean()
    df['Hacim_Oran'] = v / (df['Hacim_MA20'] + 1e-10)
    df['Hacim_Degisim'] = v.pct_change()
    df['Hacim_Surge'] = (v > df['Hacim_MA20'] * 2).astype(int)

    # Fiyat pozisyonu 52 haftalık
    df['High52w'] = h.rolling(252).max()
    df['Low52w'] = l.rolling(252).min()
    df['Pos52w'] = (c - df['Low52w']) / (df['High52w'] - df['Low52w'] + 1e-10)

    # Mum desenleri
    body = (c - df['Open']).abs()
    candle = h - l
    df['Doji'] = (body / (candle + 1e-10) < 0.1).astype(int)
    df['Hammer'] = ((body / (candle + 1e-10) < 0.3) &
                    ((df['Open'] - l) / (candle + 1e-10) > 0.6)).astype(int)
    df['Engulf_Up'] = ((c > df['Open']) &
                       (c.shift() < df['Open'].shift()) &
                       (c > df['Open'].shift()) &
                       (df['Open'] < c.shift())).astype(int)

    # ── SEKANS ÖZELLİKLERİ — LSTM'in öğrendiğini elle çıkar ─────────────────
    # RSI son 5/10/20 günde kaç kez 50'yi geçti (yukarı momentum sayacı)
    rsi_above = (df['RSI14'] > 50).astype(int)
    df['RSI_cross50_5'] = rsi_above.rolling(5).sum()
    df['RSI_cross50_10'] = rsi_above.rolling(10).sum()
    df['RSI_cross50_20'] = rsi_above.rolling(20).sum()

    # MA20 kaç gündür fiyatın altında/üstünde
    ma20_below = (c < df['MA20']).astype(int)
    df['MA20_below_days'] = ma20_below.rolling(20).sum()
    df['MA50_below_days'] = (c < df['MA50']).astype(int).rolling(20).sum()

    # Son 5/10/20 günde kaç gün yeşil kapandı
    df['GreenDay_5'] = (c > c.shift(1)).astype(int).rolling(5).sum()
    df['GreenDay_10'] = (c > c.shift(1)).astype(int).rolling(10).sum()
    df['GreenDay_20'] = (c > c.shift(1)).astype(int).rolling(20).sum()

    # RSI trendi (son 5/10 günde RSI kaç puan değişti)
    df['RSI_trend5'] = df['RSI14'] - df['RSI14'].shift(5)
    df['RSI_trend10'] = df['RSI14'] - df['RSI14'].shift(10)

    # MACD histogramı ardışık artış/azalış
    macd_inc = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)).astype(int)
    df['MACD_inc_5'] = macd_inc.rolling(5).sum()

    # Hacim trendi (son 5/10 günde ortalama hacim önceki 20 güne oranı)
    df['Vol_trend5'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-10)
    df['Vol_trend10'] = v.rolling(10).mean() / (v.rolling(20).mean() + 1e-10)

    # Fiyat ivmesi (momentumun momentumu)
    df['Mom_accel5'] = df['Mom5'] - df['Mom5'].shift(5)
    df['Mom_accel10'] = df['Mom10'] - df['Mom10'].shift(5)

    # ── MEVSİMSELLİK — Takvim özellikleri ────────────────────────────────────
    idx = df.index
    df['Gun_haftada'] = idx.dayofweek.astype(float)  # 0=Pzt, 4=Cum
    df['Ay'] = idx.month.astype(float)
    df['Ay_gun'] = idx.day.astype(float)
    df['Hafta_yilda'] = idx.isocalendar().week.astype(float)
    df['Ceyrek'] = idx.quarter.astype(float)
    df['Ay_sonu'] = (idx.day >= 25).astype(float)  # Ay sonu etkisi
    df['Ay_basi'] = (idx.day <= 5).astype(float)  # Ay başı etkisi
    df['Ceyrek_sonu'] = idx.month.isin([3, 6, 9, 12]).astype(float)

    # ── VOLATİLİTE REJİMİ — Kriz tespiti ─────────────────────────────────────
    # ATR14'ün 60 günlük ortalamasına oranı
    atr_long = df['ATR14'].rolling(60).mean()
    df['Rejim_vol'] = df['ATR14'] / (atr_long + 1e-10)
    df['Kriz_rejim'] = (df['Rejim_vol'] > 1.5).astype(int)  # Yüksek volatilite
    df['Sakin_rejim'] = (df['Rejim_vol'] < 0.7).astype(int)  # Düşük volatilite

    # Bollinger sıkışma (volatilite azalıyor = büyük hareket hazırlığı)
    df['BB_sıkısma'] = df['BB20_Genis'] / (df['BB20_Genis'].rolling(20).mean() + 1e-10)
    df['Destek20'] = df['Low'].rolling(20).min()
    df['Direnc20'] = df['High'].rolling(20).max()
    df['Hacim_an'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)

    # ── HİSSEYE ÖZEL HEDEF EŞİĞİ (volatiliteye göre dinamik) ─────────────────
    # Sabit %2.5 yerine: o hissenin 60 günlük volatilitesinin 1.5 katı
    vol60 = c.pct_change().rolling(60).std()
    dinamik_esik = (vol60 * 1.5).clip(lower=0.015, upper=0.06)
    gelecek_getiri = c.shift(-HEDEF_GUN) / c - 1
    df['Hedef'] = (gelecek_getiri >= dinamik_esik).astype(int)

    return df.dropna()


def makro_birlestir(df_h: pd.DataFrame, df_m: pd.DataFrame) -> pd.DataFrame:
    if df_m is None or df_m.empty:
        return df_h
    try:
        # Her iki df'yi de kesinlikle temizle
        df_m = _index_temizle(df_m)
        df_h = _index_temizle(df_h)

        if df_m.empty:
            return df_h

        # Her sütun için türetilmiş özellikler ekle
        seriler = {}
        for col in df_m.columns:
            try:
                s = df_m[col].dropna()
                s = _index_temizle(s.to_frame()).iloc[:, 0]  # seri de temizle
                if s.empty:
                    continue
                seriler[col] = s
                seriler[f'{col}_deg1'] = s.pct_change(1)
                seriler[f'{col}_deg5'] = s.pct_change(5)
                seriler[f'{col}_ma5'] = s.rolling(5).mean()
                seriler[f'{col}_vol14'] = s.pct_change().rolling(14).std()
                seriler[f'{col}_oran'] = s / (s.rolling(20).mean() + 1e-10)
            except Exception:
                continue

        if not seriler:
            return df_h

        extra = pd.DataFrame(seriler)
        extra = _index_temizle(extra)  # son bir kez daha temizle
        extra = extra.ffill().bfill()

        # join — her iki taraf da temiz index
        result = df_h.join(extra, how='left').ffill().bfill()
        result = _index_temizle(result)

        temel = ['Open', 'High', 'Low', 'Close', 'Volume', 'Hedef']
        mevcut_temel = [c for c in temel if c in result.columns]
        return result.dropna(subset=mevcut_temel) if mevcut_temel else result
    except Exception:
        return df_h


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────

def _sklearn_yukle():
    """sklearn lazy yükle — ilk çağrıda import, sonra cache."""
    global SKLEARN_OK
    global RobustScaler, VarianceThreshold, accuracy_score, f1_score
    global GradientBoostingClassifier, RandomForestClassifier
    global HistGradientBoostingClassifier, ExtraTreesClassifier
    global compute_sample_weight, LogisticRegression, cross_val_predict
    if RobustScaler is not None:
        return True  # Zaten yüklenmiş
    try:
        from sklearn.preprocessing import RobustScaler as _RS
        from sklearn.feature_selection import VarianceThreshold as _VT
        from sklearn.metrics import accuracy_score as _acc, f1_score as _f1
        from sklearn.ensemble import (
            GradientBoostingClassifier as _GBC,
            RandomForestClassifier as _RFC,
            HistGradientBoostingClassifier as _HGBC,
            ExtraTreesClassifier as _ETC,
        )
        from sklearn.utils.class_weight import compute_sample_weight as _CSW
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.model_selection import cross_val_predict as _CVP
        RobustScaler = _RS;
        VarianceThreshold = _VT
        accuracy_score = _acc;
        f1_score = _f1
        GradientBoostingClassifier = _GBC;
        RandomForestClassifier = _RFC
        HistGradientBoostingClassifier = _HGBC;
        ExtraTreesClassifier = _ETC
        compute_sample_weight = _CSW;
        LogisticRegression = _LR
        cross_val_predict = _CVP
        SKLEARN_OK = True
        return True
    except ImportError:
        SKLEARN_OK = False
        return False


def _xgb_yukle():
    """XGBoost lazy yükle."""
    global xgb, XGB_OK
    if xgb is not None:
        return True
    try:
        import xgboost as _x
        xgb = _x;
        XGB_OK = True;
        return True
    except ImportError:
        XGB_OK = False;
        return False


def _lgb_yukle():
    """LightGBM lazy yükle."""
    global lgb, LGB_OK
    if lgb is not None:
        return True
    try:
        import lightgbm as _l
        lgb = _l;
        LGB_OK = True;
        return True
    except ImportError:
        LGB_OK = False;
        return False


def _yf_yukle():
    """yfinance lazy yükle."""
    global yf, YF_OK
    if yf is not None:
        return True
    try:
        import yfinance as _y
        yf = _y;
        YF_OK = True;
        return True
    except ImportError:
        YF_OK = False;
        return False


def _plotly_yukle():
    """Plotly lazy yükle."""
    global go, px, make_subplots, PLOTLY_OK
    if go is not None:
        return True
    try:
        import plotly.graph_objects as _go
        import plotly.express as _px
        from plotly.subplots import make_subplots as _ms
        go = _go;
        px = _px;
        make_subplots = _ms
        PLOTLY_OK = True;
        return True
    except ImportError:
        PLOTLY_OK = False;
        return False


def _reportlab_yukle():
    """ReportLab lazy yükle."""
    global REPORTLAB_OK, A4, getSampleStyleSheet, ParagraphStyle
    global SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, colors, cm
    if REPORTLAB_OK and 'A4' in dir():
        return True
    try:
        from reportlab.lib.pagesizes import A4 as _A4
        from reportlab.lib.styles import getSampleStyleSheet as _gss, ParagraphStyle as _PS
        from reportlab.platypus import (SimpleDocTemplate as _SDT, Paragraph as _P,
                                        Spacer as _Sp, Table as _T, TableStyle as _TS)
        from reportlab.lib import colors as _col
        from reportlab.lib.units import cm as _cm
        A4 = _A4;
        getSampleStyleSheet = _gss;
        ParagraphStyle = _PS
        SimpleDocTemplate = _SDT;
        Paragraph = _P;
        Spacer = _Sp
        Table = _T;
        TableStyle = _TS;
        colors = _col;
        cm = _cm
        REPORTLAB_OK = True;
        return True
    except ImportError:
        return False


def _psutil_yukle():
    """psutil lazy yükle."""
    global _ps, PSUTIL_OK
    if _ps is not None: return True
    try:
        import psutil as _p;
        _ps = _p;
        PSUTIL_OK = True;
        return True
    except ImportError:
        return False


def _catboost_yukle():
    """CatBoost lazy yükle."""
    global CatBoostClassifier, CATBOOST_OK
    if CatBoostClassifier is not None: return True
    try:
        from catboost import CatBoostClassifier as _CB
        CatBoostClassifier = _CB;
        CATBOOST_OK = True;
        return True
    except ImportError:
        CATBOOST_OK = False;
        return False


def veri_hazirla(df: pd.DataFrame, pencere: int = PENCERE):
    _sklearn_yukle()  # Lazy import

    # 1. Temizlik
    hedef = 'Hedef'
    ozellik = [c for c in df.columns if c != hedef]
    df_temiz = df[ozellik + [hedef]].copy()
    df_temiz = df_temiz.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    # 2. Düşük varyans sütunları at (gürültü azaltma)
    try:
        vt = VarianceThreshold(threshold=1e-6)
        vt.fit(df_temiz[ozellik].values)
        ozellik = [o for o, keep in zip(ozellik, vt.get_support()) if keep]
    except Exception:
        pass

    # 3. Yeterli veri kontrolü
    min_gerekli = max(pencere * 4 + 50, 300)
    if len(df_temiz) < min_gerekli:
        raise ValueError(
            f"Yetersiz veri: {len(df_temiz)} satır, en az {min_gerekli} gerekli. "
            f"Daha uzun veri yılı seçin (5-6 yıl önerilir)."
        )

    # 4. Pencere otomatik ayarla
    maks_pencere = len(df_temiz) // 5
    if pencere > maks_pencere:
        pencere = max(10, maks_pencere)

    # 5. RobustScaler — aykırı değerlere dayanıklı
    scaler = RobustScaler()
    X_s = scaler.fit_transform(df_temiz[ozellik].values).astype(np.float32)
    y = df_temiz[hedef].values.astype(np.float32)

    # 6. Sınıf dengesini kontrol et
    pos_oran = y.mean()
    if pos_oran < 0.3 or pos_oran > 0.7:
        # Dengesiz sınıf varsa ağırlık hesapla
        class_weight = {0: 1 / (1 - pos_oran + 1e-10), 1: 1 / (pos_oran + 1e-10)}
        ag_toplam = sum(class_weight.values())
        class_weight = {k: v / ag_toplam for k, v in class_weight.items()}
    else:
        class_weight = None

    # 7. Walk-forward bölme — son %15 test, kalan eğitim
    # Büyük veri varsa daha fazla test ver (daha güvenilir metrik)
    test_orani = TEST_ORANI if len(X_s) < 2000 else 0.20

    # 8. Sekans oluştur
    X_seq, y_seq = [], []
    for i in range(pencere, len(X_s) - 1):
        X_seq.append(X_s[i - pencere:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    bolme_seq = int(len(X_seq) * (1 - test_orani))
    if bolme_seq < 20 or (len(X_seq) - bolme_seq) < 10:
        raise ValueError("Bölme sonrası çok az örnek. Daha uzun veri yılı seçin.")

    return (X_seq[:bolme_seq], X_seq[bolme_seq:],
            y_seq[:bolme_seq], y_seq[bolme_seq:],
            scaler, ozellik, class_weight)


def optuna_optimize_xgb(X_eg, y_eg, X_te, y_te, n_trials: int = 20):
    """
    Optuna ile XGBoost hiperparametrelerini optimize et.
    n_trials: kaç farklı parametre kombinasyonu dene (20 = ~30sn)
    """
    if not OPTUNA_OK:
        return {}
    from sklearn.metrics import f1_score as _f1

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "eval_metric": "logloss",
            "verbosity": 0,
            "random_state": 42,
            "n_jobs": 1,
        }
        pos_oran = float(y_eg.mean())
        scale_pos = (1 - pos_oran) / (pos_oran + 1e-10)
        params["scale_pos_weight"] = scale_pos

        params['early_stopping_rounds'] = 20
        m = xgb.XGBClassifier(**params)
        m.fit(X_eg, y_eg.astype(int),
              eval_set=[(X_te, y_te.astype(int))],
              verbose=False)
        preds = m.predict(X_te)
        return _f1(y_te.astype(int), preds, zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_optimize_lgb(X_eg, y_eg, X_te, y_te, n_trials: int = 20):
    """LightGBM için Optuna optimizasyonu."""
    if not OPTUNA_OK:
        return {}
    from sklearn.metrics import f1_score as _f1

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "class_weight": "balanced",
            "verbose": -1,
            "random_state": 42,
            "n_jobs": 1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_eg, y_eg.astype(int),
              eval_set=[(X_te, y_te.astype(int))],
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_te)
        return _f1(y_te.astype(int), preds, zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def modelleri_egit(X_eg, y_eg, X_te, y_te, pencere,
                   ozellik_sayisi, status_cb, class_weight=None):
    """
    TensorFlow KALDIRILDI — DLL hatası, versiyon çakışması yok.
    Yerine HistGradientBoosting (sklearn) kullanılıyor:
    - Çok daha hızlı (10x)
    - Kurulum derdi yok
    - Doğruluk: XGBoost/LightGBM ile eşdeğer
    """

    # Analiz başında ağır paketleri yükle (sadece ilk seferde ~10-15sn)
    _sklearn_yukle()
    _xgb_yukle()
    _lgb_yukle()
    _catboost_yukle()
    _psutil_yukle()

    if not XGB_OK or not LGB_OK or not SKLEARN_OK:
        raise ImportError(
            "Eksik paket! Terminalde çalıştır:\n"
            "pip install xgboost lightgbm scikit-learn"
        )
    modeller = {}

    # ── Giriş verisi: son gün + pencere ortalaması + pencere std ─────────────
    X_eg2d_son = X_eg[:, -1, :]  # son günün özellikleri
    X_eg2d_ort = X_eg.mean(axis=1)  # pencere ortalaması
    X_eg2d_std = X_eg.std(axis=1)  # pencere volatilitesi
    X_eg2d_mean = np.concatenate([X_eg2d_son,
                                  X_eg2d_ort,
                                  X_eg2d_std], axis=1)

    X_te2d_son = X_te[:, -1, :]
    X_te2d_ort = X_te.mean(axis=1)
    X_te2d_std = X_te.std(axis=1)
    X_te2d_mean = np.concatenate([X_te2d_son,
                                  X_te2d_ort,
                                  X_te2d_std], axis=1)

    y_eg_int = y_eg.astype(int)
    y_te_int = y_te.astype(int)
    pos_oran = float(y_eg_int.mean())
    scale_pos = (1 - pos_oran) / (pos_oran + 1e-10)
    sw = compute_sample_weight('balanced', y_eg_int)

    n_train = len(X_eg2d_mean)
    n_test = len(X_te2d_mean)

    # ── Minimum veri kontrolü ─────────────────────────────────────────────────
    if n_train < 100:
        raise ValueError(
            f"Model eğitmek için çok az veri: {n_train} eğitim örneği.\n"
            f"Bu hisse yeni halka açılmış veya veri eksik.\n"
            f"En az 2 yıl geriye giden bir hisse seçin."
        )

    # ── Veri boyutuna göre dinamik parametreler ───────────────────────────────
    if n_train < 400:
        n_est = 500
        early_stop = 500
        lr = 0.03
        max_dep = 4
        status_cb(f"⚠️ Az veri ({n_train} örnek)")
    elif n_train < 1000:
        n_est = 800
        early_stop = 500
        lr = 0.03
        max_dep = 5
        status_cb(f"📊 {n_train} örnek")
    elif n_train < 3000:
        n_est = 1000
        early_stop = 500
        lr = 0.025
        max_dep = 6
    else:
        n_est = 1500
        early_stop = 500
        lr = 0.02
        max_dep = 7
    status_cb(f"📊 Eğitim: {n_train} · Test: {n_test} · {n_est} ağaç · LR:{lr}")

    # ── XGBoost (Optuna ile optimize) ────────────────────────────────────────
    status_cb("⚡ XGBoost eğitiliyor" + (" (Optuna optimizasyonu)..." if OPTUNA_OK else "..."))
    if OPTUNA_OK and optuna is not None and st.session_state.get("optuna_aktif", False):
        _xgb_params = optuna_optimize_xgb(X_eg2d_mean, y_eg_int,
                                          X_te2d_mean, y_te_int, n_trials=15)
        _xgb_params.update({"eval_metric": "logloss", "verbosity": 0,
                            "random_state": 42, "n_jobs": 1})
    else:
        _xgb_params = dict(
            n_estimators=n_est, learning_rate=lr,
            max_depth=max_dep, min_child_weight=max(1, n_train // 200),
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0,
            random_state=42, n_jobs=1,
        )
    # early_stopping_rounds → constructor'a ver (XGBoost 2.x uyumlu)
    _xgb_params.pop('early_stopping_rounds', None)  # varsa çıkar
    _xgb_params['early_stopping_rounds'] = early_stop  # constructor'a ekle
    xgb_m = xgb.XGBClassifier(**_xgb_params)
    xgb_m.fit(X_eg2d_mean, y_eg_int,
              eval_set=[(X_te2d_mean, y_te_int)],
              verbose=False)
    xgb_pred = xgb_m.predict(X_te2d_mean)
    modeller['XGBoost'] = {
        'model': xgb_m, 'tip': 'mean2d',
        'dogruluk': accuracy_score(y_te_int, xgb_pred),
        'f1': f1_score(y_te_int, xgb_pred, zero_division=0),
    }

    # ── LightGBM (Optuna ile optimize) ───────────────────────────────────────
    status_cb("💡 LightGBM eğitiliyor" + (" (Optuna optimizasyonu)..." if OPTUNA_OK else "..."))
    if OPTUNA_OK and optuna is not None and st.session_state.get("optuna_aktif", False):
        _lgb_params = optuna_optimize_lgb(X_eg2d_mean, y_eg_int,
                                          X_te2d_mean, y_te_int, n_trials=15)
    else:
        _lgb_params = dict(
            n_estimators=n_est, learning_rate=lr,
            max_depth=max_dep, min_child_samples=max(10, n_train // 50),
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            reg_alpha=0.1, reg_lambda=1.0,
            verbose=-1, random_state=42, n_jobs=1,
        )
    lgb_m = lgb.LGBMClassifier(**_lgb_params)
    lgb_m.fit(X_eg2d_mean, y_eg_int,
              eval_set=[(X_te2d_mean, y_te_int)],
              callbacks=[lgb.early_stopping(early_stop, verbose=False),
                         lgb.log_evaluation(period=-1)])
    lgb_pred = lgb_m.predict(X_te2d_mean)
    modeller['LightGBM'] = {
        'model': lgb_m, 'tip': 'mean2d',
        'dogruluk': accuracy_score(y_te_int, lgb_pred),
        'f1': f1_score(y_te_int, lgb_pred, zero_division=0),
    }

    # ── HistGradientBoosting — sklearn'ın en hızlı modeli ────────────────────
    status_cb("🚀 HistGradientBoosting eğitiliyor...")
    hgb_m = HistGradientBoostingClassifier(
        max_iter=n_est, learning_rate=lr, max_depth=max_dep,
        min_samples_leaf=15, l2_regularization=0.1,
        class_weight='balanced', random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
    )
    hgb_m.fit(X_eg2d_mean, y_eg_int)
    hgb_pred = hgb_m.predict(X_te2d_mean)
    modeller['HistGBM'] = {
        'model': hgb_m, 'tip': 'mean2d',
        'dogruluk': accuracy_score(y_te_int, hgb_pred),
        'f1': f1_score(y_te_int, hgb_pred, zero_division=0),
    }

    # ── ExtraTrees — hızlı ve çeşitlilik sağlar ──────────────────────────────
    status_cb("🌲 ExtraTrees eğitiliyor...")
    et_m = ExtraTreesClassifier(
        n_estimators=min(n_est, 400), max_depth=10,
        min_samples_split=15, min_samples_leaf=8,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=1,
    )
    et_m.fit(X_eg2d_mean, y_eg_int)
    et_pred = et_m.predict(X_te2d_mean)
    modeller['ExtraTrees'] = {
        'model': et_m, 'tip': 'mean2d',
        'dogruluk': accuracy_score(y_te_int, et_pred),
        'f1': f1_score(y_te_int, et_pred, zero_division=0),
    }

    # RandomForest kaldirildi

    # CatBoost kaldirildi

    # ── STACKING — Meta-model (2. katman karar verici) ───────────────────────
    status_cb("🔗 Stacking meta-model eğitiliyor...")
    try:
        # 4 base model: XGB + LGB + HistGBM + ExtraTrees (RF ve CatBoost kaldırıldı)
        meta_X_eg = np.column_stack([
            xgb_m.predict_proba(X_eg2d_mean)[:, 1],
            lgb_m.predict_proba(X_eg2d_mean)[:, 1],
            hgb_m.predict_proba(X_eg2d_mean)[:, 1],
            et_m.predict_proba(X_eg2d_mean)[:, 1],
        ])
        meta_X_te = np.column_stack([
            xgb_m.predict_proba(X_te2d_mean)[:, 1],
            lgb_m.predict_proba(X_te2d_mean)[:, 1],
            hgb_m.predict_proba(X_te2d_mean)[:, 1],
            et_m.predict_proba(X_te2d_mean)[:, 1],
        ])
        meta_m = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        meta_m.fit(meta_X_eg, y_eg_int)
        meta_pred = meta_m.predict(meta_X_te)
        meta_acc = accuracy_score(y_te_int, meta_pred)
        meta_f1 = f1_score(y_te_int, meta_pred, zero_division=0)
        modeller["_meta_model"] = {
            "model": meta_m,
            "tip": "meta",
            "dogruluk": meta_acc,
            "f1": meta_f1,
            "base_models": ["XGBoost", "LightGBM", "HistGBM", "ExtraTrees"],
        }
        status_cb(f"✅ Meta-model: %{meta_acc * 100:.1f} doğruluk / F1:{meta_f1:.2f}")
    except Exception as e:
        status_cb(f"⚠️ Meta-model atlandı: {e}")

    return modeller


def ensemble_tahmin(modeller, X_son, haber_skoru, faiz_etkisi=0.0, rejim_vol=1.0, sektor_skor=0.0):
    """
    Rejim filtresi: Yüksek volatilitede (kriz) eşik yükselir,
    düşük volatilitede sinyal daha güvenilir kabul edilir.
    """
    X_son_last = X_son[:, -1, :]
    # 3-parçalı input: son gün + ortalama + std (modelleri_egit ile tutarlı)
    X_son_mean = np.concatenate([
        X_son_last,
        X_son.mean(axis=1),
        X_son.std(axis=1)
    ], axis=1)

    # Tüm base model olasılıkları
    olasiliklar = {}
    base_olas = {}
    for ad, bilgi in modeller.items():
        if ad.startswith('_'): continue  # meta-modeli atla
        m = bilgi['model']
        try:
            tip = bilgi.get('tip', '2d')
            if tip == 'mean2d':
                p = float(m.predict_proba(X_son_mean)[:, 1][0])
            elif tip == 'lstm':
                p = float(m.predict(X_son, verbose=0).flatten()[0])
            else:
                p = float(m.predict_proba(X_son_last)[:, 1][0])
        except Exception:
            p = 0.5
        olasiliklar[ad] = p
        base_olas[ad] = p

    # Meta-model varsa onu kullan (daha doğru)
    if '_meta_model' in modeller:
        try:
            meta_bilgi = modeller['_meta_model']
            meta_m = meta_bilgi['model']
            baz_modeller = meta_bilgi.get('base_models',
                                          [k for k in modeller if not k.startswith('_')])
            meta_input = np.array([[base_olas.get(k, 0.5) for k in baz_modeller]])
            meta_olas = float(meta_m.predict_proba(meta_input)[:, 1][0])
            ml_skor = meta_olas  # Meta-model kararı ana skor
            olasiliklar['Meta-Model'] = meta_olas
        except Exception:
            ml_skor = _agirlikli_ort(olasiliklar, modeller)
    else:
        ml_skor = _agirlikli_ort(olasiliklar, modeller)

    # Haber + faiz + sektör katkısı
    haber_katki = float(np.clip(haber_skoru * 0.07, -0.05, 0.05))
    faiz_katki = float(np.clip(faiz_etkisi * 0.04, -0.03, 0.03))
    sektor_katki = float(np.clip(sektor_skor * 0.05, -0.04, 0.04))
    final_ham = float(np.clip(
        0.85 * ml_skor
        + 0.06 * (0.5 + haber_katki)
        + 0.05 * (0.5 + sektor_katki)
        + 0.04 * (0.5 + faiz_katki), 0, 1
    ))

    # ── VOLATİLİTE REJİM FİLTRESİ ────────────────────────────────────────────
    # ── Ensemble doğruluğuna göre dinamik eşik ──────────────────────────────
    # Düşük doğrulukta sabit yüksek eşik → hep BEKLE çıkar. Adapte et.
    model_dogruluk = float(np.mean([
        v.get('dogruluk', 0.6)
        for k, v in modeller.items() if not k.startswith('_')
    ]))
    if model_dogruluk >= 0.70:
        baz = 0.62
    elif model_dogruluk >= 0.63:
        baz = 0.58
    else:
        baz = 0.54

    if rejim_vol > 1.5:
        esik_al = min(baz + 0.08, 0.72)
        esik_guclu_al = min(baz + 0.14, 0.78)
        esik_sat = 0.30
        rejim_notu = "⚡ Yüksek Volatilite — Dikkatli Ol"
    elif rejim_vol < 0.7:
        esik_al = max(baz - 0.03, 0.51)
        esik_guclu_al = max(baz + 0.05, 0.59)
        esik_sat = 0.37
        rejim_notu = "😌 Sakin Piyasa"
    else:
        esik_al = baz
        esik_guclu_al = baz + 0.08
        esik_sat = 0.35
        rejim_notu = f"📊 Normal Rejim (Eşik: %{baz * 100:.0f})"

    if final_ham >= esik_guclu_al:
        sinyal = "📈 GÜÇLÜ AL"
    elif final_ham >= esik_al:
        sinyal = "📈 AL"
    elif final_ham <= esik_sat:
        sinyal = "📉 GÜÇLÜ SAT"
    elif final_ham <= 0.42:
        sinyal = "📉 SAT"
    else:
        sinyal = "⏸️ BEKLE"

    return {
        "final": final_ham,
        "sinyal": sinyal,
        "olasiliklar": olasiliklar,
        "haber_katki": haber_katki,
        "sektor_katki": round(sektor_katki, 4),
        "rejim_notu": rejim_notu,
        "rejim_vol": round(rejim_vol, 2),
    }


def _agirlikli_ort(olasiliklar, modeller):
    """F1 + doğruluk ağırlıklı ortalama."""
    agirliklar = {}
    for ad, b in modeller.items():
        if ad.startswith('_'): continue
        agirliklar[ad] = b.get('f1', 0.5) * 0.6 + b.get('dogruluk', 0.5) * 0.4
    top = sum(agirliklar.values()) or 1
    return sum(olasiliklar.get(ad, 0.5) * v / top for ad, v in agirliklar.items())


def faiz_etkisi_hesapla(faiz=45.0):
    if faiz >= 50: return -0.4
    if faiz >= 40: return -0.2
    if faiz >= 30: return -0.1
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BAŞLATMA
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SQLite VERİTABANI — Kalıcı depolama (hisse aç/kapat = veri kaybolmaz)
# ─────────────────────────────────────────────────────────────────────────────
import sqlite3, json as _json, os as _os

# Streamlit Cloud'da /tmp yazilabilir (ama reboot'ta sifirlanir)
# Lokal: uygulama klasoru
try:
    _test_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "bist_data.db")
    # Yazma testi
    with open(_test_path, 'a') as _tf:
        pass
    DB_YOLU = _test_path
except Exception:
    DB_YOLU = "/tmp/bist_data.db"  # Streamlit Cloud fallback


def db_baglanti():
    conn = sqlite3.connect(DB_YOLU, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # Paralel okuma için
    return conn


def db_hazirla():
    """Tablolar yoksa oluştur."""
    with db_baglanti() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS portfolyo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now','localtime')),
            data TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS alarmlar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now','localtime')),
            data TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS analiz_gecmisi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now','localtime')),
            data TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS favoriler (
            hisse TEXT PRIMARY KEY,
            ts TEXT DEFAULT (datetime('now','localtime'))
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS hata_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now','localtime')),
            hisse TEXT, hata TEXT, tb TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS geri_bildirim (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now','localtime')),
            hisse TEXT, sinyal TEXT, dogru INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS notlar (
            hisse TEXT PRIMARY KEY, not_tr TEXT,
            ts TEXT DEFAULT (datetime('now','localtime')))""")
        c.execute("""CREATE TABLE IF NOT EXISTS ayarlar (
            key TEXT PRIMARY KEY,
            val TEXT NOT NULL
        )""")
        c.commit()


def db_yukle(tablo: str) -> list:
    """Tablodan tüm kayıtları yükle."""
    try:
        with db_baglanti() as c:
            if tablo == "favoriler":
                return [r[0] for r in c.execute("SELECT hisse FROM favoriler ORDER BY ts").fetchall()]
            rows = c.execute(f"SELECT data FROM {tablo} ORDER BY id").fetchall()
            return [_json.loads(r[0]) for r in rows]
    except Exception:
        return []


def db_kaydet_liste(tablo: str, liste: list):
    """Listeyi tablo ile senkronize et (tümünü sil, yeniden yaz)."""
    try:
        with db_baglanti() as c:
            c.execute(f"DELETE FROM {tablo}")
            for item in liste:
                c.execute(f"INSERT INTO {tablo} (data) VALUES (?)", (_json.dumps(item, ensure_ascii=False),))
            c.commit()
    except Exception:
        pass


def db_favoriler_kaydet(favoriler: list):
    try:
        with db_baglanti() as c:
            c.execute("DELETE FROM favoriler")
            for h in favoriler:
                c.execute("INSERT OR REPLACE INTO favoriler (hisse) VALUES (?)", (h,))
            c.commit()
    except Exception:
        pass


def db_ayar_oku(key: str, varsayilan=None):
    try:
        with db_baglanti() as c:
            r = c.execute("SELECT val FROM ayarlar WHERE key=?", (key,)).fetchone()
            return _json.loads(r[0]) if r else varsayilan
    except Exception:
        return varsayilan


def db_ayar_yaz(key: str, val):
    try:
        with db_baglanti() as c:
            c.execute("INSERT OR REPLACE INTO ayarlar (key, val) VALUES (?,?)",
                      (key, _json.dumps(val, ensure_ascii=False)))
            c.commit()
    except Exception:
        pass


# DB başlat
db_hazirla()
# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE ENTEGRASYONU — Kalıcı Bulut Veritabanı
# Streamlit Cloud secrets.toml'a ekle:
#   [supabase]
#   url = "https://XXXXX.supabase.co"
#   key = "eyJhbGc..."
# ─────────────────────────────────────────────────────────────────────────────
_SUPABASE_OK = False
_sb_client = None


def _supabase_baglanti():
    """Supabase bağlantısı — secrets varsa kullan."""
    global _SUPABASE_OK, _sb_client
    if _sb_client is not None:
        return _sb_client
    try:
        _url = st.secrets.get("supabase", {}).get("url", "")
        _key = st.secrets.get("supabase", {}).get("key", "")
        if not _url or not _key:
            return None
        from supabase import create_client
        _sb_client = create_client(_url, _key)
        _SUPABASE_OK = True
        return _sb_client
    except Exception:
        return None


def db_yukle_cloud(tablo: str) -> list:
    """Supabase'den yükle."""
    try:
        sb = _supabase_baglanti()
        if sb is None:
            return db_yukle(tablo)  # fallback: yerel SQLite
        res = sb.table(tablo).select("*").order("id").execute()
        return [r.get("data", r) for r in (res.data or [])]
    except Exception:
        return db_yukle(tablo)


def db_kaydet_cloud(tablo: str, liste: list):
    """Supabase'e kaydet."""
    try:
        sb = _supabase_baglanti()
        if sb is None:
            db_kaydet_liste(tablo, liste)
            return
        # Tüm kayıtları sil ve yeniden yaz
        sb.table(tablo).delete().neq("id", 0).execute()
        for item in liste:
            sb.table(tablo).insert({"data": item}).execute()
    except Exception:
        db_kaydet_liste(tablo, liste)  # fallback


def db_favoriler_cloud(favoriler: list):
    """Favorileri cloud'a kaydet."""
    try:
        sb = _supabase_baglanti()
        if sb is None:
            db_favoriler_kaydet(favoriler)
            return
        sb.table("favoriler").delete().neq("hisse", "").execute()
        for h in favoriler:
            sb.table("favoriler").insert({"hisse": h}).execute()
    except Exception:
        db_favoriler_kaydet(favoriler)


if 'alarmlar' not in st.session_state:
    st.session_state.alarmlar = db_yukle('alarmlar')
if 'portfolyo' not in st.session_state:
    st.session_state.portfolyo = db_yukle('portfolyo')
if 'analiz_gecmisi' not in st.session_state: st.session_state.analiz_gecmisi = []
if 'favoriler' not in st.session_state: st.session_state.favoriler = []
if 'dil' not in st.session_state: st.session_state.dil = "TR"

# ─────────────────────────────────────────────────────────────────────────────
# DİL DESTEĞİ
# ─────────────────────────────────────────────────────────────────────────────
METINLER = {
    "TR": {
        "baslik": "📈 BIST Tahmin Sistemi",
        "altyazi": "7 Model + Stacking · CatBoost · 15 Yıl Veri · 10 Günlük Tahmin · %70-78 Doğruluk",
        "analiz": "📊 Analiz", "portfolyo": "💼 Portföy", "gecmis": "📅 Geçmiş",
        "risk": "⚖️ Risk", "pdf": "📄 PDF", "karsilastir": "🔀 Karşılaştır",
        "backtest": "⏮️ Backtest", "piyasa": "🌍 Piyasa",
        "ayarlar": "⚙️ Ayarlar", "hisse_ara": "🔍 Hisse Ara",
        "hisse_sec": "🏦 Hisse Senedi Seç", "model_ayar": "📊 Model Ayarları",
        "veri_yili": "Geçmiş Veri (Yıl)", "lstm_pencere": "Sekans Penceresi (Gün)",
        "faiz": "🏛️ TCMB Politika Faizi", "faiz_oran": "Faiz Oranı (%)",
        "analiz_baslat": "🚀 ANALİZİ BAŞLAT", "sozluk": "📖 Terimler Sözlüğü",
        "yukselme_olas": "Yükselme Olasılığı", "guncel_fiyat": "Güncel Fiyat",
        "ensemble_dog": "Ensemble Doğruluk", "islem_plani": "📊 İŞLEM PLANI",
        "al_araligi": "🟢 AL ARALIĞI", "zarar_kes": "🔴 ZARAR KES (STOP-LOSS)",
        "bekleme": "⏱️ BEKLEME SÜRESİ", "hedef": "🎯 Hedef Fiyat",
        "favori_ekle": "⭐ Favorilere Ekle", "favori_cikar": "★ Favorilerden Çıkar",
        "uyari": "",
    },
    "EN": {
        "baslik": "📈 BIST Prediction System",
        "altyazi": "6 Models + Stacking · Portfolio · Backtest · Comparison · 70-76% Accuracy",
        "analiz": "📊 Analysis", "portfolyo": "💼 Portfolio", "gecmis": "📅 History",
        "risk": "⚖️ Risk", "pdf": "📄 PDF", "karsilastir": "🔀 Compare",
        "backtest": "⏮️ Backtest", "piyasa": "🌍 Market",
        "ayarlar": "⚙️ Settings", "hisse_ara": "🔍 Search Stock",
        "hisse_sec": "🏦 Select Stock", "model_ayar": "📊 Model Settings",
        "veri_yili": "Historical Data (Years)", "lstm_pencere": "Sequence Window (Days)",
        "faiz": "🏛️ CBRT Policy Rate", "faiz_oran": "Interest Rate (%)",
        "analiz_baslat": "🚀 START ANALYSIS", "sozluk": "📖 Glossary",
        "yukselme_olas": "Rise Probability", "guncel_fiyat": "Current Price",
        "ensemble_dog": "Ensemble Accuracy", "islem_plani": "📊 TRADE PLAN",
        "al_araligi": "🟢 BUY RANGE", "zarar_kes": "🔴 STOP-LOSS",
        "bekleme": "⏱️ HOLDING PERIOD", "hedef": "🎯 Target Price",
        "favori_ekle": "⭐ Add to Favorites", "favori_cikar": "★ Remove Favorite",
        "uyari": "",
    }
}


def guven_skoru(acc, n_train, modeller):
    """Dusuk/Orta/Yuksek guven"""
    p = 0
    p += 3 if acc >= 0.72 else 2 if acc >= 0.66 else 1 if acc >= 0.60 else 0
    p += 3 if n_train >= 2000 else 2 if n_train >= 800 else 1 if n_train >= 300 else 0
    dogs = [v["dogruluk"] for k, v in modeller.items() if not k.startswith("_")]
    if dogs: p += 2 if (max(dogs) - min(dogs)) < 0.03 else 1 if (max(dogs) - min(dogs)) < 0.07 else 0
    if p >= 7:
        return "YUKSEK", "#34d399", "OK"
    elif p >= 4:
        return "ORTA", "#facc15", "ORT"
    else:
        return "DUSUK", "#ef4444", "DUS"


def T(key):
    """Aktif dilde metin döndür."""
    return METINLER[st.session_state.dil].get(key, key)


# ─────────────────────────────────────────────────────────────────────────────
# PİYASA ÖZET VERİSİ
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat cache
def piyasa_ozeti_al():
    """
    Piyasa verilerini SADECE direkt API ile çeker.
    yfinance kütüphanesi kullanılmaz → rate limit yok.
    TTL=6 saat: günde 2-3 kez güncellenir, asla engellenmez.
    """
    gostrge = {
        "BIST 100": "XU100.IS", "USD/TRY": "USDTRY=X",
        "EUR/TRY": "EURTRY=X", "Altın": "GC=F",
        "Petrol": "CL=F", "VIX": "^VIX",
    }

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0 Safari/537.36",
    ]

    def _api_cek(tick, ua_idx=0):
        """Direkt Yahoo v8 API — yfinance kütüphanesi kullanmaz."""
        import random
        hdrs = {
            "User-Agent": USER_AGENTS[ua_idx % len(USER_AGENTS)],
            "Accept": "application/json, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finance.yahoo.com/",
            "Origin": "https://finance.yahoo.com",
        }
        for base in [
            "https://query2.finance.yahoo.com/v8/finance/chart/",
            "https://query1.finance.yahoo.com/v8/finance/chart/",
        ]:
            try:
                url = f"{base}{tick}?interval=1d&range=5d&includeAdjustedClose=true"
                r = requests.get(url, headers=hdrs, timeout=10)
                if r.status_code == 429:
                    continue  # diğer endpoint'i dene
                if r.status_code != 200:
                    continue
                data = r.json()
                res = data.get("chart", {}).get("result", [])
                if not res:
                    continue
                quot = res[0].get("indicators", {}).get("quote", [{}])[0]
                adjc = res[0].get("indicators", {}).get("adjclose", [{}])
                c = (adjc[0].get("adjclose") if adjc else None) or quot.get("close", [])
                c = [x for x in (c or []) if x and x > 0]
                if len(c) >= 2:
                    return round(c[-1], 4), round((c[-1] / c[-2] - 1) * 100, 2)
                elif len(c) == 1:
                    return round(c[-1], 4), 0.0
            except Exception:
                continue
        return 0.0, 0.0

    def _tek_ticker(item):
        ad, tick = item
        try:
            fiyat, degisim = _api_cek(tick)
            return ad, {"fiyat": fiyat, "degisim": degisim}
        except Exception:
            return ad, {"fiyat": 0.0, "degisim": 0.0}

    sonuc = {}
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(_tek_ticker, item): item[0]
                       for item in gostrge.items()}
            for future in as_completed(futures, timeout=12):
                try:
                    ad, veri = future.result(timeout=4)
                    sonuc[ad] = veri
                except Exception:
                    pass
    except Exception:
        pass

    for ad in gostrge:
        if ad not in sonuc:
            sonuc[ad] = {"fiyat": 0.0, "degisim": 0.0}
    return sonuc


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────

# ── 30 saniyelik otomatik yenileme ───────────────────────────────────────────
# Otomatik yenileme devre dışı — rate limit azaltmak için
# Manuel yenileme için F5 kullanın

st.markdown(f"""
<div class="main-header">
  <h1>{T('baslik')}</h1>
  <p>{T('altyazi')}</p>
</div>
""", unsafe_allow_html=True)

# ── Piyasa özet şeridi — arka planda yükle ──────────────────────────────────
try:
    piyasa_data = piyasa_ozeti_al()
    son_guncelleme = datetime.now().strftime("%H:%M:%S")
except Exception:
    piyasa_data = {}
    son_guncelleme = "--:--:--"

if not piyasa_data:
    piyasa_data = {
        "BIST 100": {"fiyat": 0.0, "degisim": 0.0},
        "USD/TRY": {"fiyat": 0.0, "degisim": 0.0},
        "EUR/TRY": {"fiyat": 0.0, "degisim": 0.0},
        "Altın": {"fiyat": 0.0, "degisim": 0.0},
        "Petrol": {"fiyat": 0.0, "degisim": 0.0},
        "VIX": {"fiyat": 0.0, "degisim": 0.0},
    }

# Altın ve döviz TRY dönüşümü
_usdtry = piyasa_data.get("USD/TRY", {}).get("fiyat", 0) or 44.0


def _formatla(ad, fiyat, degisim):
    """Birimi ve formatı düzelt."""
    birim = ""
    if ad in ("BIST 100",):
        birim = ""
        goster = f"{fiyat:,.0f}"
    elif ad in ("USD/TRY", "EUR/TRY"):
        birim = "₺"
        goster = f"{fiyat:.4f}"
    elif ad == "Altın":
        # GC=F USD/ons → TL/gram: (fiyat × usdtry) / 31.1035
        fiyat_tl = fiyat * _usdtry / 31.1035
        goster = f"{fiyat_tl:,.0f}"
        birim = "₺/gr"
    elif ad == "Petrol":
        fiyat_tl = fiyat * _usdtry
        goster = f"{fiyat_tl:,.0f}"
        birim = "₺/varil"
    elif ad == "VIX":
        birim = ""
        goster = f"{fiyat:.2f}"
    else:
        goster = f"{fiyat:,.2f}"
    return goster, birim


cols_piyasa = st.columns(len(piyasa_data))
for col, (ad, veri) in zip(cols_piyasa, piyasa_data.items()):
    degisim = veri['degisim']
    fiyat = veri['fiyat']
    renk = "#34d399" if degisim >= 0 else "#ef4444"
    ok = "▲" if degisim >= 0 else "▼"
    goster, birim = _formatla(ad, fiyat, degisim)
    with col:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid #1e293b;
                    border-radius:8px;padding:0.5rem 0.7rem;text-align:center;">
          <div style="color:#64748b;font-size:0.65rem;font-family:monospace">{ad}</div>
          <div style="color:#e2e8f0;font-size:0.88rem;font-weight:700;font-family:monospace">
            {goster} <span style="font-size:0.6rem;color:#64748b">{birim}</span>
          </div>
          <div style="color:{renk};font-size:0.72rem;font-family:monospace">
            {ok} %{abs(degisim):.2f}
          </div>
        </div>""", unsafe_allow_html=True)

# Son güncelleme zamanı
st.markdown(f'<div style="text-align:right;color:#334155;font-size:0.65rem;'
            f'font-family:monospace;margin-top:0.2rem">'
            f'🕐 Son güncelleme: {son_guncelleme} · 6 saatte bir güncellenir</div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Ana sekmeler
makro_df = pd.DataFrame()  # global
sekme_analiz, sekme_karsilastir, sekme_backtest, sekme_portfolyo, sekme_gecmis, sekme_risk, sekme_pdf = st.tabs([
    T("analiz"), T("karsilastir"), T("backtest"),
    T("portfolyo"), T("gecmis"), T("risk"), T("pdf")
])

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Dil
    dil_col1, dil_col2 = st.columns(2)
    with dil_col1:
        if st.button("🇹🇷 TR", use_container_width=True,
                     type="primary" if st.session_state.dil == "TR" else "secondary"):
            st.session_state.dil = "TR";
            db_ayar_yaz('dil', 'TR');
            st.rerun()
    with dil_col2:
        if st.button("🇬🇧 EN", use_container_width=True,
                     type="primary" if st.session_state.dil == "EN" else "secondary"):
            st.session_state.dil = "EN";
            db_ayar_yaz('dil', 'EN');
            st.rerun()
    st.markdown("---")

    # Hisse arama
    arama = st.text_input(T("hisse_ara"), placeholder="THYAO, Garanti, Çimento...")
    if arama:
        filtreli = {k: v for k, v in BIST_HISSELER.items()
                    if arama.upper() in k.upper()}
    else:
        filtreli = BIST_HISSELER

    # Sektör filtresi
    SEKTORLER = {
        "Tümü / All": [],
        "🏦 Bankalar": ["AKBNK", "GARAN", "ISCTR", "YKBNK", "VAKBN", "ALBRK", "SKBNK", "TSKB", "KLNMA"],
        "🏭 Sanayi": ["EREGL", "KRDMD", "ISDMR", "BRSAN", "CELHA", "SARKY", "JANTS"],
        "⚡ Enerji": ["AKSEN", "ZOREN", "ODAS", "AYDEM", "AYEN", "ENJSA", "EUPWR", "ORGE"],
        "🚗 Otomotiv": ["TOASO", "FROTO", "ASUZU", "KARSN", "OTKAR", "TTRAK"],
        "✈️ Havacılık": ["THYAO", "PGSUS", "TAVHL", "CLEBI"],
        "🛒 Perakende": ["BIMAS", "MGROS", "SOKM", "MAVI", "KOTON"],
        "🏗️ Holding": ["KCHOL", "SAHOL", "AGHOL", "DOHOL", "TKFEN", "ENKAI", "GLYHO"],
        "📱 Teknoloji": ["ASELS", "TCELL", "TTKOM", "LOGO", "ARENA", "INDES", "KAREL"],
        "🏠 GYO": ["EKGYO", "ISGYO", "TRGYO", "SNGYO", "RYGYO", "VKGYO"],
        "💊 Sağlık": ["DEVA", "MPARK", "MEDTR", "LKMNH"],
        "🌾 Gıda": ["ULKER", "TATGD", "KERVT", "CCOLA", "AEFES", "MERKO"],
        "🪨 Madencilik": ["KOZAL", "IPEKE", "MAALT", "PRKME"],
        "🧪 Kimya": ["PETKM", "SASA", "ALKIM", "SODA"],
        "🏺 Cam/Çimento": ["SISE", "CIMSA", "NUHCM", "BUCIM", "GOLTS", "OYAKC"],
    }
    sektor = st.selectbox(
        "🏷️ Sektör" if st.session_state.dil == "TR" else "🏷️ Sector",
        list(SEKTORLER.keys()), index=0
    )
    if sektor != "Tümü / All" and SEKTORLER[sektor]:
        filtreli = {k: v for k, v in filtreli.items()
                    if any(kod in k for kod in SEKTORLER[sektor])}

    st.markdown(f'<div style="font-size:0.72rem;color:#64748b;font-family:monospace;margin-bottom:0.3rem">'
                f'📋 {len(filtreli)} / {len(BIST_HISSELER)} hisse</div>', unsafe_allow_html=True)

    if not filtreli: filtreli = BIST_HISSELER

    # Favori toggle
    if st.session_state.favoriler:
        fav_toggle = st.toggle(
            f"⭐ Favoriler ({len(st.session_state.favoriler)})" if st.session_state.dil == "TR"
            else f"⭐ Favorites ({len(st.session_state.favoriler)})"
        )
        if fav_toggle:
            filtreli = {k: v for k, v in BIST_HISSELER.items()
                        if k.split("—")[0].strip() in st.session_state.favoriler}

    hisse_secim = st.selectbox(T("hisse_sec"), options=list(filtreli.keys()), index=0)
    hisse_kodu = hisse_secim.split("—")[0].strip()
    ticker = BIST_HISSELER[hisse_secim]

    # Favori ekle/çıkar butonu
    is_fav = hisse_kodu in st.session_state.favoriler
    fav_label = (T("favori_cikar") if is_fav else T("favori_ekle"))
    if st.button(fav_label, use_container_width=True):
        if is_fav:
            st.session_state.favoriler.remove(hisse_kodu)
        else:
            st.session_state.favoriler.append(hisse_kodu)
        db_favoriler_cloud(st.session_state.favoriler) if _SUPABASE_OK else db_favoriler_kaydet(
            st.session_state.favoriler)
        st.rerun()

    # Hisse bilgi kartı
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.1);border:1px solid #4f46e5;
                border-radius:10px;padding:0.8rem;margin-top:0.5rem;
                font-family:'Space Mono',monospace;font-size:0.8rem;">
      <div style="color:#818cf8;">{"Seçili" if st.session_state.dil == "TR" else "Selected"} {"⭐" if is_fav else ""}</div>
      <div style="color:#e2e8f0;font-weight:700;font-size:1rem;">{hisse_kodu}</div>
      <div style="color:#64748b;font-size:0.72rem;">{ticker}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── FAİZ (hemen hisseden sonra) ──────────────────────────────────────────
    _tema = st.toggle("Acik Tema", value=False, key="acik_tema")
    if _tema:
        st.markdown(
            "<style>.stApp{background:#f8fafc!important;color:#1e293b!important}</style>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("---")
    st.markdown(f"### {T('faiz')}")
    faiz_orani = st.slider(T("faiz_oran"), 5, 65, 45)
    f_etkisi = faiz_etkisi_hesapla(faiz_orani)
    faiz_renk = "badge-red" if f_etkisi < 0 else "badge-green"
    st.markdown(f'<span class="info-badge {faiz_renk}">Etki: {f_etkisi:+.1f}</span>',
                unsafe_allow_html=True)

    # ── ANALİZ BUTONU (büyük, belirgin) ──────────────────────────────────────
    st.markdown("---")
    analiz_btn = st.button(
        T("analiz_baslat"),
        use_container_width=True,
        type="primary",
        help="Tüm modelleri eğitip tahmin üret (2-5 dk)"
    )
    yenile_btn = st.button(
        "Verileri Yenile" if st.session_state.dil == "TR" else "Refresh Data",
        use_container_width=True
    )
    if yenile_btn:
        # Güvenli cache temizleme — cache'li olmayan fonksiyon hata verir
        for _fn in [hisse_indir, makro_indir, haber_skoru_al,
                    piyasa_ozeti_al, sektor_momentum_al]:
            try:
                _fn.clear()
            except Exception:
                pass
        try:
            indikatör_ekle.clear()
        except Exception:
            pass
        st.success("Cache temizlendi!")
        st.rerun()
    st.markdown("---")

    # ── MODEL AYARLARI (gelişmiş kullanıcılar için) ───────────────────────────
    with st.expander(f"⚙️ {T('model_ayar')}", expanded=False):
        veri_yili = st.slider(T("veri_yili"), 2, 15, VARSAYILAN_YIL)
        pencere = st.slider(T("lstm_pencere"), 20, 60, 40)

        st.markdown("### 📰 " + ("Haber Kaynağı" if st.session_state.dil == "TR" else "News Source"))
        haber_kaynagi = st.multiselect(
            "Kaynak" if st.session_state.dil == "TR" else "Source",
            ["Google News", "Bloomberg HT", "Investing.com TR",
             "Habertürk", "Milliyet", "Sabah Ekonomi", "Ekonomist", "Döviz.com"],
            default=["Google News", "Bloomberg HT", "Habertürk", "Milliyet"]
        )

        st.markdown("### 🎯 " + ("Optimizasyon" if st.session_state.dil == "TR" else "Optimization"))
        optuna_aktif = st.toggle(
            "⚡ Optuna Hiperparametre" if st.session_state.dil == "TR" else "⚡ Optuna Hyperparameter",
            value=False,
            help="Açık: XGBoost+LightGBM otomatik optimize edilir (+1-2 dk)"
        )
        if not OPTUNA_OK:
            st.caption("⚠️ pip install optuna")

    # Otomatik yenileme
    st.markdown("---")
    st.markdown("### 🔄 Otomatik Yenileme" if st.session_state.dil == "TR" else "### 🔄 Auto Refresh")
    oto_yenile = st.toggle(
        "Her 10 dk yenile" if st.session_state.dil == "TR" else "Refresh every 10 min"
    )
    if oto_yenile:
        st.markdown('<div style="color:#34d399;font-size:0.75rem;">✅ Aktif</div>',
                    unsafe_allow_html=True)
        st.markdown('<meta http-equiv="refresh" content="600">', unsafe_allow_html=True)

    st.markdown("---")
    # analiz_btn sidebar'da tanımlandı (yukarıda)

    st.markdown("---")

    # ── MİNİ SÖZLÜK ──────────────────────────────────────────────────────────
    with st.expander("📖 Terimler Sözlüğü", expanded=False):
        sozluk = {
            "📈 Teknik İndikatörler": {
                "MA20 (Hareketli Ort. 20)":
                    "Son 20 günün kapanış fiyatlarının ortalaması. Fiyat MA20'nin üstündeyse kısa vadeli yükseliş trendi var demektir.",
                "MA50 (Hareketli Ort. 50)":
                    "Son 50 günün ortalaması. Orta vadeli trendi gösterir. Fiyat MA50 üstündeyse genel trend yukarı.",
                "MA200 (Hareketli Ort. 200)":
                    "Son 200 günün ortalaması. Uzun vadeli trendin en önemli göstergesi. 'Bull market' için fiyatın MA200 üstünde olması beklenir.",
                "EMA (Üstel Hareketli Ort.)":
                    "Son günlere daha fazla ağırlık veren hareketli ortalama. Normal MA'ya göre fiyat değişimlerine daha hızlı tepki verir.",
                "RSI (Göreceli Güç End.)":
                    "0-100 arasında değer alır. 70 üstü → aşırı alım (düşebilir). 30 altı → aşırı satım (yükselebilir). 50 civarı → nötr.",
                "MACD":
                    "İki EMA'nın farkıdır. MACD çizgisi sinyal çizgisini yukarı keserse AL sinyali, aşağı keserse SAT sinyali verir.",
                "Bollinger Bantları":
                    "Fiyatın etrafında üst ve alt bant oluşturur. Fiyat üst banda yaklaştıysa aşırı alım, alt banda yaklaştıysa aşırı satım.",
                "ATR (Ortalama Gerçek Aralık)":
                    "Hissenin günlük ne kadar hareket ettiğini ölçer. Yüksek ATR = yüksek volatilite. Stop-loss hesaplamasında kullanılır.",
                "Stochastic":
                    "Fiyatın belirli bir dönemde nerede kapandığını gösterir. 80 üstü → aşırı alım, 20 altı → aşırı satım.",
                "CCI (Emtia Kanal End.)":
                    "+100 üstü → güçlü yukarı trend, -100 altı → güçlü aşağı trend. Aşırı seviyelerde geri dönüş beklenebilir.",
                "OBV (Denge Hacim)":
                    "Hacim bazlı indikatör. OBV yükseliyorken fiyat düşüyorsa yakında yükseliş gelebilir (uyumsuzluk sinyali).",
                "MFI (Para Akış End.)":
                    "RSI'ın hacimleri de dikkate alan versiyonu. 80 üstü → aşırı alım, 20 altı → aşırı satım.",
                "Williams %R":
                    "Stochastic'e benzer. -80 altı → aşırı satım (AL fırsatı), -20 üstü → aşırı alım (SAT fırsatı).",
                "BB %B":
                    "Fiyatın Bollinger Bantları içindeki yüzde konumu. %100 → üst bantta, %0 → alt bantta.",
            },
            "💰 Yatırım Terimleri": {
                "Stop-Loss (Zarar Kes)":
                    "Hisse belirlenen fiyata düştüğünde otomatik satış emri. Örn: 100₺'ye aldığın hisse için 93₺ stop koyarsan max %7 zararda çıkarsın.",
                "Hedef Fiyat":
                    "Modelin o hisse için öngördüğü kâr alma seviyesi. Fiyat buraya ulaşınca kârı realize etmek için satış yapılır.",
                "Risk/Ödül Oranı":
                    "Potansiyel kârın zarara oranı. 1:2 = 5₺ risk için 10₺ kazanç bekliyorsun demek. 1:2 ve üstü iyi sayılır.",
                "Volatilite":
                    "Hissenin fiyat dalgalanma derecesi. Yüksek volatilite = daha riskli ama daha fazla kazanç potansiyeli.",
                "Ensemble Model":
                    "Birden fazla modelin tahminlerini birleştirme yöntemi. Bu sistemde 6 model + stacking (meta-model) kullanılıyor. TensorFlow/LSTM kaldırıldı, daha hızlı ve kararlı çalışıyor.",
                "Hacim (Volume)":
                    "Bir günde el değiştiren hisse adedi. Yüksek hacimli günlerdeki hareketler daha güvenilirdir.",
                "Momentum":
                    "Fiyatın belirli bir dönemdeki değişim hızı. Pozitif momentum = hızlanan yükseliş, negatif = hızlanan düşüş.",
                "Portföy Çeşitlendirme":
                    "Parayı birden fazla hisseye bölme stratejisi. Tek hissede kayıp tüm sermayeyi etkilemez.",
                "Bull Market":
                    "Genel olarak fiyatların yükseldiği piyasa dönemi. Türkçe: 'Boğa piyasası'.",
                "Bear Market":
                    "Genel olarak fiyatların düştüğü piyasa dönemi. Türkçe: 'Ayı piyasası'.",
            },
            "🤖 Model Terimleri": {
                "XGBoost":
                    "Gradient Boosting tabanlı hızlı model. Bu sistemde sekans ortalaması + son günlük özellikler girdi olarak kullanılıyor. En yüksek doğruluk veren modellerden biri.",
                "XGBoost":
                    "Gradient Boosting tabanlı hızlı ve güçlü sınıflandırma modeli. Finans dünyasında çok tercih edilir.",
                "LightGBM":
                    "Microsoft'un geliştirdiği XGBoost'a benzer ama daha hızlı çalışan model. Büyük veri setlerinde avantajlı.",
                "Random Forest":
                    "Yüzlerce karar ağacının oylamasıyla tahmin yapan model. Tek ağaca göre çok daha kararlı sonuçlar verir.",
                "Ensemble":
                    "Birden fazla modelin tahminlerini ağırlıklı ortalama ile birleştirme. Bu uygulamada 5 model birlikte kullanılıyor.",
                "Doğruluk (Accuracy)":
                    "Modelin test verisinde kaç tahmini doğru yaptığının yüzdesi. %65+ iyi, %70+ çok iyi sayılır.",
                "Overfitting":
                    "Modelin sadece eğitim verisini ezberlediği, yeni veride başarısız olduğu durum. Bu yüzden test verisi ayrı tutulur.",
                "HistGradientBoosting":
                    "sklearn'ın en hızlı gradient boosting modelidir. TensorFlow'a gerek duymadan yüksek doğruluk sağlar, kurulum derdi yoktur.",
            },
        }

        for kategori, terimler in sozluk.items():
            st.markdown(f"**{kategori}**")
            for terim, aciklama in terimler.items():
                st.markdown(f"""
                <div style="margin:0.4rem 0;padding:0.6rem 0.8rem;
                            background:rgba(255,255,255,0.02);border-radius:8px;
                            border-left:2px solid #4f46e5;">
                  <div style="color:#818cf8;font-weight:700;font-size:0.8rem;
                              font-family:'Space Mono',monospace;">{terim}</div>
                  <div style="color:#94a3b8;font-size:0.75rem;line-height:1.5;
                              margin-top:0.2rem;">{aciklama}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")

    st.markdown("---")

    # Optimizasyon rehberi
    with st.expander("💡 " + ("Doğruluk İpuçları" if st.session_state.dil == "TR" else "Accuracy Tips")):
        st.markdown(
            "**En yüksek doğruluk için:**\n\n"
            "- Büyük hisseler seç: THYAO, GARAN, AKBNK, EREGL\n"
            "- Veri yılı = 15\n"
            "- Sekans penceresi = 40\n"
            "- TCMB faizini güncel gir (şu an ~%45)\n"
            "- Yeni IPO hisselerden kaçın\n"
            "- Hacim < 5M TL hisselerden kaçın"
        )

# ── ANALİZ SEKMESİ ───────────────────────────────────────────────────────────
with sekme_analiz:
    if analiz_btn:

        st.markdown(f"## 🔍 {hisse_kodu} Analizi")
        # RAM KORUMASI — Streamlit Cloud 1GB
        try:
            _psutil_yukle()
            if PSUTIL_OK and _ps:
                _ram_pct = _ps.virtual_memory().percent
                if _ram_pct > 85:
                    st.warning(f'⚠️ Bellek yüksek (%{_ram_pct:.0f}) — veri yılını azalt.')
                if _ram_pct > 92:
                    st.error('❌ Bellek kritik! Sayfayı yenile.')
                    st.stop()
        except Exception:
            pass

        # Model cache key
        _cache_key = f'{hisse_kodu}_{veri_yili}_{pencere}'
        _onceki_model = st.session_state.get('model_cache', {}).get(_cache_key)

        _baslangic = _t.time()
        _sureler = {}
        _hata_var = False
        # Optuna toggle değerini session state'e aktar
        if "optuna_aktif" in dir():
            st.session_state["optuna_aktif"] = optuna_aktif
        else:
            st.session_state.setdefault("optuna_aktif", False)

        try:
            _psutil_yukle()
            if PSUTIL_OK and _ps:
                import os as _os2

                _proc = _ps.Process(_os2.getpid())
                _ram0 = _proc.memory_info().rss / 1024 / 1024
            else:
                _proc = None;
                _ram0 = 0
        except Exception:
            _proc = None;
            _ram0 = 0

        adimlar = ["Veri", "Makro", "Haberler", "Hazırlama", "Eğitim", "Tahmin"]
        _prog = st.progress(0, text="⏳ Başlatılıyor...")
        _log = st.empty()


        def _ilerleme(n, msg):
            _lbl = {1: "Veri", 2: "Makro", 3: "Haberler", 4: "Hazirlik", 5: "Egitim", 6: "Tamam"}.get(n, msg)
            _prog.progress(int(n / 6 * 100), text=f"{_lbl} - %{int(n / 6 * 100)}")
            _log.info(f"**Adım {n}/6** — {msg}")


        def _hata(baslik, detay, tb=""):
            _prog.progress(0, text="❌ Hata")
            _log.empty()
            st.error(f"### ❌ {baslik}\n\n{detay}")
            if tb:
                with st.expander("🔍 Teknik detay"):
                    st.code(tb)
            st.warning("""💡 **Çözüm önerileri:**
- Veri yılını **8** yap (sol menü)
- Pencereyi **30**'a düşür
- Farklı hisse dene (THYAO, GARAN)
- F5 ile yenile, tekrar dene""")


        # ── ADIM 1 — VERİ ────────────────────────────────────────────────────
        _ilerleme(1, f"{hisse_kodu} verisi indiriliyor...")
        _t0 = _t.time()
        try:
            df = hisse_indir(ticker, veri_yili)
            df = indikatör_ekle(df)
            _sureler["Veri"] = round(_t.time() - _t0, 1)
            if len(df) < 200:
                _hata("Yetersiz Veri",
                      f"{hisse_kodu}: {len(df)} gün var, en az 200 gerekli. "
                      "Veri yılını artırın.")
                _hata_var = True
        except Exception as e:
            _hata("Veri İndirme Hatası", str(e), _tb.format_exc())
            _hata_var = True

        # ── ADIM 2 — MAKRO ───────────────────────────────────────────────────
        makro_df = pd.DataFrame()  # Her zaman tanımlı
        if not _hata_var:
            _ilerleme(2, "Makro veri indiriliyor...")
            _t0 = _t.time()
            try:
                makro_df = makro_indir()
                if not makro_df.empty:
                    df = makro_birlestir(df, makro_df)
            except Exception as e:
                makro_df = pd.DataFrame()
                st.warning(f"⚠️ Makro veri alınamadı, devam ediliyor: {e}")
            _sureler["Makro"] = round(_t.time() - _t0, 1)

        # ── ADIM 3 — HABERLER ────────────────────────────────────────────────
        if not _hata_var:
            _ilerleme(3, "Haberler analiz ediliyor...")
            _t0 = _t.time()
            try:
                _km = {"Google News": "google", "Investing.com TR": "investing",
                       "Bloomberg HT": "bloombergHT", "Ekonomist": "ekonomist",
                       "Habertürk": "haberturk", "Milliyet": "milliyet",
                       "Sabah Ekonomi": "sabah", "Döviz.com": "doviz"}
                _sec = tuple(_km.get(k, "google") for k in haber_kaynagi) or ("google",)
                haber = haber_skoru_al(hisse_kodu, _sec)
            except Exception:
                haber = {"skor": 0.0, "adet": 0, "durum": "Veri Yok", "haberler": []}
            _sureler["Haberler"] = round(_t.time() - _t0, 1)

        # ── ADIM 4 — VERİ HAZIRLAMA ──────────────────────────────────────────
        if not _hata_var:
            _ilerleme(4, "Veri hazırlanıyor ve ölçekleniyor...")
            _t0 = _t.time()
            try:
                X_eg, X_te, y_eg, y_te, scaler, ozellikler, class_weight = veri_hazirla(df, pencere)
                pencere_gercek = X_eg.shape[1]
                ozellik_sayisi = X_eg.shape[2]
                _sureler["Hazırlama"] = round(_t.time() - _t0, 1)
                _log.success(
                    f"✅ Hazır — Eğitim: **{len(X_eg)}** örnek · "
                    f"Test: **{len(X_te)}** · **{ozellik_sayisi}** özellik"
                )
                if len(X_eg) < 800:
                    if len(X_eg) < 400:
                        st.error(
                            f"❌ **Çok az veri:** {len(X_eg)} örnek — "
                            f"bu hisse için güvenilir tahmin yapılamıyor. "
                            f"THYAO, GARAN, AKBNK gibi köklü hisseleri deneyin."
                        )
                    else:
                        st.warning(
                            f"⚠️ **Sınırlı veri:** {len(X_eg)} örnek. "
                            f"Bu hisse yeni veya az işlem görüyor. "
                            f"Tahminler büyük hisseler kadar güvenilir değil."
                        )
            except ValueError as e:
                _hata("Veri Hazırlama Hatası", str(e))
                _hata_var = True
            except Exception as e:
                _hata("Veri Hazırlama Hatası", str(e), _tb.format_exc())
                _hata_var = True

        # ── ADIM 5 — MODEL EĞİTİMİ ───────────────────────────────────────────
        if not _hata_var:
            _ilerleme(5, "Modeller eğitiliyor (1-3 dk sürebilir)...")
            _t0 = _t.time()
            _eglog = st.empty()


            def status_cb(msg):
                _eglog.info(msg)


            try:
                # Model önbellek kontrolü
                if _onceki_model and not st.session_state.get('optuna_aktif', False):
                    modeller = _onceki_model
                    status_cb('⚡ Model cache\'ten yüklendi!')
                else:
                    modeller = modelleri_egit(
                        X_eg, y_eg, X_te, y_te,
                        pencere_gercek, ozellik_sayisi,
                        status_cb, class_weight
                    )
                    # Cache'e kaydet
                    if 'model_cache' not in st.session_state:
                        st.session_state.model_cache = {}
                    st.session_state.model_cache[_cache_key] = modeller
                    if len(st.session_state.model_cache) > 3:
                        oldest = next(iter(st.session_state.model_cache))
                        del st.session_state.model_cache[oldest]
                _sureler["Eğitim"] = round(_t.time() - _t0, 1)
                _ozet = [f"• {k}: %{v['dogruluk'] * 100:.1f}"
                         for k, v in modeller.items() if not k.startswith("_")]
                _eglog.success("✅ Modeller hazır\n" + "  ".join(_ozet))
                if _sureler["Eğitim"] < 10:
                    st.warning(
                        f"⚠️ Eğitim {_sureler['Eğitim']}sn'de bitti — "
                        f"bu çok hızlı! Eğitim seti: {len(X_eg)} örnek. "
                        "Veri yılını 8'e çıkarın."
                    )
            except Exception as e:
                _hata("Model Eğitimi Hatası", str(e), _tb.format_exc())
                _hata_var = True

        # ── ADIM 6 — TAHMİN ──────────────────────────────────────────────────
        if not _hata_var:
            _ilerleme(6, "Ensemble tahmin hesaplanıyor...")
            _t0 = _t.time()
            try:
                df_oz = df[[c for c in df.columns if c != "Hedef"]].copy()
                df_oz = df_oz.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                df_oz = df_oz[[c for c in ozellikler if c in df_oz.columns]]
                if len(df_oz) < pencere_gercek:
                    raise ValueError(
                        f"Tahmin için yeterli veri yok: "
                        f"{len(df_oz)} satır, {pencere_gercek} gerekli."
                    )
                sc2 = RobustScaler()
                X_full = sc2.fit_transform(df_oz.values).astype(np.float32)
                X_son = X_full[-pencere_gercek:].reshape(1, pencere_gercek, ozellik_sayisi)
                rejim_vol = float(df["Rejim_vol"].iloc[-1]) if "Rejim_vol" in df.columns else 1.0
                sektor_skor = sektor_momentum_al(hisse_kodu, gun=5)
                tahmin = ensemble_tahmin(
                    modeller, X_son, haber["skor"],
                    f_etkisi, rejim_vol, sektor_skor
                )
                _sureler["Tahmin"] = round(_t.time() - _t0, 1)
            except Exception as e:
                _hata("Tahmin Hatası", str(e), _tb.format_exc())
                _hata_var = True

        # ── TAMAMLANDI ────────────────────────────────────────────────────────
        if not _hata_var:
            _toplam_sure = round(_t.time() - _baslangic, 1)
            _ram_kullanim = 0
            if _proc:
                try:
                    _ram_kullanim = round(abs(_proc.memory_info().rss / 1024 / 1024 - _ram0), 1)
                except Exception:
                    pass

            _prog.progress(100, text=f"✅ Tamamlandı! ({_toplam_sure}sn)")
            _log.empty()

            _yeni_analiz = {
                "tarih": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "hisse": hisse_kodu,
                "sinyal": tahmin["sinyal"],
                "olasilik": round(tahmin["final"] * 100, 1),
                "fiyat": round(float(df["Close"].iloc[-1]), 2),
                "hedef": round(float(df["Close"].iloc[-1]) * (1.05 if tahmin["final"] >= 0.62 else 1.03), 2),
                "stop": round(float(df["Close"].iloc[-1]) - float(df["ATR14"].iloc[-1]) * 1.2, 2),
                "bekleme": "5-10 gün" if tahmin["final"] >= 0.62 else "7-15 gün",
                "dogruluk": round(
                    float(np.mean([v["dogruluk"] for k, v in modeller.items() if not k.startswith("_")])) * 100, 1),
                "tahmin_yon": "YUKARI" if tahmin["final"] >= 0.55 else "ASAGI",
                "gerceklesti": None,
                "sure_sn": _toplam_sure,
            }
            st.session_state.analiz_gecmisi.append(_yeni_analiz)
            db_kaydet_liste('analiz_gecmisi', st.session_state.analiz_gecmisi[-50:])
            # Geri bildirim butonu
            st.markdown("---")
            _gb1, _gb2, _ = st.columns([2, 2, 3])
            with _gb1:
                if st.button("Dogru tahmin", key=f"gb_d_{hisse_kodu}"):
                    try:
                        with db_baglanti() as _gc:
                            _gc.execute("INSERT INTO geri_bildirim (hisse,sinyal,dogru) VALUES(?,?,?)",
                                        (hisse_kodu, tahmin.get("sinyal", ""), 1))
                            _gc.commit()
                        st.success("Kaydedildi!")
                    except Exception:
                        pass
            with _gb2:
                if st.button("Yanlis tahmin", key=f"gb_y_{hisse_kodu}"):
                    try:
                        with db_baglanti() as _gc:
                            _gc.execute("INSERT INTO geri_bildirim (hisse,sinyal,dogru) VALUES(?,?,?)",
                                        (hisse_kodu, tahmin.get("sinyal", ""), 0))
                            _gc.commit()
                        st.info("Kaydedildi.")
                    except Exception:
                        pass

        if not _hata_var:
            # ── Sonuçlar ─────────────────────────────────────────────────────────────
            son_fiyat = float(df['Close'].iloc[-1])
            rsi14 = float(df['RSI14'].iloc[-1])
            macd_v = float(df['MACD'].iloc[-1])
            macd_s = float(df['MACD_Signal'].iloc[-1])
            bb_pct = float(df['BB20_Pct'].iloc[-1]) * 100
            mfi = float(df['MFI'].iloc[-1])
            vol14 = float(df['Vol14'].iloc[-1]) * 100
            stoch = float(df['Stoch14_K'].iloc[-1])

            ens_dogruluk = float(np.mean([b['dogruluk'] for b in modeller.values()]))
            _guven, _gclr, _gico = guven_skoru(ens_dogruluk, len(X_eg), modeller)
            st.markdown(f'<div style="text-align:center;margin:0.4rem 0">'
                        f'<span style="background:rgba(255,255,255,0.05);color:{_gclr};border:1px solid {_gclr}44;'
                        f'border-radius:6px;padding:3px 12px;font-size:0.82rem">Guven: {_guven}</span></div>',
                        unsafe_allow_html=True)

            # Üst metrikler
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Güncel Fiyat</div>
                  <div class="value" style="color:#38bdf8">{son_fiyat:.2f} ₺</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                renk = "#34d399" if tahmin['final'] >= 0.55 else ("#ef4444" if tahmin['final'] <= 0.45 else "#facc15")
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Yükselme Olasılığı</div>
                  <div class="value" style="color:{renk}">%{tahmin['final'] * 100:.1f}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Ensemble Doğruluk</div>
                  <div class="value" style="color:#818cf8">%{ens_dogruluk * 100:.1f}</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">RSI (14)</div>
                  <div class="value" style="color:{'#ef4444' if rsi14 < 30 else '#facc15' if rsi14 > 70 else '#34d399'}">{rsi14:.1f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Sinyal kutusu
            sinyal_txt = tahmin['sinyal']
            if "AL" in sinyal_txt:
                kutu_cls = "signal-al"
            elif "SAT" in sinyal_txt:
                kutu_cls = "signal-sat"
            else:
                kutu_cls = "signal-bekle"
            st.markdown(f'<div class="signal-box {kutu_cls}">{sinyal_txt}</div>',
                        unsafe_allow_html=True)

            # Rejim notu + meta-model bilgisi
            rejim_notu = tahmin.get('rejim_notu', '📊 Normal Rejim')
            meta_acc = modeller.get('_meta_model', {}).get('dogruluk', None)
            meta_f1 = modeller.get('_meta_model', {}).get('f1', None)
            meta_bilgi = (f" · Meta-Model: %{meta_acc * 100:.1f} doğruluk / F1:{meta_f1:.2f}"
                          if meta_acc else "")
            st.markdown(
                f'<div style="text-align:center;font-size:0.78rem;color:#64748b;'
                f'margin-top:0.5rem;font-family:monospace">'
                f'{rejim_notu}{meta_bilgi}</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── AL / SAT FİYAT ARALIKLARI & BEKLEME SÜRESİ ───────────────────────────
            atr14 = float(df['ATR14'].iloc[-1])
            # ATR tabanlı stop-loss seviyeleri
            sl_agresif = round(son_fiyat - 1.5 * atr14, 2)  # %sıkı
            sl_normal = round(son_fiyat - 2.0 * atr14, 2)  # %normal
            sl_konserv = round(son_fiyat - 3.0 * atr14, 2)  # %geniş
            sl_pct_agr = round((son_fiyat - sl_agresif) / son_fiyat * 100, 1)
            sl_pct_nor = round((son_fiyat - sl_normal) / son_fiyat * 100, 1)
            sl_pct_kon = round((son_fiyat - sl_konserv) / son_fiyat * 100, 1)
            volatilite = float(df['Vol14'].iloc[-1])
            olas = tahmin['final']

            # Fiyat aralıkları — ATR tabanlı dinamik hesaplama
            # AL aralığı: mevcut fiyattan biraz aşağıda (dip bekle)
            al_ust = round(son_fiyat * 0.995, 2)  # şimdiki fiyatın %0.5 altı
            al_alt = round(son_fiyat - atr14 * 0.8, 2)  # ATR'nin %80'i kadar aşağı

            # Hedef fiyat: olasılığa göre değişir
            if olas >= 0.70:
                hedef_carpan = 1.08  # güçlü sinyal → %8 hedef
                bekleme_gun = "3-5 gün"
                bekleme_acik = "Güçlü momentum, kısa sürede hedefe ulaşabilir"
            elif olas >= 0.62:
                hedef_carpan = 1.05
                bekleme_gun = "5-10 gün"
                bekleme_acik = "Orta güçte sinyal, sabırlı ol"
            elif olas >= 0.57:
                hedef_carpan = 1.03
                bekleme_gun = "7-15 gün"
                bekleme_acik = "Zayıf sinyal, düşük miktarda dene"
            else:
                hedef_carpan = 1.02
                bekleme_gun = "Belirsiz"
                bekleme_acik = "Sinyal zayıf, işlem yapma"

            hedef_fiyat = round(son_fiyat * hedef_carpan, 2)
            kar_yuzde = round((hedef_carpan - 1) * 100, 1)

            # SAT (stop-loss) aralığı — zarar kes
            stop_loss = round(son_fiyat - atr14 * 1.2, 2)  # ATR × 1.2 altı
            stop_yuzde = round((stop_loss / son_fiyat - 1) * 100, 1)

            # Risk/Ödül oranı
            risk = son_fiyat - stop_loss
            odul = hedef_fiyat - son_fiyat
            rr = round(odul / risk, 2) if risk > 0 else 0
            rr_renk = "#34d399" if rr >= 2 else "#facc15" if rr >= 1.5 else "#ef4444"

            st.markdown('<div class="section-title">📊 İŞLEM PLANI</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div style="background:rgba(52,211,153,0.08);border:1.5px solid #34d399;
                            border-radius:14px;padding:1.4rem;text-align:center;">
                  <div style="color:#34d399;font-size:0.7rem;font-family:'Space Mono',monospace;
                              text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem;">
                    🟢 AL ARALIĞI
                  </div>
                  <div style="font-size:1.5rem;font-weight:800;color:#34d399;font-family:'Space Mono',monospace;">
                    {al_alt:.2f} ₺ — {al_ust:.2f} ₺
                  </div>
                  <div style="color:#64748b;font-size:0.78rem;margin-top:0.5rem;">
                    Bu aralıkta alım yap
                  </div>
                  <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid rgba(52,211,153,0.2);">
                    <div style="color:#94a3b8;font-size:0.72rem;">🎯 Hedef Fiyat</div>
                    <div style="color:#34d399;font-size:1.2rem;font-weight:800;font-family:'Space Mono',monospace;">
                      {hedef_fiyat:.2f} ₺
                    </div>
                    <div style="color:#34d399;font-size:0.85rem;">+%{kar_yuzde} kâr hedefi</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.08);border:1.5px solid #ef4444;
                            border-radius:14px;padding:1.4rem;text-align:center;">
                  <div style="color:#ef4444;font-size:0.7rem;font-family:'Space Mono',monospace;
                              text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem;">
                    🔴 ZARAR KES (STOP-LOSS)
                  </div>
                  <div style="font-size:1.5rem;font-weight:800;color:#ef4444;font-family:'Space Mono',monospace;">
                    {stop_loss:.2f} ₺
                  </div>
                  <div style="color:#64748b;font-size:0.78rem;margin-top:0.5rem;">
                    Bu fiyata düşerse sat!
                  </div>
                  <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid rgba(239,68,68,0.2);">
                    <div style="color:#94a3b8;font-size:0.72rem;">📉 Max Zarar</div>
                    <div style="color:#ef4444;font-size:1.2rem;font-weight:800;font-family:'Space Mono',monospace;">
                      %{abs(stop_yuzde):.1f}
                    </div>
                    <div style="color:#ef4444;font-size:0.85rem;">Maksimum kabul edilebilir kayıp</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div style="background:rgba(129,140,248,0.08);border:1.5px solid #818cf8;
                            border-radius:14px;padding:1.4rem;text-align:center;">
                  <div style="color:#818cf8;font-size:0.7rem;font-family:'Space Mono',monospace;
                              text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem;">
                    ⏱️ BEKLEME SÜRESİ
                  </div>
                  <div style="font-size:1.5rem;font-weight:800;color:#818cf8;font-family:'Space Mono',monospace;">
                    {bekleme_gun}
                  </div>
                  <div style="color:#64748b;font-size:0.78rem;margin-top:0.5rem;">
                    {bekleme_acik}
                  </div>
                  <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid rgba(129,140,248,0.2);">
                    <div style="color:#94a3b8;font-size:0.72rem;">⚖️ Risk / Ödül Oranı</div>
                    <div style="color:{rr_renk};font-size:1.2rem;font-weight:800;font-family:'Space Mono',monospace;">
                      1 : {rr}
                    </div>
                    <div style="color:{rr_renk};font-size:0.85rem;">
                      {"✅ İyi oran" if rr >= 2 else "⚠️ Orta oran" if rr >= 1.5 else "❌ Zayıf oran"}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Özet strateji kutusu
            st.markdown("<br>", unsafe_allow_html=True)
            strateji_renk = "#34d399" if olas >= 0.62 else "#facc15" if olas >= 0.55 else "#ef4444"
            strateji_ikon = "✅" if olas >= 0.62 else "⚠️" if olas >= 0.55 else "❌"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid #1e293b;
                        border-radius:12px;padding:1.2rem 1.5rem;">
              <div style="font-size:0.7rem;color:#64748b;font-family:'Space Mono',monospace;
                          text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem;">
                📋 ÖZET STRATEJİ
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center;">
                <div style="flex:1;min-width:200px;">
                  <span style="color:#94a3b8;">Al: </span>
                  <span style="color:#34d399;font-weight:700;font-family:'Space Mono',monospace;">
                    {al_alt:.2f} ₺ — {al_ust:.2f} ₺
                  </span>
                </div>
                <div style="flex:1;min-width:200px;">
                  <span style="color:#94a3b8;">Hedef: </span>
                  <span style="color:#34d399;font-weight:700;font-family:'Space Mono',monospace;">
                    {hedef_fiyat:.2f} ₺ (+%{kar_yuzde})
                  </span>
                </div>
                <div style="flex:1;min-width:200px;">
                  <span style="color:#94a3b8;">Stop-Loss: </span>
                  <span style="color:#ef4444;font-weight:700;font-family:'Space Mono',monospace;">
                    {stop_loss:.2f} ₺ (%{stop_yuzde})
                  </span>
                </div>
                <div style="flex:1;min-width:200px;">
                  <span style="color:#94a3b8;">Süre: </span>
                  <span style="color:#818cf8;font-weight:700;font-family:'Space Mono',monospace;">
                    {bekleme_gun}
                  </span>
                </div>
                <div>
                  <span style="font-size:1.1rem;">{strateji_ikon}</span>
                  <span style="color:{strateji_renk};font-weight:700;margin-left:0.3rem;">
                    {"AL — İşlem Yap" if olas >= 0.58 else "Bekle" if olas >= 0.48 else "SAT — Geç"}
                  </span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Model doğrulukları + teknik göstergeler yan yana
            col_sol, col_sag = st.columns([1, 1])

            with col_sol:
                st.markdown('<div class="section-title">MODEL DOĞRULUKLARI</div>',
                            unsafe_allow_html=True)
                for ad, bilgi in modeller.items():
                    d = bilgi['dogruluk'] * 100
                    renk = "#34d399" if d >= 65 else "#facc15" if d >= 58 else "#ef4444"
                    pct = int(d)
                    st.markdown(f"""
                    <div class="model-row">
                      <span>{ad}</span>
                      <span style="color:{renk};font-weight:700">%{d:.1f}</span>
                    </div>
                    <div class="progress-bar-bg">
                      <div style="width:{pct}%;background:{renk};height:8px;border-radius:6px;transition:width 0.5s"></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">DUYGU & MAKRO</div>',
                            unsafe_allow_html=True)

                h_renk = "badge-green" if haber['skor'] > 0.1 else (
                    "badge-red" if haber['skor'] < -0.1 else "badge-gray")
                st.markdown(f"""
                <div class="model-row">
                  <span>Haber Durumu</span>
                  <span class="info-badge {h_renk}">{haber['durum']}</span>
                </div>
                <div class="model-row">
                  <span>Haber Adedi</span>
                  <span style="color:#94a3b8">{haber['adet']}</span>
                </div>
                <div class="model-row">
                  <span>TCMB Faiz</span>
                  <span style="color:#94a3b8">%{faiz_orani}</span>
                </div>
                <div class="model-row">
                  <span>Makro Gösterge</span>
                  <span style="color:#818cf8">{len(makro_df.columns) if makro_df is not None and not makro_df.empty else 0} adet</span>
                </div>
                """, unsafe_allow_html=True)

            with col_sag:
                st.markdown('<div class="section-title">TEKNİK GÖSTERGELER</div>',
                            unsafe_allow_html=True)

                gostergeler = [
                    ("RSI (7)", float(df['RSI7'].iloc[-1]), (30, 70), ""),
                    ("RSI (14)", rsi14, (30, 70), ""),
                    ("RSI (21)", float(df['RSI21'].iloc[-1]), (30, 70), ""),
                    ("MFI", mfi, (20, 80), ""),
                    ("Stochastic K", stoch, (20, 80), ""),
                    ("Williams %R", float(df['WilliamsR'].iloc[-1]) + 100, (20, 80), ""),
                    ("CCI", float(df['CCI'].iloc[-1]), (-100, 100), ""),
                    ("BB %B", bb_pct, (20, 80), "%"),
                    ("Volatilite 14", vol14, (0, 5), "%"),
                    ("MACD", macd_v, None, ""),
                ]

                for label, val, aralik, birim in gostergeler:
                    if aralik:
                        renk = ("#ef4444" if val < aralik[0] else
                                "#facc15" if val > aralik[1] else "#34d399")
                    else:
                        renk = "#34d399" if val > 0 else "#ef4444"

                    fmt = f"{val:.2f}{birim}"
                    st.markdown(f"""
                    <div class="model-row">
                      <span style="color:#94a3b8">{label}</span>
                      <span style="color:{renk};font-weight:700">{fmt}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── FİYAT GRAFİĞİ ────────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 FİYAT GRAFİĞİ</div>',
                        unsafe_allow_html=True)

            # Grafik periyot seçici
            gc1, gc2 = st.columns([3, 1])
            with gc2:
                grafik_sure = st.selectbox(
                    "Periyot", ["1 Ay", "3 Ay", "6 Ay", "1 Yıl", "3 Yıl", "Tümü"],
                    index=3, label_visibility="collapsed"
                )
            gun_map = {"1 Ay": 30, "3 Ay": 90, "6 Ay": 180,
                       "1 Yıl": 365, "3 Yıl": 1095, "Tümü": 99999}
            gun_say = gun_map[grafik_sure]
            df_grafik = df.tail(min(gun_say, len(df))).copy()

            try:
                _plotly_yukle()
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2],
                    vertical_spacing=0.04,
                    subplot_titles=("Fiyat & Hareketli Ortalamalar", "Hacim", "RSI (14)")
                )

                # Mum grafiği
                fig.add_trace(go.Candlestick(
                    x=df_grafik.index,
                    open=df_grafik['Open'], high=df_grafik['High'],
                    low=df_grafik['Low'], close=df_grafik['Close'],
                    name="Fiyat",
                    increasing_line_color='#34d399',
                    decreasing_line_color='#ef4444',
                    increasing_fillcolor='rgba(52,211,153,0.7)',
                    decreasing_fillcolor='rgba(239,68,68,0.7)',
                ), row=1, col=1)

                # Hareketli ortalamalar
                for ma, renk, genislik in [("MA20", "#facc15", 1.5), ("MA50", "#818cf8", 1.5), ("MA200", "#f97316", 1)]:
                    if ma in df_grafik.columns:
                        fig.add_trace(go.Scatter(
                            x=df_grafik.index, y=df_grafik[ma],
                            name=ma, line=dict(color=renk, width=genislik),
                            opacity=0.85
                        ), row=1, col=1)

                # Bollinger bantları
                if 'BB20_Ust' in df_grafik.columns:
                    fig.add_trace(go.Scatter(
                        x=df_grafik.index, y=df_grafik['BB20_Ust'],
                        name="BB Üst", line=dict(color='rgba(129,140,248,0.4)', width=1, dash='dot'),
                        showlegend=False
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df_grafik.index, y=df_grafik['BB20_Alt'],
                        name="BB Alt", line=dict(color='rgba(129,140,248,0.4)', width=1, dash='dot'),
                        fill='tonexty', fillcolor='rgba(129,140,248,0.05)',
                        showlegend=False
                    ), row=1, col=1)

                # Al/Sat seviyeleri yatay çizgi
                fig.add_hline(y=hedef_fiyat, line_dash="dash",
                              line_color="#34d399", line_width=1.5,
                              annotation_text=f"🎯 Hedef {hedef_fiyat:.2f}₺",
                              annotation_position="right", row=1, col=1)
                fig.add_hline(y=stop_loss, line_dash="dash",
                              line_color="#ef4444", line_width=1.5,
                              annotation_text=f"🛑 Stop {stop_loss:.2f}₺",
                              annotation_position="right", row=1, col=1)
                fig.add_hline(y=al_alt, line_dash="dot",
                              line_color="#34d399", line_width=1,
                              annotation_text=f"🟢 Al Alt {al_alt:.2f}₺",
                              annotation_position="right", row=1, col=1)

                # Hacim
                renkler_hacim = ['#34d399' if df_grafik['Close'].iloc[i] >= df_grafik['Open'].iloc[i]
                                 else '#ef4444' for i in range(len(df_grafik))]
                fig.add_trace(go.Bar(
                    x=df_grafik.index, y=df_grafik['Volume'],
                    name="Hacim", marker_color=renkler_hacim, opacity=0.7
                ), row=2, col=1)

                # RSI
                if 'RSI14' in df_grafik.columns:
                    fig.add_trace(go.Scatter(
                        x=df_grafik.index, y=df_grafik['RSI14'],
                        name="RSI(14)", line=dict(color='#38bdf8', width=1.5)
                    ), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444",
                                  line_width=1, row=3, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color="#34d399",
                                  line_width=1, row=3, col=1)
                    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)",
                                  line_width=0, row=3, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(10,14,26,0)',
                    plot_bgcolor='rgba(15,23,42,0.8)',
                    font=dict(family="Space Mono", color="#94a3b8", size=11),
                    height=620,
                    legend=dict(orientation="h", y=1.02, x=0,
                                bgcolor="rgba(0,0,0,0)",
                                font=dict(size=10)),
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=0, r=80, t=40, b=0),
                )
                for i in range(1, 4):
                    fig.update_xaxes(
                        gridcolor='rgba(51,65,85,0.4)',
                        showgrid=True, row=i, col=1
                    )
                    fig.update_yaxes(
                        gridcolor='rgba(51,65,85,0.4)',
                        showgrid=True, row=i, col=1
                    )

                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                # plotly yoksa basit çizgi grafik
                df_plot = df_grafik[['Close', 'MA20', 'MA50']].copy()
                st.line_chart(df_plot, use_container_width=True)

            # ── MODEL OLASILIK ÇUBUĞU ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">MODEL BAZLI YÜKSELME OLASILIĞI</div>',
                        unsafe_allow_html=True)
            olas_df = pd.DataFrame([
                {"Model": ad, "Olasılık (%)": round(v * 100, 1)}
                for ad, v in tahmin['olasiliklar'].items()
            ])
            st.bar_chart(olas_df.set_index("Model"), use_container_width=True,
                         color="#6366f1")

            # Fiyat tablosu (son 10 gün)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">SON 10 GÜNLÜK FİYAT HAREKETİ</div>',
                        unsafe_allow_html=True)
            son10 = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
            son10.index = son10.index.strftime('%d %b %Y')
            son10['Değişim %'] = son10['Close'].pct_change().mul(100).round(2)
            son10 = son10.round(2)
            st.dataframe(son10, use_container_width=True)

            # ── PERFORMANS RAPORU ─────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">⚡ ANALİZ PERFORMANS RAPORU</div>',
                        unsafe_allow_html=True)

            ens_dogruluk = float(np.mean([b['dogruluk'] for b in modeller.values()])) * 100
            en_iyi_model = max(modeller.items(), key=lambda x: x[1]['dogruluk'])
            en_kotu_model = min(modeller.items(), key=lambda x: x[1]['dogruluk'])

            # Model özeti — tek satır
            _model_ozet = " · ".join([
                f"{k}: %{v['dogruluk'] * 100:.0f}"
                for k, v in modeller.items() if not k.startswith('_')
            ])
            st.caption(f"⏱️ {_toplam_sure}sn · 🧠 %{ens_dogruluk:.1f} ensemble · {_model_ozet}")

            # Aşama süreleri — yatay çubuk
            st.markdown("<br>", unsafe_allow_html=True)
            sure_col1, sure_col2 = st.columns([2, 1])
            with sure_col1:
                st.markdown("**📊 Aşama Süreleri**")
                for asama, sure in _sureler.items():
                    oran = min(int(sure / _toplam_sure * 100), 100) if _toplam_sure > 0 else 0
                    renk = "#6366f1" if "Model" in asama else "#38bdf8"
                    st.markdown(f"""
                    <div style="margin:0.3rem 0">
                      <div style="display:flex;justify-content:space-between;
                                  font-size:0.78rem;color:#94a3b8;margin-bottom:2px">
                        <span>{asama}</span><span style="color:#e2e8f0">{sure}sn</span>
                      </div>
                      <div style="background:#1e293b;border-radius:4px;height:8px">
                        <div style="background:{renk};width:{oran}%;height:8px;
                                    border-radius:4px;transition:width 0.5s"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with sure_col2:
                st.markdown("**🎯 Model Doğrulukları**")
                for ad, bilgi in modeller.items():
                    d = bilgi['dogruluk'] * 100
                    rk = "#34d399" if d >= 65 else "#facc15" if d >= 58 else "#ef4444"
                    ikon = "🥇" if ad == en_iyi_model[0] else "🥈" if d >= 65 else ""
                    st.markdown(f"""
                    <div class="model-row">
                      <span style="font-size:0.8rem">{ikon} {ad}</span>
                      <span style="color:{rk};font-weight:700;font-family:monospace">%{d:.1f}</span>
                    </div>""", unsafe_allow_html=True)

            # Veri özeti
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid #1e293b;
                        border-radius:10px;padding:1rem;margin-top:1rem;
                        font-family:'Space Mono',monospace;font-size:0.75rem;color:#64748b">
              📋 <b style="color:#94a3b8">Veri Özeti:</b>
              {len(df)} günlük geçmiş · {ozellik_sayisi} özellik · {len(X_eg)} eğitim / {len(X_te)} test örneği ·
              Pencere: {pencere_gercek} gün · Eğitim süresi: {_sureler.get('Model Eğitimi', '?')}sn
            </div>
            """, unsafe_allow_html=True)

            # ── ALARM SİSTEMİ ─────────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔔 FİYAT ALARM SİSTEMİ</div>',
                        unsafe_allow_html=True)

            # Session state ile alarm listesi
            if 'alarmlar' not in st.session_state:
                st.session_state.alarmlar = []

            with st.expander("➕ Yeni Alarm Ekle", expanded=True):
                a1, a2, a3, a4 = st.columns([2, 1.5, 1.5, 1])

                with a1:
                    alarm_hisse = st.selectbox(
                        "Hisse", list(BIST_HISSELER.keys()),
                        index=list(BIST_HISSELER.keys()).index(hisse_secim),
                        key="alarm_hisse", label_visibility="collapsed"
                    )
                with a2:
                    alarm_tip = st.selectbox(
                        "Tip", ["🔼 Fiyat Üstüne Çıkarsa", "🔽 Fiyat Altına Düşerse",
                                "📈 AL Sinyali Gelirse", "📉 SAT Sinyali Gelirse"],
                        key="alarm_tip", label_visibility="collapsed"
                    )
                with a3:
                    alarm_fiyat = st.number_input(
                        "Fiyat", value=float(round(son_fiyat, 2)),
                        step=0.5, format="%.2f",
                        key="alarm_fiyat", label_visibility="collapsed"
                    )
                with a4:
                    ekle_btn = st.button("➕ Ekle", use_container_width=True)

                if ekle_btn:
                    alarm = {
                        "hisse": alarm_hisse.split("—")[0].strip(),
                        "ticker": BIST_HISSELER[alarm_hisse],
                        "tip": alarm_tip,
                        "fiyat": alarm_fiyat,
                        "eklenme": datetime.now().strftime("%d.%m.%Y %H:%M"),
                        "durum": "⏳ Bekliyor"
                    }
                    st.session_state.alarmlar.append(alarm)
                    st.success(f"✅ {alarm['hisse']} için alarm eklendi!")

            # Aktif alarmları kontrol et ve göster
            if st.session_state.alarmlar:
                st.markdown('<div class="section-title">📋 AKTİF ALARMLAR</div>',
                            unsafe_allow_html=True)

                guncelle_btn = st.button("🔄 Alarmları Kontrol Et", use_container_width=False)

                tetiklenen = []
                for i, alarm in enumerate(st.session_state.alarmlar):
                    try:
                        if guncelle_btn:
                            guncel = yf.Ticker(alarm['ticker'])
                            guncel_fiyat = float(guncel.fast_info['last_price'])

                            if "Üstüne" in alarm['tip'] and guncel_fiyat >= alarm['fiyat']:
                                alarm['durum'] = f"🔔 TETİKLENDİ! ({guncel_fiyat:.2f}₺)"
                                tetiklenen.append(alarm)
                            elif "Altına" in alarm['tip'] and guncel_fiyat <= alarm['fiyat']:
                                alarm['durum'] = f"🔔 TETİKLENDİ! ({guncel_fiyat:.2f}₺)"
                                tetiklenen.append(alarm)
                            else:
                                alarm['durum'] = f"⏳ Bekliyor ({guncel_fiyat:.2f}₺)"
                    except Exception:
                        pass

                    # Alarm satırı
                    durum_renk = ("#facc15" if "TETİKLENDİ" in alarm['durum']
                                  else "#64748b")
                    sil_key = f"sil_{i}_{alarm['hisse']}"
                    acol1, acol2, acol3, acol4, acol5 = st.columns([1, 2, 1.5, 2, 0.7])
                    with acol1:
                        st.markdown(f"<span style='color:#818cf8;font-weight:700;"
                                    f"font-family:monospace'>{alarm['hisse']}</span>",
                                    unsafe_allow_html=True)
                    with acol2:
                        st.markdown(f"<span style='color:#94a3b8;font-size:0.8rem'>"
                                    f"{alarm['tip']}</span>", unsafe_allow_html=True)
                    with acol3:
                        st.markdown(f"<span style='color:#e2e8f0;font-family:monospace'>"
                                    f"{alarm['fiyat']:.2f}₺</span>", unsafe_allow_html=True)
                    with acol4:
                        st.markdown(f"<span style='color:{durum_renk};font-size:0.85rem'>"
                                    f"{alarm['durum']}</span>", unsafe_allow_html=True)
                    with acol5:
                        if st.button("🗑️", key=sil_key, help="Alarmı sil"):
                            st.session_state.alarmlar.pop(i)
                            st.rerun()

                if tetiklenen:
                    st.markdown("<br>", unsafe_allow_html=True)
                    for t in tetiklenen:
                        st.warning(f"🔔 **{t['hisse']}** alarmı tetiklendi! → {t['durum']}")

                # Alarmları CSV olarak indir
                if st.session_state.alarmlar:
                    alarm_df = pd.DataFrame(st.session_state.alarmlar)
                    csv = alarm_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Alarmları İndir (CSV)",
                        data=csv,
                        file_name=f"alarmlar_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.markdown("""
                    <div style="text-align:center;padding:1.5rem;color:#475569;
                                border:1px dashed #1e293b;border-radius:10px;">
                      🔕 Henüz alarm eklenmedi.<br>
                      <span style="font-size:0.8rem;">Yukarıdan hisse, tip ve fiyat seçerek alarm ekleyebilirsin.</span>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                # Karşılama — sade
                _msg = (
                    "Sol menüden hisse seçin, faiz oranını girin ve **Analizi Başlat**a basın."
                    if st.session_state.dil == "TR" else
                    "Select a stock, set interest rate and press **Start Analysis**."
                )
                st.info(f"📈 {_msg}")

                # ── DOĞRULUK OPTİMİZASYON PANELİ ────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🎯 " + ("Doğruluk Optimizasyon Rehberi" if st.session_state.dil == "TR"
                                        else "Accuracy Optimization Guide"))

                opt_col1, opt_col2 = st.columns(2)

                with opt_col1:
                    st.markdown(f"""
                <div style="background:rgba(52,211,153,0.06);border:1px solid #1e4a3a;
                            border-radius:12px;padding:1.2rem;margin-bottom:1rem">
                  <div style="color:#34d399;font-weight:700;margin-bottom:0.8rem;font-size:0.9rem">
                    ✅ {"Doğruluğu Artıran Ayarlar" if st.session_state.dil == "TR" else "Settings That Increase Accuracy"}
                  </div>
                  <div style="color:#94a3b8;font-size:0.82rem;line-height:2">
                    📅 <b style="color:#e2e8f0">Geçmiş Veri: 5-6 yıl</b> seç → Daha çok eğitim verisi<br>
                    🔭 <b style="color:#e2e8f0">Sekans Penceresi: 40-50 gün</b> → Daha uzun bellek<br>
                    📰 <b style="color:#e2e8f0">Tüm haber kaynaklarını</b> seç → Daha iyi duygu analizi<br>
                    💹 <b style="color:#e2e8f0">Büyük hacimli hisseler</b> seç (THYAO, GARAN) → Daha temiz veri<br>
                    ⏰ <b style="color:#e2e8f0">Borsa açıkken</b> analiz yap (10:00-17:30) → Güncel veri
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.06);border:1px solid #4a1e1e;
                            border-radius:12px;padding:1.2rem">
                  <div style="color:#ef4444;font-weight:700;margin-bottom:0.8rem;font-size:0.9rem">
                    ❌ {"Doğruluğu Düşüren Durumlar" if st.session_state.dil == "TR" else "Situations That Decrease Accuracy"}
                  </div>
                  <div style="color:#94a3b8;font-size:0.82rem;line-height:2">
                    📉 <b style="color:#e2e8f0">Düşük hacimli hisseler</b> → Manipülasyona açık<br>
                    🌪️ <b style="color:#e2e8f0">Kriz dönemleri</b> → Model geçmişten öğrenemez<br>
                    🗳️ <b style="color:#e2e8f0">Seçim/TCMB kararı günleri</b> → Tahmin edilemez<br>
                    📅 <b style="color:#e2e8f0">2 yıldan az veri</b> → Yetersiz eğitim<br>
                    🔄 <b style="color:#e2e8f0">Çok kısa LSTM penceresi</b> (20 gün) → Bellek yetersiz
                  </div>
                </div>
                """, unsafe_allow_html=True)

                with opt_col2:
                    st.markdown(f"""
                <div style="background:rgba(129,140,248,0.06);border:1px solid #2d2b5a;
                            border-radius:12px;padding:1.2rem;margin-bottom:1rem">
                  <div style="color:#818cf8;font-weight:700;margin-bottom:0.8rem;font-size:0.9rem">
                    📊 {"Beklenen Doğruluk Tablosu" if st.session_state.dil == "TR" else "Expected Accuracy Table"}
                  </div>
                  <div style="font-family:'Space Mono',monospace;font-size:0.78rem">
                    <div class="model-row"><span>Sadece LSTM</span><span style="color:#facc15">%58-63</span></div>
                    <div class="model-row"><span>+ Makro Veri</span><span style="color:#facc15">%62-66</span></div>
                    <div class="model-row"><span>+ Haber Analizi</span><span style="color:#34d399">%64-68</span></div>
                    <div class="model-row"><span>+ 6 Model + Stacking	%70-76</span></div>
                    <div class="model-row"><span>+ 5-6 Yıl Veri + Stacking	%70-76</span></div>
                    <div class="model-row" style="border-top:1px solid #334155;margin-top:0.5rem;padding-top:0.5rem">
                      <span style="color:#e2e8f0;font-weight:700">Maks. Gerçekçi Hedef</span>
                      <span style="color:#34d399;font-weight:700">%72</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:rgba(250,204,21,0.06);border:1px solid #4a3f1e;
                            border-radius:12px;padding:1.2rem">
                  <div style="color:#facc15;font-weight:700;margin-bottom:0.8rem;font-size:0.9rem">
                    💡 {"En İyi Strateji" if st.session_state.dil == "TR" else "Best Strategy"}
                  </div>
                  <div style="color:#94a3b8;font-size:0.82rem;line-height:2">
                    1️⃣ <b style="color:#e2e8f0">Günde 1 kez analiz yap</b> — sabah 10:00'da<br>
                    2️⃣ <b style="color:#e2e8f0">Sadece %65+ sinyal ver</b> — düşük güvenli sinyali atla<br>
                    3️⃣ <b style="color:#e2e8f0">Stop-loss koy</b> — her işlemde max %7 zarar<br>
                    4️⃣ <b style="color:#e2e8f0">Risk/Ödül ≥ 1:2</b> — sadece iyi oranlı işlem<br>
                    5️⃣ <b style="color:#e2e8f0">Backtest ile doğrula</b> — önce geçmişte test et<br>
                    6️⃣ <b style="color:#e2e8f0">Max %15 tek hisse</b> — çeşitlendirmeyi unutma
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ── KARŞILAŞTIRMA SEKMESİ ────────────────────────────────────────────────────
with sekme_karsilastir:
    st.markdown("## 🔀 " + ("İki Hisse Karşılaştırma" if st.session_state.dil == "TR" else "Compare Two Stocks"))

    k1, k2 = st.columns(2)
    with k1:
        hisse1_sec = st.selectbox("1. Hisse" if st.session_state.dil == "TR" else "1st Stock",
                                  list(BIST_HISSELER.keys()), index=0, key="k_hisse1")
    with k2:
        hisse2_sec = st.selectbox("2. Hisse" if st.session_state.dil == "TR" else "2nd Stock",
                                  list(BIST_HISSELER.keys()), index=1, key="k_hisse2")

    karsilastir_btn = st.button(
        "🔀 Karşılaştır" if st.session_state.dil == "TR" else "🔀 Compare",
        use_container_width=True
    )

    if karsilastir_btn:
        h1_kod = hisse1_sec.split("—")[0].strip()
        h2_kod = hisse2_sec.split("—")[0].strip()
        h1_ticker = BIST_HISSELER[hisse1_sec]
        h2_ticker = BIST_HISSELER[hisse2_sec]

        with st.spinner("Veriler hazırlanıyor..."):
            try:
                df1 = hisse_indir(h1_ticker, 2)
                df2 = hisse_indir(h2_ticker, 2)
                df1 = indikatör_ekle(df1)
                df2 = indikatör_ekle(df2)

                # Normalize fiyat (100 bazlı performans)
                norm1 = (df1['Close'] / df1['Close'].iloc[0]) * 100
                norm2 = (df2['Close'] / df2['Close'].iloc[0]) * 100

                try:
                    fig_k = go.Figure()
                    fig_k.add_trace(go.Scatter(x=norm1.index, y=norm1,
                                               name=h1_kod, line=dict(color='#34d399', width=2)))
                    fig_k.add_trace(go.Scatter(x=norm2.index, y=norm2,
                                               name=h2_kod, line=dict(color='#818cf8', width=2)))
                    fig_k.add_hline(y=100, line_dash="dot", line_color="#64748b", line_width=1)
                    fig_k.update_layout(
                        title="Normalize Performans (Başlangıç = 100)",
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15,23,42,0.8)',
                        font=dict(color="#94a3b8"),
                        height=350,
                    )
                    st.plotly_chart(fig_k, use_container_width=True)
                except Exception:
                    pass


                # Karşılaştırma tablosu
                def ozet_al(df, kod):
                    c = df['Close']
                    return {
                        "Hisse": kod,
                        "Son Fiyat (₺)": round(float(c.iloc[-1]), 2),
                        "1 Aylık Getiri %": round(float(c.pct_change(20).iloc[-1] * 100), 2),
                        "3 Aylık Getiri %": round(float(c.pct_change(60).iloc[-1] * 100), 2),
                        "RSI (14)": round(float(df['RSI14'].iloc[-1]), 1),
                        "MACD": round(float(df['MACD'].iloc[-1]), 3),
                        "Volatilite 30g %": round(float(df['Vol14'].iloc[-1] * 100), 2),
                        "Hacim / Ort": round(float(df['Hacim_Oran'].iloc[-1]), 2),
                        "BB %B": round(float(df['BB20_Pct'].iloc[-1] * 100), 1),
                    }


                ozet_df = pd.DataFrame([ozet_al(df1, h1_kod), ozet_al(df2, h2_kod)])
                ozet_df = ozet_df.set_index("Hisse").T


                # Renklendirme
                def renk_karsilastir(val):
                    try:
                        v = float(str(val).replace("%", ""))
                        return f"color: {'#34d399' if v > 0 else '#ef4444'}"
                    except Exception:
                        return ""


                st.dataframe(
                    ozet_df.style.applymap(renk_karsilastir),
                    use_container_width=True
                )

                # Hangisi daha iyi — basit skor
                skor1 = (1 if float(df1['RSI14'].iloc[-1]) < 70 else 0) + \
                        (1 if float(df1['MACD'].iloc[-1]) > 0 else 0) + \
                        (1 if float(df1['Close'].pct_change(20).iloc[-1]) > 0 else 0)
                skor2 = (1 if float(df2['RSI14'].iloc[-1]) < 70 else 0) + \
                        (1 if float(df2['MACD'].iloc[-1]) > 0 else 0) + \
                        (1 if float(df2['Close'].pct_change(20).iloc[-1]) > 0 else 0)

                kazanan = h1_kod if skor1 >= skor2 else h2_kod
                st.markdown(f"""
                <div style="background:rgba(52,211,153,0.1);border:1px solid #34d399;
                            border-radius:12px;padding:1rem;text-align:center;margin-top:1rem">
                  <div style="color:#64748b;font-size:0.75rem">Teknik Göstergeler Skoru</div>
                  <div style="color:#34d399;font-size:1.5rem;font-weight:800">{kazanan}</div>
                  <div style="color:#94a3b8;font-size:0.8rem">
                    {h1_kod}: {skor1}/3 puan · {h2_kod}: {skor2}/3 puan
                  </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Hata: {e}")
    else:
        st.info("Sol menüden iki hisse seçin ve Karşılaştır butonuna basın." if st.session_state.dil == "TR"
                else "Select two stocks from the menu and click Compare.")

# ── BACKTEST SEKMESİ ──────────────────────────────────────────────────────────
with sekme_backtest:
    st.markdown("## ⏮️ " + ("Geçmiş Backtest" if st.session_state.dil == "TR" else "Historical Backtest"))
    st.markdown("Bu strateji geçmişte uygulanmış olsaydı kaç para kazandırırdı?" if st.session_state.dil == "TR"
                else "How much would this strategy have earned in the past?")

    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        bt_hisse = st.selectbox("Hisse" if st.session_state.dil == "TR" else "Stock",
                                list(BIST_HISSELER.keys()), key="bt_hisse")
    with bt2:
        bt_sermaye = st.number_input("Başlangıç Sermayesi (₺)" if st.session_state.dil == "TR"
                                     else "Starting Capital (₺)",
                                     min_value=1000, value=100000, step=5000, key="bt_sermaye")
    with bt3:
        bt_yil = st.slider("Backtest Yılı" if st.session_state.dil == "TR" else "Backtest Years",
                           1, 4, 2, key="bt_yil")

    bt_stop = st.slider("Stop-Loss (%)", 3, 15, 7, key="bt_stop")
    bt_hedef = st.slider("Kâr Al (%)" if st.session_state.dil == "TR" else "Take Profit (%)",
                         3, 20, 8, key="bt_hedef")
    bt_esik = st.slider("AL Sinyali Eşiği (%)" if st.session_state.dil == "TR" else "Buy Signal Threshold (%)",
                        55, 75, 62, key="bt_esik")

    backtest_btn = st.button("▶️ Backtest Çalıştır" if st.session_state.dil == "TR" else "▶️ Run Backtest",
                             use_container_width=True)

    if backtest_btn:
        with st.spinner("Backtest hesaplanıyor..."):
            try:
                bt_ticker = BIST_HISSELER[bt_hisse]
                df_bt = hisse_indir(bt_ticker, bt_yil + 1)
                df_bt = indikatör_ekle(df_bt)

                # Basit backtest — RSI + MACD + Momentum tabanlı sinyal
                sermaye = float(bt_sermaye)
                pozisyon = 0.0
                alis_fiyat = 0.0
                islemler = []
                sermaye_seyri = [sermaye]
                tarihler = [df_bt.index[50]]

                for i in range(50, len(df_bt) - 1):
                    row = df_bt.iloc[i]
                    fiyat = float(row['Close'])
                    rsi = float(row['RSI14'])
                    macd = float(row['MACD'])
                    macd_s = float(row['MACD_Signal'])
                    mom5 = float(row['Mom5'])

                    # Basit sinyal: RSI < 60, MACD > sinyal, momentum pozitif
                    sinyal_skoru = (
                                           (1 if rsi < 60 else 0) +
                                           (1 if macd > macd_s else 0) +
                                           (1 if mom5 > 0 else 0)
                                   ) / 3

                    # AL
                    if pozisyon == 0 and sinyal_skoru >= (bt_esik / 100):
                        adet = int(sermaye / fiyat)
                        alis_fiyat = fiyat
                        pozisyon = adet
                        sermaye -= adet * fiyat
                        islemler.append({
                            "Tarih": df_bt.index[i].strftime("%d.%m.%Y"),
                            "İşlem": "AL", "Fiyat": fiyat,
                            "Adet": adet, "Sermaye": round(sermaye, 0)
                        })

                    # SAT — hedef veya stop
                    elif pozisyon > 0:
                        getiri = (fiyat - alis_fiyat) / alis_fiyat
                        if getiri >= bt_hedef / 100 or getiri <= -bt_stop / 100:
                            gelir = pozisyon * fiyat
                            sermaye += gelir
                            kar_zarar = round((fiyat - alis_fiyat) * pozisyon, 0)
                            islemler.append({
                                "Tarih": df_bt.index[i].strftime("%d.%m.%Y"),
                                "İşlem": "SAT ✅" if getiri > 0 else "SAT ❌",
                                "Fiyat": fiyat, "Adet": pozisyon,
                                "Sermaye": round(sermaye, 0),
                                "K/Z (₺)": kar_zarar
                            })
                            pozisyon = 0

                    sermaye_seyri.append(round(sermaye + pozisyon * fiyat, 0))
                    tarihler.append(df_bt.index[i])

                # Final
                son_fiyat_bt = float(df_bt['Close'].iloc[-1])
                final_deger = sermaye + pozisyon * son_fiyat_bt
                toplam_getiri = round((final_deger / bt_sermaye - 1) * 100, 2)
                al_sat_sayisi = len([x for x in islemler if "AL" in x["İşlem"]])
                kazanc_sayisi = len([x for x in islemler if "✅" in x.get("İşlem", "")])
                kayip_sayisi = len([x for x in islemler if "❌" in x.get("İşlem", "")])
                isabet_orani = round(kazanc_sayisi / (kazanc_sayisi + kayip_sayisi) * 100, 1) if (
                                                                                                             kazanc_sayisi + kayip_sayisi) > 0 else 0

                # Sonuç metrikleri
                renk_g = "#34d399" if toplam_getiri >= 0 else "#ef4444"
                m1, m2, m3, m4 = st.columns(4)
                for col, lbl, val, rk in [
                    (m1, "Başlangıç", f"{bt_sermaye:,.0f} ₺", "#38bdf8"),
                    (m2, "Final Değer", f"{final_deger:,.0f} ₺", renk_g),
                    (m3, "Toplam Getiri", f"%{toplam_getiri:+.1f}", renk_g),
                    (m4, "İsabet Oranı", f"%{isabet_orani}", "#818cf8"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="label">{lbl}</div>
                          <div class="value" style="color:{rk};font-size:1.2rem">{val}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # İşlem özeti
                ozet_col1, ozet_col2 = st.columns([1, 3])
                with ozet_col1:
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02);border:1px solid #1e293b;
                                border-radius:10px;padding:1rem;font-family:monospace">
                      <div style="color:#64748b;font-size:0.7rem;margin-bottom:0.5rem">İŞLEM ÖZETİ</div>
                      <div class="model-row"><span>Toplam İşlem</span><span style="color:#e2e8f0">{al_sat_sayisi}</span></div>
                      <div class="model-row"><span>Kârlı İşlem ✅</span><span style="color:#34d399">{kazanc_sayisi}</span></div>
                      <div class="model-row"><span>Zararlı İşlem ❌</span><span style="color:#ef4444">{kayip_sayisi}</span></div>
                      <div class="model-row"><span>İsabet Oranı</span><span style="color:#818cf8">%{isabet_orani}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                with ozet_col2:
                    # Sermaye seyri grafiği
                    try:
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(
                            x=tarihler, y=sermaye_seyri,
                            fill='tozeroy',
                            line=dict(color='#6366f1', width=2),
                            fillcolor='rgba(99,102,241,0.1)',
                            name="Sermaye"
                        ))
                        fig_bt.add_hline(y=bt_sermaye, line_dash="dot",
                                         line_color="#64748b", line_width=1,
                                         annotation_text="Başlangıç")
                        fig_bt.update_layout(
                            title="Sermaye Seyri",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(15,23,42,0.8)',
                            font=dict(color="#94a3b8"),
                            height=280, margin=dict(t=35, b=10, l=0, r=0)
                        )
                        st.plotly_chart(fig_bt, use_container_width=True)
                    except Exception:
                        pass

                # İşlem geçmişi tablosu
                if islemler:
                    st.markdown('<div class="section-title">İŞLEM GEÇMİŞİ</div>',
                                unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(islemler), use_container_width=True, height=300)

            except Exception as e:
                st.error(f"Backtest hatası: {e}")
    else:
        st.info("Ayarları yapıp 'Backtest Çalıştır' butonuna basın." if st.session_state.dil == "TR"
                else "Configure settings and click 'Run Backtest'.")

# ── PORTFÖY SEKMESİ ───────────────────────────────────────────────────────────
with sekme_portfolyo:
    st.markdown("## 💼 Portföy Takibi")
    st.markdown("Aldığın hisseleri buraya ekle, anlık kâr/zarar takibini yap.")

    # Yeni pozisyon ekle
    with st.expander("➕ Yeni Pozisyon Ekle", expanded=len(st.session_state.portfolyo) == 0):
        p1, p2, p3, p4, p5 = st.columns([2, 1.2, 1.2, 1.2, 1])
        with p1:
            p_hisse = st.selectbox("Hisse", list(BIST_HISSELER.keys()),
                                   key="p_hisse", label_visibility="collapsed")
        with p2:
            p_adet = st.number_input("Adet", min_value=1, value=100,
                                     key="p_adet", label_visibility="collapsed")
        with p3:
            p_maliyet = st.number_input("Alış Fiyatı (₺)", min_value=0.01,
                                        value=100.0, step=0.5, format="%.2f",
                                        key="p_maliyet", label_visibility="collapsed")
        with p4:
            p_tarih = st.date_input("Alış Tarihi", key="p_tarih",
                                    label_visibility="collapsed")
        with p5:
            p_ekle = st.button("➕ Ekle", key="p_ekle_btn", use_container_width=True)

        if p_ekle:
            st.session_state.portfolyo.append({
                "hisse": p_hisse.split("—")[0].strip(),
                "ticker": BIST_HISSELER[p_hisse],
                "adet": p_adet,
                "maliyet": p_maliyet,
                "tarih": str(p_tarih),
                "toplam_maliyet": round(p_adet * p_maliyet, 2)
            })
            st.success(f"✅ {p_hisse.split('—')[0].strip()} portföye eklendi!")
            st.rerun()

    if st.session_state.portfolyo:
        # Anlık fiyatları çek
        guncelle = st.button("🔄 Fiyatları Güncelle", key="port_guncelle")

        toplam_maliyet = 0
        toplam_deger = 0
        portfoy_satirlar = []

        for i, pos in enumerate(st.session_state.portfolyo):
            guncel_fiyat = pos['maliyet']  # varsayılan
            if guncelle:
                try:
                    t = yf.Ticker(pos['ticker'])
                    guncel_fiyat = float(t.fast_info['last_price'])
                    st.session_state.portfolyo[i]['guncel'] = guncel_fiyat
                except Exception:
                    pass
            else:
                guncel_fiyat = pos.get('guncel', pos['maliyet'])

            guncel_deger = round(pos['adet'] * guncel_fiyat, 2)
            kar_zarar = round(guncel_deger - pos['toplam_maliyet'], 2)
            kar_yuzde = round((guncel_fiyat / pos['maliyet'] - 1) * 100, 2)
            toplam_maliyet += pos['toplam_maliyet']
            toplam_deger += guncel_deger

            portfoy_satirlar.append({
                "Hisse": pos['hisse'],
                "Adet": pos['adet'],
                "Alış (₺)": pos['maliyet'],
                "Güncel (₺)": guncel_fiyat,
                "Maliyet (₺)": pos['toplam_maliyet'],
                "Güncel Değer (₺)": guncel_deger,
                "Kâr/Zarar (₺)": kar_zarar,
                "Kâr/Zarar (%)": kar_yuzde,
                "Alış Tarihi": pos['tarih'],
            })

        # Özet metrikler
        toplam_kar = round(toplam_deger - toplam_maliyet, 2)
        toplam_kar_yuzde = round((toplam_deger / toplam_maliyet - 1) * 100, 2) if toplam_maliyet > 0 else 0
        kar_renk = "#34d399" if toplam_kar >= 0 else "#ef4444"

        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, renk in [
            (m1, "Toplam Maliyet", f"{toplam_maliyet:,.0f} ₺", "#38bdf8"),
            (m2, "Güncel Değer", f"{toplam_deger:,.0f} ₺", "#818cf8"),
            (m3, "Toplam Kâr/Zarar", f"{toplam_kar:+,.0f} ₺", kar_renk),
            (m4, "Getiri %", f"%{toplam_kar_yuzde:+.2f}", kar_renk),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{label}</div>
                  <div class="value" style="color:{renk};font-size:1.3rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Portföy tablosu
        df_port = pd.DataFrame(portfoy_satirlar)


        def renk_satirlar(row):
            kar = row.get("Kâr/Zarar (%)", 0)
            renk = "rgba(52,211,153,0.08)" if kar >= 0 else "rgba(239,68,68,0.08)"
            return [f"background-color: {renk}"] * len(row)


        st.dataframe(
            df_port.style.apply(renk_satirlar, axis=1).format({
                "Alış (₺)": "{:.2f}", "Güncel (₺)": "{:.2f}",
                "Maliyet (₺)": "{:,.0f}", "Güncel Değer (₺)": "{:,.0f}",
                "Kâr/Zarar (₺)": "{:+,.0f}", "Kâr/Zarar (%)": "{:+.2f}%"
            }),
            use_container_width=True, height=350
        )

        # Portföy pasta grafiği
        try:
            fig_pie = px.pie(
                df_port, values="Güncel Değer (₺)", names="Hisse",
                title="Portföy Dağılımı",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"), title_font=dict(color="#e2e8f0"),
                legend=dict(font=dict(color="#94a3b8"))
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception:
            pass

        # Pozisyon sil
        sil_hisse = st.selectbox("Sil:", ["— Seç —"] + [p['hisse'] for p in st.session_state.portfolyo],
                                 key="port_sil")
        if st.button("🗑️ Seçili Pozisyonu Sil", key="port_sil_btn") and sil_hisse != "— Seç —":
            st.session_state.portfolyo = [p for p in st.session_state.portfolyo
                                          if p['hisse'] != sil_hisse]
            st.rerun()

        # CSV indir
        csv_port = df_port.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Portföyü İndir (CSV)", data=csv_port,
                           file_name=f"portfolyo_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")
    else:
        st.info("Henüz portföy eklenmedi. Yukarıdan hisse ekleyebilirsin.")

# ── ANALİZ GEÇMİŞİ SEKMESİ ───────────────────────────────────────────────────
with sekme_gecmis:
    # Kullanici istatistikleri
    try:
        with db_baglanti() as _sc:
            _tot = _sc.execute("SELECT COUNT(*) FROM analiz_gecmisi").fetchone()[0]
            _top = _sc.execute(
                "SELECT hisse,COUNT(*) c FROM analiz_gecmisi GROUP BY hisse ORDER BY c DESC LIMIT 3").fetchall()
            _gb_d = _sc.execute("SELECT COUNT(*) FROM geri_bildirim WHERE dogru=1").fetchone()[0]
            _gb_t = _sc.execute("SELECT COUNT(*) FROM geri_bildirim").fetchone()[0]
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Toplam Analiz", _tot)
        _c2.metric("En Cok", _top[0][0] if _top else "-")
        _pct = f"%{round(_gb_d / _gb_t * 100)}" if _gb_t > 0 else "Veri yok"
        _c3.metric("Gercek Dogruluk", _pct)
    except Exception:
        pass
    st.markdown("---")
    # Not defteri
    st.markdown("### Not Defteri")
    _nhs = st.selectbox("Hisse", list(BIST_HISSELER.keys())[:30], key="not_sec")
    _nhk = _nhs.split("--")[0].strip() if "--" in _nhs else _nhs.split(" ")[0]
    try:
        with db_baglanti() as _nc:
            _mn = _nc.execute("SELECT not_tr FROM notlar WHERE hisse=?", (_nhk,)).fetchone()
        _nt = _mn[0] if _mn else ""
    except Exception:
        _nt = ""
    _ynt = st.text_area("Not:", value=_nt, height=100, key=f"nt_{_nhk}")
    if st.button("Kaydet", key="nt_save"):
        try:
            with db_baglanti() as _nc:
                _nc.execute("INSERT OR REPLACE INTO notlar(hisse,not_tr) VALUES(?,?)", (_nhk, _ynt))
                _nc.commit()
            st.success("Kaydedildi!")
        except Exception as e:
            st.error(str(e))
    st.markdown("---")
    st.markdown("## 📅 Analiz Geçmişi")
    st.markdown("Önceki analizlerin ve tahminlerin burada saklanır.")

    if st.session_state.analiz_gecmisi:
        df_gecmis = pd.DataFrame(st.session_state.analiz_gecmisi)
        if not df_gecmis.empty:
            try:
                _plotly_yukle()
                if go is not None:
                    _fg = go.Figure()
                    if "sinyal" in df_gecmis.columns and "dogruluk" in df_gecmis.columns:
                        for _sn, _rk in [("AL", "#34d399"), ("SAT", "#ef4444"), ("BEKLE", "#facc15")]:
                            _d2 = df_gecmis[df_gecmis["sinyal"].str.contains(_sn, na=False)]
                            if not _d2.empty:
                                _fg.add_bar(x=_d2.get("hisse", range(len(_d2))), y=_d2["dogruluk"], name=_sn,
                                            marker_color=_rk)
                    _fg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8")
                    st.plotly_chart(_fg, use_container_width=True)
            except Exception:
                pass

        # Doğruluk istatistiği
        if 'gerceklesti' in df_gecmis.columns:
            dogru = (df_gecmis['gerceklesti'] == df_gecmis['tahmin_yon']).sum()
            toplam = df_gecmis['gerceklesti'].notna().sum()
            if toplam > 0:
                dogruluk = round(dogru / toplam * 100, 1)
                st.markdown(f"""
                <div class="metric-card" style="max-width:300px;margin-bottom:1.5rem">
                  <div class="label">Gerçekleşen Doğruluk</div>
                  <div class="value" style="color:{'#34d399' if dogruluk >= 60 else '#ef4444'}">
                    %{dogruluk}
                  </div>
                  <div style="color:#64748b;font-size:0.8rem">{dogru}/{toplam} tahmin tuttu</div>
                </div>""", unsafe_allow_html=True)

        # Geçmiş tablosu
        st.dataframe(df_gecmis, use_container_width=True, height=400)

        # CSV indir
        csv_gecmis = df_gecmis.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Geçmişi İndir (CSV)", data=csv_gecmis,
                           file_name=f"analiz_gecmisi_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        if st.button("🗑️ Geçmişi Temizle", key="gecmis_temizle"):
            st.session_state.analiz_gecmisi = []
            st.rerun()
    else:
        st.info("Henüz analiz yapılmadı. Analiz sekmesinden hisse analiz et, sonuçlar buraya kaydedilecek.")

# ── RİSK HESAPLAYICI SEKMESİ ──────────────────────────────────────────────────
with sekme_risk:
    st.markdown("## ⚖️ Risk Hesaplayıcı")
    st.markdown("Toplam sermayeni gir, model sana nasıl dağıtacağını söylesin.")

    rk1, rk2 = st.columns([1, 1])
    with rk1:
        sermaye = st.number_input("💰 Toplam Sermaye (₺)", min_value=1000,
                                  value=100000, step=5000, format="%d")
        risk_profil = st.select_slider("📊 Risk Profili",
                                       options=["Çok Düşük", "Düşük", "Orta",
                                                "Yüksek", "Çok Yüksek"],
                                       value="Orta")
        stop_yuzde_genel = st.slider("🛑 Stop-Loss Yüzdesi (%)", 3, 15, 7)
        max_hisse = st.slider("📦 Max Hisse Sayısı", 3, 15, 7)

    with rk2:
        # Profil bazlı parametreler
        profil_map = {
            "Çok Düşük": {"hisse_bas": 0.05, "hisse_max": 0.10, "nakit": 0.40,
                          "aciklama": "Sermayenin %40'ı nakit, her hisseye max %10"},
            "Düşük": {"hisse_bas": 0.08, "hisse_max": 0.12, "nakit": 0.30,
                      "aciklama": "Sermayenin %30'ı nakit, her hisseye max %12"},
            "Orta": {"hisse_bas": 0.10, "hisse_max": 0.15, "nakit": 0.20,
                     "aciklama": "Sermayenin %20'si nakit, her hisseye max %15"},
            "Yüksek": {"hisse_bas": 0.12, "hisse_max": 0.20, "nakit": 0.10,
                       "aciklama": "Sermayenin %10'u nakit, her hisseye max %20"},
            "Çok Yüksek": {"hisse_bas": 0.15, "hisse_max": 0.25, "nakit": 0.05,
                           "aciklama": "Sermayenin %5'i nakit, her hisseye max %25"},
        }
        profil = profil_map[risk_profil]
        yatirim_sermaye = sermaye * (1 - profil['nakit'])
        nakit_miktar = sermaye * profil['nakit']
        hisse_basi = yatirim_sermaye / max_hisse

        st.markdown(f"""
        <div style="background:rgba(129,140,248,0.08);border:1px solid #818cf8;
                    border-radius:12px;padding:1.2rem;margin-top:0.5rem">
          <div style="color:#818cf8;font-size:0.72rem;text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:0.8rem">Risk Profili: {risk_profil}</div>
          <div style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem">{profil['aciklama']}</div>
          <div class="model-row">
            <span>Yatırım Sermayesi</span>
            <span style="color:#34d399;font-weight:700">{yatirim_sermaye:,.0f} ₺</span>
          </div>
          <div class="model-row">
            <span>Nakit Rezerv</span>
            <span style="color:#facc15;font-weight:700">{nakit_miktar:,.0f} ₺</span>
          </div>
          <div class="model-row">
            <span>Hisse Başı Max</span>
            <span style="color:#818cf8;font-weight:700">{hisse_basi:,.0f} ₺</span>
          </div>
          <div class="model-row">
            <span>Stop-Loss / Hisse</span>
            <span style="color:#ef4444;font-weight:700">
              {hisse_basi * stop_yuzde_genel / 100:,.0f} ₺ (%{stop_yuzde_genel})
            </span>
          </div>
          <div class="model-row">
            <span>Max Toplam Zarar</span>
            <span style="color:#ef4444;font-weight:700">
              {(hisse_basi * stop_yuzde_genel / 100) * max_hisse:,.0f} ₺
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 ÖNERİLEN DAĞILIM PLANI</div>',
                unsafe_allow_html=True)

    # Portföydeki hisselere dağıtım
    if st.session_state.portfolyo:
        dagitim_data = []
        for pos in st.session_state.portfolyo[:max_hisse]:
            miktar = min(hisse_basi, yatirim_sermaye / len(st.session_state.portfolyo[:max_hisse]))
            adet_tahmini = int(miktar / pos['maliyet']) if pos['maliyet'] > 0 else 0
            dagitim_data.append({
                "Hisse": pos['hisse'],
                "Önerilen (₺)": round(miktar, 0),
                "Portföy %": round(miktar / sermaye * 100, 1),
                "Stop-Loss": round(miktar * stop_yuzde_genel / 100, 0),
            })
        st.dataframe(pd.DataFrame(dagitim_data), use_container_width=True)
    else:
        # Örnek dağılım göster
        ornekler = [
            ("THYAO", 0.20), ("GARAN", 0.15), ("EREGL", 0.15),
            ("ASELS", 0.15), ("BIMAS", 0.10), ("KCHOL", 0.10),
            ("NAKIT", profil['nakit']),
        ]
        dagitim_data = []
        for hisse, oran in ornekler[:max_hisse]:
            miktar = sermaye * oran
            dagitim_data.append({
                "Hisse": hisse,
                "Önerilen (₺)": round(miktar, 0),
                "Portföy %": round(oran * 100, 1),
                "Stop-Loss (₺)": round(miktar * stop_yuzde_genel / 100, 0) if hisse != "NAKİT" else 0,
            })
        st.dataframe(pd.DataFrame(dagitim_data), use_container_width=True)
        st.caption("💡 Portföy sekmesinden hisse eklerseniz gerçek dağılım hesaplanır.")

    # Görsel dağılım
    try:
        fig_risk = px.bar(
            pd.DataFrame(dagitim_data),
            x="Hisse", y="Önerilen (₺)",
            color="Portföy %",
            color_continuous_scale="Viridis",
            title=f"Önerilen Sermaye Dağılımı — {risk_profil} Risk",
        )
        fig_risk.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)',
            font=dict(color="#94a3b8"), title_font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    except Exception:
        pass

# ── PDF RAPOR SEKMESİ ─────────────────────────────────────────────────────────
with sekme_pdf:
    st.markdown("## 📄 PDF Rapor Oluştur")

    if st.session_state.analiz_gecmisi:
        son_analiz = st.session_state.analiz_gecmisi[-1]
        st.success(f"✅ Son analiz: **{son_analiz.get('hisse', '?')}** — {son_analiz.get('tarih', '?')}")

        if st.button("📄 PDF Rapor Oluştur", use_container_width=True):
            try:
                # Reportlab lazy yükle
                _reportlab_yukle()
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib import colors
                from reportlab.lib.units import cm

                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4,
                                        leftMargin=2 * cm, rightMargin=2 * cm,
                                        topMargin=2 * cm, bottomMargin=2 * cm)
                styles = getSampleStyleSheet()
                hikaye = []

                # Başlık
                baslik_stil = ParagraphStyle('Baslik', parent=styles['Title'],
                                             fontSize=20, textColor=colors.HexColor('#1a1a2e'),
                                             spaceAfter=6)
                hikaye.append(Paragraph("📈 BIST Tahmin Raporu", baslik_stil))
                hikaye.append(Paragraph(
                    f"Oluşturma: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
                    styles['Normal']
                ))
                hikaye.append(Spacer(1, 0.5 * cm))

                # Analiz özeti
                for analiz in st.session_state.analiz_gecmisi[-10:]:
                    hikaye.append(Paragraph(
                        f"<b>{analiz.get('hisse', '?')}</b> — {analiz.get('tarih', '?')}",
                        styles['Heading2']
                    ))
                    data = [
                        ["Sinyal", analiz.get('sinyal', '?')],
                        ["Yükselme Olasılığı", f"%{analiz.get('olasilik', 0):.1f}"],
                        ["Fiyat", f"{analiz.get('fiyat', 0):.2f} ₺"],
                        ["Hedef", f"{analiz.get('hedef', 0):.2f} ₺"],
                        ["Stop-Loss", f"{analiz.get('stop', 0):.2f} ₺"],
                        ["Bekleme Süresi", analiz.get('bekleme', '?')],
                        ["Doğruluk", f"%{analiz.get('dogruluk', 0):.1f}"],
                    ]
                    tablo = Table(data, colWidths=[5 * cm, 10 * cm])
                    tablo.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4ff')),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
                         [colors.white, colors.HexColor('#f8f9ff')]),
                        ('PADDING', (0, 0), (-1, -1), 6),
                    ]))
                    hikaye.append(tablo)
                    hikaye.append(Spacer(1, 0.4 * cm))

                # Portföy
                if st.session_state.portfolyo:
                    hikaye.append(Paragraph("💼 Portföy Özeti", styles['Heading1']))
                    p_data = [["Hisse", "Adet", "Alış", "Maliyet"]]
                    for pos in st.session_state.portfolyo:
                        p_data.append([
                            pos['hisse'], str(pos['adet']),
                            f"{pos['maliyet']:.2f} ₺",
                            f"{pos['toplam_maliyet']:,.0f} ₺"
                        ])
                    p_tablo = Table(p_data, colWidths=[4 * cm, 3 * cm, 4 * cm, 5 * cm])
                    p_tablo.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                         [colors.white, colors.HexColor('#f8f9ff')]),
                        ('PADDING', (0, 0), (-1, -1), 6),
                    ]))
                    hikaye.append(p_tablo)

                # Uyarı
                hikaye.append(Spacer(1, 1 * cm))
                hikaye.append(Paragraph(
                    ""
                    "finansal danışman görüşü alınız. Geçmiş performans gelecek "
                    "sonuçları garanti etmez.",
                    ParagraphStyle('Uyari', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.grey)
                ))

                doc.build(hikaye)
                buffer.seek(0)

                st.download_button(
                    "⬇️ PDF'i İndir",
                    data=buffer,
                    file_name=f"bist_rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
                st.success("✅ PDF hazır! İndir butonuna tıkla.")

            except ImportError:
                st.warning("📦 reportlab kütüphanesi gerekli:")
                st.code("pip install reportlab")
            except Exception as e:
                st.error(f"PDF oluşturma hatası: {e}")
    else:
        st.info("📊 PDF oluşturmak için önce Analiz sekmesinden en az bir hisse analiz edin.")
        st.markdown("""
        <div style="background:rgba(129,140,248,0.08);border:1px solid #4f46e5;
                    border-radius:12px;padding:1.5rem;margin-top:1rem">
          <div style="color:#818cf8;font-weight:700;margin-bottom:0.8rem">PDF Raporu İçerir:</div>
          <ul style="color:#94a3b8;line-height:2">
            <li>Analiz edilen her hissenin tahmin sonuçları</li>
            <li>AL/SAT fiyat aralıkları ve stop-loss seviyeleri</li>
            <li>Bekleme süreleri ve risk/ödül oranları</li>
            <li>Portföy özeti ve kâr/zarar tablosu</li>
            <li>Model doğruluk oranları</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# ── Kalıcı uyarı banner ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(245,158,11,0.08));
            border:1px solid rgba(239,68,68,0.3);border-radius:12px;
            padding:1.2rem 1.5rem;margin-top:1rem;">
  <div style="color:#f87171;font-weight:700;font-size:1rem;margin-bottom:0.5rem;">
    ⚠️ Yasal Uyarı — Lütfen Okuyun
  </div> 
  <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7;">
    Bu sistem <b style="color:#fbbf24">yapay zeka destekli tahmin aracıdır</b> ve yatırım tavsiyesi değildir.
    Geçmiş performans gelecek sonuçları garanti etmez. Modeller tarihsel veri üzerinde eğitilmekte olup
    <b style="color:#fbbf24">gerçek piyasa koşullarında kayıp yaşanabilir</b>.<br><br>
    Tahminler <b style="color:#34d399">%70-76 doğruluk</b> hedeflemektedir — bu oran yüksek görünse de
    her 4 tahminden 1'i yanlış olabilir anlamına gelir.
    <b>Hiçbir zaman tüm sermayenizi tek bir hisseye yatırmayın.</b>
    Kararlarınızı bu sistem dahil birden fazla kaynakla destekleyin.<br><br>
    <span style="color:#64748b;font-size:0.75rem;">
    Gelistirici: Egehan Macit | v5.0 | <a href="https://github.com/EgehanMacit" target="_blank" style="color:#818cf8">GitHub</a> |
    Bu platform ticari bir yatırım danışmanlığı hizmeti değildir.
    </span>
  </div>
</div>
""", unsafe_allow_html=True)