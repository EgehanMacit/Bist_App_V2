"""BIST Tahmin Sistemi - Veritabani (SQLite + Supabase)"""
import streamlit as st
import json as _json, os as _os

import sqlite3, json as _json, os as _os

# Streamlit Cloud'da /tmp yazilabilir (ama reboot'ta sifirlanir)
# Lokal: uygulama klasoru
try:
    _test_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "bist_data.db")
    # Yazma testi
    with open(_test_path, 'a') as _t: pass
    DB_YOLU = _test_path
except Exception:
    DB_YOLU = "/tmp/bist_data.db"  # Streamlit Cloud fallback

def db_baglanti():
    conn = sqlite3.connect(DB_YOLU, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # Paralel okuma için
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
_sb_client   = None

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
            return db_yukle(tablo)   # fallback: yerel SQLite
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
        db_kaydet_liste(tablo, liste)   # fallback

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



if 'alarmlar'        not in st.session_state:
    st.session_state.alarmlar = db_yukle('alarmlar')
if 'portfolyo'       not in st.session_state:
    st.session_state.portfolyo = db_yukle('portfolyo')
if 'analiz_gecmisi'  not in st.session_state: st.session_state.analiz_gecmisi = []
if 'favoriler'       not in st.session_state: st.session_state.favoriler = []
if 'dil'             not in st.session_state: st.session_state.dil = "TR"

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


