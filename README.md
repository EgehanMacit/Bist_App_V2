# 📈 BIST Tahmin Sistemi v5.0

**Yapay Zeka Destekli BIST Hisse Analizi ve Tahmin Platformu**

Geliştirici: **Egehan Macit** | [github.com/EgehanMacit](https://github.com/EgehanMacit)

---

## 🚀 Hızlı Başlangıç

### Windows Kurulum (İlk Kez)

1. **Python 3.11.9** kur → [İndir](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
   - ✅ "Add Python to PATH" kutusunu işaretle
   - ✅ "Install for all users" kutusunu işaretle

2. Tüm dosyaları aynı klasöre koy:
```
EBorsa_Tahmin/
├── bist_streamlit_app.py
├── requirements.txt
├── KURULUM.bat
├── CALISTIR.bat
├── utils/
│   ├── __init__.py
│   ├── sabitler.py
│   ├── veri.py
│   ├── modeller.py
│   └── veritabani.py
└── .streamlit/
    └── config.toml
```

3. **`KURULUM.bat`** dosyasına çift tıkla (5-10 dakika sürer)
4. **`CALISTIR.bat`** dosyasına çift tıkla
5. Tarayıcıda aç: **http://localhost:8765**

---

### Manuel Kurulum (CMD)

```cmd
cd C:\Users\...\EBorsa_Tahmin
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
streamlit run bist_streamlit_app.py
```

---

## ☁️ Streamlit Cloud Yayını

1. GitHub'a yükle (tüm dosyalar)
2. [share.streamlit.io](https://share.streamlit.io) → New app
3. Repository seç → `bist_streamlit_app.py` → Deploy
4. [uptimerobot.com](https://uptimerobot.com) ile uyku modunu engelle (ücretsiz)

### Supabase (Kalıcı Veri)
Streamlit Cloud Secrets kısmına ekle:
```toml
[supabase]
url = "https://XXXXX.supabase.co"
key = "eyJhbGc..."
```

---

## 📊 Özellikler

| Özellik | Açıklama |
|---------|----------|
| 325+ Hisse | Tüm BIST hisseleri, 14 sektör |
| 5 ML Modeli | XGBoost + LightGBM + HistGBM + ExtraTrees + Stacking |
| 10 Günlük Tahmin | AL / SAT / BEKLE sinyali |
| %70-78 Doğruluk | Büyük hacimli hisselerde |
| 120+ İndikatör | RSI, MACD, Bollinger, ATR... |
| Haber Analizi | 8 kaynak, duygu skoru |
| Makro Veri | USD/TRY, Altın, VIX, BIST100 |
| Model Güven Skoru | Düşük / Orta / Yüksek |
| Stop-Loss | ATR tabanlı 3 seviye |
| Destek/Direnç | 20 ve 50 günlük seviyeler |
| Portföy Takibi | Alarm + geçmiş |
| Not Defteri | Hisse başına SQLite notları |
| Geri Bildirim | Tahmin doğruluk takibi |
| PDF Rapor | Analiz raporu indirme |
| TR / EN | İki dil desteği |
| Mobil Uyumlu | Responsive CSS |
| Light/Dark Tema | Sidebar toggle |

---

## 🤖 ML Modelleri

```
XGBoost      → %70-76  ~60-90 sn
LightGBM     → %69-75  ~30-45 sn
HistGBM      → %68-74  ~20-30 sn
ExtraTrees   → %65-72  ~15-20 sn
─────────────────────────────────
Stacking     → %72-78  ~5 sn (meta-model)

Toplam: ~3-6 dakika (THYAO, 15 yıl veri)
Early stopping: 500 round
```

---

## 📁 Dosya Yapısı

```
bist_streamlit_app.py   Ana uygulama — 4,592 satır
utils/
  sabitler.py           BIST hisse listesi, sektörler
  veri.py               Veri indirme, indikatörler
  modeller.py           ML eğitim ve tahmin
  veritabani.py         SQLite + Supabase
requirements.txt        Python paketleri
config.toml             Streamlit ayarları (port: 8765)
KURULUM.bat             İlk kurulum
CALISTIR.bat            Günlük kullanım
```

---

## 🔧 Gereksinimler

```
Python 3.11.9 (önerilen)
streamlit==1.40.0
xgboost==2.1.1
lightgbm==4.5.0
scikit-learn==1.5.2
pandas==2.2.2
numpy==1.26.4
plotly==5.24.1
yfinance>=0.2.54
reportlab==4.2.5
supabase>=2.0.0
```

---

## 🗄️ Veri Kaynakları

| # | Kaynak | Durum |
|---|--------|-------|
| 1 | Yahoo Finance v8 (query2) | Ana |
| 2 | Yahoo Finance v8 (query1) | Yedek |
| 3 | yfinance kütüphanesi | Fallback |
| 4 | Investing.com | Yedek |
| 5 | Stooq.com | Son çare |

Cache TTL: 6 saat · "Verileri Yenile" butonu ile sıfırlanır

---

## 🐛 Sık Karşılaşılan Sorunlar

**WinError 10013 — Port sorunu**
```cmd
streamlit run bist_streamlit_app.py --server.port 8765
```
Veya `CALISTIR.bat` kullan (otomatik port 8765).

**Yahoo Finance rate limit**
Sidebar'dan "Verileri Yenile" → birkaç dakika bekle.

**Model eğitimi çok hızlı bitiyor (10-20 sn)**
Veri yılını 10-15'e çıkar, büyük hisse seç (THYAO, GARAN).

**Streamlit Cloud RAM hatası**
Veri yılını 8'e düşür, pencere boyutunu 30'a düşür.

**"Geç bu hisseyi" sinyali**
Normal — model %48 altında sinyal üretiyor. Farklı hisse veya farklı tarihte dene.

---

## ⚠️ Yasal Uyarı

Bu sistem **yatırım tavsiyesi değildir.**
Tahminler geçmiş veriye dayalıdır, gelecek performansı garanti etmez.
Her 4 tahminden 1'i yanlış olabilir. Tüm sermayenizi tek hisseye yatırmayın.

---

*BIST Tahmin Sistemi v5.0 — Egehan Macit — [github.com/EgehanMacit](https://github.com/EgehanMacit)*