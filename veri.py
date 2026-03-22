"""BIST Tahmin Sistemi - Veri ve Indikatörler"""
import streamlit as st
import numpy as np, pandas as pd, requests, warnings, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

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
               f"{ticker}?interval=1d&range={min(gun,730)}d")
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r    = requests.get(url, headers=hdrs, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        res  = data.get("chart", {}).get("result", [])
        if not res:
            return pd.DataFrame()
        res  = res[0]
        ts   = res.get("timestamp", [])
        quot = res.get("indicators", {}).get("quote", [{}])[0]
        adjc = res.get("indicators", {}).get("adjclose", [{}])
        close_data = (adjc[0].get("adjclose") if adjc else None) or quot.get("close")
        if not ts or not close_data:
            return pd.DataFrame()
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
        df  = pd.DataFrame({
            "Open":   quot.get("open",   [None]*len(ts)),
            "High":   quot.get("high",   [None]*len(ts)),
            "Low":    quot.get("low",    [None]*len(ts)),
            "Close":  close_data,
            "Volume": quot.get("volume", [0]*len(ts)),
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
            if q.get('exchange') in ('BIST','ISE') and bist_kod in q.get('symbol','').upper():
                pair_id = q.get('pairId') or q.get('id'); break
        if not pair_id: return pd.DataFrame()
        import time as _tt2
        now = int(_tt2.time()); start = now - gun * 86400
        url2 = f"https://tvc4.investing.com/{pair_id}/1/{start}/{now}/1/history?symbol={pair_id}&resolution=D&from={start}&to={now}"
        r2 = requests.get(url2, headers=hdrs, timeout=12)
        if r2.status_code != 200: return pd.DataFrame()
        h = r2.json()
        if h.get('s') != 'ok': return pd.DataFrame()
        df = pd.DataFrame({'Open':h.get('o',[]),'High':h.get('h',[]),'Low':h.get('l',[]),
                           'Close':h.get('c',[]),'Volume':h.get('v',[])},
                          index=pd.to_datetime(h.get('t',[]),unit='s').normalize())
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
        end   = str(bitis.date())     if hasattr(bitis, 'date')     else str(bitis)
        url   = (f"https://stooq.com/q/d/l/?s={stooq_ticker}"
                 f"&d1={start.replace('-','')}&d2={end.replace('-','')}&i=d")
        r     = requests.get(url, timeout=15,
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
    gun       = max(int((bitis - baslangic).days), 30)
    start_str = str(baslangic.date()) if hasattr(baslangic, 'date') else str(baslangic)
    end_str   = str(bitis.date())     if hasattr(bitis, 'date')     else str(bitis)
    import time as _tl

    # ── Yöntem 1: Direkt Yahoo v8 API (rate limit neredeyse yok) ─────────────
    try:
        import time as _tt
        p1  = int(_tt.mktime(baslangic.timetuple()))
        p2  = int(_tt.mktime(bitis.timetuple()))
        for base_url in [
            "https://query2.finance.yahoo.com/v8/finance/chart/",
            "https://query1.finance.yahoo.com/v8/finance/chart/",
        ]:
            try:
                url  = (f"{base_url}{ticker}?period1={p1}&period2={p2}"
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
                res  = data.get("chart", {}).get("result", [])
                if not res:
                    continue
                ts   = res[0].get("timestamp", [])
                quot = res[0].get("indicators", {}).get("quote", [{}])[0]
                adjc = res[0].get("indicators", {}).get("adjclose", [{}])
                close = (adjc[0].get("adjclose") if adjc else None) or quot.get("close")
                if ts and close:
                    idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
                    df  = pd.DataFrame({
                        "Open":   quot.get("open",   [None]*len(ts)),
                        "High":   quot.get("high",   [None]*len(ts)),
                        "Low":    quot.get("low",    [None]*len(ts)),
                        "Close":  close,
                        "Volume": quot.get("volume", [0]*len(ts)),
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
            t  = yf.Ticker(ticker)
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
        base    = "https://query2.finance.yahoo.com/v8/finance/chart/"
        p1      = int(_tt.mktime(baslangic.timetuple())) if hasattr(baslangic, 'timetuple') else int(baslangic)
        p2      = int(_tt.mktime(bitis.timetuple()))     if hasattr(bitis, 'timetuple')     else int(bitis)
        url     = (f"{base}{ticker}?period1={p1}&period2={p2}"
                   f"&interval={interval}&events=history&includeAdjustedClose=true")
        hdrs    = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/123.0 Safari/537.36",
            "Accept":     "application/json",
            "Referer":    "https://finance.yahoo.com",
        }
        r    = requests.get(url, headers=hdrs, timeout=15)
        data = r.json()
        res  = data.get("chart", {}).get("result", [{}])[0]
        ts   = res.get("timestamp", [])
        quot = res.get("indicators", {}).get("quote", [{}])[0]
        adj  = res.get("indicators", {}).get("adjclose", [{}])[0]
        if ts and quot.get("close"):
            import pandas as _pd
            idx = _pd.to_datetime(ts, unit="s", utc=True).tz_convert("Europe/Istanbul").tz_localize(None)
            df  = _pd.DataFrame({
                "Open":   quot.get("open",   [None]*len(ts)),
                "High":   quot.get("high",   [None]*len(ts)),
                "Low":    quot.get("low",    [None]*len(ts)),
                "Close":  adj.get("adjclose",[None]*len(ts)) or quot.get("close",[None]*len(ts)),
                "Volume": quot.get("volume", [0]*len(ts)),
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
    bitis     = datetime.today()
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

    gerekli = ['Open','High','Low','Close','Volume']
    mevcut  = [c for c in gerekli if c in df.columns]
    df = df[mevcut].copy()
    df = df.replace(0, np.nan)
    df['Volume'] = df['Volume'].fillna(0)
    df = df.dropna(subset=['Open','High','Low','Close'])

    if len(df) < 100:
        raise ValueError(
            f"{ticker} için yeterli veri yok ({len(df)} satır). "
            f"Daha uzun veri yılı seçin (5-6 yıl önerilir)."
        )
    return df

@st.cache_data(ttl=21600, show_spinner=False)   # 6 saat cache
def makro_indir() -> pd.DataFrame:
    """
    Makro verileri paralel çeker, 6 saat cache'ler.
    Rate limit olursa boş df döner — analiz durmuyor.
    """
    bitis     = datetime.today()
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
    "rekor":2,"zirve":2,"fırladı":2,"coştu":2,"beklentileri aştı":2,
    "tarihi yüksek":2,"güçlü büyüme":2,"kâr açıkladı":2,"temettü artışı":2,
    "anlaşma imzaladı":2,"ihale kazandı":2,"kapasite artırıyor":2,
    "yüksek kâr":2,"borsada lider":2,"yatırımcı ilgisi":2,
    # Orta pozitif (ağırlık 1)
    "yükseliş":1,"artış":1,"büyüme":1,"kâr":1,"kazanç":1,"güçlü":1,
    "başarı":1,"olumlu":1,"yatırım":1,"ihracat":1,"talep":1,
    "toparlanma":1,"iyileşme":1,"temettü":1,"anlaşma":1,"sözleşme":1,
    "sipariş":1,"kapasite":1,"ihracat artışı":1,"pazar payı":1,
    "alım":1,"destek":1,"pozitif":1,"rally":1,"momentum":1,
    "ralli":1,"yeni müşteri":1,"büyük proje":1,"güvenli liman":1,
}
NEGATIF = {
    # Güçlü negatif (ağırlık 2)
    "çöktü":2,"iflas":2,"kriz":2,"soruşturma":2,"dava":2,
    "manipülasyon":2,"haciz":2,"büyük zarar":2,"sermaye kaybı":2,
    "ihracat yasağı":2,"yaptırım":2,"olağanüstü hal":2,
    "devalüasyon":2,"yüksek enflasyon":2,"faiz artışı şoku":2,
    # Orta negatif (ağırlık 1)
    "düşüş":1,"kayıp":1,"zarar":1,"risk":1,"enflasyon":1,
    "gerileme":1,"olumsuz":1,"endişe":1,"panik":1,"baskı":1,
    "zayıf":1,"daralma":1,"küçülme":1,"uyarı":1,"volatilite":1,
    "satış baskısı":1,"zayıf talep":1,"gelir düşüşü":1,"maliyet artışı":1,
    "borç":1,"yükümlülük":1,"temerrüt":1,"erteleme":1,"iptal":1,
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
            device=-1,          # CPU
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
        for metin in metinler[:15]:          # max 15 metin
            metin_k = metin[:512]            # token limiti
            sonuc   = pipe(metin_k)[0]
            label   = sonuc["label"].lower()
            score   = sonuc["score"]         # güven skoru
            deger   = label_map.get(label, 0.0) * score
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
    return 0.0 if toplam == 0 else float(np.clip((poz-neg)/(toplam+2), -1, 1))

def _tek_kaynak_cek(sorgu: str, url_sablonu: str, limit: int = 15) -> list:
    """Tek bir RSS/HTML kaynaktan haber çeker."""
    haberler = []
    headers  = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/122.0 Safari/537.36"),
        "Accept-Language": "tr-TR,tr;q=0.9",
    }
    try:
        url = url_sablonu.format(q=requests.utils.quote(sorgu))
        r   = requests.get(url, headers=headers, timeout=8)
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

@st.cache_data(ttl=21600, show_spinner=False)
def indikatör_ekle(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    # Hareketli ortalamalar
    for p in [5,10,20,50,100,200]:
        df[f'MA{p}']      = c.rolling(p).mean()
        df[f'MA{p}_oran'] = c / (df[f'MA{p}'] + 1e-10)
    for p in [9,12,21,26,50]:
        df[f'EMA{p}'] = c.ewm(span=p, adjust=False).mean()

    # MA çaprazları (golden/death cross sinyalleri)
    df['MA_GC_20_50']  = (df['MA20']  > df['MA50']).astype(int)
    df['MA_GC_50_200'] = (df['MA50']  > df['MA200']).astype(int)
    df['EMA_GC_9_21']  = (df['EMA9']  > df['EMA21']).astype(int)

    # RSI çoklu periyot
    for p in [7,14,21]:
        d  = c.diff()
        g  = d.clip(lower=0).rolling(p).mean()
        ls = (-d.clip(upper=0)).rolling(p).mean()
        df[f'RSI{p}'] = 100 - 100/(1 + g/(ls+1e-10))
    df['RSI_diverg'] = df['RSI14'] - df['RSI14'].shift(5)  # RSI ivmesi

    # MACD
    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    df['MACD_Cross']  = (df['MACD'] > df['MACD_Signal']).astype(int)

    # Bollinger bantları
    for p in [10,20]:
        ort = c.rolling(p).mean(); std = c.rolling(p).std()
        df[f'BB{p}_Ust'] = ort + 2*std
        df[f'BB{p}_Alt'] = ort - 2*std
        df[f'BB{p}_Pct'] = (c-df[f'BB{p}_Alt'])/(df[f'BB{p}_Ust']-df[f'BB{p}_Alt']+1e-10)
        df[f'BB{p}_Genis'] = (df[f'BB{p}_Ust']-df[f'BB{p}_Alt'])/(ort+1e-10)  # Bant genişliği

    # ATR ve volatilite
    for p in [7,14,21]:
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        df[f'ATR{p}']      = tr.rolling(p).mean()
        df[f'ATR{p}_norm'] = df[f'ATR{p}'] / (c+1e-10)  # Normalize ATR

    # Stochastic
    for p in [9,14]:
        lp = l.rolling(p).min(); hp = h.rolling(p).max()
        df[f'Stoch{p}_K'] = 100*(c-lp)/(hp-lp+1e-10)
        df[f'Stoch{p}_D'] = df[f'Stoch{p}_K'].rolling(3).mean()

    # CCI
    tp = (h+l+c)/3
    df['CCI']    = (tp-tp.rolling(20).mean())/(0.015*tp.rolling(20).std()+1e-10)
    df['CCI_14'] = (tp-tp.rolling(14).mean())/(0.015*tp.rolling(14).std()+1e-10)

    # OBV
    obv = (np.sign(c.diff())*v).fillna(0).cumsum()
    df['OBV']      = obv
    df['OBV_MA']   = obv.rolling(10).mean()
    df['OBV_oran'] = obv / (df['OBV_MA']+1e-10)
    df['OBV_trend']= obv.pct_change(5)

    # Williams %R
    hh = h.rolling(14).max(); ll = l.rolling(14).min()
    df['WilliamsR'] = -100*(hh-c)/(hh-ll+1e-10)

    # MFI
    mf  = tp*v
    pos = mf.where(tp>tp.shift(),0).rolling(14).sum()
    neg = mf.where(tp<tp.shift(),0).rolling(14).sum()
    df['MFI'] = 100 - 100/(1+pos/(neg+1e-10))

    # Momentum çoklu periyot
    for p in [3,5,10,20,60]:
        df[f'Mom{p}'] = c.pct_change(p)

    # Volatilite
    for p in [7,14,30]:
        df[f'Vol{p}'] = c.pct_change().rolling(p).std()
    df['Vol_oran'] = df['Vol7'] / (df['Vol30']+1e-10)

    # Hacim
    df['Hacim_MA5']     = v.rolling(5).mean()
    df['Hacim_MA20']    = v.rolling(20).mean()
    df['Hacim_Oran']    = v / (df['Hacim_MA20']+1e-10)
    df['Hacim_Degisim'] = v.pct_change()
    df['Hacim_Surge']   = (v > df['Hacim_MA20']*2).astype(int)

    # Fiyat pozisyonu 52 haftalık
    df['High52w'] = h.rolling(252).max()
    df['Low52w']  = l.rolling(252).min()
    df['Pos52w']  = (c - df['Low52w']) / (df['High52w'] - df['Low52w']+1e-10)

    # Mum desenleri
    body   = (c - df['Open']).abs()
    candle = h - l
    df['Doji']      = (body/(candle+1e-10) < 0.1).astype(int)
    df['Hammer']    = ((body/(candle+1e-10) < 0.3) &
                       ((df['Open']-l)/(candle+1e-10) > 0.6)).astype(int)
    df['Engulf_Up'] = ((c > df['Open']) &
                       (c.shift() < df['Open'].shift()) &
                       (c > df['Open'].shift()) &
                       (df['Open'] < c.shift())).astype(int)

    # ── SEKANS ÖZELLİKLERİ — LSTM'in öğrendiğini elle çıkar ─────────────────
    # RSI son 5/10/20 günde kaç kez 50'yi geçti (yukarı momentum sayacı)
    rsi_above = (df['RSI14'] > 50).astype(int)
    df['RSI_cross50_5']  = rsi_above.rolling(5).sum()
    df['RSI_cross50_10'] = rsi_above.rolling(10).sum()
    df['RSI_cross50_20'] = rsi_above.rolling(20).sum()

    # MA20 kaç gündür fiyatın altında/üstünde
    ma20_below = (c < df['MA20']).astype(int)
    df['MA20_below_days'] = ma20_below.rolling(20).sum()
    df['MA50_below_days'] = (c < df['MA50']).astype(int).rolling(20).sum()

    # Son 5/10/20 günde kaç gün yeşil kapandı
    df['GreenDay_5']  = (c > c.shift(1)).astype(int).rolling(5).sum()
    df['GreenDay_10'] = (c > c.shift(1)).astype(int).rolling(10).sum()
    df['GreenDay_20'] = (c > c.shift(1)).astype(int).rolling(20).sum()

    # RSI trendi (son 5/10 günde RSI kaç puan değişti)
    df['RSI_trend5']  = df['RSI14'] - df['RSI14'].shift(5)
    df['RSI_trend10'] = df['RSI14'] - df['RSI14'].shift(10)

    # MACD histogramı ardışık artış/azalış
    macd_inc = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)).astype(int)
    df['MACD_inc_5'] = macd_inc.rolling(5).sum()

    # Hacim trendi (son 5/10 günde ortalama hacim önceki 20 güne oranı)
    df['Vol_trend5']  = v.rolling(5).mean() / (v.rolling(20).mean()+1e-10)
    df['Vol_trend10'] = v.rolling(10).mean() / (v.rolling(20).mean()+1e-10)

    # Fiyat ivmesi (momentumun momentumu)
    df['Mom_accel5']  = df['Mom5']  - df['Mom5'].shift(5)
    df['Mom_accel10'] = df['Mom10'] - df['Mom10'].shift(5)

    # ── MEVSİMSELLİK — Takvim özellikleri ────────────────────────────────────
    idx = df.index
    df['Gun_haftada']  = idx.dayofweek.astype(float)        # 0=Pzt, 4=Cum
    df['Ay']           = idx.month.astype(float)
    df['Ay_gun']       = idx.day.astype(float)
    df['Hafta_yilda']  = idx.isocalendar().week.astype(float)
    df['Ceyrek']       = idx.quarter.astype(float)
    df['Ay_sonu']      = (idx.day >= 25).astype(float)      # Ay sonu etkisi
    df['Ay_basi']      = (idx.day <= 5).astype(float)       # Ay başı etkisi
    df['Ceyrek_sonu']  = idx.month.isin([3,6,9,12]).astype(float)

    # ── VOLATİLİTE REJİMİ — Kriz tespiti ─────────────────────────────────────
    # ATR14'ün 60 günlük ortalamasına oranı
    atr_long = df['ATR14'].rolling(60).mean()
    df['Rejim_vol']   = df['ATR14'] / (atr_long+1e-10)
    df['Kriz_rejim']  = (df['Rejim_vol'] > 1.5).astype(int)  # Yüksek volatilite
    df['Sakin_rejim'] = (df['Rejim_vol'] < 0.7).astype(int)  # Düşük volatilite

    # Bollinger sıkışma (volatilite azalıyor = büyük hareket hazırlığı)
    df['BB_sıkısma'] = df['BB20_Genis'] / (df['BB20_Genis'].rolling(20).mean()+1e-10)
    df['Destek20'] = df['Low'].rolling(20).min()
    df['Direnc20'] = df['High'].rolling(20).max()
    df['Hacim_an'] = (df['Volume']>df['Volume'].rolling(20).mean()*2).astype(int)

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
                seriler[col]              = s
                seriler[f'{col}_deg1']    = s.pct_change(1)
                seriler[f'{col}_deg5']    = s.pct_change(5)
                seriler[f'{col}_ma5']     = s.rolling(5).mean()
                seriler[f'{col}_vol14']   = s.pct_change().rolling(14).std()
                seriler[f'{col}_oran']    = s / (s.rolling(20).mean() + 1e-10)
            except Exception:
                continue

        if not seriler:
            return df_h

        extra = pd.DataFrame(seriler)
        extra = _index_temizle(extra)   # son bir kez daha temizle
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
        return True   # Zaten yüklenmiş
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
        RobustScaler = _RS; VarianceThreshold = _VT
        accuracy_score = _acc; f1_score = _f1
        GradientBoostingClassifier = _GBC; RandomForestClassifier = _RFC
        HistGradientBoostingClassifier = _HGBC; ExtraTreesClassifier = _ETC
        compute_sample_weight = _CSW; LogisticRegression = _LR
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
        xgb = _x; XGB_OK = True; return True
    except ImportError:
        XGB_OK = False; return False


def _lgb_yukle():
    """LightGBM lazy yükle."""
    global lgb, LGB_OK
    if lgb is not None:
        return True
    try:
        import lightgbm as _l
        lgb = _l; LGB_OK = True; return True
    except ImportError:
        LGB_OK = False; return False


def _yf_yukle():
    """yfinance lazy yükle."""
    global yf, YF_OK
    if yf is not None:
        return True
    try:
        import yfinance as _y
        yf = _y; YF_OK = True; return True
    except ImportError:
        YF_OK = False; return False


def _plotly_yukle():
    """Plotly lazy yükle."""
    global go, px, make_subplots, PLOTLY_OK
    if go is not None:
        return True
    try:
        import plotly.graph_objects as _go
        import plotly.express as _px
        from plotly.subplots import make_subplots as _ms
        go = _go; px = _px; make_subplots = _ms
        PLOTLY_OK = True; return True
    except ImportError:
        PLOTLY_OK = False; return False


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
        A4=_A4; getSampleStyleSheet=_gss; ParagraphStyle=_PS
        SimpleDocTemplate=_SDT; Paragraph=_P; Spacer=_Sp
        Table=_T; TableStyle=_TS; colors=_col; cm=_cm
        REPORTLAB_OK = True; return True
    except ImportError:
        return False

def _psutil_yukle():
    """psutil lazy yükle."""
    global _ps, PSUTIL_OK
    if _ps is not None: return True
    try:
        import psutil as _p; _ps = _p; PSUTIL_OK = True; return True
    except ImportError:
        return False

def _catboost_yukle():
    """CatBoost lazy yükle."""
    global CatBoostClassifier, CATBOOST_OK
    if CatBoostClassifier is not None: return True
    try:
        from catboost import CatBoostClassifier as _CB
        CatBoostClassifier = _CB; CATBOOST_OK = True; return True
    except ImportError:
        CATBOOST_OK = False; return False



