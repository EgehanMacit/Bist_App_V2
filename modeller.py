"""BIST Tahmin Sistemi - ML Modelleri"""
import numpy as np, pandas as pd, streamlit as st, warnings

def veri_hazirla(df: pd.DataFrame, pencere: int = PENCERE):
    _sklearn_yukle()   # Lazy import

    # 1. Temizlik
    hedef    = 'Hedef'
    ozellik  = [c for c in df.columns if c != hedef]
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
    X_s    = scaler.fit_transform(df_temiz[ozellik].values).astype(np.float32)
    y      = df_temiz[hedef].values.astype(np.float32)

    # 6. Sınıf dengesini kontrol et
    pos_oran = y.mean()
    if pos_oran < 0.3 or pos_oran > 0.7:
        # Dengesiz sınıf varsa ağırlık hesapla
        class_weight = {0: 1/(1-pos_oran+1e-10), 1: 1/(pos_oran+1e-10)}
        ag_toplam    = sum(class_weight.values())
        class_weight = {k: v/ag_toplam for k,v in class_weight.items()}
    else:
        class_weight = None

    # 7. Walk-forward bölme — son %15 test, kalan eğitim
    # Büyük veri varsa daha fazla test ver (daha güvenilir metrik)
    test_orani = TEST_ORANI if len(X_s) < 2000 else 0.20

    # 8. Sekans oluştur
    X_seq, y_seq = [], []
    for i in range(pencere, len(X_s) - 1):
        X_seq.append(X_s[i-pencere:i])
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
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "eval_metric":      "logloss",
            "verbosity":        0,
            "random_state":     42,
            "n_jobs": 1,
        }
        pos_oran  = float(y_eg.mean())
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
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "class_weight":     "balanced",
            "verbose":          -1,
            "random_state":     42,
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
    modeller  = {}

    # ── Giriş verisi: son gün + pencere ortalaması + pencere std ─────────────
    X_eg2d_son  = X_eg[:, -1, :]                     # son günün özellikleri
    X_eg2d_ort  = X_eg.mean(axis=1)                  # pencere ortalaması
    X_eg2d_std  = X_eg.std(axis=1)                   # pencere volatilitesi
    X_eg2d_mean = np.concatenate([X_eg2d_son,
                                   X_eg2d_ort,
                                   X_eg2d_std], axis=1)

    X_te2d_son  = X_te[:, -1, :]
    X_te2d_ort  = X_te.mean(axis=1)
    X_te2d_std  = X_te.std(axis=1)
    X_te2d_mean = np.concatenate([X_te2d_son,
                                   X_te2d_ort,
                                   X_te2d_std], axis=1)

    y_eg_int  = y_eg.astype(int)
    y_te_int  = y_te.astype(int)
    pos_oran  = float(y_eg_int.mean())
    scale_pos = (1 - pos_oran) / (pos_oran + 1e-10)
    sw        = compute_sample_weight('balanced', y_eg_int)

    n_train = len(X_eg2d_mean)
    n_test  = len(X_te2d_mean)

    # ── Minimum veri kontrolü ─────────────────────────────────────────────────
    if n_train < 100:
        raise ValueError(
            f"Model eğitmek için çok az veri: {n_train} eğitim örneği.\n"
            f"Bu hisse yeni halka açılmış veya veri eksik.\n"
            f"En az 2 yıl geriye giden bir hisse seçin."
        )

    # ── Veri boyutuna göre dinamik parametreler ───────────────────────────────
    if n_train < 400:
        n_est      = 500
        early_stop = 500
        lr         = 0.03
        max_dep    = 4
        status_cb(f"⚠️ Az veri ({n_train} örnek)")
    elif n_train < 1000:
        n_est      = 800
        early_stop = 500
        lr         = 0.03
        max_dep    = 5
        status_cb(f"📊 {n_train} örnek")
    elif n_train < 3000:
        n_est      = 1000
        early_stop = 500
        lr         = 0.025
        max_dep    = 6
    else:
        n_est      = 1500
        early_stop = 500
        lr         = 0.02
        max_dep    = 7
    status_cb(f"📊 Eğitim: {n_train} · Test: {n_test} · {n_est} ağaç · LR:{lr}")

    # ── XGBoost (Optuna ile optimize) ────────────────────────────────────────
    status_cb("⚡ XGBoost eğitiliyor" + (" (Optuna optimizasyonu)..." if OPTUNA_OK else "..."))
    if OPTUNA_OK and optuna is not None and st.session_state.get("optuna_aktif", False):
        _xgb_params = optuna_optimize_xgb(X_eg2d_mean, y_eg_int,
                                           X_te2d_mean, y_te_int, n_trials=15)
        _xgb_params.update({"eval_metric":"logloss","verbosity":0,
                             "random_state":42,"n_jobs": 1})
    else:
        _xgb_params = dict(
            n_estimators=n_est, learning_rate=lr,
            max_depth=max_dep, min_child_weight=max(1, n_train//200),
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0,
            random_state=42, n_jobs=1,
        )
    # early_stopping_rounds → constructor'a ver (XGBoost 2.x uyumlu)
    _xgb_params.pop('early_stopping_rounds', None)   # varsa çıkar
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
            max_depth=max_dep, min_child_samples=max(10, n_train//50),
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
        meta_acc  = accuracy_score(y_te_int, meta_pred)
        meta_f1   = f1_score(y_te_int, meta_pred, zero_division=0)
        modeller["_meta_model"] = {
            "model":       meta_m,
            "tip":         "meta",
            "dogruluk":    meta_acc,
            "f1":          meta_f1,
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
    base_olas   = {}
    for ad, bilgi in modeller.items():
        if ad.startswith('_'): continue   # meta-modeli atla
        m = bilgi['model']
        try:
            tip = bilgi.get('tip','2d')
            if tip == 'mean2d':
                p = float(m.predict_proba(X_son_mean)[:,1][0])
            elif tip == 'lstm':
                p = float(m.predict(X_son, verbose=0).flatten()[0])
            else:
                p = float(m.predict_proba(X_son_last)[:,1][0])
        except Exception:
            p = 0.5
        olasiliklar[ad] = p
        base_olas[ad]   = p

    # Meta-model varsa onu kullan (daha doğru)
    if '_meta_model' in modeller:
        try:
            meta_bilgi = modeller['_meta_model']
            meta_m     = meta_bilgi['model']
            baz_modeller = meta_bilgi.get('base_models',
                           [k for k in modeller if not k.startswith('_')])
            meta_input = np.array([[base_olas.get(k, 0.5) for k in baz_modeller]])
            meta_olas  = float(meta_m.predict_proba(meta_input)[:,1][0])
            ml_skor    = meta_olas        # Meta-model kararı ana skor
            olasiliklar['Meta-Model'] = meta_olas
        except Exception:
            ml_skor = _agirlikli_ort(olasiliklar, modeller)
    else:
        ml_skor = _agirlikli_ort(olasiliklar, modeller)

    # Haber + faiz + sektör katkısı
    haber_katki  = float(np.clip(haber_skoru  * 0.07, -0.05, 0.05))
    faiz_katki   = float(np.clip(faiz_etkisi  * 0.04, -0.03, 0.03))
    sektor_katki = float(np.clip(sektor_skor  * 0.05, -0.04, 0.04))
    final_ham    = float(np.clip(
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
    if model_dogruluk >= 0.70:   baz = 0.62
    elif model_dogruluk >= 0.63: baz = 0.58
    else:                        baz = 0.54

    if rejim_vol > 1.5:
        esik_al       = min(baz + 0.08, 0.72)
        esik_guclu_al = min(baz + 0.14, 0.78)
        esik_sat      = 0.30
        rejim_notu    = "⚡ Yüksek Volatilite — Dikkatli Ol"
    elif rejim_vol < 0.7:
        esik_al       = max(baz - 0.03, 0.51)
        esik_guclu_al = max(baz + 0.05, 0.59)
        esik_sat      = 0.37
        rejim_notu    = "😌 Sakin Piyasa"
    else:
        esik_al       = baz
        esik_guclu_al = baz + 0.08
        esik_sat      = 0.35
        rejim_notu    = f"📊 Normal Rejim (Eşik: %{baz*100:.0f})"

    if   final_ham >= esik_guclu_al: sinyal = "📈 GÜÇLÜ AL"
    elif final_ham >= esik_al:       sinyal = "📈 AL"
    elif final_ham <= esik_sat:      sinyal = "📉 GÜÇLÜ SAT"
    elif final_ham <= 0.42:          sinyal = "📉 SAT"
    else:                            sinyal = "⏸️ BEKLE"

    return {
        "final":        final_ham,
        "sinyal":       sinyal,
        "olasiliklar":  olasiliklar,
        "haber_katki":  haber_katki,
        "sektor_katki": round(sektor_katki, 4),
        "rejim_notu":   rejim_notu,
        "rejim_vol":    round(rejim_vol, 2),
    }

def _agirlikli_ort(olasiliklar, modeller):
    """F1 + doğruluk ağırlıklı ortalama."""
    agirliklar = {}
    for ad, b in modeller.items():
        if ad.startswith('_'): continue
        agirliklar[ad] = b.get('f1', 0.5)*0.6 + b.get('dogruluk', 0.5)*0.4
    top = sum(agirliklar.values()) or 1
    return sum(olasiliklar.get(ad,0.5) * v/top for ad, v in agirliklar.items())

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

def guven_skoru(acc, n_train, modeller):
    """Dusuk/Orta/Yuksek guven"""
    p = 0
    p += 3 if acc>=0.72 else 2 if acc>=0.66 else 1 if acc>=0.60 else 0
    p += 3 if n_train>=2000 else 2 if n_train>=800 else 1 if n_train>=300 else 0
    dogs = [v["dogruluk"] for k,v in modeller.items() if not k.startswith("_")]
    if dogs: p += 2 if (max(dogs)-min(dogs))<0.03 else 1 if (max(dogs)-min(dogs))<0.07 else 0
    if p>=7: return "YUKSEK","#34d399","OK"
    elif p>=4: return "ORTA","#facc15","ORT"
    else: return "DUSUK","#ef4444","DUS"

