"""
explainability.py
-----------------
SHAP ve LIME kullanarak model kararlarini aciklayan modul.

Pipeline:
    1. compute_shap_values()     -> Tum test seti icin SHAP degerleri
    2. get_shap_for_patient()    -> Tek hasta icin SHAP ozeti (dict)
    3. compute_lime_explanation()-> Tek hasta icin LIME aciklamasi
    4. explain_patient()         -> SHAP + LIME birlesik dict (LLM'e gidecek)

Model-Explainer eslesmesi:
    XGBoost / RandomForest -> TreeExplainer  (hizli, kesin)
    LogisticRegression     -> LinearExplainer (katsayilara dayali)
"""

import numpy as np
import shap
import lime
import lime.lime_tabular

from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.config import RANDOM_STATE

# ---------------------------------------------------------------------------
# Tip tanimlari
# ---------------------------------------------------------------------------

# Tek hasta icin SHAP ozeti: {"Glucose": 0.42, "BMI": 0.31, ...}
ShapSummary = dict[str, float]

# LLM'e gidecek tam aciklama paketi
PatientExplanation = dict


# ---------------------------------------------------------------------------
# Adim 1: SHAP Degerleri Hesapla
# ---------------------------------------------------------------------------

def build_shap_explainer(
    model,
    X_train: np.ndarray,
) -> shap.Explainer:
    """
    Model tipine gore dogru SHAP Explainer'i olusturur.

    Neden farkli Explainer'lar?
        TreeExplainer:
            Agac yapisini (bolunme noktalari, yaprak degerleri) dogrudan okur.
            Cok hizli — O(n) karmasiklik.
            XGBoost ve RandomForest icin kullan.

        LinearExplainer:
            Logistic Regression'in agirlik vektoru (coef_) uzerinden
            tam analitik SHAP degerleri hesaplar.
            Hizli ve kesin.

        KernelExplainer (burada kullanmiyoruz):
            Her model icin calisir ama cok yavas — Monte Carlo ornekleme.
            768 satir veri icin bile dakikalar surebilir.

    Args:
        model:   Egitilmis sklearn veya xgboost modeli.
        X_train: Egitim verisi — background distribution icin gerekli.

    Returns:
        Uygun SHAP Explainer nesnesi.
    """
    if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
        # Agac modelleri icin — hizli ve kesin
        explainer = shap.TreeExplainer(model)
        logger.info(f"TreeExplainer olusturuldu: {type(model).__name__}")

    elif isinstance(model, LogisticRegression):
        # Dogrusal model icin — arka plan dagilimi icin X_train ornegi
        # Tum X_train'i vermek yerine 100 ornek yeterli (hiz icin)
        background = shap.sample(X_train, 100, random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(model, background)
        logger.info("LinearExplainer olusturuldu: LogisticRegression")

    else:
        # Tanimsiz model tipi — KernelExplainer ile fallback
        logger.warning(
            f"Tanimsiz model tipi: {type(model).__name__}. "
            "KernelExplainer kullaniliyor (yavas olabilir)."
        )
        background = shap.sample(X_train, 50, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
) -> np.ndarray:
    """
    Verilen veri seti icin SHAP degerlerini hesaplar.

    SHAP degeri ne anlatiyor:
        Her satir bir ornek (hasta), her sutun bir ozelliktir.
        Deger > 0 : Bu ozellik "diyabetli" tahminini artirdi
        Deger < 0 : Bu ozellik "diyabetli" tahminini azaltti
        |Deger|   : Etkinin buyuklugu

    Args:
        explainer: build_shap_explainer() ile olusturulmus explainer.
        X:         SHAP hesaplanacak ornekler. Shape: (n, 8)

    Returns:
        SHAP degerleri matrisi. Shape: (n, 8)
    """
    shap_values = explainer.shap_values(X)

    # RandomForest 2 sinif icin [sinif_0, sinif_1] listesi dondurur
    # Biz pozitif sinif (diyabetli=1) ile ilgileniyoruz
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # sinif 1 (diyabetli)

    logger.info(f"SHAP degerleri hesaplandi: {shap_values.shape}")
    return shap_values


# ---------------------------------------------------------------------------
# Adim 2: Tek Hasta icin SHAP Ozeti
# ---------------------------------------------------------------------------

def get_shap_for_patient(
    shap_values: np.ndarray,
    patient_index: int,
    feature_names: list[str],
    top_n: int = 5,
) -> ShapSummary:
    """
    Tek bir hastanin SHAP degerlerini isimli sozluge cevir.

    LLM'e tum 8 ozelligi gondermek yerine en etkili top_n ozelligi
    gondermek hem daha odakli hem de token tasarruflu.

    Args:
        shap_values:   compute_shap_values() ciktisi. Shape: (n, 8)
        patient_index: Kac numarali hastanin aciklamasi isteniyor?
        feature_names: Ozellik isimlerinin sirali listesi.
        top_n:         Kac ozellik donulecek? (mutlak SHAP buyuklugune gore)

    Returns:
        {"Glucose": 0.42, "BMI": 0.31, ...} — buyukten kucuge sirali
    """
    patient_shap = shap_values[patient_index]  # Shape: (8,)

    # Isim-deger eslesmesi yap (dict comprehension)
    shap_dict: ShapSummary = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, patient_shap)
    }

    # Mutlak degere gore sirala (en etkili once)
    shap_sorted = dict(
        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Sadece top_n ozellik al
    top_shap = dict(list(shap_sorted.items())[:top_n])

    logger.info(f"Hasta {patient_index} icin top-{top_n} SHAP: {top_shap}")
    return top_shap


# ---------------------------------------------------------------------------
# Adim 3: LIME Aciklamasi
# ---------------------------------------------------------------------------

def build_lime_explainer(
    X_train: np.ndarray,
    feature_names: list[str],
) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Tabular veri icin LIME Explainer olusturur.

    LIME nasil calisir:
        1. Aciklanacak noktanin (hastanin) etrafinda rastgele ornekler uretir
        2. Bu ornekleri asil modele verir, tahminleri alir
        3. Bu kucuk komsuluktaki davranisi yakalamak icin
           basit bir dogrusal model (LASSO) egitir
        4. Bu dogrusal modelin katsayilari = LIME aciklamalari

    SHAP'tan farki:
        SHAP: Global referansa gore katkiyi olcer (daha tutarli)
        LIME: Lokal komsulugu modelleyen yaklasim (o hastaya ozgun)
        Ikisi ayni sonucu vermezse -> model davranisi karmasik demektir

    Args:
        X_train:       Egitim verisi — LIME ornekleme icin referans dagilim.
        feature_names: Ozellik isimleri.

    Returns:
        LimeTabularExplainer nesnesi.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Saglikli", "Diyabetli"],
        mode="classification",
        random_state=RANDOM_STATE,
    )
    logger.info("LIME LimeTabularExplainer olusturuldu")
    return explainer


def compute_lime_explanation(
    lime_explainer: lime.lime_tabular.LimeTabularExplainer,
    model,
    patient_data: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> ShapSummary:
    """
    Tek bir hasta icin LIME aciklamasi uretir.

    Args:
        lime_explainer: build_lime_explainer() ile olusturulmus explainer.
        model:          Egitilmis model (predict_proba metodu olmali).
        patient_data:   Tek hastanin ozellikleri. Shape: (8,) veya (1, 8)
        feature_names:  Ozellik isimleri — LIME kural parse icin kullanilir.
        top_n:          Kac ozellik donulecek?

    Returns:
        {"Glucose": 0.38, "BMI": 0.27, ...} — LIME agirliklarini icerir.
    """
    if patient_data.ndim == 1:
        patient_data = patient_data.reshape(1, -1)

    explanation = lime_explainer.explain_instance(
        data_row=patient_data[0],
        predict_fn=model.predict_proba,
        num_features=top_n,
        num_samples=1000,  # Daha fazla ornek = daha stabil aciklama
    )

    # LIME ciktisi: [("Glucose > 130", 0.38), ("0.14 < BMI <= 30", 0.27), ...]
    # Kural ifadelerini sadece ozellik adina indirgiyoruz.
    # Onemli: LIME bazen "0.14 < BMI <= 30" gibi kurallarda sayiyla baslar.
    # Bu durumda ilk kelime bir sayi olur ("0.14") — bu bir ozellik adi degil.
    # Cozum: Kural icinde feature_names listesindeki bir isim ara.
    lime_weights: ShapSummary = {}
    for feature_rule, weight in explanation.as_list():
        matched_feature = None
        for fname in feature_names:
            if fname in feature_rule:
                matched_feature = fname
                break
        if matched_feature:
            lime_weights[matched_feature] = round(float(weight), 4)
        else:
            logger.warning(f"LIME kural parse edilemedi, atlandi: '{feature_rule}'")

    logger.info(f"LIME aciklamasi: {lime_weights}")
    return lime_weights


# ---------------------------------------------------------------------------
# Adim 4: Birlesik Aciklama (LLM'e Gidecek)
# ---------------------------------------------------------------------------

def explain_patient(
    patient_index: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    shap_explainer: shap.Explainer,
    lime_explainer: lime.lime_tabular.LimeTabularExplainer,
    feature_names: list[str],
    scaler,                   # MinMaxScaler — ham degerlere donmek icin
    top_n: int = 5,
) -> PatientExplanation:
    """
    Tek bir hasta icin SHAP ve LIME aciklamalarini birlestirir.

    Bu fonksiyonun ciktisi dogrudan llm_handler.py'e gider.
    LLM bu dict'i okuyarak Turkce, empatik, medikal disclaimer
    iceren bir rapor uretir.

    Args:
        patient_index:   Test setindeki hasta indeksi.
        X_test:          Olceklenmis test ozellikleri.
        y_test:          Gercek etiketler.
        model:           Aciklanacak egitilmis model.
        shap_explainer:  SHAP explainer nesnesi.
        lime_explainer:  LIME explainer nesnesi.
        feature_names:   Ozellik isimlerinin listesi.
        scaler:          Egitimde kullanilan MinMaxScaler —
                         inverse_transform ile gercek degerlere donmek icin.
        top_n:           Her yontem icin kac ozellik alinacak.

    Returns:
        {
            "patient_index": 42,
            "true_label": 1,
            "prediction": {"label": "Diyabetli", "probability": 0.78},
            "raw_values": {"Glucose": 147.0, "BMI": 32.1, ...},  # Gercek degerler
            "shap": {"Glucose": 0.42, "BMI": 0.31, ...},
            "lime": {"Glucose": 0.38, "BMI": 0.27, ...},
            "agreement": ["Glucose", "BMI"],
        }
    """
    patient_data = X_test[patient_index]
    true_label   = int(y_test[patient_index])

    # --- Tahmin ---
    prob = float(model.predict_proba(patient_data.reshape(1, -1))[0][1])
    label = "Diyabetli" if prob >= 0.5 else "Saglikli"

    # --- SHAP ---
    shap_values  = compute_shap_values(shap_explainer, X_test)
    shap_summary = get_shap_for_patient(shap_values, patient_index, feature_names, top_n)

    # --- LIME ---
    lime_summary = compute_lime_explanation(lime_explainer, model, patient_data, feature_names, top_n)

    # --- Ham degerler: inverse_transform ile gercek degerlere don ---
    # Neden scaler.inverse_transform?
    #   X_test MinMaxScaler'dan gecmis: Glucose=147 -> 0.72
    #   LLM'e 0.72 gondermek anlamsiz — "Glikoz 0.72" yazamaz.
    #   inverse_transform: 0.72 -> 147 (tam tersine cevir)
    X_original = scaler.inverse_transform(X_test)
    raw_values: dict[str, float] = {
        name: round(float(val), 2)
        for name, val in zip(feature_names, X_original[patient_index])
    }

    # --- Uzlasma: Her iki yontemin de ust N'de saydigi ozellikler ---
    shap_top_features = set(shap_summary.keys())
    lime_top_features = set(lime_summary.keys())
    agreement = list(shap_top_features & lime_top_features)

    explanation: PatientExplanation = {
        "patient_index": patient_index,
        "true_label":    true_label,
        "prediction": {
            "label":       label,
            "probability": round(prob, 4),
        },
        "raw_values":  raw_values,   # Gercek klinik degerler (mg/dL, kg/m2 vb.)
        "shap":        shap_summary,
        "lime":        lime_summary,
        "agreement":   agreement,
    }

    logger.info(
        f"Hasta {patient_index} aciklamasi hazirlandi | "
        f"Tahmin: {label} ({prob:.2%}) | "
        f"SHAP-LIME uzlasma: {agreement}"
    )
    return explanation

