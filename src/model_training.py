"""
model_training.py
-----------------
Pima Diabetes veri seti icin model egitimi ve degerlendirme modulu.

Egitilen modeller:
    1. Logistic Regression  — baseline, yorumlanabilir
    2. Random Forest        — ensemble, guclu baseline
    3. XGBoost              — gradient boosting, genellikle en iyi performans

Temel kararlar:
    - class_weight="balanced" / scale_pos_weight: sinif dengesizligini ele alir
    - Recall ana metrik: False Negative (kacan diyabetli) klinik olarak daha tehlikeli
    - Tum modeller ayni egitim/test verisiyle degerlendirilir (adil karsilastirma)
    - Egitilen modeller models/ klasorune .joblib olarak kaydedilir
"""

import numpy as np
import joblib
from pathlib import Path
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

from src.config import MODELS_DIR, RANDOM_STATE

# ---------------------------------------------------------------------------
# Tip Kisaltmalari
# ---------------------------------------------------------------------------
ModelMetrics = dict[str, float]          # {"recall": 0.82, "f1": 0.76, ...}
ConfusionMtx = np.ndarray                # 2x2 numpy array


# ---------------------------------------------------------------------------
# Adim 1: Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """
    Logistic Regression modelini egitir.

    Neden Logistic Regression ilk?
        - En basit siniflandirici — baseline referans noktasi olusturur.
        - Katsayilari dogrudan yorumlanabilir (XAI adimiyla uyumlu).
        - Egitimi saniyeler icerisinde tamamlanir.

    Parametre kararlari:
        class_weight="balanced":
            Sinif agirliklarini otomatik hesaplar.
            Azinlik sinifi (diyabetli=1) daha yuksek agirlik alir.
            Formul: w_i = n_samples / (n_classes * n_samples_i)

        max_iter=1000:
            Varsayilan 100 iterasyon bu veri seti icin yetersiz kalabilir.
            Convergence uyarisi almamak icin arttirildi.

        random_state:
            Tekrarlanabilirlik icin sabit tohum.

    Args:
        X_train: Olceklenmis egitim ozellikleri. Shape: (n, 8)
        y_train: Egitim etiketleri. Shape: (n,)

    Returns:
        Egitilmis LogisticRegression modeli.
    """
    logger.info("Logistic Regression egitiliyor...")

    model = LogisticRegression(
        class_weight="balanced",   # Sinif dengesizligini ele al
        max_iter=1000,             # Convergence icin yeterli iterasyon
        random_state=RANDOM_STATE,
        solver="lbfgs",            # Kucuk-orta veri setleri icin verimli
    )

    model.fit(X_train, y_train)
    logger.info("Logistic Regression egitimi tamamlandi")
    return model


# ---------------------------------------------------------------------------
# Adim 2: Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Random Forest modelini egitir.

    Random Forest nedir?
        Cok sayida karar agaci (n_estimators) ayri ayri egitilir.
        Her agac verinin rastgele bir alt kumesi ve rastgele ozellikler
        uzerinde egitilir (bootstrap + feature randomness).
        Final tahmin: tum agaclarin oy cogunlugu (majority vote).

    Neden guclu bir baseline?
        - Tek agaca gore cok daha az overfit eder (ensemble etkisi)
        - Scaling gerektirmez (ama biz yine de uyguladik — tutarlilik icin)
        - Feature importance dogal olarak cikar (SHAP ile uyumlu)

    Parametre kararlari:
        n_estimators=200:
            200 agac. 100 cok az, 500 gereksiz yavas.
            Bu veri boyutu icin 200 iyi denge noktasi.

        max_depth=None:
            Agaclarin derinligi sinirlanmaz — her agac tam buyur.
            Overfitting riski var ama ensemble etkisi dengeler.

        min_samples_leaf=2:
            Her yaprak dugumunde en az 2 ornek olmali.
            Asiri ozellesmis (overfitted) agaclari hafifce kisitlar.

        class_weight="balanced":
            Logistic Regression ile ayni mantik — azinlik sinifina
            daha yuksek agirlik verir.

    Args:
        X_train: Egitim ozellikleri. Shape: (n, 8)
        y_train: Egitim etiketleri. Shape: (n,)

    Returns:
        Egitilmis RandomForestClassifier modeli.
    """
    logger.info("Random Forest egitiliyor...")

    model = RandomForestClassifier(
        n_estimators=200,           # Agac sayisi
        max_depth=None,             # Sinirsiz derinlik
        min_samples_leaf=2,         # Hafif overfitting kisiti
        class_weight="balanced",    # Sinif dengesizligi
        random_state=RANDOM_STATE,
        n_jobs=-1,                  # Tum CPU cekirdeklerini kullan
    )

    model.fit(X_train, y_train)
    logger.info("Random Forest egitimi tamamlandi")
    return model


# ---------------------------------------------------------------------------
# Adim 3: XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> xgb.XGBClassifier:
    """
    XGBoost modelini egitir.

    XGBoost nedir?
        Gradient Boosted Trees — agaclar sirayla eklenir, her yeni agac
        bir oncekinin hatalarini duzeltmeye calisir (boosting).
        Random Forest'in aksine agaclar PARALEL degil, SIRAYLA egitilir.

    Neden genellikle en iyi performans?
        - Her iterasyonda kayip fonksiyonu minimize edilir (gradyan)
        - Regularization (L1/L2) yerlesik gelir — overfit riski dusuk
        - Kategorik veride de guclu

    Parametre kararlari:
        scale_pos_weight = n_negative / n_positive ≈ 500/268 ≈ 1.87:
            XGBoost'un class_weight karsiligi.
            "Pozitif sinifi tahmin etmek, negatif sinifi tahmin etmekten
             bu kadar daha degerli" anlamina gelir.
            Bunu hep data'dan hesaplariz, sabit kodlamayiz.

        n_estimators=200:
            Random Forest'teki gibi 200 agac — ama burada sirayla eklenir.

        learning_rate=0.1:
            Her yeni agacin katkilari bu katsayiyla olceklenir.
            Dusuk LR → yavas ogrenme ama genellikle daha iyi genelleme.
            Yuksek LR → hizli ama overfit riski.

        max_depth=4:
            Bireysel agaclarin derinligi. Boosting'de agaclar genellikle
            sığ (3-6) tutulur — deep tree'ye gerek yok cunku boosting zaten
            katmanli ogrenme yapiyor.

        subsample=0.8:
            Her agac icin verilerin %80'i rastgele secilir.
            Hem overfitting azalir hem de cesitlilik artar.

        colsample_bytree=0.8:
            Her agac icin ozelliklerin %80'i rastgele secilir.
            Random Forest'in feature randomness mantigi.

        use_label_encoder=False, eval_metric="logloss":
            Sklearn API uyumlulugu icin — uyari mesajlarini onler.

    Args:
        X_train: Egitim ozellikleri. Shape: (n, 8)
        y_train: Egitim etiketleri. Shape: (n,)

    Returns:
        Egitilmis XGBClassifier modeli.
    """
    logger.info("XGBoost egitiliyor...")

    # scale_pos_weight'i veriden hesapla (sabit kodlama yapmiyoruz)
    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())
    scale_pos_weight = n_negative / n_positive
    logger.info(
        f"scale_pos_weight = {n_negative}/{n_positive} = {scale_pos_weight:.3f}"
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,              # Gereksiz XGBoost loglarini sustur
    )

    model.fit(X_train, y_train)
    logger.info("XGBoost egitimi tamamlandi")
    return model


# ---------------------------------------------------------------------------
# Adim 4: Model Degerlendirme
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> ModelMetrics:
    """
    Egitilmis modeli test seti uzerinde degerlendirir.

    Neden bu metrik seti?
        Recall     : Ana metrik — klinik olarak kritik (FN'yi minimize et)
        Precision  : FP dengesini takip et — cok fazla yanlis alarm da sorun
        F1-Score   : Recall ve Precision'in harmonik ortalamasi
        ROC-AUC    : Threshold'dan bagimsiz model gucu — modeller arasi karsilastirma
        Accuracy   : Baglam icin — tek basina yaniltici olabilir

    Confusion Matrix:
        [[TN  FP]
         [FN  TP]]
        TN: Doğru tahmin, Diyabeti YOK
        FP: Saglikli, yanlis "diyabetli" tahmin  (False Alarm)
        FN: Diyabetli, yanlis "saglikli" tahmin  (Tehlikeli!)
        TP: Diyabetli, dogru "diyabetli" tahmin

    Args:
        model:      Egitilmis sklearn/xgboost modeli.
        X_test:     Test ozellikleri.
        y_test:     Gercek test etiketleri.
        model_name: Log ve rapor icin model adi.

    Returns:
        Metrik sozlugu: {"accuracy": ..., "recall": ..., ...}
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Pozitif sinif olasiligi

    metrics: ModelMetrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_pred_prob), 4),
    }

    cm: ConfusionMtx = confusion_matrix(y_test, y_pred)

    # Log ciktilari
    logger.info(f"\n{'='*45}")
    logger.info(f"MODEL: {model_name}")
    logger.info(f"{'='*45}")
    logger.info(f"  Accuracy  : {metrics['accuracy']}")
    logger.info(f"  Precision : {metrics['precision']}")
    logger.info(f"  Recall    : {metrics['recall']}  <- Ana metrik")
    logger.info(f"  F1-Score  : {metrics['f1']}")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']}")
    logger.info(f"\n  Confusion Matrix:\n"
                f"    TN={cm[0,0]}  FP={cm[0,1]}\n"
                f"    FN={cm[1,0]}  TP={cm[1,1]}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return metrics


# ---------------------------------------------------------------------------
# Adim 5: Modeli Kaydet
# ---------------------------------------------------------------------------

def save_model(model, model_name: str) -> Path:
    """
    Egitilmis modeli models/ klasorune .joblib olarak kaydeder.

    Neden joblib?
        - pickle'a gore buyuk numpy array'leri daha verimli serialize eder
        - sklearn ve xgboost modelleri icin standart tercih

    Args:
        model:      Kaydedilecek egitilmis model nesnesi.
        model_name: Dosya adi prefix'i (orn: "logistic_regression")

    Returns:
        Kaydedilen dosyanin tam yolu.
    """
    out_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, out_path)
    logger.info(f"Model kaydedildi: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Ana Pipeline: Tum Modelleri Egit ve Degerlendir
# ---------------------------------------------------------------------------

def run_training_pipeline(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> dict[str, tuple]:
    """
    Uc modeli sirayla egitir, degerlendirir ve kaydeder.

    Args:
        X_train, X_test: Olceklenmis ozellik matrisleri.
        y_train, y_test: Etiket vektorleri.

    Returns:
        {
            "logistic_regression": (model, metrics),
            "random_forest":       (model, metrics),
            "xgboost":             (model, metrics),
        }
    """
    logger.info("=" * 50)
    logger.info("MODEL EGITIM PIPELINE BASLIYOR")
    logger.info("=" * 50)

    results: dict[str, tuple] = {}

    # --- Logistic Regression ---
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    save_model(lr_model, "logistic_regression")
    results["logistic_regression"] = (lr_model, lr_metrics)

    # --- Random Forest ---
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    save_model(rf_model, "random_forest")
    results["random_forest"] = (rf_model, rf_metrics)

    # --- XGBoost ---
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    save_model(xgb_model, "xgboost")
    results["xgboost"] = (xgb_model, xgb_metrics)

    # --- Karsilastirma Ozeti ---
    logger.info("\n" + "=" * 50)
    logger.info("MODEL KARSILASTIRMA OZETI")
    logger.info("=" * 50)
    logger.info(f"{'Model':<25} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    logger.info("-" * 52)
    for name, (_, m) in results.items():
        logger.info(
            f"{name:<25} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )

    logger.info("=" * 50)
    logger.info("MODEL EGITIM PIPELINE TAMAMLANDI")
    logger.info("=" * 50)

    return results
