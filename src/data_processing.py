"""
data_processing.py
------------------
Pima Diabetes veri seti için ön işleme pipeline'ı.

Pipeline sırası:
    1. Veri yükleme
    2. Geçersiz 0'ları NaN'a çevirme
    3. Outlier tespiti ve işleme (IQR clip)
    4. Eksik değerleri doldurma (medyan imputation)
    5. Öznitelik / hedef ayrımı ve train-test split
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

# Config modülünden proje geneli sabitler
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Tıbbi olarak 0 değeri ANLAMSIZ olan sütunlar — bu sütunlardaki 0'lar
# "veri toplanamamış" anlamına gelir ve NaN olarak işaretlenmeli.
# NOT: "Pregnancies" bu listede YOK çünkü hiç hamile kalmamış olmak geçerli.
INVALID_ZERO_COLUMNS: list[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


# ---------------------------------------------------------------------------
# Adım 1: Veri Yükleme
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Ham CSV dosyasını yükler ve temel bilgileri loglar.

    Args:
        path: CSV dosyasının yolu. Varsayılan: config'deki RAW_DATA_PATH.

    Returns:
        Ham DataFrame.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Veri dosyasi bulunamadi: {path}\n"
            "data/raw/diabetes.csv dosyasini Kaggle'dan indirip buraya koy."
        )

    df = pd.read_csv(path)
    logger.info(f"Veri yuklendi -> {df.shape[0]} satir, {df.shape[1]} sutun")
    logger.info(f"Sutunlar: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Adım 2: Geçersiz 0'ları NaN'a Çevirme
# ---------------------------------------------------------------------------

def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tıbbi olarak imkânsız olan 0 değerlerini NaN ile değiştirir.

    Etkilenen sütunlar: Glucose, BloodPressure, SkinThickness, Insulin, BMI
    Dokunulmayan sütun : Pregnancies (0 geçerli bir değer — hiç hamile olmamak)

    Args:
        df: Ham DataFrame.

    Returns:
        Geçersiz 0'ların NaN'a çevrildiği DataFrame (kopya).
    """
    df = df.copy()  # Orijinal veriyi asla değiştirme

    for col in INVALID_ZERO_COLUMNS:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            df[col] = df[col].replace(0, np.nan)
            logger.info(f"  {col}: {zero_count} adet 0 → NaN'a çevrildi")

    total_nan = df[INVALID_ZERO_COLUMNS].isnull().sum().sum()
    logger.info(f"Toplam {total_nan} deger NaN olarak isaretlendi")
    return df


# ---------------------------------------------------------------------------
# Adim 3: Outlier Tespiti ve IQR Clip
# ---------------------------------------------------------------------------

# Hedef ve kimlik sütunlara outlier işlemi YAPMA.
# Outcome binary (0/1), Pregnancies doğal sayı — clip uygulanmaz.
OUTLIER_SKIP_COLUMNS: list[str] = ["Outcome", "Pregnancies"]


def handle_outliers(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Numerik sütunlardaki outlier değerleri IQR yöntemiyle tespit edip
    alt/üst sınıra kırpar (clip). Hiçbir satır silinmez.

    Yöntem:
        Alt sınır = Q1 - iqr_multiplier * IQR
        Üst sınır = Q3 + iqr_multiplier * IQR
        Bu sınırlar dışındaki değerler sınır değeriyle değiştirilir.

    Neden clip, drop değil:
        - 768 satırlık küçük veri setinde her satır değerli.
        - data/raw/ hiç değişmez; sadece işlenmiş kopya etkilenir.

    Args:
        df:              Geçersiz 0'ları NaN'a çevrilmiş DataFrame.
        iqr_multiplier:  IQR çarpanı. Varsayılan 1.5 (standart).
                         Daha az agresif kesmek için 3.0 kullanılabilir.

    Returns:
        Outlier değerlerin clip edildiği DataFrame (kopya).
    """
    df = df.copy()

    # Sadece sayısal sütunlara uygula, hedef ve Pregnancies'i atla
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in OUTLIER_SKIP_COLUMNS
    ]

    outlier_report: dict[str, dict] = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        # Clip oncesi kac outlier var?
        n_lower = (df[col] < lower).sum()
        n_upper = (df[col] > upper).sum()
        n_total = n_lower + n_upper

        if n_total > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            outlier_report[col] = {
                "alt_sinir": round(lower, 2),
                "ust_sinir": round(upper, 2),
                "alt_outlier": int(n_lower),
                "ust_outlier": int(n_upper),
                "toplam": int(n_total),
            }
            logger.info(
                f"  {col}: {n_total} outlier clip edildi "
                f"(alt={lower:.2f}, ust={upper:.2f})"
            )
        else:
            logger.info(f"  {col}: outlier yok")

    toplam_outlier = sum(r["toplam"] for r in outlier_report.values())
    logger.info(f"Toplam {toplam_outlier} deger clip edildi | "
                f"Satir kaybi: 0")
    return df


# ---------------------------------------------------------------------------
# Adim 4: Eksik Deger Doldurma (Median Imputation)
# ---------------------------------------------------------------------------

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    NaN degerleri her sutunun medyaniyla doldurur.

    Neden medyan, mean degil:
        - Saglik verisi genellikle saga carpik dagilim gosterir.
        - IQR clip sonrasi bile carpiklik tamamen kalkmaz.
        - Medyan carpikliktan etkilenmez; mean etkilenir.
        - Mean ve medyanin birbirini tutarliligi olan sutunlarda
          pratikte sonuc cok farki olmaz — ama medyan her durumda
          daha guvenli secimdir.

    Onemli: Medyan degerleri EGITIM verisinden hesaplanmali,
    test verisine uygulanmalidir. Bu fonksiyon ham DataFrame
    uzerinde calisir; train/test ayriminda daha ileri bir
    adimda sklearn Pipeline ile entegre edilecektir.

    Args:
        df: Outlier islemi tamamlanmis DataFrame.

    Returns:
        NaN degerlerin medyanla dolduruldugu DataFrame (kopya).
    """
    df = df.copy()

    # Hedef sutunu imputation'a dahil etme
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]

    nan_before = df[feature_cols].isnull().sum()
    toplam_nan = nan_before.sum()

    if toplam_nan == 0:
        logger.info("Imputation: NaN deger bulunamadi, islem atlandi")
        return df

    for col in feature_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(
                f"  {col}: {n_missing} NaN -> medyan={median_val:.2f} ile dolduruldu"
            )

    nan_after = df[feature_cols].isnull().sum().sum()
    logger.info(f"Imputation tamamlandi: {toplam_nan} NaN -> {nan_after} NaN kaldi")
    return df


# ---------------------------------------------------------------------------
# Adim 5: Train/Test Split ve Feature Scaling
# ---------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def scale_and_split(df: pd.DataFrame) -> tuple[
    np.ndarray, np.ndarray,   # X_train, X_test
    np.ndarray, np.ndarray,   # y_train, y_test
    MinMaxScaler,             # fit edilmis scaler (modelde tekrar kullanmak icin)
    list[str],                # feature isimlerinin sirali listesi
]:
    """
    Veriyi ozellik (X) / hedef (y) olarak ayirir, train-test split
    uygular ve X uzerinde MinMaxScaler ile olcekler.

    Neden MinMaxScaler:
        - IQR clip uygulandiktan sonra min/max degerleri artik asiri
          degerlerden arinmistir — MinMaxScaler'in tek zayif noktasi
          giderilmis olur.
        - Tum degerleri [0, 1] araligina tasir; dagilim seklinden
          bagimsizdir. Saglik verisi gibi carpik dagilimli verilerde
          StandardScaler'dan daha uygun.
        - StandardScaler mean kullanir; carpik dagilimda mean
          yaniltici bir merkez olcusu olur (biz imputation'da da
          ayni sebepten medyan sectik).

    Kritik Kural — Data Leakage:
        Scaler SADECE X_train uzerinde fit edilir.
        X_test'e yalnizca transform uygulanir (fit edilmez).
        Aksi halde test verisi egitim surecine sizar (data leakage).

    Args:
        df: Tum on isleme adimlari tamamlanmis DataFrame.

    Returns:
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    # Ozellik ve hedef ayirimi
    feature_names: list[str] = [
        col for col in df.columns if col != TARGET_COLUMN
    ]
    X: np.ndarray = df[feature_names].values
    y: np.ndarray = df[TARGET_COLUMN].values

    logger.info(f"X sekli: {X.shape} | y sekli: {y.shape}")
    logger.info(f"Sinif dagilimi -> 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    # Train / Test ayirimi
    # stratify=y: sinif oranini train ve test'te esit korur
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f"Train: {X_train.shape[0]} satir | Test: {X_test.shape[0]} satir"
    )

    # MinMaxScaler — YALNIZCA train uzerinde fit et
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)   # fit + transform
    X_test  = scaler.transform(X_test)        # sadece transform (data leakage onleme)

    logger.info("MinMaxScaler uygulandi -> aralik: [0, 1] (fit: train, transform: train+test)")
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# Yardimci: Islenmis Veriyi Kaydet
# ---------------------------------------------------------------------------

def save_processed_data(df: pd.DataFrame, filename: str = "diabetes_processed.csv") -> None:
    """
    On isleme tamamlanmis DataFrame'i data/processed/ klasorune kaydeder.
    Ham veri (data/raw/) hic degistirilmez.

    Args:
        df:       Kaydedilecek islenmis DataFrame.
        filename: Dosya adi. Varsayilan: diabetes_processed.csv
    """
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info(f"Islenmis veri kaydedildi: {out_path}")


# ---------------------------------------------------------------------------
# Ana Pipeline: Tum Adimlari Sirasyla Calistir
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    StandardScaler,
    list[str],
]:
    """
    Veri on isleme pipeline'ini bastan sona calistirir.

    Adimlar:
        1. Ham veriyi yukle
        2. Gecersiz 0'lari NaN'a cevir
        3. Outlier'lari IQR ile clip et
        4. NaN'lari medyanla doldur
        5. Islenmis veriyi kaydet
        6. Train/test split + StandardScaler

    Returns:
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    logger.info("=" * 50)
    logger.info("ON ISLEME PIPELINE BASLIYOR")
    logger.info("=" * 50)

    df = load_raw_data()
    df = replace_invalid_zeros(df)
    df = handle_outliers(df)
    df = impute_missing_values(df)

    save_processed_data(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = scale_and_split(df)

    logger.info("=" * 50)
    logger.info("ON ISLEME PIPELINE TAMAMLANDI")
    logger.info("=" * 50)

    return X_train, X_test, y_train, y_test, scaler, feature_names
