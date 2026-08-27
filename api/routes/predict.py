"""
api/routes/predict.py
---------------------
POST /predict  -> Hasta verisini al, rapor uret
GET  /features -> Ozellik listesi ve normal araliklarini dondur
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import PatientInput, PredictionResult, FeaturesResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ---------------------------------------------------------------------------
# Uygulama baslarken pipeline'i yukle (main.py'den inject edilecek)
# ---------------------------------------------------------------------------

# Bu degiskenler main.py'deki lifespan fonksiyonu tarafindan doldurulur.
# Global state: API ayaga kalkarken bir kez yuklenir, her istekte yeniden
# yuklenmez. Boylece her /predict cagrisi icin model yeniden egitilmez.
_pipeline = {}


def set_pipeline(pipeline: dict):
    """main.py tarafindan cagrilir, pipeline nesnelerini inject eder."""
    _pipeline.update(pipeline)


# ---------------------------------------------------------------------------
# GET /predict/features
# ---------------------------------------------------------------------------

@router.get(
    "/features",
    response_model=FeaturesResponse,
    summary="Ozellik listesi",
    description="Modelin kullandigi ozelliklerin isimlerini ve "
                "klinik normal araliklerini dondurur.",
)
async def get_features():
    """
    Neden bu endpoint var?
        Bir frontend (web, mobil) hangi alanlari formda gosterecegini
        dinamik olarak ogrenmek ister. Bu endpoint sayesinde frontend
        kodu degismeden ozellikler guncellenebilir.
    """
    feature_names = _pipeline.get("feature_names", [])

    reference_ranges = {
        "Pregnancies":              {"alt": 0,    "ust": 17,   "birim": "adet"},
        "Glucose":                  {"alt": 70,   "ust": 99,   "birim": "mg/dL"},
        "BloodPressure":            {"alt": 60,   "ust": 80,   "birim": "mmHg"},
        "SkinThickness":            {"alt": 10,   "ust": 40,   "birim": "mm"},
        "Insulin":                  {"alt": 16,   "ust": 166,  "birim": "uU/mL"},
        "BMI":                      {"alt": 18.5, "ust": 24.9, "birim": "kg/m2"},
        "DiabetesPedigreeFunction": {"alt": None, "ust": None, "birim": ""},
        "Age":                      {"alt": None, "ust": None, "birim": "yil"},
    }

    return FeaturesResponse(
        features=feature_names,
        reference_ranges=reference_ranges,
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "",                            # POST /predict
    response_model=PredictionResult,
    summary="Diyabet risk tahmini",
    description="Hasta klinik degerlerini alir, ML modeli + LLM ile "
                "klinik karar destek raporu uretir.",
)
async def predict(patient: PatientInput):
    """
    Bu fonksiyon nasil calisir?

    1. FastAPI, gelen JSON'u PatientInput Pydantic modeline donusturur.
       Eksik alan veya yanlis tip varsa otomatik 422 hatasi doner —
       biz hicbir kontrol yazmak zorunda kalmayiz.

    2. Pipeline nesnelerini aliyoruz (scaler, model, explainerlar).

    3. Ham degerleri scaler ile normalize ediyoruz.
       Neden? Model MinMaxScaler ile egitildi, ayni olcege ihtiyac var.

    4. explain_patient() ile SHAP + LIME hesapliyoruz.

    5. Istege gore RAG baglami getiriyoruz.

    6. generate_report() ile LLM raporu uretiyoruz.

    7. PredictionResult olarak donduruyoruz.

    Args:
        patient: Doktor tarafindan gonderilen hasta verileri (PatientInput)

    Returns:
        PredictionResult: Olasilik + risk seviyesi + LLM raporu

    Raises:
        HTTPException 503: Pipeline hazir degilse
        HTTPException 500: Beklenmedik hata
    """
    # Pipeline hazir mi?
    if not _pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline henuz hazir degil. Birkac saniye bekleyip tekrar deneyin."
        )
        # HTTPException nedir?
        # FastAPI'nin hata mekanizmasi.
        # status_code: HTTP durum kodu (503 = Service Unavailable)
        # detail: Hata mesaji (JSON olarak doner)

    try:
        scaler        = _pipeline["scaler"]
        model         = _pipeline["model"]
        shap_explainer = _pipeline["shap_explainer"]
        lime_explainer = _pipeline["lime_explainer"]
        feature_names = _pipeline["feature_names"]
        vector_store  = _pipeline.get("vector_store")

        # 1. Ham degerleri dogru siraya diz
        raw = np.array([[
            patient.Pregnancies,
            patient.Glucose,
            patient.BloodPressure,
            patient.SkinThickness,
            patient.Insulin,
            patient.BMI,
            patient.DiabetesPedigreeFunction,
            patient.Age,
        ]])

        # 2. Scaler ile normalize et
        # Neden? Model [0,1] araligindaki veriyle egitildi.
        # Ham degeri modele versek yanlis tahmin uretir.
        scaled = scaler.transform(raw)

        # 3. explain_patient numpy array bekliyor (X_test[index] = satir erisimi)
        # DataFrame'de df[0] = sutun erisimi olur → KeyError
        # Bu yuzden scaled numpy array'i direkt kullaniyoruz
        X_single = scaled          # shape: (1, 8) — numpy array
        y_single = np.array([-1])  # gercek etiket yok, -1 koyuyoruz

        # 4. Aciklama hesapla
        from src.explainability import explain_patient
        explanation = explain_patient(
            patient_index=0,
            X_test=X_single,
            y_test=y_single,
            model=model,
            shap_explainer=shap_explainer,
            lime_explainer=lime_explainer,
            feature_names=feature_names,
            scaler=scaler,
            top_n=5,
        )

        # 5. RAG baglami
        rag_context = ""
        if patient.use_rag and vector_store is not None:
            from src.rag_handler import retrieve_context
            rag_context = retrieve_context(explanation, vector_store)

        # 6. LLM raporu
        from src.llm_handler import generate_report
        report = generate_report(explanation, rag_context=rag_context)

        # 7. Cevabi dondur
        return PredictionResult(
            probability=round(explanation["prediction"]["probability"], 4),
            risk_level=report.risk_level,
            summary=report.summary,
            key_factors=report.key_factors,
            recommendation=report.recommendation,
            disclaimer=report.disclaimer,
            rag_used=bool(rag_context),
        )

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        logger.error(f"Tahmin hatasi:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sirasinda hata: {str(e)}"
        )
