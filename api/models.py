"""
api/models.py
-------------
FastAPI'nin Request ve Response modelleri.

Neden Pydantic kullaniyoruz?
    FastAPI Pydantic ile tam entegre calisir.
    Bir endpoint'e yanlis tip gonderirsen (orn: string yerine float beklenen yerde)
    FastAPI otomatik olarak 422 Unprocessable Entity hatasi dondurur.
    Sen hata kontrolu yazmak zorunda kalmazsin — Pydantic halleder.

Neden ayri dosya?
    models.py -> sadece veri yapilari
    routes/   -> sadece endpoint mantigi
    Sorumluluk ayristirmasi: Her dosyanin tek bir gorevi var.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# REQUEST MODELI
# Doktorun POST /predict'e gonderdigi veri
# ---------------------------------------------------------------------------

class PatientInput(BaseModel):
    """
    Hasta ozellikleri — 8 klinik deger.

    Field(...) ne demek?
        ... -> bu alan zorunlu, varsayilan deger yok
        ge=0 -> greater or equal to 0 (negatif deger kabul edilmez)
        description -> Swagger dokumantasyonunda gorunur

    Neden validation (dogrulama) ekledik?
        "Glucose: -50" ya da "BMI: 500" gonderilirse ne olur?
        Model anlamsiz tahmin uretir. Validasyon bunu engeller.
    """

    Pregnancies: float = Field(
        ..., ge=0, le=20,
        description="Gebelik sayisi (0-20)"
    )
    Glucose: float = Field(
        ..., ge=0, le=300,
        description="Plazma glukozu mg/dL (0-300)"
    )
    BloodPressure: float = Field(
        ..., ge=0, le=200,
        description="Diyastolik kan basinci mmHg (0-200)"
    )
    SkinThickness: float = Field(
        ..., ge=0, le=100,
        description="Triseps deri katlantisi kalinligi mm (0-100)"
    )
    Insulin: float = Field(
        ..., ge=0, le=1000,
        description="2 saatlik serum insülin uU/mL (0-1000)"
    )
    BMI: float = Field(
        ..., ge=0, le=100,
        description="Vucut kitle indeksi kg/m2 (0-100)"
    )
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0, le=3,
        description="Diyabet soyagaci fonksiyonu (0-3)"
    )
    Age: float = Field(
        ..., ge=21, le=120,
        description="Yas (21-120)"
    )
    use_rag: bool = Field(
        default=True,
        description="True ise RAG bilgi tabani kullanilir"
    )

    # Swagger'da ornek deger gosterir
    # Doktor API dokumantasyonunu actiginda hazir bir ornek gorur
    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 4,
                "Glucose": 171,
                "BloodPressure": 72,
                "SkinThickness": 32,
                "Insulin": 169.5,
                "BMI": 43.6,
                "DiabetesPedigreeFunction": 0.48,
                "Age": 26,
                "use_rag": True
            }
        }
    }


# ---------------------------------------------------------------------------
# RESPONSE MODELLERI
# API'nin dondurukleri
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    """
    /predict endpoint'inin dondurukleri.

    Neden response modeli de Pydantic?
        1. API'nin ne dondurecegi dokumante edilir (Swagger'da gorunur)
        2. Yanlis formatta cikti uretilirse hata alirsin — sessizce gecmez
        3. Tip guvenligi: probability her zaman float, risk_level her zaman str
    """
    probability: float = Field(
        description="Diyabet olasiligi (0.0 - 1.0)"
    )
    risk_level: str = Field(
        description="Risk seviyesi: DUSUK | ORTA | YUKSEK"
    )
    summary: str = Field(
        description="LLM tarafindan uretilen klinik ozet"
    )
    key_factors: list[str] = Field(
        description="En etkili klinik faktorler"
    )
    recommendation: str = Field(
        description="LLM tarafindan uretilen yasam tarzi onerisi"
    )
    disclaimer: str = Field(
        description="Tibbi sorumluluk reddi beyani"
    )
    rag_used: bool = Field(
        description="RAG bilgi tabani kullanildi mi?"
    )


class HealthResponse(BaseModel):
    """GET /health endpoint'inin dondurukleri."""
    status: str = Field(description="API durumu: ok | error")
    model: str   = Field(description="Aktif LLM modeli")
    rag:   bool  = Field(description="RAG hazir mi?")


class FeaturesResponse(BaseModel):
    """GET /features endpoint'inin dondurukleri."""
    features: list[str] = Field(description="Ozellik isimleri")
    reference_ranges: dict = Field(
        description="Her ozellik icin normal aralik bilgisi"
    )
