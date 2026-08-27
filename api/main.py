"""
api/main.py
-----------
FastAPI uygulamasinin kalbi.

Burada ne oluyor?
    1. FastAPI app nesnesi olusturulur
    2. Uygulama baslarken (startup) pipeline yuklenir
    3. Router'lar eklenir (/health, /predict)
    4. CORS ayarlari yapilir
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import health, predict


# ---------------------------------------------------------------------------
# Lifespan: Uygulama yasam dongusu
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan nedir?
        Uygulamanin dogum ve olum anlari.
        yield'den ONCE olan kod -> startup (uygulama baslarken)
        yield'den SONRA olan kod -> shutdown (uygulama kapanirken)

    Neden pipeline'i burada yukluyoruz?
        Her /predict isteginde modeli egitmek => her istek 30 saniye surar.
        Bir kez yukle, her istekte hazir kullan => milisaniyeler.

        Bu pattern'in adi: "Application State" veya "Dependency Injection"
    """
    # --- STARTUP ---
    logger.info("API basliyor: Pipeline yukleniyor...")

    from src.data_processing import run_preprocessing_pipeline
    from src.model_training import run_training_pipeline
    from src.explainability import build_shap_explainer, build_lime_explainer
    from src.rag_handler import build_vector_store

    # Veri + model
    X_train, X_test, y_train, y_test, scaler, feature_names = (
        run_preprocessing_pipeline()
    )
    results = run_training_pipeline(X_train, X_test, y_train, y_test)
    xgb_model, _ = results["xgboost"]

    # Explainer'lar
    shap_exp = build_shap_explainer(xgb_model, X_train)
    lime_exp = build_lime_explainer(X_train, feature_names)

    # RAG
    vector_store = build_vector_store()

    # predict.py'ye inject et
    predict.set_pipeline({
        "scaler":         scaler,
        "model":          xgb_model,
        "shap_explainer": shap_exp,
        "lime_explainer": lime_exp,
        "feature_names":  feature_names,
        "vector_store":   vector_store,
    })

    logger.info("API hazir!")
    yield   # <-- Burasi uygulamanin calistigi sure

    # --- SHUTDOWN ---
    logger.info("API kapaniyor...")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Diyabet Risk Analizi API",
    description="""
    Pima diyabet veri seti uzerinde egitilmis XGBoost modeli +
    LLM (llama3.1:8b) + RAG (ChromaDB) ile klinik karar destek sistemi.

    ## Endpoint'ler
    - **GET /health** — Servis saglik kontrolu
    - **GET /predict/features** — Ozellik listesi ve normal araliolar
    - **POST /predict** — Hasta verisi gonder, rapor al
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# ---------------------------------------------------------------------------

# CORS nedir?
#   Tarayici guvenlik kurali: Bir web sayfasi baska domain'e istek atarsa
#   tarayici bunu engeller. CORS ayarlariyla hangi domain'lerin API'ye
#   erisebilecegini belirleriz.
#
#   allow_origins=["*"] -> Herkese izin ver (gelistirme icin OK,
#                          production'da spesifik domain yaz)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Router'lari ekle
# ---------------------------------------------------------------------------

# Router nedir?
#   APIRouter ile tanimlanmis endpoint gruplarini ana app'e baglar.
#   /health   -> health.router
#   /predict  -> predict.router
app.include_router(health.router)
app.include_router(predict.router)


# ---------------------------------------------------------------------------
# Kok endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    """API'ye gelen koku karsilar, Swagger linkini gosterir."""
    return {
        "message": "Diyabet Risk Analizi API'sine hosgeldiniz.",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
    }
