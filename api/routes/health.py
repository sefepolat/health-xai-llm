"""
api/routes/health.py
--------------------
GET /health endpoint'i.

Bu endpoint ne ise yarar?
    Production'da bir servis calisip calismadigini anlamak icin
    "health check" yapilir. Load balancer'lar, Docker, Kubernetes
    periyodik olarak bu endpoint'i cagirir.
    "200 OK" -> servis saglikli
    "500"    -> servis bozuk, trafik yonlendirme

Neden ayri dosya?
    Her endpoint grubunun kendi dosyasi olur.
    health.py -> saglik kontrolleri
    predict.py -> tahmin isleri
    Buyudukce yonetimi kolaylasir.
"""

from fastapi import APIRouter
from api.models import HealthResponse

# APIRouter nedir?
#   FastAPI'nin "alt router" mekanizmasi.
#   main.py'deki ana app'e bu router'i dahil ediyoruz.
#   prefix="/health" -> bu dosyadaki tum endpoint'ler /health ile baslar.
#   tags=["Health"] -> Swagger'da gruplanir.
router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",                          # GET /health
    response_model=HealthResponse,
    summary="API saglik kontrolu",
    description="Servisin ayakta olup olmadigini kontrol eder.",
)
async def health_check():
    """
    Neden bu endpoint async await kullanmiyor?
        Burada I/O yok — sadece sabit bir dict donduruyoruz.
        Ama async tanimlamak hic bir zararini yapmaz,
        tutarlilik icin async yaziyoruz.
    """
    from src.config import OLLAMA_MODEL

    return HealthResponse(
        status="ok",
        model=OLLAMA_MODEL,
        rag=True,
    )
