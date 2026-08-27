"""
config.py
---------
Merkezi yapılandırma modülü.
Tüm ortam değişkenlerini, sabit yolları ve proje genelinde
kullanılan ayarları tek bir yerden yönetir.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# .env dosyasını yükle (proje kök dizinindeki)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent  # health-risk-project/
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Dizin Yolları
# ---------------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

# Tüm çıktı dizinlerini otomatik oluştur (yoksa)
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Veri Seti Ayarları
# ---------------------------------------------------------------------------
RAW_DATA_PATH: Path = RAW_DATA_DIR / "diabetes.csv"  # Pima Diabetes
TARGET_COLUMN: str = "Outcome"                         # Hedef sütun adı

# ---------------------------------------------------------------------------
# Model Ayarları
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42       # Tekrarlanabilirlik için sabit tohum
TEST_SIZE: float = 0.2       # Eğitim/test bölme oranı
CV_FOLDS: int = 5            # Cross-validation kat sayısı

# ---------------------------------------------------------------------------
# OpenAI API Ayarları
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# ---------------------------------------------------------------------------
# Ollama (Yerel LLM) Ayarları
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

# ---------------------------------------------------------------------------
# Aktif LLM Backend Seçimi
# "openai" → OpenAI API | "ollama" → Yerel Ollama
# ---------------------------------------------------------------------------
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")  # Varsayılan: Ollama

# ---------------------------------------------------------------------------
# Loglama Ayarları
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = OUTPUTS_DIR / "app.log"

logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation="10 MB",   # 10 MB dolunca yeni dosyaya geç
    retention="7 days", # 7 günden eski logları sil
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# Doğrulama: Kritik ayarların kontrolü
# ---------------------------------------------------------------------------
if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
    logger.warning(
        "LLM_BACKEND='openai' seçili ama OPENAI_API_KEY boş! "
        ".env dosyasını kontrol et."
    )

logger.info(f"Config yüklendi → LLM Backend: {LLM_BACKEND} | Model: "
            f"{OPENAI_MODEL if LLM_BACKEND == 'openai' else OLLAMA_MODEL}")
