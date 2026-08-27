# 🩺 Diyabet Risk Tahmin Sistemi
### LLM-Assisted Explainable AI for Health Risk Prediction

Pima Indians Diabetes veri seti üzerinde eğitilmiş, XGBoost tabanlı bir makine öğrenmesi modelini; SHAP + LIME açıklanabilirlik katmanlarıyla, RAG destekli LLM rapor üretici ile ve REST API arayüzüyle bir araya getiren klinik karar destek sistemi.

---

## ✨ Özellikler

- **XGBoost** ile diyabet risk tahmini (Recall: 0.85, ROC-AUC: 0.95)
- **SHAP + LIME** ile model kararlarının açıklanması
- **RAG** (Retrieval-Augmented Generation) destekli klinik rapor üretimi
  - ChromaDB vector store
  - `intfloat/multilingual-e5-small` embedding modeli
  - ADA/WHO kılavuzlarından oluşan bilgi tabanı
- **Ollama** üzerinden yerel LLM (llama3.1:8b)
- **FastAPI** ile REST API (`/health`, `/predict`, `/predict/features`)
- **Streamlit** ile interaktif kullanıcı arayüzü
- **Docker Compose** ile tek komutla çalıştırma

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────┐
│                    Kullanıcı Arayüzü                    │
│           Streamlit (8501) │ FastAPI (8000)              │
└───────────────────┬─────────────────┬───────────────────┘
                    │                 │
         ┌──────────▼─────────────────▼──────────┐
         │           Pipeline                      │
         │  data_processing → model_training       │
         │  explainability (SHAP + LIME)           │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │         LLM + RAG Katmanı                │
         │  ChromaDB ← multilingual-e5-small        │
         │  Ollama (llama3.1:8b) ← LangChain        │
         └─────────────────────────────────────────┘
```

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| ML Modeli | XGBoost, Scikit-learn, imbalanced-learn |
| XAI | SHAP, LIME |
| LLM | Ollama (llama3.1:8b), LangChain |
| Embedding | intfloat/multilingual-e5-small |
| Vector DB | ChromaDB |
| API | FastAPI, Uvicorn, Pydantic |
| UI | Streamlit |
| Containerization | Docker, Docker Compose |

---

## 🚀 Kurulum

### Ön Koşullar
- [Ollama](https://ollama.com/) kurulu ve çalışıyor olmalı
- `ollama pull llama3.1:8b` ile model indirilmiş olmalı

### Seçenek 1: Docker ile (Önerilen)

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADI/health-risk-project.git
cd health-risk-project

# 2. Ortam değişkenlerini ayarla
cp .env.example .env
# .env dosyasını düzenle (gerekirse)

# 3. Ollama'yı başlat
ollama serve

# 4. Container'ları başlat
docker compose up -d
```

- **Streamlit:** http://localhost:8501
- **FastAPI:** http://localhost:8000
- **Swagger:** http://localhost:8000/docs

### Seçenek 2: Yerel Kurulum

```bash
# 1. Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# 2. Bağımlılıkları kur
pip install -r requirements.txt

# 3. Streamlit arayüzünü başlat
streamlit run app.py

# 4. (İsteğe bağlı) FastAPI'yi başlat
python run_api.py
```

---

## 🔌 API Kullanımı

### Sağlık Kontrolü
```bash
GET http://localhost:8000/health
```

### Diyabet Risk Tahmini
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "Pregnancies": 4,
  "Glucose": 171,
  "BloodPressure": 72,
  "SkinThickness": 32,
  "Insulin": 169.5,
  "BMI": 43.6,
  "DiabetesPedigreeFunction": 0.48,
  "Age": 26,
  "use_rag": true
}
```

**Yanıt:**
```json
{
  "probability": 0.9988,
  "risk_level": "YUKSEK",
  "summary": "Diyabet riskiniz %99.9'dir...",
  "key_factors": ["Glucose: COK YUKSEK", "BMI: COK YUKSEK"],
  "recommendation": "...",
  "rag_used": true
}
```

---

## 📁 Proje Yapısı

```
health-risk-project/
├── api/                    # FastAPI endpoint'leri
│   ├── main.py             # Uygulama ve lifespan
│   ├── models.py           # Pydantic request/response modelleri
│   └── routes/
│       ├── health.py       # GET /health
│       └── predict.py      # POST /predict
├── data/
│   ├── raw/                # Ham veri seti (diabetes.csv)
│   └── knowledge_base/     # RAG bilgi tabanı (Markdown)
├── src/
│   ├── config.py           # Merkezi yapılandırma
│   ├── data_processing.py  # Ön işleme pipeline'ı
│   ├── model_training.py   # Model eğitimi ve değerlendirme
│   ├── explainability.py   # SHAP + LIME açıklamaları
│   ├── llm_handler.py      # LangChain + Ollama entegrasyonu
│   └── rag_handler.py      # ChromaDB + embedding
├── app.py                  # Streamlit uygulaması
├── run_api.py              # FastAPI başlatma scripti
├── Dockerfile
├── docker-compose.yml
├── requirements.txt        # Yerel kurulum (Python 3.14)
├── requirements.docker.txt # Docker kurulum (Python 3.12)
└── .env.example            # Ortam değişkenleri şablonu
```

---

## ⚙️ Ortam Değişkenleri

`.env.example` dosyasını `.env` olarak kopyalayıp düzenle:

```env
# LLM Backend: "ollama" veya "openai"
LLM_BACKEND=ollama

# Ollama ayarları
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# OpenAI (isteğe bağlı)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o
```

---

## 📊 Model Performansı

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.833 | 0.726 | 0.842 |
| Random Forest | 0.852 | 0.821 | 0.943 |
| **XGBoost** ✅ | **0.852** | **0.844** | **0.949** |

> Metrik olarak **Recall** ön plana alındı: Hasta bir bireyi sağlıklı olarak sınıflandırmak (FN), sağlıklı bir bireyi hasta olarak sınıflandırmaktan (FP) daha maliyetlidir.

---

## 📝 Lisans

MIT
