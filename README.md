# 🩺 Clinical Decision Support System for Diabetes Risk Prediction
### LLM-Assisted Explainable AI (XAI) & RAG-Powered Clinical Reporting

<p align="center">
  <a href="#-english">English</a> •
  <a href="#-türkçe">Türkçe</a>
</p>

---

<a name="english"></a>
## 🇬🇧 English

An end-to-end clinical decision support system built on the **Pima Indians Diabetes Dataset**. The project integrates an **XGBoost** classification model, **SHAP & LIME** explainability frameworks, a **RAG (Retrieval-Augmented Generation)** knowledge retrieval pipeline using **ChromaDB**, and local LLM inference via **Ollama (`llama3.1:8b`)** to generate evidence-based, actionable clinical risk reports. Delivered through both a **FastAPI** backend and an interactive **Streamlit** dashboard, fully containerized with **Docker Compose**.

### 🌟 Key Highlights

- **Predictive Modeling:** Optimized XGBoost classifier prioritizing high recall (**Recall: 0.852**, **ROC-AUC: 0.949**) to minimize critical false negatives in medical screening.
- **Explainable AI (XAI):** Feature-level explanations via global/local **SHAP** values and instance-level **LIME** weights to ensure transparency in AI decisions.
- **Domain-Specific RAG:** Grounded clinical context retrieved from ADA (American Diabetes Association) and WHO guidelines using `intfloat/multilingual-e5-small` embeddings stored in ChromaDB.
- **Structured LLM Reporting:** Prompt engineering with LangChain and Pydantic validation to produce structured reports (risk tier, key contributing biomarkers, clinical recommendations, and medical disclaimers).
- **Production-Ready Delivery:** RESTful API with automated OpenAPI/Swagger documentation (FastAPI) alongside a multi-tab analytical UI (Streamlit), orchestrated via Docker Compose.

---

### 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          User Interfaces                               │
│        Streamlit Dashboard (:8501)   │    FastAPI REST Backend (:8000) │
└───────────────────┬───────────────────┴─────────────────┬──────────────┘
                    │                                     │
         ┌──────────▼─────────────────────────────────────▼──────────────┐
         │                    Core ML & XAI Pipeline                     │
         │  • Data Preprocessing (Median Imputation, IQR Clipping, Scaler)│
         │  • XGBoost Inference Engine                                   │
         │  • SHAP (TreeExplainer) + LIME (Tabular Explainer)             │
         └──────────────────────────────┬────────────────────────────────┘
                                        │
         ┌──────────────────────────────▼────────────────────────────────┐
         │                  RAG & LLM Clinical Reporting                 │
         │  • ChromaDB Vector Store (multilingual-e5-small)              │
         │  • ADA / WHO Guidelines Knowledge Base                        │
         │  • LangChain + Ollama (llama3.1:8b) → Structured Output       │
         └───────────────────────────────────────────────────────────────┘
```

---

### 💻 Technologies & Libraries

- **Machine Learning & Preprocessing:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`
- **Explainable AI (XAI):** `shap`, `lime`
- **LLM & Orchestration:** `langchain`, `langchain-ollama`, `pydantic`
- **Embeddings & Vector Storage:** `sentence-transformers` (`intfloat/multilingual-e5-small`), `chromadb`
- **API & Backend:** `fastapi`, `uvicorn`
- **Frontend & Visualization:** `streamlit`, `plotly`, `seaborn`, `matplotlib`
- **DevOps & Containerization:** `docker`, `docker compose`

---

### 📊 Model Evaluation

In clinical risk assessment, **Recall** is chosen as the primary evaluation metric because failing to identify a diabetic individual (False Negative) carries a significantly higher medical cost than a False Positive.

| Model | Recall (Primary) | F1-Score | ROC-AUC | Precision |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.833 | 0.726 | 0.842 | 0.643 |
| Random Forest | 0.852 | 0.821 | 0.943 | 0.793 |
| **XGBoost (Best)** | **0.852** | **0.844** | **0.949** | **0.836** |

---

### 🚀 Getting Started

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for containerized setup)
- [Ollama](https://ollama.com/) running locally with the target model:
  ```bash
  ollama pull llama3.1:8b
  ollama serve
  ```

#### Method 1: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/USERNAME/health-risk-project.git
cd health-risk-project

# 2. Setup environment variables
cp .env.example .env

# 3. Start services
docker compose up -d
```

- **Interactive Dashboard:** [http://localhost:8501](http://localhost:8501)
- **API Base:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

#### Method 2: Local Python Environment

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate   # On Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit dashboard
streamlit run app.py

# 4. (Optional) Run FastAPI server in a separate terminal
python run_api.py
```

---

### 🔌 API Reference

#### Health Check
`GET /health`
```json
{
  "status": "ok",
  "model": "llama3.1:8b",
  "rag": true
}
```

#### Predict & Generate Report
`POST /predict`
```json
// Request Body
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

```json
// Response Body
{
  "probability": 0.9988,
  "risk_level": "YUKSEK",
  "summary": "Diyabet riskiniz %99.9'dir. Kan şekeri ve insülin seviyeniz yüksek, vücut kitle indeksiniz çok yüksektir.",
  "key_factors": [
    "Glucose: COK YUKSEK (171.0 mg/dL)",
    "Insulin: YUKSEK (169.5 uU/mL)",
    "BMI: COK YUKSEK (43.6 kg/m2)"
  ],
  "recommendation": "Diyabet riskini azaltmak için hekim kontrolünde beslenme ve egzersiz düzenlemesi önerilir.",
  "disclaimer": "Bu rapor yalnızca tahmin niteliğinde olup kesin tıbbi teşhis içermez.",
  "rag_used": true
}
```

---
---

<a name="türkçe"></a>
## 🇹🇷 Türkçe

**Pima Indians Diabetes Veri Seti** üzerinde geliştirilmiş uçtan uca klinik karar destek sistemi. Proje; **XGBoost** sınıflandırma modelini, **SHAP ve LIME** açıklanabilirlik yöntemlerini, **ChromaDB** tabanlı **RAG (Retrieval-Augmented Generation)** bilgi tabanını ve yerel **Ollama (`llama3.1:8b`)** LLM entegrasyonunu bir araya getirerek kanıta dayalı klinik risk raporları üretir. Sistem hem **FastAPI** REST arayüzü hem de **Streamlit** paneli üzerinden **Docker Compose** ile tek komutla çalıştırılabilir.

### 🌟 Öne Çıkan Özellikler

- **Tahminleme Modeli:** Klinik taramada kritik olan hatalı negatifleri (False Negative) en aza indirmek için Recall metriği optimize edilmiş XGBoost modeli (**Recall: 0.852**, **ROC-AUC: 0.949**).
- **Açıklanabilir Yapay Zeka (XAI):** Model kararlarının güvenilirliğini ve şeffaflığını sağlamak üzere **SHAP** ve **LIME** ile öznitelik düzeyinde etki analizi.
- **Klinik Bilgi Tabanı (RAG):** ADA (Amerikan Diyabet Birliği) ve DSÖ (WHO) kılavuzlarından derlenen bilgilerin `intfloat/multilingual-e5-small` ile vektörleştirilip ChromaDB üzerinden bağlama dahil edilmesi.
- **Yapılandırılmış LLM Raporu:** LangChain ve Pydantic ile standardize edilmiş çıktı şablonu (risk seviyesi, tetikleyici biyobelirteçler, yaşam tarzı önerileri ve yasal sorumluluk reddi).
- **Üretim Odaklı Mimari:** Otomatik Swagger/OpenAPI dokümantasyonuna sahip FastAPI servisi ve kullanıcı dostu Streamlit web paneli (Docker Compose ile konteynerize edilmiş).

---

### 🏛️ Sistem Mimarisi

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Kullanıcı Arayüzleri                          │
│        Streamlit Paneli (:8501)      │    FastAPI REST Servisi (:8000) │
└───────────────────┬───────────────────┴─────────────────┬──────────────┘
                    │                                     │
         ┌──────────▼─────────────────────────────────────▼──────────────┐
         │                  Makine Öğrenmesi & XAI Hattı                 │
         │  • Veri Ön İşleme (Medyan Doldurma, IQR Kırpma, Ölçekleme)     │
         │  • XGBoost Tahmin Motoru                                      │
         │  • SHAP (TreeExplainer) + LIME (Tabular Explainer)             │
         └──────────────────────────────┬────────────────────────────────┘
                                        │
         ┌──────────────────────────────▼────────────────────────────────┐
         │                  RAG & LLM Raporlama Katmanı                  │
         │  • ChromaDB Vektör Veritabanı (multilingual-e5-small)         │
         │  • ADA / DSÖ Klinik Kılavuz Bilgi Tabanı                      │
         │  • LangChain + Ollama (llama3.1:8b) → Pydantic Formatı         │
         └───────────────────────────────────────────────────────────────┘
```

---

### 💻 Kullanılan Teknolojiler ve Kütüphaneler

- **Makine Öğrenmesi & Veri İşleme:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`
- **Açıklanabilir Yapay Zeka (XAI):** `shap`, `lime`
- **LLM & RAG Orkestrasyonu:** `langchain`, `langchain-ollama`, `pydantic`
- **Embedding & Vektör Veritabanı:** `sentence-transformers` (`intfloat/multilingual-e5-small`), `chromadb`
- **API & Sunucu:** `fastapi`, `uvicorn`
- **Arayüz & Görselleştirme:** `streamlit`, `plotly`, `seaborn`, `matplotlib`
- **Konteynerleştirme & Dağıtım:** `docker`, `docker compose`

---

### 📊 Model Performansı ve Karşılaştırma

Tıbbi karar destek sistemlerinde, hasta bir bireye sağlıklı teşhisi koymanın (False Negative) hayati riski yüksek olduğundan ana optimizasyon metriği olarak **Recall** seçilmiştir.

| Model | Recall (Ana Metrik) | F1-Score | ROC-AUC | Precision |
|---|:---:|:---:|:---:|:---:|
| Lojistik Regresyon | 0.833 | 0.726 | 0.842 | 0.643 |
| Random Forest | 0.852 | 0.821 | 0.943 | 0.793 |
| **XGBoost (En İyi)** | **0.852** | **0.844** | **0.949** | **0.836** |

---

### 🚀 Kurulum ve Çalıştırma

#### Gereksinimler
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (konteyner ile çalıştırmak için)
- Yerel [Ollama](https://ollama.com/) servisi ve indirilen model:
  ```bash
  ollama pull llama3.1:8b
  ollama serve
  ```

#### Yöntem 1: Docker Compose ile (Önerilen)

```bash
# 1. Projeyi klonla
git clone https://github.com/KULLANICI_ADI/health-risk-project.git
cd health-risk-project

# 2. Ortam değişkenleri şablonunu kopyala
cp .env.example .env

# 3. Servisleri arka planda başlat
docker compose up -d
```

- **Streamlit Paneli:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Kök Dizin:** [http://localhost:8000](http://localhost:8000)
- **Swagger API Dokümantasyonu:** [http://localhost:8000/docs](http://localhost:8000/docs)

#### Yöntem 2: Yerel Python Ortamı ile

```bash
# 1. Sanal ortamı oluştur ve aktif et
python -m venv .venv
.venv\Scripts\activate      # Windows için
source .venv/bin/activate   # Linux/macOS için

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Streamlit arayüzünü başlat
streamlit run app.py

# 4. (İsteğe bağlı) FastAPI sunucusunu ayrı bir terminalde çalıştır
python run_api.py
```

---

### 📁 Proje Dosya Yapısı

```
health-risk-project/
├── api/                    # FastAPI modülü
│   ├── main.py             # Uygulama yaşam döngüsü (lifespan) ve CORS
│   ├── models.py           # Pydantic girdi ve çıktı sözleşmeleri
│   └── routes/
│       ├── health.py       # GET /health
│       └── predict.py      # POST /predict ve GET /predict/features
├── data/
│   ├── raw/                # Ham veri seti (diabetes.csv)
│   └── knowledge_base/     # RAG için klinik bilgi dokümanları (Markdown)
├── src/
│   ├── config.py           # Merkezi konfigürasyon ve ortam değişkenleri
│   ├── data_processing.py  # Eksik veri doldurma, IQR ve ölçekleme hattı
│   ├── model_training.py   # Model eğitimi, hiperparametre ve metrikler
│   ├── explainability.py   # SHAP ve LIME açıklanabilirlik hesaplamaları
│   ├── llm_handler.py      # LangChain, Ollama ve Pydantic çıktı şablonu
│   └── rag_handler.py      # Embedding üretimi ve ChromaDB arama motoru
├── app.py                  # Streamlit web uygulaması
├── run_api.py              # FastAPI uvicorn başlatıcısı
├── Dockerfile              # Docker imaj tarifi (Python 3.12-slim tabanlı)
├── docker-compose.yml      # Çoklu konteyner orkestrasyonu (API + UI)
├── requirements.txt        # Yerel ortam bağımlılıkları
├── requirements.docker.txt # Docker ortamı için hafif (CPU-only) bağımlılıklar
└── .env.example            # Örnek ortam değişkenleri dosyası
```

---

### 📝 Lisans

Bu proje [MIT](LICENSE) lisansı altında sunulmaktadır.
