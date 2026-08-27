"""
app.py
------
Streamlit arayüzü — Pima Diabetes Health Risk Analyzer

Sekmeler:
    1. Hasta Raporu    : Tahmin + LLM raporu + ham degerler
    2. Model Aciklamasi: SHAP + LIME gorsellestirilmesi
    3. Model Performansi: Uc modelin karsilastirma tablosu + confusion matrix

Calistirmak icin:
    streamlit run app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data_processing import run_preprocessing_pipeline
from src.model_training import run_training_pipeline
from src.explainability import (
    build_shap_explainer,
    build_lime_explainer,
    explain_patient,
    compute_shap_values,
)
from src.llm_handler import generate_report
from src.rag_handler import build_vector_store, retrieve_context

# ---------------------------------------------------------------------------
# Sayfa ayarlari
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diyabet Risk Analizi",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Pipeline'i bir kez yukle ve cache'le
# st.cache_resource: Model nesneleri gibi paylasilabilen kaynaklar icin
# st.cache_data: Saf veri donduran fonksiyonlar icin
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Veri isleniyor ve modeller egitiliyor...")
def load_pipeline():
    """
    On isleme + model egitimi + explainer'lari bir kez calistir ve cache'le.
    Streamlit her yeniden cizimde (rerun) bu fonksiyonu tekrar CALISTIRMAZ —
    cache sayesinde saniyeler icinde geri doner.
    """
    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing_pipeline()
    results = run_training_pipeline(X_train, X_test, y_train, y_test)

    models = {
        "XGBoost":             results["xgboost"][0],
        "Random Forest":       results["random_forest"][0],
        "Logistic Regression": results["logistic_regression"][0],
    }
    metrics = {
        "XGBoost":             results["xgboost"][1],
        "Random Forest":       results["random_forest"][1],
        "Logistic Regression": results["logistic_regression"][1],
    }

    shap_explainers = {
        name: build_shap_explainer(model, X_train)
        for name, model in models.items()
    }
    lime_explainer = build_lime_explainer(X_train, feature_names)

    return (
        X_train, X_test, y_train, y_test,
        scaler, feature_names,
        models, metrics,
        shap_explainers, lime_explainer,
    )


@st.cache_resource(show_spinner="RAG bilgi tabani yukleniyor...")
def load_vector_store():
    """
    ChromaDB vector store'u bir kez yukle ve cache'le.
    Ayri cache: Pipeline'dan bagimsiz yuklenebilsin.
    """
    return build_vector_store()


# ---------------------------------------------------------------------------
# Pipeline ve RAG yukle
# ---------------------------------------------------------------------------
(
    X_train, X_test, y_train, y_test,
    scaler, feature_names,
    models, metrics,
    shap_explainers, lime_explainer,
) = load_pipeline()

# ---------------------------------------------------------------------------
# Sidebar: Kontroller
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Ayarlar")
    st.divider()

    selected_model_name = st.selectbox(
        "Model",
        options=list(models.keys()),
        index=0,
    )

    patient_index = st.slider(
        "Hasta (Test Seti)",
        min_value=0,
        max_value=len(X_test) - 1,
        value=3,
        help="Test setindeki hasta indeksini secin",
    )

    analyze_btn = st.button("🔍 Analiz Et", use_container_width=True, type="primary")

    st.divider()
    use_rag = st.toggle(
        "📚 Bilgi Tabanı (RAG)",
        value=True,
        help="Aktif: LLM raporu ADA/WHO kılavuzlarına dayanarak yazar.\n"
             "Pasif: Sadece model verisine bakarak yazar.",
    )

    st.divider()
    true_label = int(y_test[patient_index])
    st.markdown(f"**Gercek Etiket:** {'🔴 Diyabetli' if true_label == 1 else '🟢 Saglikli'}")
    st.caption(f"Toplam test hastasi: {len(X_test)}")

# ---------------------------------------------------------------------------
# Session state: Analiz sonuclarini sakla
# ---------------------------------------------------------------------------
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "report" not in st.session_state:
    st.session_state.report = None
if "last_patient" not in st.session_state:
    st.session_state.last_patient = -1
if "last_model" not in st.session_state:
    st.session_state.last_model = ""

# Analiz butonuna basilinca calistir
if analyze_btn or (
    st.session_state.last_patient != patient_index or
    st.session_state.last_model != selected_model_name
):
    with st.spinner("XAI aciklamasi hesaplaniyor..."):
        explanation = explain_patient(
            patient_index=patient_index,
            X_test=X_test,
            y_test=y_test,
            model=models[selected_model_name],
            shap_explainer=shap_explainers[selected_model_name],
            lime_explainer=lime_explainer,
            feature_names=feature_names,
            scaler=scaler,
            top_n=5,
        )
        st.session_state.explanation = explanation
        st.session_state.last_patient = patient_index
        st.session_state.last_model = selected_model_name
        st.session_state.report = None  # Model veya hasta degistiyse raporu sifirla

explanation = st.session_state.explanation

# ---------------------------------------------------------------------------
# Baslik
# ---------------------------------------------------------------------------
st.title("🏥 Diyabet Risk Analizi")
st.caption("Pima Indians Diabetes Dataset — XAI Destekli Klinik Karar Destek Sistemi")

if explanation is None:
    st.info("Sol panelden bir hasta seçin ve 'Analiz Et' butonuna tıklayın.")
    st.stop()

# ---------------------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏥 Hasta Raporu", "📊 Model Açıklaması", "📈 Model Performansı"])


# ============================================================
# SEKME 1: HASTA RAPORU
# ============================================================
with tab1:
    pred = explanation["prediction"]
    prob = pred["probability"]
    label = pred["label"]

    # --- Tahmin Sonucu (buyuk, renkli) ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if label == "Diyabetli":
            st.error(f"## 🔴 {label}")
        else:
            st.success(f"## 🟢 {label}")
        st.metric("Model Güven Skoru", f"%{prob * 100:.1f}")
        st.caption(f"Model: {selected_model_name}")

    st.divider()

    # --- LLM Raporu ---
    col_rapor, col_degerler = st.columns([3, 2])

    with col_rapor:
        st.subheader("📄 Klinik Karar Destek Raporu")

        if st.session_state.report is None:
            with st.spinner("LLM raporu hazırlanıyor (llama3.1:8b)..."):
                try:
                    # RAG toggle'a gore baglam getir
                    rag_context = ""
                    if use_rag:
                        vector_store = load_vector_store()
                        rag_context = retrieve_context(explanation, vector_store)

                    report = generate_report(explanation, rag_context=rag_context)
                    st.session_state.report = report
                    st.session_state.rag_used = use_rag
                except Exception as e:
                    st.error(f"LLM hatası: {e}")
                    st.session_state.report = None

        report = st.session_state.report
        if report:
            # Risk seviyesi badge
            risk_colors = {"DUSUK": "green", "ORTA": "orange", "YUKSEK": "red"}
            risk_color = risk_colors.get(report.risk_level, "gray")
            st.markdown(
                f"**Risk Seviyesi:** :{risk_color}[**{report.risk_level}**]"
            )

            st.markdown("**Özet:**")
            st.info(report.summary)

            st.markdown("**Temel Faktörler:**")
            for i, factor in enumerate(report.key_factors, 1):
                st.markdown(f"{i}. {factor}")

            st.markdown("**Öneri:**")
            st.success(report.recommendation)

            st.warning(f"⚠️ {report.disclaimer}")

    with col_degerler:
        st.subheader("🔢 Hasta Değerleri")
        raw = explanation["raw_values"]

        # Referans araligini da goster
        reference = {
            "Pregnancies":             {"birim": "adet",  "normal": "0-17"},
            "Glucose":                 {"birim": "mg/dL", "normal": "70-99"},
            "BloodPressure":           {"birim": "mmHg",  "normal": "60-80"},
            "SkinThickness":           {"birim": "mm",    "normal": "10-40"},
            "Insulin":                 {"birim": "μU/mL", "normal": "16-166"},
            "BMI":                     {"birim": "kg/m²", "normal": "18.5-24.9"},
            "DiabetesPedigreeFunction":{"birim": "",       "normal": "0.08-2.42"},
            "Age":                     {"birim": "yil",   "normal": "-"},
        }

        rows = []
        for name, val in raw.items():
            ref = reference.get(name, {})
            rows.append({
                "Özellik": name,
                "Değer": f"{val} {ref.get('birim', '')}".strip(),
                "Normal Aralık": ref.get("normal", "-"),
            })

        df_raw = pd.DataFrame(rows)
        st.dataframe(df_raw, width='stretch', hide_index=True)


# ============================================================
# SEKME 2: MODEL ACIKLAMASI
# ============================================================
with tab2:
    st.subheader(f"📊 {selected_model_name} — XAI Açıklaması")

    shap_data = explanation["shap"]
    lime_data = explanation["lime"]
    agreement = explanation["agreement"]

    col_shap, col_lime = st.columns(2)

    def make_bar_chart(data: dict, title: str, color_pos: str, color_neg: str):
        """Yatay bar chart uretir — pozitif/negatif degerleri farkli renkte gosterir."""
        features = list(data.keys())
        values   = list(data.values())
        colors   = [color_pos if v > 0 else color_neg for v in values]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Etki Degeri")
        fig.tight_layout()
        return fig

    with col_shap:
        st.markdown("**SHAP Değerleri**")
        st.caption("Pozitif → riski artırıyor | Negatif → riski azaltıyor")
        fig_shap = make_bar_chart(shap_data, "SHAP", "#e74c3c", "#2ecc71")
        st.pyplot(fig_shap)
        plt.close()

    with col_lime:
        st.markdown("**LIME Değerleri**")
        st.caption("Lokal komşuluk tabanlı açıklama")
        fig_lime = make_bar_chart(lime_data, "LIME", "#e67e22", "#3498db")
        st.pyplot(fig_lime)
        plt.close()

    st.divider()

    # Uzlasma vurgusu
    st.markdown("### 🤝 SHAP & LIME Uzlaşması")
    if agreement:
        st.success(
            f"Her iki yöntem de şu özelliklerde hemfikir: **{', '.join(agreement)}**  \n"
            "Bu özellikler bu hasta için güçlü kanıt taşıyor."
        )
    else:
        st.warning("SHAP ve LIME bu hasta için farklı özellikler öne çıkardı — yoruma dikkat edin.")

    # JSON gosterimi (gelistirici modu)
    with st.expander("🔍 Ham Açıklama Verisi (JSON)"):
        st.json(explanation)


# ============================================================
# SEKME 3: MODEL PERFORMANSI
# ============================================================
with tab3:
    st.subheader("📈 Model Karşılaştırması")

    # Metrik tablosu
    perf_rows = []
    for model_name, m in metrics.items():
        perf_rows.append({
            "Model":     model_name,
            "Recall ↑":  m["recall"],
            "F1 ↑":      m["f1"],
            "ROC-AUC ↑": m["roc_auc"],
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
        })

    df_perf = pd.DataFrame(perf_rows).set_index("Model")

    st.dataframe(
        df_perf.style.highlight_max(
            subset=["Recall ↑", "F1 ↑", "ROC-AUC ↑"],
            color="#d4edda",
            axis=0,
        ).format("{:.4f}"),
        use_container_width=True,
    )

    st.caption(
        "↑ Yüksek iyi | **Recall** ana metrik — diyabetliyi kaçırmamak kritik"
    )

    st.divider()

    # Notlar
    st.markdown("""
    ### 📌 Metrik Açıklamaları
    | Metrik | Anlam |
    |---|---|
    | **Recall** | Gerçek diyabetlilerin kaçta kaçı yakalandı — klinik açıdan kritik |
    | **Precision** | "Diyabetli" dediğimizde kaç kere haklıyız |
    | **F1** | Recall ve Precision dengesi |
    | **ROC-AUC** | Eşikten bağımsız ayırt etme gücü |
    | **Accuracy** | Genel doğruluk — sınıf dengesizliğinde yanıltıcı olabilir |
    """)
