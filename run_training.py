"""
RAG pipeline testi:
1. Vector store olustur (ilk seferde embedding modeli indirir)
2. Hasta 63 icin baglam getir
3. RAG ile rapor uret
4. RAG olmadan rapor uret
5. Karsilastir
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.data_processing import run_preprocessing_pipeline
from src.model_training import run_training_pipeline
from src.explainability import build_shap_explainer, build_lime_explainer, explain_patient
from src.rag_handler import build_vector_store, retrieve_context
from src.llm_handler import generate_report, print_report

# 1. Pipeline
print("=== Pipeline yukleniyor... ===")
X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing_pipeline()
results = run_training_pipeline(X_train, X_test, y_train, y_test)
xgb_model, _ = results["xgboost"]
shap_exp = build_shap_explainer(xgb_model, X_train)
lime_exp = build_lime_explainer(X_train, feature_names)

# 2. Hasta aciklamasi (BMI=43.6 olan hasta)
explanation = explain_patient(63, X_test, y_test, xgb_model, shap_exp, lime_exp, feature_names, scaler, 5)

# 3. Vector store
print("\n=== RAG: Vector Store olusturuluyor... ===")
vector_store = build_vector_store()

# 4. Baglam getir
print("\n=== RAG: Baglam getiriliyor... ===")
context = retrieve_context(explanation, vector_store, top_k=3)
print(f"\n--- Getirilen RAG Baglami (ilk 600 karakter) ---")
print(context[:600])
print("...")

# 5. RAG ile rapor
print("\n=== RAG AKTIF rapor uretiliyor... ===")
report_rag = generate_report(explanation, rag_context=context)
print_report(report_rag, 63)

# 6. RAG olmadan rapor
print("\n=== RAG DEVRE DISI rapor uretiliyor... ===")
report_no_rag = generate_report(explanation, rag_context="")
print_report(report_no_rag, 63)
