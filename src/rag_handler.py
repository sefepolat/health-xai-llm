"""
rag_handler.py
--------------
RAG (Retrieval Augmented Generation) katmani.

Akis:
    data/knowledge_base/*.md
        -> Belgeler chunk'lara bolunur
        -> multilingual-e5-small ile embed edilir
        -> ChromaDB'ye yazilir (persist edilir)

    Rapor uretimi sirasinda:
        Hasta ozellikleri + anormal degerler -> sorgu
        -> ChromaDB'den en alakali 3 chunk
        -> LLM prompt'una "KLINIK BILGI TABANI" bölumu olarak eklenir

Teknik kararlar:
    - ChromaDB: Local, sunucu gerektirmez, .chromadb/ klasorune yazar
    - multilingual-e5-small: ~118MB, Turkce + Ingilizce destekler
    - Chunk boyutu 600 token, 100 overlap: Klinik metin icin ideal
      (cok kucuk -> baglam kaybi, cok buyuk -> alakasiz bilgi girer)
"""

import os
from pathlib import Path
from loguru import logger

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import BASE_DIR

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
CHROMA_PERSIST_DIR = str(BASE_DIR / ".chromadb")
COLLECTION_NAME    = "diabetes_knowledge"
EMBEDDING_MODEL    = "intfloat/multilingual-e5-small"

# Chunk ayarlari
# Neden 600/100?
#   600 token: Bir klinik paragraf icin yeterli baglam
#   100 overlap: Chunk sinirlarinda bilgi kaybolmasin diye
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100

# Kac chunk getirilecek
TOP_K = 3


# ---------------------------------------------------------------------------
# Embedding modeli (once yuklenince cache'lenir)
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    multilingual-e5-small embedding modelini yukler.

    Neden HuggingFaceEmbeddings?
        sentence-transformers kutuphanesi uzerinden calisir.
        Model ilk seferde Hugging Face'den indirilir (~118MB),
        sonrasinda local cache'den yuklenir.

    model_kwargs: {"device": "cpu"} — GPU yoksa CPU kullan
    encode_kwargs: normalize_true → kosinüs benzerligini saglar
        Normalize edilmis vektorler [0,1] araliginda benzerlik verir.
        ChromaDB kosinüs mesafesi kullanirken bu gerekli.
    """
    logger.info(f"Embedding modeli yukleniyor: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Bilgi Tabani Olusturma
# ---------------------------------------------------------------------------

def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Bilgi tabanini okur, chunk'lar, embed eder ve ChromaDB'ye yazar.

    Neden persist?
        Her uygulama baslatildiginda embedding hesaplamak yavas.
        ChromaDB .chromadb/ klasorune yazarak sonraki acilislarda
        aninda yukler — embedding maliyeti bir kez odenir.

    force_rebuild=True:
        Bilgi tabani guncellendiyse mevcut ChromaDB'yi sil, yeniden olustur.
        Yoksa mevcut varsa direkt yukle.

    Args:
        force_rebuild: True ise mevcut ChromaDB'yi sil ve yeniden olustur.

    Returns:
        Chroma vector store nesnesi.
    """
    embeddings = get_embeddings()

    # Mevcut ChromaDB varsa ve force_rebuild False ise direkt yukle
    chroma_path = Path(CHROMA_PERSIST_DIR)
    if chroma_path.exists() and not force_rebuild:
        logger.info("Mevcut ChromaDB yukleniyor...")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vector_store._collection.count()
        if count > 0:
            logger.info(f"ChromaDB yuklendi: {count} chunk mevcut")
            return vector_store
        logger.info("ChromaDB bos, yeniden olusturuluyor...")

    # Belgeleri yukle
    logger.info(f"Bilgi tabani okunuyor: {KNOWLEDGE_BASE_DIR}")
    documents = []
    md_files = list(KNOWLEDGE_BASE_DIR.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"Bilgi tabani bos: {KNOWLEDGE_BASE_DIR} icinde .md dosyasi yok"
        )

    for md_file in md_files:
        loader = TextLoader(str(md_file), encoding="utf-8")
        docs = loader.load()
        # Her belgeye kaynak bilgisi ekle
        for doc in docs:
            doc.metadata["source"] = md_file.name
        documents.extend(docs)
        logger.info(f"  Yuklendi: {md_file.name}")

    logger.info(f"Toplam {len(documents)} belge yuklendi")

    # Chunk'lara bol
    # RecursiveCharacterTextSplitter:
    #   Once paragrafa bol (\n\n), sonra satira (\n), sonra bosluga (" ")
    #   Bu sirada bol — boylece anlamli sinirlar korunur
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"{len(chunks)} chunk olusturuldu (boyut={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # ChromaDB'ye yaz (embed + persist)
    logger.info("ChromaDB'ye yaziliyor (embedding hesaplaniyor)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info(f"ChromaDB olusturuldu: {CHROMA_PERSIST_DIR}")
    return vector_store


# ---------------------------------------------------------------------------
# Baglam Getirme (Retrieve)
# ---------------------------------------------------------------------------

def retrieve_context(
    explanation: dict,
    vector_store: Chroma,
    top_k: int = TOP_K,
) -> str:
    """
    Hasta aciklama diktinden sorgu olusturur ve alakali chunk'lari getirir.

    RAG'in kalbi burasi:
        1. Hasta verisinden anlamli bir sorgu metni olustur
        2. Bu metni embed et (ayni embedding modeli)
        3. ChromaDB'de kosinüs benzerligine gore en yakin top_k chunk'i bul
        4. Bu chunk'lari string olarak birlestir -> LLM prompt'una ekle

    Neden bu hasta verisinden sorgu olusturuyoruz?
        LLM "Glucose 171 mg/dL" gorduğünde ne anlama geldigini bilmeyebilir.
        Biz "Glucose elevated, BMI obesity, insulin resistance" diye sorgu atarsak
        ChromaDB bu konulardaki chunk'lari dondurur.
        LLM artık "ADA'ya gore Glucose >= 126 = diyabet..." baglamıyla rapor yazar.

    Args:
        explanation: explain_patient() tarafindan uretilen dict.
        vector_store: Mevcut ChromaDB nesnesi.
        top_k: Kac chunk getirilecegi.

    Returns:
        Birlestirilmis context string'i (LLM prompt'una eklenecek).
    """
    # Sorgu metnini olustur: anormal degerleri ve tahmini vurgula
    raw = explanation.get("raw_values", {})
    pred_label = explanation.get("prediction", {}).get("label", "")
    shap = explanation.get("shap", {})

    # Sorgu: "yuksek deger ne anlama gelir?" seklinde dogal dil sorgusu
    # Klinik terimleri kullanmak embedding kalitesini arttirir
    query_parts = [f"diabetes risk assessment prediction: {pred_label}"]

    feature_terms = {
        "Glucose":                  "plasma glucose blood sugar hyperglycemia",
        "BMI":                      "body mass index obesity overweight",
        "Insulin":                  "insulin resistance hyperinsulinemia",
        "BloodPressure":            "blood pressure hypertension",
        "DiabetesPedigreeFunction": "diabetes family history genetic risk",
        "Pregnancies":              "gestational diabetes pregnancy history",
        "SkinThickness":            "skin fold thickness body fat adiposity",
        "Age":                      "age diabetes risk",
    }

    # Anormal ya da SHAP'ta one cikan ozellikleri sorguya ekle
    for feature in shap.keys():
        term = feature_terms.get(feature, feature)
        query_parts.append(term)

    query = " ".join(query_parts)
    logger.debug(f"RAG sorgusu: {query[:100]}...")

    # Similarity search
    docs = vector_store.similarity_search(query, k=top_k)

    if not docs:
        logger.warning("RAG: Hic alakali chunk bulunamadi")
        return ""

    # Chunk'lari birlestir
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[Kaynak {i}: {source}]\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)
    logger.info(f"RAG: {len(docs)} chunk getirildi (kaynak: {[d.metadata.get('source') for d in docs]})")
    return context
