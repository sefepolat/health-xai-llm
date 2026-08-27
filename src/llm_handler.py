"""
llm_handler.py
--------------
LLM entegrasyonu — Ollama uzerinde calistiran llama3.1:8b modeli ile
hasta aciklama raporlari uretir.

Akis:
    explain_patient() ciktisi (dict)
        -> build_prompt()
        -> LLM (ChatOllama / llama3.1:8b)
        -> JsonOutputParser + Pydantic dogrulama
        -> PatientReport

Temel kararlar:
    - ChatOllama: Yerel LLM, gizlilik sorunu yok, API maliyeti yok
    - Pydantic sema: LLM ciktisini yapilandirma + runtime dogrulama
    - Temperature=0.2: Tutarli, yaratici olmayan medikal rapor
    - Guardrails: System prompt ile teshis koymak yasaklandi
    - Turkce: Hasta raporlari Turkce uretilir
"""

import json
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from loguru import logger

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TEMPERATURE


# ---------------------------------------------------------------------------
# Adim 1: Pydantic Cikti Semasi
# ---------------------------------------------------------------------------

class PatientReport(BaseModel):
    """
    LLM'in uretecegi hasta raporunun yapisi.

    Neden Pydantic?
        1. format_instructions: Sema LLM'e prompt olarak gonder
           "Bu alanlari bu tipte doldur" -> LLM formati taklit eder
        2. Runtime dogrulama: LLM yanlis tip dondururse Pydantic yakalar
           risk_level="maybe" -> ValidationError -> retry mekanizmasi
        3. IDE destegi: Kod yazarken otomatik tamamlama calisir
    """
    risk_level: Literal["DUSUK", "ORTA", "YUKSEK"] = Field(
        description="Hastanin diyabet risk seviyesi. Sadece: DUSUK, ORTA veya YUKSEK"
    )
    summary: str = Field(
        description="Modelin tahminini ve temel nedenleri aciklayan 2-3 cumle. "
                    "Teknik jargon kullanma, hasta anlayabilmeli."
    )
    key_factors: list[str] = Field(
        description="Tahmini en cok etkileyen 3 faktor, kisa cumleler halinde. "
                    "Ornek: ['Kan sekeri yuksek', 'Vucut kitle indeksi normal']"
    )
    recommendation: str = Field(
        description="Hastaya yonelik genel saglik onerisi. "
                    "Kesin tani veya ilac onerme."
    )
    disclaimer: str = Field(
        description="Zorunlu medikal uyari cumlesi. Her raporda olmali."
    )


# ---------------------------------------------------------------------------
# Adim 2: Prompt Sablonu
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Sen bir klinik karar destek sistemisin. Gorev: makine ogrenme \
modelinin urettigi SHAP ve LIME aciklamalarini, hastanin anlayabilecegi sade \
Turkce ile yorumlamak.

KESIN KURALLAR:
1. Kesin tibbi teshis KOYMA ("Diyabetiniz var" gibi ifadeler yasak)
2. Ilac veya tedavi onerme
3. Veri setinde olmayan olculere referans verme (HbA1c, kolesterol vb.)
4. Her raporda mutlaka bir medikal uyari cumlesi bulundur
5. Teknik terimler kullanma (SHAP, LIME, model gibi kelimeler hastaya anlamsiz)
6. Empatik, sakin ve bilgilendirici bir dil kullan

HASTA BILGISI YORUMLAMA KURALLARI (COK ONEMLI):
7. "Durum: NORMAL" olan ozellikleri ASLA temel faktor olarak listeleme.
   Ornek: Pregnancies=4, Durum=NORMAL ise bunu raporda hic belirtme.
8. "Durum: COK YUKSEK" veya "Durum: YUKSEK" olan ozellikleri MUTLAKA belirt.
   Bunlar SHAP listesinde olmasa bile, klinik olarak anlamli oldugu icin rapora dahil et.
9. "Durum: COK YUKSEK" ve "Durum: YUKSEK" arasindaki farki vurgula:
   COK YUKSEK = cok ciddi sapma, kesinlikle bahset
   YUKSEK = sapma var, bahset
   NORMAL = sapma yok, bahsetme

SHAP DEGERI YORUMU:
- Pozitif SHAP: O ozellik diyabet riskini artiriyor
- Negatif SHAP: O ozellik diyabet riskini azaltiyor
- Mutlak buyukluk: Etkinin gucunu gosteriyor

AGREEMENT (UZLASMA):
- Hem SHAP hem LIME ayni ozelligi one cikardiysa -> guclu kanit
- Sadece birinde ciktiysa -> zayif kanit, dikkatli yorum yap"""


USER_PROMPT_TEMPLATE = """Asagidaki hasta icin diyabet risk raporu uret.

MODEL TAHMINI:
- Sonuc: {prediction_label}
- Olasilik: %{prediction_probability}

RAPORA DAHIL EDILECEK FAKTORLER — SADECE BUNLARI YAZ, BASKASINI EKLEME:
{curated_factors}

LIME DOGRULAMASI (ek bilgi):
{lime_factors}

HER IKI YONTEMIN UZLASTIGI FAKTORLER: {agreement}

{rag_context_section}

TALIMATLAR:
- key_factors listesine YUKARIDAKI FAKTORLER'den her birini ekle
- "Durum: NORMAL" yazan hicbir satiri key_factors'a EKLEME
- "Durum: COK YUKSEK" veya "Durum: YUKSEK" olan her satiri key_factors'a MUTLAKA ekle
- Ozette de bu anormal degerlere yer ver
- Klinik bilgi tabanindaki referans degerleri kullanarak ozeti zenginlestir

{format_instructions}"""


# ---------------------------------------------------------------------------
# Adim 3: LLM ve Parser Kurulumu
# ---------------------------------------------------------------------------

def build_chain():
    """
    LangChain zinciri olusturur: Prompt -> LLM -> Parser

    Zincir mimarisi (LCEL — LangChain Expression Language):
        prompt | llm | parser

    | operatoru nedir?
        Python'da normali bitwise OR, ama LangChain bunu override etti.
        Anlami: "soldakinin ciktisini sagdakinin girdisine ver"
        prompt.invoke(data) -> llm.invoke(prompt_ciktisi) -> parser.invoke(llm_ciktisi)
        Tek satirda zincir: prompt | llm | parser

    Neden JsonOutputParser?
        LLM metin uretir — biz dict istiyoruz.
        JsonOutputParser: LLM'den gelen JSON string'i Python dict'e cevirir.
        Pydantic sema ile kullanildiginda format_instructions otomatik uretilir.

    Returns:
        (chain, parser) — chain.invoke() ile kullanilir
    """
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        format="json",   # Ollama'ya: "ciktin JSON olmali"
    )

    parser = JsonOutputParser(pydantic_object=PatientReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])

    chain = prompt | llm | parser
    logger.info(f"LLM zinciri olusturuldu: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    return chain, parser


# ---------------------------------------------------------------------------
# Adim 4: Aciklama Dict'ini LLM'e Hazirla
# ---------------------------------------------------------------------------

def format_explanation_for_prompt(explanation: dict) -> dict:
    """
    explain_patient() ciktisini LLM prompt sablonuna uygun stringe donusturur.

    Temel tasarim karari:
        LLM'e SHAP listesi + ayri klinik alarm gondermek cakisma yaratiyordu.
        LLM ikisini uzlastirmaya calisip yanlis sonuc veriyordu.

        Cozum: Biz programatik olarak temiz tek bir faktor listesi olusturuyoruz:
            1. SHAP top-N icinden sadece klinik durumu NORMAL OLMAYANLAR alinir
            2. COK YUKSEK / YUKSEK / DUSUK olan her ozellik listeye eklenir
               (SHAP'ta olmasa bile)
            3. Duplicate'lar temizlenir
        LLM bu hazir listeyi gorur ve "sadece bunu yaz" seklinde kullanir.

    Args:
        explanation: explain_patient() tarafindan uretilen dict.

    Returns:
        ChatPromptTemplate'in bekledigini degiskenler sozlugu.
    """
    raw_values = explanation["raw_values"]
    prediction = explanation["prediction"]
    shap       = explanation["shap"]
    lime       = explanation["lime"]
    agreement  = explanation["agreement"]

    # Klinik referans araliklar
    REFERENCE = {
        "Pregnancies":              {"birim": "adet",  "alt": 0,    "ust": 17,   "label": "0-17"},
        "Glucose":                  {"birim": "mg/dL", "alt": 70,   "ust": 99,   "label": "70-99"},
        "BloodPressure":            {"birim": "mmHg",  "alt": 60,   "ust": 80,   "label": "60-80"},
        "SkinThickness":            {"birim": "mm",    "alt": 10,   "ust": 40,   "label": "10-40"},
        "Insulin":                  {"birim": "uU/mL", "alt": 16,   "ust": 166,  "label": "16-166"},
        "BMI":                      {"birim": "kg/m2", "alt": 18.5, "ust": 24.9, "label": "18.5-24.9"},
        "DiabetesPedigreeFunction": {"birim": "",      "alt": None, "ust": None, "label": "0.08-2.42"},
        "Age":                      {"birim": "yil",   "alt": None, "ust": None, "label": "-"},
    }

    def get_status(name: str, val: float) -> str:
        ref = REFERENCE.get(name, {})
        alt, ust = ref.get("alt"), ref.get("ust")
        if alt is None or ust is None:
            return "?"
        if val < alt:
            return "DUSUK"
        if val > ust:
            return "COK YUKSEK" if (val - ust) / ust > 0.5 else "YUKSEK"
        return "NORMAL"

    # --- Ham degerler tablosu (klinik durum ile birlikte) ---
    raw_lines = []
    for name, val in raw_values.items():
        ref    = REFERENCE.get(name, {})
        birim  = ref.get("birim", "")
        aralik = ref.get("label", "-")
        durum  = get_status(name, val)
        raw_lines.append(
            f"  - {name}: {val} {birim} | Normal: {aralik} | Durum: {durum}"
        )
    raw_str = "\n".join(raw_lines)

    # --- Programatik faktor birlestirme ---
    # Adim 1: SHAP top-N'den sadece anormal olanlari al
    curated: dict[str, dict] = {}   # {feature: {shap, durum, val}}
    for feature, shap_val in shap.items():
        raw_val = raw_values.get(feature)
        if raw_val is None:
            continue
        durum = get_status(feature, raw_val)
        if durum != "NORMAL":   # NORMAL ise listeye ekleme
            curated[feature] = {"shap": shap_val, "durum": durum, "val": raw_val}

    # Adim 2: COK YUKSEK / YUKSEK / DUSUK olan her ozelligi ekle
    # (SHAP top-N'de olmasa bile klinik olarak onemli)
    for feature, val in raw_values.items():
        durum = get_status(feature, val)
        if durum in ("COK YUKSEK", "YUKSEK", "DUSUK") and feature not in curated:
            curated[feature] = {"shap": None, "durum": durum, "val": val}

    # Adim 3: Temiz faktor listesini stringe cevir
    factor_lines = []
    for feature, info in curated.items():
        ref    = REFERENCE.get(feature, {})
        birim  = ref.get("birim", "")
        aralik = ref.get("label", "-")
        durum  = info["durum"]
        val    = info["val"]
        shap_v = info["shap"]

        if shap_v is not None:
            yon = "riski artiriyor" if shap_v > 0 else "riski azaltiyor"
            factor_lines.append(
                f"  - {feature} = {val} {birim} | Durum: {durum} | "
                f"Model etkisi: {yon} (SHAP={shap_v:+.3f})"
            )
        else:
            factor_lines.append(
                f"  - {feature} = {val} {birim} | Durum: {durum} | "
                f"Normal aralik: {aralik} — klinik olarak anormal"
            )

    factors_str = (
        "\n".join(factor_lines)
        if factor_lines
        else "  Klinik olarak anormal faktor bulunamadi."
    )

    # --- LIME ozeti (dogrulama icin) ---
    lime_lines = []
    for feature, val in lime.items():
        direction = "artirici" if val > 0 else "azaltici"
        lime_lines.append(f"  - {feature}: {val:+.4f} ({direction})")
    lime_str = "\n".join(lime_lines)

    return {
        "raw_values":              raw_str,
        "curated_factors":         factors_str,
        "prediction_label":        prediction["label"],
        "prediction_probability":  f"{prediction['probability'] * 100:.1f}",
        "lime_factors":            lime_str,
        "agreement":               ", ".join(agreement) if agreement else "Uzlasma yok",
    }


# ---------------------------------------------------------------------------
# Adim 5: Rapor Uret
# ---------------------------------------------------------------------------

def generate_report(
    explanation: dict,
    rag_context: str = "",
) -> PatientReport:
    """
    Hasta aciklama diktini LLM'e gonderir ve yapılandırılmis rapor uretir.

    Hata yonetimi:
        LLM bazen gecersiz JSON uretebilir (orn: eksik alan, yanlis tip).
        Bu durumda JsonOutputParser hata firlatir.
        Biz bunu yakalar ve fallback rapor dondururuz.

    Args:
        explanation:  explain_patient() tarafindan uretilen dict.
        rag_context:  rag_handler.retrieve_context() tarafindan uretilen
                      klinik bilgi tabani metni. Bos string = RAG devre disi.

    Returns:
        PatientReport — validate edilmis Pydantic nesnesi.
    """
    chain, parser = build_chain()
    prompt_vars = format_explanation_for_prompt(explanation)

    # RAG baglami: varsa prompt'a ekle, yoksa bos birak
    if rag_context:
        prompt_vars["rag_context_section"] = (
            "KLINIK BILGI TABANI (ADA/WHO kilavuzlari — bu bilgilere dayanarak yaz):\n"
            + rag_context
        )
    else:
        prompt_vars["rag_context_section"] = ""

    prompt_vars["format_instructions"] = parser.get_format_instructions()

    logger.info(
        f"LLM rapor uretimi basliyor | "
        f"Hasta: {explanation['patient_index']} | "
        f"Tahmin: {explanation['prediction']['label']} | "
        f"RAG: {'Aktif' if rag_context else 'Devre disi'}"
    )

    try:
        result = chain.invoke(prompt_vars)
        if isinstance(result, dict):
            report = PatientReport(**result)
        else:
            report = result
        logger.info(f"Rapor uretildi | Risk: {report.risk_level}")
        return report

    except Exception as e:
        logger.error(f"LLM rapor uretimi basarisiz: {e}")
        return PatientReport(
            risk_level="ORTA",
            summary="Rapor otomatik olarak uretilirken bir hata olustu. "
                    "Lutfen bir saglik uzmaniyla gorusin.",
            key_factors=["Rapor uretimi basarisiz"],
            recommendation="Lutfen bir saglik profesyoneliyle iletisime gecin.",
            disclaimer="Bu sistem bir karar destek aracıdır. "
                       "Kesin tani icin hekiminize basvurun.",
        )


# ---------------------------------------------------------------------------
# Yardimci: Raporu Goster
# ---------------------------------------------------------------------------

def print_report(report: PatientReport, patient_index: int) -> None:
    """
    PatientReport nesnesini terminal'de okunabilir formatta yazdirir.
    Streamlit entegrasyonunda bu fonksiyon kullanilmaz — direkt alanlara erisir.
    """
    print(f"\n{'='*55}")
    print(f"  HASTA {patient_index} — KLiNiK KARAR DESTEK RAPORU")
    print(f"{'='*55}")
    print(f"  Risk Seviyesi : {report.risk_level}")
    print(f"\n  Ozet:\n  {report.summary}")
    print(f"\n  Temel Faktorler:")
    for i, factor in enumerate(report.key_factors, 1):
        print(f"    {i}. {factor}")
    print(f"\n  Oneri:\n  {report.recommendation}")
    print(f"\n  Uyari:\n  {report.disclaimer}")
    print(f"{'='*55}\n")
