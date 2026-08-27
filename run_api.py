"""
run_api.py
----------
FastAPI'yi baslatir.

Calistirmak icin:
    .venv\Scripts\python.exe run_api.py

Swagger dokumantasyonu:
    http://localhost:8000/docs

Uvicorn nedir?
    FastAPI bir "ASGI" uygulamasi. ASGI = Asynchronous Server Gateway Interface.
    Uvicorn bu ASGI uygulamasini HTTP sunucusuna donusturur.
    Yani FastAPI kodu yazar, Uvicorn onu internete acar.

    Benzetme:
        FastAPI = Asci (yemegi yapan)
        Uvicorn = Garson (yemegi masaya getiren)

reload=True:
    Kod degistiginde sunucu otomatik yeniden baslar.
    Gelistirme sirasinda hayat kurtarir.
    Production'da False olmali.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",   # "hangi dosyadaki hangi nesne"
        host="0.0.0.0",   # 0.0.0.0 = her network arayuzunden erisim
        port=8000,
        reload=False,     # Pipeline agir, reload kapatiyoruz
        log_level="info",
    )
