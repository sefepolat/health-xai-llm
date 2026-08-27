# Dockerfile
# ----------
# Bu dosya bir "tarif" (recipe).
# "docker build" komutu bu tarifte yazılanları adım adım çalıştırır
# ve sonunda bir Image üretir.
#
# Her satır bir "katman" (layer) oluşturur.
# Docker bu katmanları cache'ler — değişmeyen katmanlar tekrar çalıştırılmaz.
# Bu yüzden sıralama önemli: nadiren değişen şeyler üstte, sık değişenler altta.

# ---------------------------------------------------------------------------
# AŞAMA 1: Base image seç
# ---------------------------------------------------------------------------
# "python:3.11-slim" → Debian tabanlı, sadece Python kurulu, minimal (~150MB)
# "slim" olmayan tam versiyon ~900MB — bize gerek yok
# Neden 3.11? Projedeki Python versiyonuyla eşleşmeli.
FROM python:3.12-slim

# ---------------------------------------------------------------------------
# AŞAMA 2: Sistem bağımlılıkları
# ---------------------------------------------------------------------------
# Bu paketler pip ile değil, işletim sistemi paket yöneticisiyle kurulur.
# build-essential: C/C++ derleyicisi — bazı pip paketleri derleme gerektirir
# curl: sağlık kontrolü için
# --no-install-recommends: tavsiye edilen ama zorunlu olmayan paketleri atla
# rm -rf /var/lib/apt/lists/*: apt cache'ini temizle → image boyutu küçülür
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# AŞAMA 3: Çalışma dizini
# ---------------------------------------------------------------------------
# Container içinde tüm komutların çalışacağı dizin.
# /app → Docker dünyasında yaygın convention.
WORKDIR /app

# ---------------------------------------------------------------------------
# AŞAMA 4: Bağımlılıkları kur (cache optimizasyonu)
# ---------------------------------------------------------------------------
# Neden önce sadece requirements.txt kopyalıyoruz?
#
# Docker layer cache mantığı:
#   Eğer requirements.txt değişmediyse bu adım cache'den gelir (hızlı).
#   Sadece kaynak kodumuz değişti → pip install tekrar çalışmaz.
#   Eğer tüm dosyaları birden kopyalasaydık, herhangi bir .py değişince
#   pip install da yeniden çalışırdı (~5 dakika).
#
# Bu pattern "dependency layer caching" olarak bilinir.
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.docker.txt

# ---------------------------------------------------------------------------
# AŞAMA 5: Kaynak kodu kopyala
# ---------------------------------------------------------------------------
# Bu adım her kod değişikliğinde yeniden çalışır — ama sadece kopyalama,
# pip install değil. Bu yüzden hızlı.
COPY src/ ./src/
COPY api/ ./api/
COPY app.py .
COPY run_api.py .
COPY data/raw/ ./data/raw/
COPY data/knowledge_base/ ./data/knowledge_base/

# ---------------------------------------------------------------------------
# AŞAMA 6: Port tanımla
# ---------------------------------------------------------------------------
# EXPOSE sadece dokümantasyon amaçlı — hangi portu kullandığını belirtir.
# Gerçek port açma docker run -p veya docker-compose'da yapılır.
EXPOSE 8000
EXPOSE 8501

# ---------------------------------------------------------------------------
# AŞAMA 7: Default komut (override edilebilir)
# ---------------------------------------------------------------------------
# docker-compose'da her servis kendi CMD'ini belirleyecek.
# Bu sadece fallback.
CMD ["python", "run_api.py"]
