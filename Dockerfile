# Образ для скринера і дашборду на сервері (docker-compose.server.yml).
# Лише Python і бібліотеки: сам код підключається з папки проєкту томом (./:/app), тому оновлення коду —
# це скопіювати файли і перезапустити контейнер, без перебудови образу. Перебудова — лише коли змінились бібліотеки.
FROM python:3.14-slim

# libgomp1 — OpenMP, без нього LightGBM не завантажується
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# вивід print() одразу в лог контейнера (без буфера); українські тексти — UTF-8
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
# точні версії — ті самі, що на ноутбуці: моделі (joblib) збережені цими версіями LightGBM / scikit-learn
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
