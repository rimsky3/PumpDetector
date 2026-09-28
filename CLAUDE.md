# Pump Detector — контекст проєкту

Пет-проєкт для портфоліо (перехід у ML/DS): модель, що розпізнає pump-and-dump на крипто-ф'ючерсах
і узагальнюється на тікери, яких не бачила під час навчання.

- Повне ТЗ: https://claude.ai/code/artifact/ced9c93e-7099-4097-934a-4d64159cd413
- Макет інтерфейсу: https://claude.ai/artifact/MAivAWzKcz2NKqPQs4JENJ

## Скоуп v1
- Тільки режим Midterm, гранулярність 1h. Shortterm (5m) і ієрархічна схема — у v2.
- Класифікація: 1 = памп триває, 0 = звичайний ринок. Не регресія ціни.
- Код пишеться і запускається локально (Windows, VS Code, Python venv, PostgreSQL).
  Сервер (Vultr VPS, Docker Compose) — пізніше, для скринера 24/7 і публічного демо.

## Дані
- Джерело: Coinalyze API (ключ у `.env`), ліміт 40 викликів/хв, кожен символ у запиті = 1 виклик.
- Біржі: Binance, Bybit, OKX, Aster, Hyperliquid, MEXC. Агрегацію між біржами робимо самі.
- 1h-історія в Coinalyze ≈ 60–80 днів, 5m ≈ тиждень.
- OI і ліквідації тягнемо в USD (convert_to_usd). Обсяг OHLCV буває в монетах або USD —
  див. `markets.volume_denominated`, приводити до однієї одиниці в `features.py`.
- Сховище: PostgreSQL, схема в `schema.sql`. Сирі таблиці: upsert по (symbol, interval, ts).

## Ознаки
OI (дельта, швидкість зміни), OHLCV (log-return, volume z-score, bv/v), funding і predicted funding,
ліквідації (обсяг, асиметрія long/short), long/short ratio.
Усі ознаки відносні до власної історії тікера (z-score, ratio, %), не абсолютні.
`features.py` — один модуль для тренування, інференсу і скринера, без копій логіки.

## Розмітка
- `pumps.csv` → таблиця `pumps`: symbol, start_ts, end_ts, pattern_type, notes. Усі часи в UTC.
- Початок: OI стійко зростає над звичайним рівнем (довга передфаза не обов'язкова).
- Кінець: різкий обвал OI після піку ціни.
- Приклади v1: LSKUSDT, GUSDT (був фальстарт без OI — цінний негативний приклад), MYXUSDT, AKEUSDT.

## Модель і валідація
- Baseline: просте правило (OI + volume) → Logistic Regression → LightGBM.
- Дисбаланс класів: class weights або SMOTE.
- Leave-ticker-out валідація; метрики precision / recall / F1, не accuracy.
- Модель мусить перевершити просте правило, інакше це чесний негативний результат.

## Вихід з позиції
- Стоп: 75-й перцентиль MAE в одиницях NATR.
- Основний вихід: ймовірність моделі нижче порогу 2–3 бари поспіль (очікуваний лаг відносно піку).
- Trailing stop (≈1.5 NATR від максимуму) — страховка.

## Структура
- `schema.sql`, `fetch.py` — готові
- далі: `pumps.csv` + завантажувач, `features.py`, `train.py`, `app/streamlit_app.py`, `screener.py`

## Стиль роботи
Спілкування українською. Власник проєкту — досвідчений трейдер, у ML новачок-практик:
пояснювати рішення, а не лише давати код.
