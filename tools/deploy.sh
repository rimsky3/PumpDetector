#!/usr/bin/env bash
# Ноутбук → сервер: код проєкту (і за прапорцями — розмітка, моделі, бектести). Запуск з Git Bash у папці проєкту:
#   bash tools/deploy.sh                 — лише код (усе, що в git, + нові файли, не ігноровані .gitignore)
#   bash tools/deploy.sh --restart       — + перезапустити скринер і дашборд (новий код починає працювати)
#   bash tools/deploy.sh --labels        — ЛИШЕ розмітка і списки (pumps.csv, tickers.txt, excluded.txt, reviews.csv)
#                                          + на сервері load_pumps.py (журнал одразу бачить нову розмітку)
#                                          і fetch.py --new (годинна історія нових тікерів навчання)
#   bash tools/deploy.sh --models        — + models/ (після «Прийняти» нову модель; потрібен ще --restart)
#   bash tools/deploy.sh --backtests     — + прогони бектесту (reports/backtests/*.json) — видно у вкладці «Бектест»
#   bash tools/deploy.sh --first         — перший раз: + models/, робочі JSON зі reports/, прогони бектесту, списки
# Прапорці можна поєднувати: bash tools/deploy.sh --models --restart
#
# Файли, які на сервері змінює сам дашборд (watchlist.txt — «＋ у watchlist», scan_config.json — пороги сканера),
# звичайне оновлення НЕ перезаписує — лише --first. Секрети (.env) скрипт не копіює ніколи.
set -euo pipefail

HOST="ubuntu@13.60.152.123"
PORT=2222
KEY="D:/keys/pd.pem"
DIR="pd"                                   # папка на сервері: /home/ubuntu/pd
SERVER_OWNED='^(watchlist\.txt|scan_config\.json)$'
SKIP='^(notes/|images_bags/|data/|ТЗ )'      # особисте і великі локальні кеші — серверу не потрібні

first=0; models=0; restart=0; labels=0; backtests=0
for a in "$@"; do
  case "$a" in
    --first) first=1; models=1; backtests=1 ;;
    --models) models=1 ;;
    --restart) restart=1 ;;
    --labels) labels=1 ;;
    --backtests) backtests=1 ;;
    *) echo "невідомий прапорець: $a"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
ssh_() { ssh -p "$PORT" -i "$KEY" -o ConnectTimeout=20 "$HOST" "$@"; }

list=$(mktemp)
if [ "$labels" = 1 ]; then
  # лише розмітка і списки, які ведуться на ноутбуці (label.py); код не чіпаємо
  for f in pumps.csv tickers.txt excluded.txt reviews.csv; do
    [ -f "$f" ] && printf '%s\0' "$f" >> "$list"
  done
else
  # -c: файли в git, -o: нові (не ігноровані); -z і quotepath=off — кирилиця і пробіли в назвах
  git -c core.quotepath=off ls-files -co --exclude-standard -z | tr '\0' '\n' | grep -v -E "$SKIP" \
    | { if [ "$first" = 1 ]; then cat; else grep -v -E "$SERVER_OWNED"; fi; } | while IFS= read -r f; do
        [ -f "$f" ] && printf '%s\0' "$f"; done > "$list" || true
fi
if [ "$models" = 1 ]; then
  find models -type f -print0 >> "$list"
fi
if [ "$first" = 1 ]; then
  for f in reports/auto_watchlist.json reports/scan_last.json reports/scan_recall.json reports/scan_turnover_avg.json \
           reports/wake_events.json reports/oof_predictions.csv; do
    [ -f "$f" ] && printf '%s\0' "$f" >> "$list"
  done
fi
if [ "$backtests" = 1 ]; then
  find reports/backtests -maxdepth 1 -name '*.json' -print0 >> "$list"
fi

n=$(tr -cd '\0' < "$list" | wc -c)
echo "→ $HOST:~/$DIR — файлів: $n"
tar --null -T "$list" -czf - | ssh_ "mkdir -p $DIR && tar -xzf - -C $DIR"
rm -f "$list"

if [ "$labels" = 1 ]; then
  # load_pumps — таблиця pumps з нового pumps.csv; fetch.py --new — годинна історія Coinalyze для тікерів з tickers.txt,
  # яких ще немає на сервері (монету завантажили в label.py на ноутбуці — інакше після pull_db її годинні дані зникли б)
  ssh_ "cd $DIR && docker compose -f docker-compose.server.yml exec -T screener sh -c 'python load_pumps.py && python fetch.py --new'"
fi
if [ "$restart" = 1 ]; then
  ssh_ "cd $DIR && docker compose -f docker-compose.server.yml restart screener dashboard"
fi
echo "готово"
