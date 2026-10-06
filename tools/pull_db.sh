#!/usr/bin/env bash
# Сервер → ноутбук: свіжа копія бази для розмітки (label.py), навчання (features.py / train.py) і бектестів.
# Запуск з Git Bash у папці проєкту (Docker Desktop має бути запущений, label.py — закритий):
#   bash tools/pull_db.sh
# Що робить: дамп бази на сервері → файл data/server.dump на ноутбуці → відновлення в ЛОКАЛЬНУ базу (замінює її вміст)
# → python load_pumps.py (таблиця pumps — з твого pumps.csv, а не з серверної копії).
# Сервер при цьому працює далі, нічого на ньому не змінюється.
set -euo pipefail
export MSYS_NO_PATHCONV=1          # Git Bash не має перетворювати /tmp/... у шляхи Windows в аргументах docker

HOST="ubuntu@13.60.152.123"
PORT=2222
KEY="D:/keys/pd.pem"
OUT="data/server.dump"

cd "$(dirname "$0")/.."
mkdir -p data
local_db=$(docker compose ps -q db 2>/dev/null || true)
if [ -z "$local_db" ]; then
  echo "Локальна база не запущена: відкрий Docker Desktop і виконай  docker compose up -d"; exit 1
fi

t0=$(date +%s)
echo "1/3 дамп бази на сервері → $OUT ..."
ssh -p "$PORT" -i "$KEY" -o ConnectTimeout=20 "$HOST" \
  "docker exec pd-db-1 pg_dump -U postgres -Fc -Z 6 pump_detector" > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "    $(du -h "$OUT" | cut -f1)"

echo "2/3 відновлення в локальну базу (стара локальна копія замінюється) ..."
docker cp "$OUT" "$local_db:/tmp/server.dump"
# --clean --if-exists: спершу видалити таблиці, потім створити з дампу; помилки «не існує» — не страшні
docker exec "$local_db" pg_restore -U postgres -d pump_detector --clean --if-exists --no-owner -j 4 /tmp/server.dump \
  || echo "    (pg_restore повідомив про попередження — перевір рядки вище; зазвичай це не страшно)"
docker exec "$local_db" rm -f /tmp/server.dump

echo "3/3 розмітка з pumps.csv → таблиця pumps ..."
.venv/Scripts/python load_pumps.py

last=$(docker exec "$local_db" psql -U postgres -d pump_detector -Atc "SELECT max(scored_at) FROM signals")
echo "готово за $(( $(date +%s) - t0 )) с; остання оцінка в копії: $last"
