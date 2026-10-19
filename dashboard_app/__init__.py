"""
dashboard_app — логіка дашборда Pump Detector, розкладена за відповідальністю (19.10.2026, notes/architecture_plan.md,
крок 2). Точка входу і HTTP-маршрути — dashboard.py; тут — дані для екранів.

    common         шляхи, State, оцінки моделей, рівні, склад списку, /api/meta
    trades         угоди моделі для списку й сповіщень, стоп наживо
    watchlist_ops  «＋ у watchlist», «×», «Повернути», картка «Сканер»
    listview       список (/api/list) і міні-графіки (/api/minicharts)
    tickerview     графік тікера (/api/ticker), «Оновити»
    twaps          TWAP · HL (/api/twaps)
    backtests      вкладка «Бектест»
    journal_view   вкладка «Журнал»
"""
