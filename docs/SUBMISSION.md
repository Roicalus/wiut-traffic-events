# Чек-лист сабмишена (по требованиям организаторов)

Порядок действий перед финальным тегом. Всё, что можно проверить
автоматически, проверяет `python tools/presubmit.py`.

1. Опорный кадр зон: `python tools/make_zone_ref.py --video samples/C3896.MP4 --frame 150`,
   проверка `python tools/check_alignment.py --videos samples`.
2. Веса лежат в `weights/` (и закоммичены — их ~25 МБ).
3. Локальный torch = 2.8.0 (как в requirements): `python -c "import torch; print(torch.__version__)"`.
4. Итоговый прогон на ВСЕХ сэмплах БЕЗ --no-risk и без WIUT_TIME_GUARD:
   `python run_submission.py --videos samples --out predictions_samples.json --team <команда>`
5. `python evaluate.py --pred predictions_samples.json --validate-only`
6. README: заполнить команду (кто что делал, ссылки), убрать `<team name>`.
7. Детерминизм: `python tools/presubmit.py --determinism samples/C3905.MP4`
8. `git add -A && git commit`, `python tools/presubmit.py` → 0 FAIL.
9. Проверка «чистой машины»: клон репозитория в новую папку, новый venv,
   `pip install -r requirements.txt`, `python run_submission.py --videos samples --out /tmp/p.json`.
   Лучше всего — в Kaggle Notebook с T4 (та же видеокарта, что у организаторов).
10. `git tag v1.0 && git push origin main --tags`; репозиторий публичный.
11. Сдать: ссылка на репозиторий + тег/хэш коммита, ссылка на сайт.
