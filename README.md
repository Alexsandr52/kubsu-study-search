# KUBSU Session Search

Локальный поиск по учебным материалам и сообщениям Telegram: Qdrant находит фрагменты, cross-encoder reranker уточняет порядок результатов, а локальная Qwen формирует ответ со ссылками на источники.

## Быстрый запуск готовой базы

Нужны Git, Git LFS, Docker с Compose, Python 3.11+ и [Ollama](https://ollama.com/download).

```bash
git clone <URL-репозитория>
cd <папка-репозитория>
git lfs install
git lfs pull
shasum -a 256 -c artifacts/qdrant/SHA256SUMS
```

Поднимите Qdrant и загрузите в него включённый в проект snapshot коллекции:

```bash
docker compose up -d qdrant
curl -f http://localhost:6333/collections
curl -X POST \
  'http://localhost:6333/collections/telegram_keep_user2_350tok_50overlap/snapshots/upload?priority=snapshot' \
  -F 'snapshot=@artifacts/qdrant/telegram_keep_user2_350tok_50overlap.snapshot'
```

Загрузите локальную Q&A-модель и установите зависимости приложения:

```bash
ollama pull qwen3.5:4b
python3 -m venv .venv-reranker
source .venv-reranker/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Streamlit/requirements.txt
```

Запустите интерфейс:

```bash
./Streamlit/run_app.sh
```

Откройте <http://127.0.0.1:8501>. При первом запросе приложение скачает USER2-base и reranker с Hugging Face. Ollama должна быть запущена локально на `http://localhost:11434`, Qdrant — на `http://localhost:6333`. На CPU модели работают медленнее; на Apple Silicon приложение использует MPS, если он доступен.

Snapshot содержит Qdrant-коллекцию целиком: векторы, тексты сообщений, метаданные и пути к вложениям. Он хранится в Git LFS. Исходные Telegram-экспорты, PDF, промежуточные таблицы и локальное хранилище Qdrant в Git не включены. Snapshot является готовой базой для демонстрационного запуска; файловые ссылки на исходные вложения откроются только если соответствующие файлы есть локально.

## Обработка собственных данных

Каталог `data/` исключён из Git. Чтобы собрать индекс заново, положите Telegram HTML-экспорты в `data/ChatExport_*` и при необходимости учебные PDF в `data/KUBSU_DOCS/`.

1. В `notebooks/EDA.ipynb` разберите экспорты и создайте CSV датированных сообщений от 1 июля 2026 года.
2. В `notebooks/qwen_chat_processing.ipynb` классифицируйте сообщения, проверьте ручную очередь, извлеките OCR-текст вложений и соберите хронологический корпус.
3. Установите зависимости, перечисленные в первых ячейках ноутбуков. Для OCR PDF нужны Poppler и Tesseract с русской языковой моделью.
4. Запустите Qdrant через `docker compose up -d qdrant`, затем выполните `notebooks/01_qdrant_ingest_user2.ipynb`. Размер чанка — 350 токенов USER2-base, перекрытие — 50.
5. Для дополнительных учебных PDF проверьте отчёт извлечения и выполните `notebooks/04_qdrant_ingest_kubsu_docs.ipynb`.
6. Ноутбуки `02_qdrant_search_eval_user2.ipynb` и `03_qdrant_ask_rerank_answer.ipynb` содержат примеры поиска и оценки ответов.

## Как работает поиск

- Приложение объединяет выдачи исходного вопроса и трёх его поисковых переформулировок, оставляет до 100 кандидатов и ранжирует их cross-encoder моделью `sshalimov04/ru-reranker-edge-150m`.
- В Qwen передаются максимум 10 наиболее сильных результатов, находящихся не дальше 1.25 по reranker score от лучшего совпадения. Это относительный отсев по оценке модели, а не вероятность релевантности.
- В раскрываемом блоке интерфейса доступны все 50 результатов реранкера; там видно, какие фрагменты использовались в ответе.
- Qwen `qwen3.5:4b` получает инструкцию отвечать только по подтверждённым источникам и сообщать, если найденных данных недостаточно. Размер контекста подстраивается под запрос, с резервом 600 токенов для ответа.

## Состав проекта

- `Streamlit/app.py` — приложение.
- `Streamlit/requirements.txt` — зависимости приложения.
- `Streamlit/run_app.sh` — запуск Streamlit.
- `notebooks/` — EDA, подготовка корпуса, индексация, поиск и Q&A.
- `artifacts/qdrant/` — snapshot готовой коллекции Qdrant (Git LFS).
- `docker-compose.yml` — локальный Qdrant.
