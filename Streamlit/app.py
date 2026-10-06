from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import requests
import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub.utils._http import hf_request_event_hook, set_client_factory
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# Hugging Face downloads use gzip/deflate to avoid broken Brotli decoding in this environment.
set_client_factory(
    lambda: httpx.Client(
        headers={"accept-encoding": "gzip, deflate"},
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
    )
)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "telegram_keep_user2_350tok_50overlap"
EMBEDDING_MODEL = "deepvk/USER2-base"
RERANKER_MODEL = "sshalimov04/ru-reranker-edge-150m"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:4b"
CANDIDATE_COUNT = 100
RERANKED_COUNT = 50
FINAL_CONTEXT_COUNT = 10
# Reranker logits are not probabilities. Use a relative margin from the best
# match so weak tail results do not dilute the answer context.
CONTEXT_SCORE_MARGIN = 1.25
QUERY_VARIANT_COUNT = 3
RRF_K = 60
QNA_MAX_CONTEXT_TOKENS = 262_144
ANSWER_RESERVE_TOKENS = 600
# Leave a small margin for differences between the Hugging Face and Ollama chat templates.
TOKENIZER_SAFETY_MARGIN = 64

st.set_page_config(page_title="KUBSU Session Search", page_icon=None, layout="centered")
st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {max-width: 760px; padding-top: 3rem;}
      div[data-testid="stTextInput"] input {font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("KUBSU Session Search")


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Any, Any, Any, Any, Any, str]:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device).eval()
    rank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    rank_model = (
        AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        .to(device)
        .eval()
    )
    qna_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    return (
        embed_tokenizer,
        embed_model,
        rank_tokenizer,
        rank_model,
        qna_tokenizer,
        device,
    )


@st.cache_resource(show_spinner=False)
def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30)


def ollama_chat(
    messages: list[dict[str, str]],
    *,
    response_format: Any = None,
    num_ctx: int = 4096,
    num_predict: int = 700,
) -> str:
    body: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "seed": 42,
        },
    }
    if response_format is not None:
        body["format"] = response_format
    response = requests.post(OLLAMA_URL, json=body, timeout=300)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()


def expand_queries(question: str) -> list[str]:
    schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": QUERY_VARIANT_COUNT,
                "maxItems": QUERY_VARIANT_COUNT,
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }
    content = ollama_chat(
        [
            {
                "role": "system",
                "content": (
                    f"Составь ровно {QUERY_VARIANT_COUNT} коротких поисковых формулировки на русском "
                    "для поиска сообщений в переписке по вопросу пользователя. Сохраняй имена, даты, "
                    "названия предметов и документов. Используй разные естественные формулировки и "
                    "ключевые слова. Не отвечай на вопрос. Верни только JSON по заданной схеме."
                ),
            },
            {"role": "user", "content": question},
        ],
        response_format=schema,
    )
    parsed = json.loads(content)
    variants = parsed.get("queries", [])
    clean: list[str] = []
    for item in variants:
        if isinstance(item, str):
            item = item.strip()
            if (
                item
                and item.casefold() != question.casefold()
                and item.casefold() not in {x.casefold() for x in clean}
            ):
                clean.append(item)
    if len(clean) < QUERY_VARIANT_COUNT:
        raise RuntimeError(
            "LLM не вернула три разных варианта поискового запроса. Попробуйте отправить вопрос ещё раз."
        )
    return [question, *clean[:QUERY_VARIANT_COUNT]]


def embed_query(question: str, tokenizer: Any, model: Any, device: str) -> list[float]:
    encoded = tokenizer(
        ["search_query: " + question],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        output = model(**encoded)
        mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .expand(output.last_hidden_state.size())
            .float()
        )
        pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1e-9
        )
        return F.normalize(pooled, p=2, dim=1)[0].cpu().tolist()


def get_answer(
    question: str, progress: Any
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    progress.progress(3, text="Подключаю поиск…")
    (
        embed_tokenizer,
        embed_model,
        rank_tokenizer,
        rank_model,
        qna_tokenizer,
        device,
    ) = load_models()
    client = qdrant_client()
    collection_names = {item.name for item in client.get_collections().collections}
    if COLLECTION not in collection_names:
        raise RuntimeError(f"Коллекция Qdrant «{COLLECTION}» не найдена.")

    progress.progress(15, text="Подбираю варианты запроса…")
    queries = expand_queries(question)

    progress.progress(30, text="Ищу по нескольким формулировкам…")
    merged: dict[str, dict[str, Any]] = {}
    for query in queries:
        vector = embed_query(query, embed_tokenizer, embed_model, device)
        hits = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=CANDIDATE_COUNT,
            with_payload=True,
        ).points
        for rank, hit in enumerate(hits, 1):
            key = str(hit.id)
            if key not in merged:
                merged[key] = {
                    "hit": hit,
                    "rrf_score": 0.0,
                    "best_vector_score": float(hit.score),
                }
            merged[key]["rrf_score"] += 1.0 / (RRF_K + rank)
            if float(hit.score) > merged[key]["best_vector_score"]:
                merged[key]["hit"] = hit
                merged[key]["best_vector_score"] = float(hit.score)

    # Fuse all query variants, then rerank the best 100 unique chunks.
    candidates = sorted(
        merged.values(),
        key=lambda item: (item["rrf_score"], item["best_vector_score"]),
        reverse=True,
    )[:CANDIDATE_COUNT]
    if not candidates:
        raise RuntimeError("По этому вопросу ничего не нашлось.")
    candidate_hits = [item["hit"] for item in candidates]

    progress.progress(48, text="Переранжирую найденное…")
    texts = [hit.payload.get("text", "") for hit in candidate_hits]
    scores: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(texts), 8):
            batch = texts[offset : offset + 8]
            inputs = rank_tokenizer(
                [question] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = rank_model(**inputs).logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())

    ranked = sorted(
        zip(candidate_hits, scores), key=lambda item: float(item[1]), reverse=True
    )[:RERANKED_COUNT]
    ranked_sources: list[dict[str, Any]] = []
    for hit, score in ranked:
        payload = hit.payload
        ranked_sources.append(
            {
                "text": payload.get("text", ""),
                "date": payload.get("date", ""),
                "chat": payload.get("chat_name", ""),
                "sender": payload.get("sender", ""),
                "file_paths": payload.get("file_paths", []),
                "source_input_file": payload.get("source_input_file", ""),
                "semantic_score": float(hit.score),
                "rerank_score": float(score),
            }
        )

    # Keep the context both small and focused. The relative score margin is
    # intentionally used instead of an absolute cutoff: reranker logits are
    # not calibrated confidence values and can shift between questions.
    best_rerank_score = ranked_sources[0]["rerank_score"]
    selected_sources = [
        source
        for source in ranked_sources
        if source["rerank_score"] >= best_rerank_score - CONTEXT_SCORE_MARGIN
    ][:FINAL_CONTEXT_COUNT]
    selected_ids = {id(source) for source in selected_sources}
    for source in ranked_sources:
        source["used_in_answer"] = id(source) in selected_ids

    progress.progress(75, text="Подбираю фрагменты под контекст модели…")
    system_prompt = (
        "Ответь именно на вопрос пользователя, кратко и по-русски. Используй только сведения, "
        "которые прямо подтверждаются релевантными фрагментами ниже. Не добавляй внешние знания, "
        "догадки, вероятные объяснения и детали, не нужные для ответа. Полностью игнорируй "
        "посторонние фрагменты. Не связывай документ, событие или человека с вопросом, если источник "
        "не устанавливает эту связь напрямую. Подкрепляй каждое фактическое утверждение ссылкой "
        "на подтверждающий фрагмент вида [S1]. Не ставь ссылку на источник, который этого не подтверждает. "
        "Если релевантные фрагменты не дают ответа или противоречат друг другу, прямо скажи: "
        "«В найденных сообщениях недостаточно данных, чтобы ответить уверенно». Не заполняй ответ "
        "пересказом посторонних результатов."
    )
    prompt_token_budget = (
        QNA_MAX_CONTEXT_TOKENS - ANSWER_RESERVE_TOKENS - TOKENIZER_SAFETY_MARGIN
    )
    sources: list[dict[str, Any]] = []
    for candidate in selected_sources:
        source = {"id": f"S{len(sources) + 1}", **candidate}
        trial_sources = [*sources, source]
        context = "\n\n".join(
            f"[{item['id']}] Дата: {item['date']} | Чат: {item['chat']} | Автор: {item['sender']}\n{item['text']}"
            for item in trial_sources
        )
        trial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\n\nИсточники:\n{context}"},
        ]
        token_count = len(
            qna_tokenizer.apply_chat_template(
                trial_messages, tokenize=True, add_generation_prompt=True
            )
        )
        if token_count > prompt_token_budget:
            # A lower-ranked passage can be shorter and still fit.
            continue
        sources.append(source)
    if not sources:
        raise RuntimeError(
            "Не удалось поместить ни одного найденного фрагмента в контекст модели."
        )

    context = "\n\n".join(
        f"[{source['id']}] Дата: {source['date']} | Чат: {source['chat']} | Автор: {source['sender']}\n{source['text']}"
        for source in sources
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Вопрос: {question}\n\nИсточники:\n{context}"},
    ]
    prompt_tokens = len(
        qna_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    )
    request_context_tokens = min(
        QNA_MAX_CONTEXT_TOKENS,
        prompt_tokens + ANSWER_RESERVE_TOKENS + TOKENIZER_SAFETY_MARGIN,
    )
    answer = ollama_chat(
        messages,
        num_ctx=request_context_tokens,
        num_predict=ANSWER_RESERVE_TOKENS,
    )
    if not answer:
        raise RuntimeError("Модель вернула пустой ответ.")
    progress.progress(100, text="Готово")
    st.session_state["context_usage"] = {
        "prompt": prompt_tokens,
        "window": request_context_tokens,
        "reserve": request_context_tokens - prompt_tokens,
    }
    return answer, sources, ranked_sources


def submit_question() -> None:
    question = st.session_state.get("question_input", "").strip()
    if question:
        st.session_state["submitted_question"] = question


def resolve_attachment_path(raw_path: str, source_input_file: str = "") -> Path | None:
    """Resolve old absolute chat paths against their current export directory."""
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path.resolve()

    normalized = path.as_posix()
    marker = "/chats/"
    if marker not in normalized:
        return None
    relative_path = normalized.split(marker, 1)[1]

    export_root = Path(__file__).resolve().parents[1] / "data"
    preferred_export = None
    match = re.search(
        r"^chatexport_(.+?)(?:_([12]))?_messages_from_\d{4}-\d{2}\.csv$",
        source_input_file,
    )
    if match:
        suffix = f" ({match.group(2)})" if match.group(2) else ""
        preferred_export = f"ChatExport_{match.group(1)}{suffix}"

    export_dirs = sorted(export_root.glob("ChatExport*"))
    if preferred_export:
        export_dirs.sort(key=lambda directory: directory.name != preferred_export)
    for export_dir in export_dirs:
        candidate = export_dir / "chats" / relative_path
        if candidate.is_file():
            return candidate.resolve()
    return None


st.text_input(
    label="Вопрос",
    label_visibility="collapsed",
    placeholder="",
    key="question_input",
    on_change=submit_question,
)

question = st.session_state.get("submitted_question")
if question and question != st.session_state.get("answered_question"):
    progress = st.progress(0, text="Подключаю поиск…")
    try:
        answer, sources, ranked_sources = get_answer(question, progress)
        st.session_state["answer"] = answer
        st.session_state["sources"] = sources
        st.session_state["ranked_sources"] = ranked_sources
        st.session_state["answered_question"] = question
    except Exception as exc:
        st.session_state.pop("answer", None)
        st.session_state.pop("sources", None)
        st.session_state.pop("ranked_sources", None)
        st.error(str(exc))
    finally:
        progress.empty()

if st.session_state.get("answer"):
    st.markdown(st.session_state["answer"])
    usage = st.session_state.get("context_usage")
    if usage:
        st.caption(
            f"Контекст модели: {usage['prompt']:,} / {usage['window']:,} токенов; "
            f"свободно {usage['reserve']:,} токенов".replace(",", " ")
        )
    cited = set(re.findall(r"\[(S\d+)\]", st.session_state["answer"]))
    source_map = {
        source["id"]: source for source in st.session_state.get("sources", [])
    }
    citations = [
        source_map[key]
        for key in sorted(cited, key=lambda x: int(x[1:]))
        if key in source_map
    ]
    links: list[str] = []
    for source in citations:
        for path in source["file_paths"]:
            if path:
                resolved = resolve_attachment_path(path, source["source_input_file"])
                if resolved:
                    file_url = "file://" + quote(str(resolved))
                    links.append(f"[{source['id']}: {resolved.name}]({file_url})")
    if links:
        st.caption(" · ".join(links))

with st.expander("Результаты поиска и источники для ответа"):
    st.caption(
        "Первые 50 фрагментов после реранжирования. Отмечено, какие именно попали в контекст ответа."
    )
    for rank, source in enumerate(st.session_state.get("ranked_sources", []), 1):
        status = "В контексте ответа" if source["used_in_answer"] else "Не передан модели"
        st.caption(
            f"{rank}. {status} · rerank {source['rerank_score']:.3f} · "
            f"{source['date']} · {source['chat']} · {source['sender']}"
        )
        st.text(source["text"])
        paths = source.get("file_paths", [])
        if paths:
            for path in paths:
                resolved = resolve_attachment_path(path, source["source_input_file"])
                if resolved:
                    file_url = "file://" + quote(str(resolved))
                    st.markdown(f"[Открыть {resolved.name}]({file_url})")
                else:
                    st.caption(f"Файл не найден: {Path(path).name}")
