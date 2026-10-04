"""Build a quote-grounded concept graph using a local llama.cpp server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from videorag.config import COLLECTION_NAME
from videorag.database.graph_store import create_graph_store
from videorag.database.vector_store import VectorStore

LLM_BASE_URL = os.getenv("GRAPH_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.getenv("GRAPH_LLM_MODEL", "")
BATCH_SIZE = 5

SYSTEM_PROMPT = """أنت مستخرج حقائق محافظ من تفريغ فيديو عربي قد يحتوي أخطاء تعرّف صوتي.
استخرج فقط علاقة صريحة يقولها النص نفسه، ولا تستنتجها من مجرد ظهور مفهومين معًا.
أعد لكل حقيقة: subject, predicate, object, chunk_id, evidence.
يجب أن يكون evidence أقصر اقتباس حرفي متصل يحتوي العلاقة كاملة من transcript للمقطع.
لزيادة الدقة، أخرج العلاقة فقط إذا ظهرت في الاقتباس بإحدى الصيغتين: subject ثم predicate ثم object، أو predicate ثم subject ثم object (صيغة الفعل أولًا).
انسخ صيغة الفعل كما وردت حرفيًا، ولا تعكس subject وobject ولا تصحح أو تكمل كلام التفريغ.
يجب أن تظهر العبارات الثلاث نفسها داخل evidence وبالترتيب الموافق لإحدى الصيغتين؛ إذا لم ينطبق ذلك فتجاهل العلاقة.
إذا ظهر في الاقتباس نفسه ترتيب يسمح بالعلاقة وعكسها، فاعتبره ملتبسًا وتجاهله.
لا تخرج أسماء أو مفاهيم لمجرد أنها وردت في النص من دون علاقة صريحة.
أعد JSON فقط بالشكل: {"facts": [{"subject":"...","predicate":"...","object":"...","chunk_id":"...","evidence":"..."}]}.
"""


def _norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def _ambiguous_direction_tokens(value: str) -> list[str]:
    """Normalize light Arabic spelling/ASR variants for contradiction checks only."""
    tokens = _norm_text(value).split()
    normalized = []
    for token in tokens:
        # ASR may alternate between forms such as "الذاتي" and "ذاتيا".
        token = token.removeprefix("ال")
        if len(token) > 3 and token.endswith("ا"):
            token = token[:-1]
        normalized.append(token)
    return normalized


def _get_model_id() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    request = Request(f"{LLM_BASE_URL}/models", method="GET")
    with urlopen(request, timeout=10) as response:
        models = json.loads(response.read().decode("utf-8")).get("data", [])
    if not models:
        raise RuntimeError("خادم llama.cpp لم يُرجع أي نموذج من /v1/models")
    return str(models[0]["id"])


def _summarize_community(model_id: str, report: str) -> str:
    """Create a concise topic summary; final answers still cite raw evidence."""
    request_body = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "لخّص مجتمع المفاهيم التالي بالعربية في 2 إلى 4 جمل. "
                    "استند حصريًا إلى العلاقات والأدلة المعروضة، ولا تضف معرفة خارجية. "
                    "هذه خلاصة للاسترجاع وليست دليلًا نهائيًا. أعد JSON فقط: "
                    '{"summary":"..."}'
                ),
            },
            {"role": "user", "content": "/no_think\n" + report[:14000]},
        ],
        "temperature": 0,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"].get("content", "")
        summary = str(json.loads(content).get("summary", "")).strip()
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError,
            TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"تعذر تلخيص تقرير المجتمع: {exc}") from exc
    if not summary:
        raise RuntimeError("النموذج أعاد خلاصة فارغة لأحد المجتمعات")
    return summary


def _summarize_communities(graph, model_id: str) -> tuple[int, int]:
    reports = graph.community_reports()
    created = reused = 0
    for index, item in enumerate(reports, start=1):
        cached = graph.cached_community_summary(
            item["community_id"], item["content_hash"], model_id
        )
        if cached:
            reused += 1
            continue
        summary = _summarize_community(model_id, item["report"])
        graph.save_community_summary(
            item["community_id"], item["content_hash"], model_id, summary
        )
        created += 1
        print(
            f"Community summaries: {index}/{len(reports)} "
            f"({created} generated, {reused} reused)",
            flush=True,
        )
    return created, reused


def _request_candidates(model_id: str, chunks: list[dict]) -> list[dict]:
    request_body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "/no_think\nاستخرج العلاقات من المقاطع التالية، وحافظ على chunk_id والتوقيت ضمن فهمك.\n"
                + json.dumps(chunks, ensure_ascii=False),
            },
        ],
        "temperature": 0,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"خادم النموذج المحلي أعاد HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"تعذر الاتصال بخادم النموذج المحلي على {LLM_BASE_URL}. "
            "شغّل llama.cpp أولًا واترك نافذته تعمل."
        ) from exc

    content = payload["choices"][0]["message"]["content"]
    try:
        facts = json.loads(content).get("facts", [])
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"النموذج لم يُرجع JSON صالحًا: {content[:500]}") from exc
    if not isinstance(facts, list):
        raise RuntimeError("حقل facts في استجابة النموذج ليس قائمة")
    return [fact for fact in facts if isinstance(fact, dict)]


def _validate_candidates(facts: list[dict], chunks: list[dict]) -> list[dict]:
    """Validate cached or fresh candidates against their exact source chunks."""
    source_by_id = {str(chunk["chunk_id"]): str(chunk["transcript"]) for chunk in chunks}
    validated = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        chunk_id = str(fact.get("chunk_id") or "")
        source = source_by_id.get(chunk_id)
        subject = str(fact.get("subject") or "").strip()
        predicate = str(fact.get("predicate") or "").strip()
        object_ = str(fact.get("object") or "").strip()
        evidence = str(fact.get("evidence") or "").strip()
        if not source or not all((subject, predicate, object_, evidence)):
            continue
        # Reject model-invented evidence; allow only normalization differences
        # such as Arabic diacritics, alef variants, and punctuation.
        normalized_evidence = _norm_text(evidence)
        if not normalized_evidence or normalized_evidence not in _norm_text(source):
            continue

        # The quote alone is not enough: reject reversed triples and predicates
        # invented by the model unless the literal subject → predicate → object
        # sequence is present in that same quote.
        components = tuple(_norm_text(part) for part in (subject, predicate, object_))
        if not all(components):
            continue

        def appears_in_order(sequence: tuple[str, str, str]) -> bool:
            cursor = 0
            for component in sequence:
                position = normalized_evidence.find(component, cursor)
                if position < 0:
                    return False
                cursor = position + len(component)
            return True

        def relation_order_is_present(subject: str, predicate: str, object_: str) -> bool:
            # Accept subject-verb-object and verb-subject-object forms.
            return appears_in_order((subject, predicate, object_)) or appears_in_order(
                (predicate, subject, object_)
            )

        def ambiguous_order_is_present(subject: str, predicate: str, object_: str) -> bool:
            # Be more tolerant only when looking for the opposite direction:
            # Arabic ASR can vary a word's article or final alef (e.g. ذاتي/ذاتيا).
            expected = [
                _ambiguous_direction_tokens(part)
                for part in (subject, predicate, object_)
            ]
            source_tokens = _ambiguous_direction_tokens(evidence)
            if not all(expected):
                return False
            cursor = 0
            for component in expected:
                found = False
                while cursor < len(source_tokens):
                    if source_tokens[cursor : cursor + len(component)] == component:
                        cursor += len(component)
                        found = True
                        break
                    cursor += 1
                if not found:
                    return False
            return True

        # A transcript can concatenate adjacent/overlapping phrases without
        # punctuation. Reject a triple when the quote also states the reverse
        # clearly in subject-predicate-object order. Do not interpret a later
        # verb-first token sequence as a second relation: repeated entities in
        # ASR text can make one S-P-O clause look like a reversed V-S-O clause.
        if not relation_order_is_present(*components):
            continue
        if ambiguous_order_is_present(components[2], components[1], components[0]):
            continue
        validated.append({
            "chunk_id": chunk_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "evidence": evidence,
        })
    return validated


def _index_video(
    graph: GraphStore,
    store: VectorStore,
    model_id: str,
    video_id: str,
    batch_size: int,
) -> tuple[int, int]:
    payloads = store.list_transcripts(video_id=video_id)
    chunks = [
        {
            "chunk_id": str(payload["chunk_id"]),
            "start_timestamp": float(payload["start_timestamp"]),
            "end_timestamp": float(payload["end_timestamp"]),
            "transcript": str(payload.get("transcript") or "").strip(),
        }
        for payload in payloads
        if payload.get("chunk_id") and str(payload.get("transcript") or "").strip()
    ]
    if not chunks:
        print(f"Skipping {video_id}: no transcript chunks found.")
        return 0, 0

    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    transcript_hashes = {
        chunk["chunk_id"]: hashlib.sha256(
            chunk["transcript"].encode("utf-8")
        ).hexdigest()
        for chunk in chunks
    }
    candidates_by_chunk: dict[str, list[dict]] = {}
    pending_chunks = []
    for chunk in chunks:
        cached = graph.get_cached_candidates(
            video_id, chunk["chunk_id"], transcript_hashes[chunk["chunk_id"]],
            model_id, prompt_hash,
        )
        if cached is None:
            cached = graph.existing_facts_as_candidates(
                video_id, chunk["chunk_id"], chunk["transcript"]
            )
            if cached is not None:
                graph.cache_candidates(
                    video_id, chunk["chunk_id"], transcript_hashes[chunk["chunk_id"]],
                    model_id, prompt_hash, cached,
                )
        if cached is None:
            pending_chunks.append(chunk)
        else:
            candidates_by_chunk[chunk["chunk_id"]] = cached

    batches = [
        pending_chunks[i:i + batch_size]
        for i in range(0, len(pending_chunks), batch_size)
    ]
    started = time.perf_counter()
    for batch_number, batch in enumerate(batches, start=1):
        candidates = _request_candidates(model_id, batch)
        candidate_batch: dict[str, list[dict]] = {
            chunk["chunk_id"]: [] for chunk in batch
        }
        for candidate in candidates:
            chunk_id = str(candidate.get("chunk_id") or "")
            if chunk_id in candidate_batch:
                candidate_batch[chunk_id].append(candidate)
        for chunk_id, chunk_candidates in candidate_batch.items():
            candidates_by_chunk[chunk_id] = chunk_candidates
            graph.cache_candidates(
                video_id, chunk_id, transcript_hashes[chunk_id], model_id,
                prompt_hash, chunk_candidates,
            )
        validated_count = len(_validate_candidates(candidates, batch))
        print(
            f"{video_id}: processed batch {batch_number}/{len(batches)} "
            f"({validated_count} validated relations; raw candidates cached)",
            flush=True,
        )

    facts_by_chunk: dict[str, list[dict]] = {}
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for chunk_id, candidates in candidates_by_chunk.items():
        facts_by_chunk[chunk_id] = _validate_candidates(
            candidates, [chunks_by_id[chunk_id]]
        )

    # Replace this video's facts only after all required model calls succeeded.
    graph_chunks = [
        {**chunk, "facts": facts_by_chunk[chunk["chunk_id"]]}
        for chunk in chunks
    ]
    graph.replace_video_facts(video_id, graph_chunks)
    print(f"Graph index updated for {video_id}: {graph.fact_summary(video_id)}")
    print(
        f"Inference cache: reused {len(chunks) - len(pending_chunks)}/{len(chunks)} "
        f"chunks; inferred {len(pending_chunks)}"
    )
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")
    return len(chunks), len(pending_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evidence-grounded graph from indexed video transcripts."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--video-id", help="Index one Qdrant video_id")
    target.add_argument("--all", action="store_true", help="Index every video in Qdrant")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, choices=range(1, 11))
    args = parser.parse_args()

    try:
        import networkx  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "NetworkX غير مثبت. نفّذ: pip install -r requirements.txt"
        ) from exc

    store = VectorStore(collection_name=COLLECTION_NAME)
    graph = create_graph_store()
    try:
        try:
            model_id = _get_model_id()
        except (URLError, TimeoutError, HTTPError) as exc:
            raise SystemExit(
                f"لا يوجد خادم نموذج محلي عند {LLM_BASE_URL}. "
                "شغّل llama.cpp أولًا، ثم أعد الأمر."
            ) from exc

        video_ids = store.list_video_ids() if args.all else [args.video_id]
        if not video_ids:
            raise SystemExit("No indexed videos found in Qdrant.")
        total_chunks = total_inferred = 0
        for video_id in video_ids:
            chunks_count, inferred_count = _index_video(
                graph, store, model_id, str(video_id), args.batch_size
            )
            total_chunks += chunks_count
            total_inferred += inferred_count

        if args.all:
            stale_videos = graph.remove_videos_except(set(video_ids))
            if stale_videos:
                print(f"Removed stale graph videos: {', '.join(stale_videos)}")

        normalized = graph.rebuild_knowledge_graph()
        communities = graph.rebuild_communities()
        summaries_created, summaries_reused = _summarize_communities(graph, model_id)
        print(f"Corpus graph rebuilt: {normalized}")
        print(f"Hierarchical communities rebuilt: {communities}")
        print(
            f"Community summaries: generated {summaries_created}, "
            f"reused {summaries_reused}"
        )
        print(f"Videos processed: {len(video_ids)}; chunks: {total_chunks}")
        print(f"Chunks requiring fresh model inference: {total_inferred}")
        print(f"Model: {model_id}")
        print(f"SQLite sidecar: {graph.path}")
        print("Relations retain source transcript evidence and video timestamps.")
    finally:
        graph.close()
        store.close()


if __name__ == "__main__":
    main()
