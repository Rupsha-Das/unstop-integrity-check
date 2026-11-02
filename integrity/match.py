"""Embed, FAISS index, peer + external matching, boilerplate (SPEC §5.3-5.4)."""
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from integrity.config import (
    BOILERPLATE_MIN_DOCS,
    BOILERPLATE_SIM,
    EMBED_MODEL,
    MATCH_THRESHOLD,
    SEARCH_K,
)

_model = None


def embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    vecs = _model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def build_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    return index


def find_matches(query_chunks, query_vecs, target_chunks, target_index,
                 exclude_same_doc=True, boilerplate=True):
    results = {}
    k = min(SEARCH_K, len(target_chunks))
    if k == 0:
        for c in query_chunks:
            results[c["chunk_id"]] = {"is_boilerplate": False, "shared_with_n_files": 0,
                                      "matches": []}
        return results
    scores, ids = target_index.search(np.ascontiguousarray(query_vecs), k)
    for i, chunk in enumerate(query_chunks):
        neighbours = [(target_chunks[j], float(s))
                      for j, s in zip(ids[i], scores[i]) if j != -1]
        if exclude_same_doc:
            neighbours = [(t, s) for t, s in neighbours if t["doc_id"] != chunk["doc_id"]]
        boiler_docs = {t["doc_id"] for t, s in neighbours if s >= BOILERPLATE_SIM}
        is_boiler = boilerplate and len(boiler_docs) >= BOILERPLATE_MIN_DOCS
        best = {}
        for t, s in neighbours:
            if s >= MATCH_THRESHOLD and s > best.get(t["doc_id"], (-1,))[0]:
                best[t["doc_id"]] = (t, s)
        results[chunk["chunk_id"]] = {
            "is_boilerplate": is_boiler,
            "shared_with_n_files": len(boiler_docs),
            "matches": sorted(best.values(), key=lambda m: -m[1]),
        }
    return results


JACCARD_CLEAN = re.compile(r"[^\w\s]")


def jaccard(a, b):
    wa = {w for w in JACCARD_CLEAN.sub("", a.lower()).split()}
    wb = {w for w in JACCARD_CLEAN.sub("", b.lower()).split()}
    if not wa and not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def match_type(a, b):
    return "near-exact copy" if jaccard(a, b) >= 0.6 else "paraphrase / semantic match"


def percentage(doc_id, chunks, results):
    own = [c for c in chunks if c["doc_id"] == doc_id]
    scorable = [c for c in own if not results[c["chunk_id"]]["is_boilerplate"]]
    if not scorable:
        return 0.0
    flagged = sum(1 for c in scorable if results[c["chunk_id"]]["matches"])
    return round(100 * flagged / len(scorable), 1)


def similarity_matrix(chunks, vecs):
    """Best cosine between any two chunks of any two distinct docs."""
    n = len(chunks)
    docs = sorted({c["doc_id"] for c in chunks})
    sims = np.ascontiguousarray(vecs) @ np.ascontiguousarray(vecs).T
    best = {}
    for i in range(n):
        for j in range(n):
            a, b = chunks[i]["doc_id"], chunks[j]["doc_id"]
            if a == b:
                continue
            if float(sims[i][j]) > best.get((a, b), -1.0):
                best[(a, b)] = float(sims[i][j])
    return docs, best


def ref_chunks_from_corpus(folder):
    """reference_corpus .txt files -> chunks; units split on blank lines."""
    from integrity.extract import chunk_doc
    chunks = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        units = [{"text": " ".join(p.split())}
                 for p in re.split(r"\n\s*\n", text) if p.strip()]
        doc = {"doc_id": path.name, "modality": "text", "status": "ok",
               "reason": None, "units": units}
        chunks.extend(chunk_doc(doc))
    return chunks


if __name__ == "__main__":
    from integrity.extract import chunk_docs, extract_folder
    docs = extract_folder("sample_submissions")
    chunks = chunk_docs(docs)
    vecs = embed([c["text"] for c in chunks])
    index = build_index(vecs)
    peer = find_matches(chunks, vecs, chunks, index)
    print("=== doc x doc best cosine (upper triangle) ===")
    _, matrix = similarity_matrix(chunks, vecs)
    names = sorted({c["doc_id"] for c in chunks})
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = matrix.get((a, b), 0.0)
            flag = " <---" if s >= MATCH_THRESHOLD else ""
            print(f"{s:.2f}  {a} <-> {b}{flag}")
    print("=== boilerplate chunks per doc ===")
    from collections import Counter
    boiler = Counter()
    for c in chunks:
        if peer[c["chunk_id"]]["is_boilerplate"]:
            boiler[c["doc_id"]] += 1
    for name in names:
        print(f"{boiler.get(name, 0):2d}  {name}")
    print("=== external (reference corpus) ===")
    ref = ref_chunks_from_corpus("reference_corpus")
    ref_vecs = embed([c["text"] for c in ref])
    ref_index = build_index(ref_vecs)
    ext = find_matches(chunks, vecs, ref, ref_index,
                       exclude_same_doc=False, boilerplate=False)
    for c in chunks:
        for t, s in ext[c["chunk_id"]]["matches"]:
            print(f"{s:.2f}  {c['doc_id']} -> {t['doc_id']}  {c['text'][:50]!r}")
