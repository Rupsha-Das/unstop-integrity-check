"""CLI entry: extract -> chunk -> embed -> match -> ai -> report (SPEC §5.7)."""
import argparse
import logging
from pathlib import Path

from integrity.ai_detect import ai_score, doc_text
from integrity.extract import chunk_docs, extract_folder
from integrity.match import (
    build_index,
    embed,
    find_matches,
    ref_chunks_from_corpus,
    similarity_matrix,
)
from integrity.report import build_report, write_markdown, write_report_json, write_summary


def main():
    ap = argparse.ArgumentParser(description="Multimodal plagiarism check")
    ap.add_argument("--input_dir", default="./sample_submissions")
    ap.add_argument("--output_dir", default="./reports")
    ap.add_argument("--reference_dir", default="./reference_corpus")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = extract_folder(args.input_dir)
    chunks = chunk_docs(docs)
    peer = {c["chunk_id"]: {"is_boilerplate": False, "shared_with_n_files": 0,
                            "matches": []} for c in chunks}
    ext = dict(peer)
    matrix = {"docs": sorted({d["doc_id"] for d in docs if d["status"] == "ok"}), "best": {}}
    if chunks:
        vecs = embed([c["text"] for c in chunks])
        idx = build_index(vecs)
        peer = find_matches(chunks, vecs, chunks, idx, exclude_same_doc=True)
        matrix = {"docs": matrix["docs"], "best": similarity_matrix(chunks, vecs)[1]}
        ref_chunks = ref_chunks_from_corpus(args.reference_dir)
        if ref_chunks:
            ref_vecs = embed([c["text"] for c in ref_chunks])
            ref_idx = build_index(ref_vecs)
            ext = find_matches(chunks, vecs, ref_chunks, ref_idx,
                               exclude_same_doc=False, boilerplate=False)

    reports = []
    for doc in docs:
        ai = (ai_score(doc_text(doc)) if doc["status"] == "ok"
              else {"probability": None, "perplexity": None, "burstiness": None,
                    "note": "skipped"})
        report = build_report(doc, chunks, peer, ext, ai)
        write_report_json(report, out)
        write_markdown(report, out)
        reports.append(report)
        logging.info("%s: %s", doc["doc_id"], report["total_integrity_risk"])

    write_summary(reports, matrix, out)
    s = ["file | status | risk | peer | external | ai | top match"]
    for r in reports:
        sc = r["scores"]
        m = r["peer_matches"] + r["multimodal_matches"] + r["external_matches"]
        top = "-" if not m else f"{m[0]['matched_file']} ({m[0]['similarity_score']})"
        s.append(f"{r['document_id']} | {r['status']} | {r['total_integrity_risk']} | "
                 f"{sc['peer_similarity_percentage']} | {sc['external_similarity_percentage']}"
                 f" | {sc['ai_generated_probability']} | {top}")
    print("\n".join(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
