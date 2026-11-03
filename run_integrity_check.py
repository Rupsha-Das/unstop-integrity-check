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
    ap = argparse.ArgumentParser(
        prog="integrity-check",
        description="Multimodal plagiarism check for PDF/DOCX/MP4 submissions. "
                    "Run it on a folder (default ./sample_submissions) or a single "
                    "file. Without arguments it checks ./sample_submissions into "
                    "./reports and evaluates the results.")
    ap.add_argument("target", nargs="?", default=None,
                    help="single PDF/DOCX/MP4 file or a folder of them "
                         "(default ./sample_submissions). A file is checked against "
                         "the other files in its folder + the reference corpus.")
    ap.add_argument("--input_dir", default=None,
                    help="(deprecated) folder of submissions; same as passing a folder as target")
    ap.add_argument("--output_dir", default="./reports",
                    help="folder for reports (default ./reports)")
    ap.add_argument("--reference_dir", default="./reference_corpus",
                    help="folder of .txt reference 'web' files (default ./reference_corpus)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the evaluation step after reporting")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="show progress/debug logs (default: quiet)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # resolve target: none -> default folder; dir -> that folder; file -> query doc
    if args.input_dir:
        args.target = args.input_dir  # old flag still works
    if args.target and not Path(args.target).exists():
        print(f"error: '{args.target}' does not exist", flush=True)
        return 2
    single = None
    if args.target and Path(args.target).is_file():
        single = Path(args.target).name
        input_dir = str(Path(args.target).parent) or "."
    else:
        input_dir = str(Path(args.target)) if args.target else "./sample_submissions"
    print(f"Checking: {input_dir}" + (f" (single file: {single})" if single else ""),
          flush=True)

    docs = extract_folder(input_dir)
    if single and not any(d["doc_id"] == single for d in docs):
        print(f"error: could not process '{single}' (unsupported type or unreadable)",
              flush=True)
        return 2
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
        if single and doc["doc_id"] != single:
            continue  # single-file mode: siblings only serve as matching context
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
    print(f"\n-> report files written to {out} "
          f"(summary: {Path(args.output_dir) / 'summary.md'})")

    if single:
        md = out / f"{reports[0]['document_id']}.md"
        print("\n" + "=" * 70 + "\n")
        print(md.read_text(encoding="utf-8"))
        return 0

    if not args.no_eval:
        try:
            from evaluate import main as evaluate_main
            evaluate_main(args.output_dir)
        except Exception as e:
            logging.warning("evaluation skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
