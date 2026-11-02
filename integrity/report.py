"""JSON + Markdown report writers and summary (SPEC §5.6)."""
import json
from pathlib import Path

from integrity.config import RISK_HIGH, RISK_MEDIUM
from integrity.match import jaccard, match_type


RISK_KEY = {"peer_similarity_percentage": "peer",
            "external_similarity_percentage": "external",
            "ai_generated_probability": "ai"}


def risk(scores):
    if any(v is not None and v >= RISK_HIGH[RISK_KEY[k]] for k, v in scores.items()):
        return "HIGH"
    if any(v is not None and v >= RISK_MEDIUM[RISK_KEY[k]] for k, v in scores.items()):
        return "MEDIUM"
    return "LOW"


def _ts(seconds):
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _loc(chunk, prefix):
    if chunk["modality"] == "video":
        return {f"{prefix}_timestamp": f"{_ts(chunk['start'])} - {_ts(chunk['end'])}"}
    if chunk.get("page") is not None:
        return {f"{prefix}_page": chunk["page"]}
    if chunk.get("paragraph") is not None:
        return {f"{prefix}_paragraph": chunk["paragraph"]}
    return {}


def build_report(doc, chunks, peer, ext, ai):
    doc_id = doc["doc_id"]
    if doc["status"] != "ok":
        return {
            "document_id": doc_id, "status": doc["status"], "skip_reason": doc["reason"],
            "total_integrity_risk": "UNPROCESSED",
            "scores": {"peer_similarity_percentage": None,
                       "external_similarity_percentage": None,
                       "ai_generated_probability": None},
            "peer_matches": [], "multimodal_matches": [], "external_matches": [],
            "excluded_boilerplate": [],
            "ai_details": {"perplexity": None, "burstiness": None, "note": "skipped"},
            "explanations": f"File could not be processed: {doc['reason']}.",
        }
    own = [c for c in chunks if c["doc_id"] == doc_id]

    peer_agg, mm_agg = {}, {}
    peer_count = {}
    for c in own:
        if peer[c["chunk_id"]]["is_boilerplate"]:
            continue
        for t, s in peer[c["chunk_id"]]["matches"]:
            bucket = peer_agg if t["modality"] == c["modality"] else mm_agg
            if t["doc_id"] not in bucket or s > bucket[t["doc_id"]][0]:
                bucket[t["doc_id"]] = (s, c, t)
            peer_count.setdefault(t["doc_id"], set()).add(c["chunk_id"])

    ext_agg, ext_count = {}, {}
    for c in own:
        if peer[c["chunk_id"]]["is_boilerplate"]:
            continue
        for t, s in ext[c["chunk_id"]]["matches"]:
            if t["doc_id"] not in ext_agg or s > ext_agg[t["doc_id"]][0]:
                ext_agg[t["doc_id"]] = (s, c, t)
            ext_count.setdefault(t["doc_id"], set()).add(c["chunk_id"])

    def mk(agg, counts):
        out = []
        for tdoc, (s, q, t) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            out.append(_entry(q, t, s, len(counts[tdoc])))
        return out

    boiler = [{"passage": c["text"][:120],
               "shared_with_n_files": peer[c["chunk_id"]]["shared_with_n_files"]}
              for c in own if peer[c["chunk_id"]]["is_boilerplate"]]

    if own:
        scorable = [c for c in own if not peer[c["chunk_id"]]["is_boilerplate"]]
        if scorable:
            flagged = sum(1 for c in scorable if peer[c["chunk_id"]]["matches"])
            peer_pct = round(100 * flagged / len(scorable), 1)
            eflag = sum(1 for c in scorable if ext[c["chunk_id"]]["matches"])
            ext_pct = round(100 * eflag / len(scorable), 1)
        else:
            flagged, peer_pct, ext_pct = 0, 0.0, 0.0
    else:
        flagged, peer_pct, ext_pct = 0, 0.0, 0.0

    scores = {"peer_similarity_percentage": peer_pct,
              "external_similarity_percentage": ext_pct,
              "ai_generated_probability": ai["probability"]}
    rep = {
        "document_id": doc_id, "status": "ok", "skip_reason": None,
        "total_integrity_risk": risk(scores),
        "scores": scores,
        "peer_matches": mk(peer_agg, peer_count),
        "multimodal_matches": mk(mm_agg, peer_count),
        "external_matches": mk(ext_agg, ext_count),
        "excluded_boilerplate": boiler,
        "ai_details": {"perplexity": ai["perplexity"], "burstiness": ai["burstiness"],
                       "note": ai["note"]},
    }
    rep["explanations"] = _explain(rep, len(own), flagged)
    return rep


def _entry(query, target, score, n_passages):
    return {
        "matched_file": target["doc_id"],
        "similarity_score": round(score, 2),
        "jaccard": round(jaccard(query["text"], target["text"]), 2),
        "match_type": match_type(query["text"], target["text"]),
        "query_passage": query["text"],
        "matched_passage": target["text"],
        "location": {**_loc(query, "query"), **_loc(target, "matched")},
        "passage_count": n_passages,
    }


def _explain(rep, total, flagged):
    pct = round(100 * flagged / total, 1) if total else 0.0
    all_mm = rep["peer_matches"] + rep["multimodal_matches"]
    if all_mm:
        src = "; ".join(
            f"{m['matched_file']} ({m['passage_count']} passage"
            f"{'s' if m['passage_count'] != 1 else ''}, {m['match_type']})"
            for m in all_mm[:3])
        lead = (f"{flagged} of {total} passages ({pct}%) closely match other submissions: "
                f"{src}.")
        if rep["multimodal_matches"]:
            mm = "; ".join(f"{m['matched_file']} (video)" for m in rep["multimodal_matches"])
            lead += f" The match against {mm} crosses modality (speech vs writing)."
    else:
        lead = f"No other submission matches ({total} passages checked)."
    mid = ""
    if rep["excluded_boilerplate"]:
        n = max(b["shared_with_n_files"] for b in rep["excluded_boilerplate"])
        c = len(rep["excluded_boilerplate"])
        mid = (f" {c} passage{'s' if c != 1 else ''} shared with up to {n} other files "
               f"w{'as' if c == 1 else 'ere'} treated as common template text and ignored.")
    if rep["external_matches"]:
        src = ", ".join(f"{m['matched_file']} ({m['similarity_score']})"
                        for m in rep["external_matches"])
        ext = f" An external source matches: {src}."
    else:
        ext = " No external source matches."
    if rep["scores"]["ai_generated_probability"] is None:
        tail = " AI likelihood not computed from this file type."
    else:
        d = rep["ai_details"]
        tail = (f" AI likelihood {round(rep['scores']['ai_generated_probability'])}% "
                f"(perplexity {d['perplexity']}, burstiness {d['burstiness']}).")
    return (lead + mid + ext + tail).strip()


def _esc(t):
    return t.replace("|", "\\|").replace("\n", " ")


def _loc_label(loc, prefix, fallback):
    for k in (f"{prefix}_page", f"{prefix}_paragraph", f"{prefix}_timestamp"):
        if k in loc:
            v = loc[k]
            if k.endswith("_page"):
                return f"page {v}"
            if k.endswith("_paragraph"):
                return f"paragraph {v}"
            return str(v)
    return fallback


def write_markdown(report, out_dir):
    d = Path(out_dir)
    name = report["document_id"]
    if report["status"] != "ok":
        text = f"# {name}: UNPROCESSED\n\nStatus: UNPROCESSED: {report['skip_reason']}\n"
        (d / f"{name}.md").write_text(text, encoding="utf-8")
        return
    s = report["scores"]
    lines = [
        f"# {name}: Risk {report['total_integrity_risk']}",
        "",
        "| Peer % | External % | AI % |",
        "|---|---|---|",
        '| ' + f"{s['peer_similarity_percentage']} | {s['external_similarity_percentage']} | "
        f"{'' if s['ai_generated_probability'] is None else s['ai_generated_probability']}" + ' |',
        "",
        f"**Summary:** {report['explanations']}",
        "",
    ]
    sections = [("## Matches", report["peer_matches"]),
                ("## Cross-modal matches", report["multimodal_matches"]),
                ("## External matches", report["external_matches"])]
    n = 0
    for title, matches in sections:
        if not matches:
            continue
        lines += [title, ""]
        for m in matches:
            n += 1
            lines.append(f"### {n}. {m['matched_file']}: {m['similarity_score']} "
                         f"({m['match_type']}, {m['passage_count']} passage"
                         f"{'s' if m['passage_count'] != 1 else ''})")
            lines.append("")
            l = m["location"]
            lines.append(f"| This file ({_loc_label(l, 'query', 'passage')}) "
                         f"| {m['matched_file']} ({_loc_label(l, 'matched', 'passage')}) |")
            lines.append("|---|---|")
            lines.append(f"| {_esc(m['query_passage'])} | {_esc(m['matched_passage'])} |")
            lines.append("")
    if report["excluded_boilerplate"]:
        lines += ["## Ignored as common template", ""]
        for b in report["excluded_boilerplate"]:
            lines.append(f"- \"{b['passage']}\" (in {b['shared_with_n_files']} other files)")
        lines.append("")
    (d / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def write_report_json(report, out_dir):
    (Path(out_dir) / f"{report['document_id']}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


def write_summary(all_reports, matrix, out_dir):
    d = Path(out_dir)
    names, best = matrix["docs"], matrix["best"]

    def top(r):
        m = r["peer_matches"] + r["multimodal_matches"] + r["external_matches"]
        return "-" if not m else f"{m[0]['matched_file']} ({m[0]['similarity_score']})"

    md = ["# Summary", "",
          "| file | status | risk | peer | external | ai | top match |",
          "|---|---|---|---|---|---|---|"]
    for r in all_reports:
        s = r["scores"]
        md.append("| " + " | ".join(
            "" if v is None else str(v)
            for v in (r["document_id"], r["status"], r["total_integrity_risk"],
                      s["peer_similarity_percentage"], s["external_similarity_percentage"],
                      s["ai_generated_probability"], top(r))) + " |")
    md += ["", "## Pairwise best chunk similarity (doc x doc)", "",
           "| " + " | ".join([""] + names) + " |",
           "|" + "---|" * (len(names) + 1)]
    for a in names:
        cells = [a]
        for b in names:
            v = "-" if a == b else f"{best.get((a, b), best.get((b, a))):.2f}"
            cells.append(v)
        md.append("| " + " | ".join(cells) + " |")
    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(
        {"reports": all_reports,
         "matrix": {f"{a} || {b}": v for (a, b), v in best.items()}},
        indent=2, default=str), encoding="utf-8")
