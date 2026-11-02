"""Compare reports/*.json to the dataset manifest and print metrics (SPEC §7)."""
import glob
import json
from pathlib import Path

MANIFEST = Path("sample_submissions/dataset_manifest.json")
REPORT_DIR = Path("reports")

SIGNALS = ("peer", "external", "ai")


def load_reports():
    reports = {}
    for path in glob.glob(str(REPORT_DIR / "*.json")):
        if "summary" in path or "evaluation" in path:
            continue
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        reports[r["document_id"]] = r
    return reports


def predicted(r):
    p = (len(r["peer_matches"]) + len(r["multimodal_matches"])) > 0
    e = len(r["external_matches"]) > 0
    a = r["scores"]["ai_generated_probability"] is not None \
        and r["scores"]["ai_generated_probability"] >= 50
    return p, e, a


def confusion(ok_names, manifest, reports, signal):
    tp = fp = fn = tn = 0
    for name in ok_names:
        expect = manifest["files"][name]["expect"]
        got = predicted(reports[name])[SIGNALS.index(signal)]
        want = expect[signal]
        if want and got:
            tp += 1
        elif want and not got:
            fn += 1
        elif not want and got:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": None if precision is None else round(precision, 2),
            "recall": None if recall is None else round(recall, 2)}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reports = load_reports()
    ok_names = [n for n, f in manifest["files"].items() if f["expect"]["status"] == "ok"]
    lines = ["# Evaluation results", ""]

    for signal in SIGNALS:
        m = confusion(ok_names, manifest, reports, signal)
        lines.append(f"## {signal}")
        lines.append("")
        lines.append("| TP | FP | FN | TN | precision | recall |")
        lines.append("|---|---|---|---|---|---|")
        lines.append(f"| {m['TP']} | {m['FP']} | {m['FN']} | {m['TN']} "
                     f"| {m['precision']} | {m['recall']} |")
        lines.append("")

    lines += ["## Pairs", "", "| pair | relation | found |", "|---|---|---|"]
    hits = 0
    for pair in manifest["pairs"]:
        a, b = pair["a"], pair["b"]
        r = reports.get(a, {})
        all_m = (r.get("peer_matches", []) + r.get("multimodal_matches", [])
                 + r.get("external_matches", []))
        found = any(m["matched_file"] == b for m in all_m)
        hits += found
        lines.append(f"| {a} <-> {b} | {pair['relation']} | {'yes' if found else 'NO'} |")
    lines.append(f"\nPair recall: **{hits}/{len(manifest['pairs'])}**")
    lines.append("")

    tpl = manifest["template_files"]
    kept = [f for f in tpl
            if reports.get(f, {}).get("excluded_boilerplate")]
    lines += ["## Template ignored as boilerplate", ""]
    lines.append(f"{len(kept)}/{len(tpl)} template files have non-empty excluded_boilerplate")

    degraded = [n for n, f in manifest["files"].items()
                if f["label"].startswith("degraded")]
    skipped = [n for n in degraded if reports.get(n, {}).get("status") == "skipped"]
    lines.append(f"\n{len(skipped)}/{len(degraded)} degraded files skipped")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    (REPORT_DIR / "evaluation_results.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
