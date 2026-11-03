from integrity.extract import extract_file
from integrity.match import build_index, embed, find_matches


def _chunks_from_texts(triples):
    from integrity.extract import _chunk
    chunks = []
    for i, (doc_id, modality, text) in enumerate(triples):
        c = _chunk(doc_id, modality, text, page=1)
        c["chunk_id"] = f"{doc_id}#{i}"
        chunks.append(c)
    return chunks


def test_no_same_doc_match(tmp_path):
    chunks = _chunks_from_texts([
        ("a.docx", "text", "The board approved the phased entry into three tier-2 cities."),
        ("a.docx", "text", "Rider incentives totalled 22 rupees per delivery last quarter."),
    ])
    vecs = embed([c["text"] for c in chunks])
    idx = build_index(vecs)
    peer = find_matches(chunks, vecs, chunks, idx, exclude_same_doc=True)
    assert all(r["matches"] == [] for r in peer.values())


def test_three_doc_chunk_is_boilerplate(tmp_path):
    shared = ("GreenCart Case Competition 2026: participants answer three questions on "
              "expansion, city selection and operational changes before the deadline.")
    chunks = _chunks_from_texts([
        ("a.docx", "text", shared),
        ("b.docx", "text", shared),
        ("c.docx", "text", shared),
        ("d.docx", "text", shared),
        ("e.docx", "text", "The vendor quoted 19% higher cold-chain rates outside the metros."),
    ])
    vecs = embed([c["text"] for c in chunks])
    idx = build_index(vecs)
    peer = find_matches(chunks, vecs, chunks, idx, exclude_same_doc=True)
    assert peer["a.docx#0"]["is_boilerplate"] is True


def test_empty_file_skipped(tmp_path):
    f = tmp_path / "empty.pdf"
    f.write_bytes(b"")
    doc = extract_file(f)
    assert doc["status"] == "skipped"
    assert doc["reason"] == "empty file"


def test_two_threshold_neighbours_in_one_target_doc():
    # a query matching TWO chunks of the same other file must keep the best
    # score (regression: previously crashed comparing float > chunk dict)
    from integrity.extract import _chunk
    base = ("The board approved a phased entry into three tier-2 cities after "
            "two quarters of pilot data and a careful review of unit economics.")
    texts = [("a.pdf", "text", base + " Internal detail one."),
             ("a.pdf", "text", base + " Internal detail two."),
             ("b.docx", "text", base)]
    chunks = []
    for i, (doc_id, modality, text) in enumerate(texts):
        c = _chunk(doc_id, modality, text, page=1)
        c["chunk_id"] = f"{doc_id}#{i}"
        chunks.append(c)
    vecs = embed([c["text"] for c in chunks])
    idx = build_index(vecs)
    peer = find_matches(chunks, vecs, chunks, idx, exclude_same_doc=True)
    hits = [m for m in peer["b.docx#2"]["matches"] if m[0]["doc_id"] == "a.pdf"]
    assert len(hits) == 1 and hits[0][1] >= 0.80


def test_evaluation_rejects_missing_reports(tmp_path, monkeypatch):
    import json
    import pytest
    import evaluate

    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps({"files": {"a.pdf": {}}}), encoding="utf-8")
    monkeypatch.setattr(evaluate, "MANIFEST", manifest_path)

    with pytest.raises(ValueError, match="Missing reports for manifest files: a.pdf"):
        evaluate.main(tmp_path / "reports")


def test_evaluation_finds_pair_reported_from_either_side(tmp_path, monkeypatch):
    import json
    import evaluate

    manifest = {
        "files": {
            "a.pdf": {"label": "original",
                      "expect": {"status": "ok", "peer": False,
                                 "external": False, "ai": False}},
            "b.docx": {"label": "original_with_template",
                       "expect": {"status": "ok", "peer": True,
                                  "external": False, "ai": False}},
        },
        "pairs": [{"a": "a.pdf", "b": "b.docx", "relation": "exact_copy"}],
        "template_files": [],
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(evaluate, "MANIFEST", manifest_path)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    common = {"status": "ok", "scores": {"ai_generated_probability": 0},
              "peer_matches": [], "multimodal_matches": [], "external_matches": [],
              "excluded_boilerplate": []}
    (reports_dir / "a.pdf.json").write_text(
        json.dumps({**common, "document_id": "a.pdf"}), encoding="utf-8")
    b_report = {**common, "document_id": "b.docx",
                "peer_matches": [{"matched_file": "a.pdf"}]}
    (reports_dir / "b.docx.json").write_text(json.dumps(b_report), encoding="utf-8")

    assert evaluate.main(reports_dir) == 0
    result = (reports_dir / "evaluation_results.md").read_text(encoding="utf-8")
    assert "| a.pdf <-> b.docx | exact_copy | yes |" in result
