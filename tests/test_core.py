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
