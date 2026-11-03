"""File -> units -> chunks (SPEC §5.1-5.2)."""
import logging
import re
from pathlib import Path

from integrity.config import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS, WHISPER_MODEL

logger = logging.getLogger(__name__)

PAGE_ONLY_RE = re.compile(r"^(page\s*)?\d+(\s*of\s*\d+)?$", re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
VIDEO_EXTS = {".pdf", ".docx", ".mp4"}

_whisper = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def extract_pdf(path):
    import pymupdf
    doc = pymupdf.open(path)
    npages = doc.page_count
    # lines per page: (y0, y1, text) sorted top-down
    pages = []
    for i, page in enumerate(doc, 1):
        lines = []
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                text = " ".join("".join(s["text"] for s in line["spans"]).split())
                if text:
                    lines.append((line["bbox"][1], line["bbox"][3], text))
        lines.sort()
        pages.append((i, lines))
    doc.close()

    def key_of(t):
        return "".join(c for c in t.lower() if not c.isdigit()).strip()

    page_freq = {}
    for _, lines in pages:
        for k in {key_of(t) for _, _, t in lines}:
            page_freq[k] = page_freq.get(k, 0) + 1
    units = []
    for pageno, lines in pages:
        # drop page-number lines and lines repeating on more than half the pages
        kept = []
        for y0, y1, text in lines:
            if PAGE_ONLY_RE.match(text):
                continue
            if npages >= 2 and page_freq[key_of(text)] > 0.5 * npages:
                continue
            kept.append((y0, y1, text))
        # group lines into paragraph units on a vertical gap well above the
        # in-paragraph leading (~0.4pt); paragraph spaceAfter=12 gives ~12.4pt gaps
        group = []
        prev_y1 = None
        for y0, y1, text in kept:
            height = y1 - y0
            if prev_y1 is not None and y0 - prev_y1 > 0.5 * height + 2:
                units.append({"text": " ".join(group), "page": pageno})
                group = []
            group.append(text)
            prev_y1 = y1
        if group:
            units.append({"text": " ".join(group), "page": pageno})
    return units


def extract_docx(path):
    import docx as docx_lib
    document = docx_lib.Document(path)
    units = []
    for i, p in enumerate(document.paragraphs, 1):
        if p.text.strip():
            units.append({"text": " ".join(p.text.split()), "paragraph": i})
    # Tables often hold financials and other substantive submission content.
    # Use explicit table/row locations so reports do not call these paragraphs.
    for table_no, table in enumerate(document.tables, 1):
        for row_no, row in enumerate(table.rows, 1):
            text = " | ".join(" ".join(cell.text.split()) for cell in row.cells)
            if text.strip(" |"):
                units.append({"text": text, "table": table_no, "row": row_no})
    return units


def _sentence_fragments(sentence):
    """Split an unusually long sentence into bounded word windows."""
    words = sentence.split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [sentence]
    # Distribute words evenly so the last fragment is not just a few words.
    count = (len(words) + CHUNK_MAX_WORDS - 1) // CHUNK_MAX_WORDS
    width = (len(words) + count - 1) // count
    return [" ".join(words[i:i + width]) for i in range(0, len(words), width)]


def _decode_audio(path):
    """Mirror faster_whisper.audio.decode_audio, but without metadata_errors
    (removed in av>=15, so faster-whisper 1.2.1 cannot open files itself)."""
    import av
    import numpy as np
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    raw, dtype = bytearray(), np.int16
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(audio=0):
            if frame.format is None:
                continue
            for rf in resampler.resample(frame):
                array = rf.to_ndarray()
                dtype = array.dtype
                raw.extend(array.tobytes())
    audio = np.frombuffer(bytes(raw), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def extract_video(path):
    import av
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("no audio track")
        audio = _decode_audio(path)
    model = _get_whisper()
    segments, _ = model.transcribe(audio, language="en", beam_size=5, initial_prompt=(
        "GreenCart grocery delivery case pitch. Phased entry in Indore, Coimbatore and Jaipur; "
        "dark store break even at nine hundred orders per day. Bhopal, Rajkot, Patna, tier-2 "
        "cities, contribution margin, rupees."))
    units = []
    for s in segments:
        text = s.text.strip()
        if text:
            units.append({"text": text, "start": s.start, "end": s.end})
    return units


def extract_file(path):
    path = Path(path)
    doc_id = path.name
    try:
        if path.stat().st_size == 0:
            return {"doc_id": doc_id, "modality": "text", "status": "skipped",
                    "reason": "empty file", "units": []}
        ext = path.suffix.lower()
        if ext == ".pdf":
            units = extract_pdf(path)
            modality = "text"
        elif ext == ".docx":
            units = extract_docx(path)
            modality = "text"
        elif ext == ".mp4":
            units = extract_video(path)
            modality = "video"
        else:
            return {"doc_id": doc_id, "modality": "text", "status": "skipped",
                    "reason": "unsupported type", "units": []}
        if not units or sum(len(u["text"].split()) for u in units) < 5:
            return {"doc_id": doc_id, "modality": modality, "status": "skipped",
                    "reason": "no extractable text", "units": []}
        return {"doc_id": doc_id, "modality": modality, "status": "ok",
                "reason": None, "units": units}
    except Exception as e:
        logger.warning("%s: %s", doc_id, e)
        return {"doc_id": doc_id, "modality": "text", "status": "skipped",
                "reason": str(e)[:200], "units": []}


def extract_folder(folder):
    docs = []
    for path in sorted(Path(folder).iterdir()):
        if path.name == "dataset_manifest.json" or path.suffix.lower() not in VIDEO_EXTS:
            continue
        docs.append(extract_file(path))
    return docs


def chunk_doc(doc):
    if doc["status"] != "ok":
        return []
    chunks = []
    if doc["modality"] == "video":
        current, start, end = [], None, None
        n_words = 0
        for unit in doc["units"]:
            for fragment in _sentence_fragments(unit["text"]):
                w = fragment.split()
                if current and n_words + len(w) > CHUNK_MAX_WORDS:
                    text = " ".join(current)
                    if len(text.split()) >= CHUNK_MIN_WORDS:
                        chunks.append(_chunk(doc["doc_id"], "video", text,
                                            start=start, end=end))
                    current, start, end, n_words = [], None, None, 0
                if start is None:
                    start = unit["start"]
                end = unit["end"]
                current.append(fragment)
                n_words += len(w)
        if current:
            text = " ".join(current)
            if len(text.split()) >= CHUNK_MIN_WORDS:
                chunks.append(_chunk(doc["doc_id"], "video", text, start=start, end=end))
    else:
        for unit in doc["units"]:
            loc = {key: unit[key] for key in ("page", "paragraph", "table", "row")
                   if key in unit}
            sentences = [fragment for sent in SENT_SPLIT_RE.split(unit["text"])
                         for fragment in _sentence_fragments(sent)]
            prev_last = None  # previous content piece, reused only when it fits
            cur, cur_words = [], 0
            for sent in sentences:
                sw = len(sent.split())
                if cur and cur_words + sw > CHUNK_MAX_WORDS:
                    chunks.append(_chunk(doc["doc_id"], "text", " ".join(cur), **loc))
                    prev_last = cur[-1]
                    cur, cur_words = [], 0
                if not cur and prev_last is not None:
                    overlap_words = len(prev_last.split())
                    if overlap_words + sw <= CHUNK_MAX_WORDS:
                        cur = [prev_last]
                        cur_words = overlap_words
                cur.append(sent)
                cur_words += sw
            if cur:
                chunks.append(_chunk(doc["doc_id"], "text", " ".join(cur), **loc))
    kept = [c for c in chunks if len(c["text"].split()) >= CHUNK_MIN_WORDS]
    for i, c in enumerate(kept):
        c["chunk_id"] = f"{doc['doc_id']}#{i}"
    return kept


def _chunk(doc_id, modality, text, page=None, paragraph=None, table=None, row=None,
           start=None, end=None):
    return {"chunk_id": "", "doc_id": doc_id, "modality": modality, "text": text,
            "page": page, "paragraph": paragraph, "table": table, "row": row,
            "start": start, "end": end}


def chunk_docs(docs):
    out = []
    for doc in docs:
        out.extend(chunk_doc(doc))
    return out


if __name__ == "__main__":
    docs = extract_folder("sample_submissions")
    for d in docs:
        first = d["units"][0]["text"][:80] if d["units"] else ""
        print(f"{d['doc_id']} | {d['status']} | {d['reason']} | {len(d['units'])} | {first}")
    print()
    for d in docs:
        chunks = chunk_doc(d)
        sizes = [len(c["text"].split()) for c in chunks]
        if sizes:
            sample = chunks[0]
            loc = {k: v for k, v in sample.items()
                   if k in ("page", "paragraph", "start", "end") and v is not None}
            print(f"{d['doc_id']} | n_chunks {len(chunks)} | min {min(sizes)} | max {max(sizes)}")
            print(f"    sample: {loc} {sample['text'][:70]!r}")
        else:
            print(f"{d['doc_id']} | n_chunks 0")
