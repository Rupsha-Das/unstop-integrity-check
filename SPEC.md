# SPEC: Multimodal Plagiarism Detection Engine

You are building a Python CLI. It reads a folder of PDF, DOCX, and MP4 files and writes one JSON and one Markdown report per file, plus a summary.
All design decisions are already made below. **Do not redesign. Do not add features.** If something is unclear, pick the simplest option and write it in PROGRESS.md under "Decisions".

---

## 0. The whole system on one page

```text
sample_submissions/*.pdf|docx|mp4
        │
        ▼
[extract.py]   file ──► units (paragraphs with page no. / speech segments with timestamps)
        │               broken/empty/silent file ──► status "skipped" + reason (never crash)
        ▼
[extract.py]   units ──► chunks of ≤120 words (never cross a paragraph or page)
        │
        ▼
[match.py]     chunks ──► 384-number vectors (all-MiniLM-L6-v2) ──► FAISS IndexFlatIP
        │
        ▼
[match.py]     for every chunk: find nearest chunks in OTHER files
        │         • same file ──► ignored (self-match exclusion)
        │         • similar in ≥3 other files ──► "boilerplate", ignored for scoring
        │         • sim ≥ 0.80 ──► a match (peer), or vs reference_corpus (external)
        ▼
[ai_detect.py] distilgpt2 perplexity + burstiness ──► AI likelihood %
        │
        ▼
[report.py]    reports/<file>.json + reports/<file>.md + reports/summary.md/json
```

Run command (must work exactly like this):

```bash
uv run run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

---

## 1. Final file tree

```text
project/
├── pyproject.toml              # deps, managed by uv
├── run_integrity_check.py      # CLI entry: argparse, calls the steps in order
├── integrity/
│   ├── config.py               # ALL thresholds and model names (only place with numbers)
│   ├── extract.py              # file -> units -> chunks
│   ├── match.py                # embed, FAISS index, peer + external matching, boilerplate
│   ├── ai_detect.py            # perplexity / burstiness score
│   └── report.py               # JSON + Markdown writers, summary
├── build_dataset.py            # texts + writes sample_submissions/ + manifest
├── evaluate.py                 # compares reports to manifest, prints metrics
├── sample_submissions/         # generated, committed (incl. dataset_manifest.json)
├── reference_corpus/           # 6 .txt files = the fake "web"
├── tests/test_core.py          # 3 tiny pytest tests
├── README.md  EVALUATION.md
└── PROGRESS.md                 # running log
```

Keep it small: one function per pseudocode block below, no classes, no helper modules, no extra files.

---

## 2. Environment

Use **uv** for everything. Create `pyproject.toml` (exact content), then run `uv sync`:

```toml
[project]
name = "integrity-check"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "pymupdf", "python-docx", "reportlab", "faster-whisper", "av",
  "sentence-transformers", "faiss-cpu", "transformers", "torch",
  "numpy", "gTTS", "imageio-ffmpeg", "pytest",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Rules:
- Setup = `uv sync`. Run anything = `uv run <file.py>` or `uv run python -m integrity.<module>`. Never pip, never activate a venv.
- Add a dependency only with `uv add <pkg>`, and log why.
- ffmpeg comes from pip: `imageio_ffmpeg.get_ffmpeg_exe()` returns its path. No system ffmpeg.
- First run downloads models (MiniLM ~90MB, whisper tiny.en ~75MB, distilgpt2 ~350MB). No API keys needed.

---

## 3. config.py (exact content)

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "tiny.en"
LM_MODEL = "distilgpt2"

CHUNK_MAX_WORDS = 120      # ~160 tokens; small chunks give precise citations
CHUNK_MIN_WORDS = 8        # drop headings / tiny fragments
SEARCH_K = 20              # neighbours fetched per chunk

MATCH_THRESHOLD = 0.80     # cosine >= this = a match
BOILERPLATE_SIM = 0.85     # cosine used to detect shared template text
BOILERPLATE_MIN_DOCS = 3   # chunk similar in >= 3 OTHER files = boilerplate

# AI score = 100 * (0.6 * ppl_part + 0.4 * burst_part), see ai_detect.py
AI_PPL_LOW, AI_PPL_HIGH = 20.0, 80.0     # perplexity at/below LOW -> looks AI
AI_CV_LOW, AI_CV_HIGH = 0.2, 0.8         # burstiness at/below LOW -> looks AI
AI_MIN_SENTENCES = 5

RISK_HIGH = {"peer": 40.0, "external": 40.0, "ai": 80.0}
RISK_MEDIUM = {"peer": 15.0, "external": 15.0, "ai": 50.0}
```

---

## 4. Data shapes (use these exact keys everywhere)

```python
# extract.py returns one Doc per file
Doc = {
  "doc_id": "sub01_alpha.pdf",       # file name
  "modality": "text" | "video",
  "status": "ok" | "skipped",
  "reason": None | "empty file" | "no audio track" | "no extractable text" | "<error message>",
  "units": [                          # text: one per paragraph; video: one per whisper segment
     {"text": "...", "page": 2}                     # pdf
     {"text": "...", "paragraph": 7}                # docx (DOCX has no real pages)
     {"text": "...", "start": 12.0, "end": 18.4}    # mp4, seconds
  ],
}

# extract.py chunk_docs() returns a flat list of Chunk
Chunk = {
  "chunk_id": "sub01_alpha.pdf#4",
  "doc_id": "sub01_alpha.pdf",
  "modality": "text" | "video",
  "text": "...",
  "page": 2 | None, "paragraph": 7 | None, "start": 12.0 | None, "end": 18.4 | None,
}
```

---

## 5. Module specs (pseudocode = what to write)

### 5.1 extract.py

```text
extract_file(path) -> Doc
  try:
    if file size == 0           -> skipped, reason "empty file"
    .pdf  -> extract_pdf
    .docx -> extract_docx
    .mp4  -> extract_video
    other -> skipped, reason "unsupported type"
    if total words in units < 5 -> skipped, reason "no extractable text"
  except Exception as e:
    logging.warning(f"{name}: {e}") ; skipped, reason str(e)[:200]

extract_pdf(path)
  doc = fitz.open(path)
  for each page (1-based no.): blocks = page.get_text("blocks"); text = block[4], whitespace collapsed
  remove headers/footers:
    - drop block if it matches ^(page\s*)?\d+(\s*of\s*\d+)?$ (case-insensitive)
    - if pages >= 2: key = block text with digits removed, lowercased;
      drop block if its key appears on >= 50% of pages
  unit = {"text", "page"}

extract_docx(path)
  for i, p in enumerate(docx.Document(path).paragraphs, 1): if p.text.strip(): unit {"text", "paragraph": i}

extract_video(path)
  with av.open(path) as c: if no c.streams.audio -> skipped, reason "no audio track"
  model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")   # load once, cache in a global
  segments, _ = model.transcribe(str(path))
  unit per segment {"text": s.text.strip(), "start": s.start, "end": s.end}

extract_folder(folder) -> list[Doc]   # sorted by file name; skips dataset_manifest.json and non-supported silently
```

`if __name__ == "__main__":` prints one line per file: `name | status | reason | n_units | first 80 chars`.

### 5.2 extract.py (chunking part)

```text
chunk_doc(doc) -> list[Chunk]
  text docs: for each unit (paragraph):
     split into sentences with re.split(r"(?<=[.!?])\s+", text)
     pack sentences into chunks of <= CHUNK_MAX_WORDS words;
     when starting a new chunk, repeat the last sentence of the previous one (1-sentence overlap)
     keep page / paragraph from the unit
  video docs: pack consecutive segments into chunks of <= CHUNK_MAX_WORDS words;
     start = first segment start, end = last segment end
  drop chunks with < CHUNK_MIN_WORDS words
  chunk_id = f"{doc_id}#{n}" (n counts from 0 inside the doc)

chunk_docs(docs) -> chunks for all docs with status "ok"
```

`__main__` of extract.py prints, after the extract lines: `name | n_chunks | min words | max words` and one sample chunk with its location.

### 5.3 match.py (embedding part)

```text
_model = None (lazy-load SentenceTransformer(EMBED_MODEL))
embed(texts) -> np.float32 array (n, 384), normalize_embeddings=True
build_index(vectors) -> faiss.IndexFlatIP(384) with vectors added
# normalized vectors + inner product == cosine similarity
```

### 5.4 match.py (matching part)

```text
find_matches(query_chunks, query_vecs, target_chunks, target_index, exclude_same_doc=True) -> dict
  k = min(SEARCH_K, number of target chunks)
  scores, ids = target_index.search(query_vecs, k)
  for each query chunk q:
     neighbours = [(target_chunks[j], s) for j, s in row if j != -1]
     if exclude_same_doc: drop neighbours where t.doc_id == q.doc_id      # SELF-MATCH EXCLUSION
     boilerplate_docs = distinct t.doc_id with s >= BOILERPLATE_SIM
     q.is_boilerplate = len(boilerplate_docs) >= BOILERPLATE_MIN_DOCS (only for peer search)
     best per other doc: keep the highest s per t.doc_id, if s >= MATCH_THRESHOLD
  returns {chunk_id: {"is_boilerplate": bool, "matches": [(target_chunk, score), ...]}}

jaccard(a, b) = |set(words a) ∩ set(words b)| / |union|  (lowercase, strip punctuation)
match_type = "near-exact copy" if jaccard >= 0.6 else "paraphrase / semantic match"

percentage(doc_chunks, results)
  scorable = chunks of this doc that are not boilerplate
  flagged = scorable chunks with >= 1 match
  return round(100 * flagged / len(scorable), 1) if scorable else 0.0
```

Peer search: query = all submission chunks, target = same index, exclude_same_doc=True.
External search: target = index of reference_corpus chunks (each .txt treated as a text Doc with `paragraph` units split on blank lines), exclude_same_doc=False, no boilerplate rule.

Main block prints a **doc × doc matrix** of the best cosine between any two files' chunks (2 decimals), and the number of boilerplate chunks per doc.

### 5.5 ai_detect.py

```text
ai_score(text) -> {"probability": float|None, "perplexity": float|None, "burstiness": float|None, "note": str}
  sentences = split like chunk_doc, keep those with >= 5 words, take first 40
  if len < AI_MIN_SENTENCES -> probability None, note "too little text"
  for each sentence: ppl = exp(model loss) using distilgpt2 (torch.no_grad, labels=input_ids)
  P = mean(ppl) ; CV = std(ppl) / mean(ppl)          # CV = burstiness
  ppl_part   = clamp((AI_PPL_HIGH - P) / (AI_PPL_HIGH - AI_PPL_LOW), 0, 1)
  burst_part = clamp((AI_CV_HIGH - CV) / (AI_CV_HIGH - AI_CV_LOW), 0, 1)
  probability = round(100 * (0.6 * ppl_part + 0.4 * burst_part), 1)
```

Why: AI text is predictable (low perplexity) and evenly predictable (low burstiness). Human text is spikier.
Main block prints `name | P | CV | probability` for every ok doc.

### 5.6 report.py

Risk:

```text
risk(scores): ignore None values
  HIGH   if any score >= RISK_HIGH[that signal]
  MEDIUM if any score >= RISK_MEDIUM[that signal]
  else LOW
  skipped files -> "UNPROCESSED"
```

JSON per file (keys from the assignment + a few extras, nothing else):

```json
{
  "document_id": "sub03_charlie.docx",
  "status": "ok",
  "skip_reason": null,
  "total_integrity_risk": "HIGH",
  "scores": {
    "peer_similarity_percentage": 85.7,
    "external_similarity_percentage": 0.0,
    "ai_generated_probability": 31.2
  },
  "peer_matches": [
    {
      "matched_file": "sub02_bravo.pdf",
      "similarity_score": 0.99,
      "jaccard": 0.97,
      "match_type": "near-exact copy",
      "query_passage": "...",
      "matched_passage": "...",
      "location": {"query_paragraph": 4, "matched_page": 1}
    }
  ],
  "multimodal_matches": [ ...same shape, location uses query_timestamp / matched_timestamp... ],
  "external_matches":   [ ...same shape, matched_file = reference_corpus file... ],
  "excluded_boilerplate": [{"passage": "first 120 chars...", "shared_with_n_files": 4}],
  "ai_details": {"perplexity": 41.3, "burstiness": 0.52, "note": ""},
  "explanations": "plain-English summary, see below"
}
```

Rules:
- A match goes in `multimodal_matches` if query modality != matched modality, else `peer_matches`.
- Location keys: put only the ones that exist: `query_page` / `query_paragraph` / `query_timestamp`, same for `matched_`.
- Timestamp format: `"00:00:12 - 00:00:24"` (HH:MM:SS - HH:MM:SS).
- Sort matches by similarity_score descending. Round scores to 2 decimals.
- `explanations` built from a template, for example:
  `"4 of 6 passages (66.7%) closely match other submissions: sub02_bravo.pdf (4 passages, near-exact copy). 1 passage shared with 4 other files was treated as common template text and ignored. No external source matches. AI likelihood 31% (perplexity 41, burstiness 0.52)."`

Markdown per file, built for a 30-second review:

```text
# sub03_charlie.docx: Risk HIGH
| Peer % | External % | AI % |
|---|---|---|
| 85.7 | 0.0 | 31.2 |

**Summary:** <explanations>

## Matches
### 1. sub02_bravo.pdf: 0.99 (near-exact copy)
| This file (paragraph 4) | sub02_bravo.pdf (page 1) |
|---|---|
| query passage | matched passage |

## Ignored as common template
- "GreenCart Case Competition 2026: Problem Statement..." (in 4 other files)
```

Skipped file Markdown: title + `Status: UNPROCESSED: <reason>`.

`write_summary(all_reports, matrix, out_dir)` writes `summary.json` and `summary.md`. summary.md holds one table (file | status | risk | peer | external | ai | top match) and the doc × doc similarity matrix.

### 5.7 run_integrity_check.py

```text
args: --input_dir, --output_dir, --reference_dir (default ./reference_corpus)
logging.basicConfig(level=INFO)
docs   = extract_folder(input_dir)
chunks = chunk_docs(docs); vecs = embed(texts); idx = build_index(vecs)
ref docs -> ref chunks -> ref vecs -> ref idx
peer = find_matches(chunks, vecs, chunks, idx, exclude_same_doc=True)
ext  = find_matches(chunks, vecs, ref_chunks, ref_idx, exclude_same_doc=False, boilerplate=False)
for each doc: scores, ai_score(full text), build report, write json + md
write summary; print the summary table to the terminal
```

If there are 0 ok docs, still write the summary and exit 0. The tool must never crash on a bad file.

---

## 6. Dataset (build_dataset.py)

Story: case competition **"GreenCart: should an online grocery startup expand to tier-2 Indian cities?"**

| # | File | What it is | Ground-truth label |
|---|---|---|---|
| 1 | sub01_alpha.pdf | original, 2 pages, TEMPLATE header, CROSSMODAL paragraph on page 2 | template + source of cross-modal copy |
| 2 | sub02_bravo.pdf | original, TEMPLATE header | template + source of exact copy |
| 3 | sub03_charlie.docx | TEMPLATE header + **identical body to sub02** | exact copy of sub02 |
| 4 | sub04_delta.docx | original, TEMPLATE header | template + source of paraphrase |
| 5 | sub05_echo.docx | TEMPLATE header + **every sentence of sub04 reworded** (synonyms, new sentence order inside paragraphs, same arguments and numbers) | paraphrase of sub04 |
| 6 | sub06_foxtrot.pdf | written in typical LLM style (see below) | ai_generated |
| 7 | sub07_golf.pdf | original, human style, different angle (argues "delay") | original |
| 8 | sub08_hotel.docx | original + **one paragraph copied verbatim** from reference_corpus/quick_commerce.txt | external copy |
| 9 | sub09_india.docx | original, human style (argues "don't expand") | original |
| 10 | sub10_juliet.pdf | **zero-byte file** | degraded |
| 11 | pitch01_kilo.mp4 | TTS voice reading the CROSSMODAL paragraph | cross-modal copy of sub01 |
| 12 | pitch02_lima.mp4 | TTS voice, original 20–30 s pitch | original |
| 13 | pitch03_mike.mp4 | 10 s black video, **no audio track** | degraded |

Counts: 5 PDF (1,2,6,7,10), 5 DOCX (3,4,5,8,9), 3 MP4. 

### Fixed texts (copy exactly into build_dataset.py)

TEMPLATE (~150 words, first paragraph(s) of files 1–5):

> GreenCart Case Competition 2026: Problem Statement. GreenCart is a five-year-old online grocery delivery company operating in six metropolitan cities. It has 1.2 million monthly active customers and an average order value of 1,450 rupees, but its contribution margin per order is still negative in two of its six cities. The board is considering expanding into twelve tier-2 cities over the next eighteen months. Expansion would require an estimated investment of 380 crore rupees in dark stores, cold-chain logistics and marketing. Participants must answer three questions. First, should GreenCart expand now, delay, or not expand at all? Second, if it expands, which three cities should it enter first and why? Third, what operational changes are needed to make unit economics positive within two years? Support your answer with clear assumptions and numbers. Submissions are limited to four pages. Cite all external sources.

CROSSMODAL (its own paragraph on page 2 of sub01; also the full TTS script of pitch01):

> Our recommendation is a phased entry, starting with Indore, Coimbatore and Jaipur. These three cities combine high smartphone penetration with dense residential clusters, which keeps delivery distances short. We estimate that a single dark store in each city can break even at around nine hundred orders per day, which is reachable within eight months if GreenCart reuses its existing supplier contracts.

### Texts you write yourself (in build_dataset.py)
- Each body: 4–6 paragraphs, 250–450 words in total, all different content. Use concrete, slightly odd numbers.
- **Human style** (sub01, 02, 04, 07, 08, 09, pitch02): first person ("we think", "honestly"), uneven sentence lengths, a short fragment or two, one small typo, specific local details.
- **AI style** (sub06): polished, balanced, uniform sentences, words like "Furthermore", "In conclusion", "it is important to note", "leverage", "holistic", "robust". No template header.
- **reference_corpus/**: 6 .txt files of 120–250 words, neutral encyclopedia style: `quick_commerce.txt`, `dark_stores.txt`, `cold_chain.txt`, `unit_economics.txt`, `tier2_cities_india.txt`, `customer_acquisition_cost.txt`. sub08 copies one full paragraph of quick_commerce.txt word for word.

### Building files
- PDF: reportlab `SimpleDocTemplate` + `Paragraph` per paragraph. Every page gets header text `"GreenCart Case Competition 2026 | Confidential"` and footer `"Page N"` drawn with `onPage` canvas (this tests header/footer stripping). sub01 uses `PageBreak()` so CROSSMODAL lands on page 2.
- DOCX: python-docx, one `add_paragraph` per paragraph.
- Zero-byte PDF: `open(path, "wb").close()`.
- MP4 with speech: `gTTS(text, lang="en", tld="co.in").save(tmp.mp3)`, then
  `ffmpeg -y -f lavfi -i color=c=black:s=640x360:r=10 -i tmp.mp3 -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac out.mp4`
- Silent MP4: `ffmpeg -y -f lavfi -i color=c=black:s=640x360:r=10 -t 10 -c:v libx264 -pix_fmt yuv420p -an out.mp4`
- Run ffmpeg with `subprocess.run([...], check=True, capture_output=True)` using `imageio_ffmpeg.get_ffmpeg_exe()`.

### dataset_manifest.json

```json
{
  "files": {
    "sub01_alpha.pdf":   {"label": "original_with_template", "expect": {"status": "ok", "peer": true,  "external": false, "ai": false}},
    "sub02_bravo.pdf":   {"label": "original_with_template", "expect": {"status": "ok", "peer": true,  "external": false, "ai": false}},
    "sub03_charlie.docx":{"label": "exact_copy",             "expect": {"status": "ok", "peer": true,  "external": false, "ai": false}},
    "sub04_delta.docx":  {"label": "original_with_template", "expect": {"status": "ok", "peer": true,  "external": false, "ai": false}},
    "sub05_echo.docx":   {"label": "paraphrase",             "expect": {"status": "ok", "peer": true,  "external": false, "ai": false}},
    "sub06_foxtrot.pdf": {"label": "ai_generated",           "expect": {"status": "ok", "peer": false, "external": false, "ai": true}},
    "sub07_golf.pdf":    {"label": "original",               "expect": {"status": "ok", "peer": false, "external": false, "ai": false}},
    "sub08_hotel.docx":  {"label": "external_copy",          "expect": {"status": "ok", "peer": false, "external": true,  "ai": false}},
    "sub09_india.docx":  {"label": "original",               "expect": {"status": "ok", "peer": false, "external": false, "ai": false}},
    "sub10_juliet.pdf":  {"label": "degraded_empty",         "expect": {"status": "skipped"}},
    "pitch01_kilo.mp4":  {"label": "cross_modal_copy",       "expect": {"status": "ok", "peer": true,  "external": false, "ai": false}},
    "pitch02_lima.mp4":  {"label": "original",               "expect": {"status": "ok", "peer": false, "external": false, "ai": false}},
    "pitch03_mike.mp4":  {"label": "degraded_silent",        "expect": {"status": "skipped"}}
  },
  "pairs": [
    {"a": "sub02_bravo.pdf",  "b": "sub03_charlie.docx", "relation": "exact_copy"},
    {"a": "sub04_delta.docx", "b": "sub05_echo.docx",    "relation": "paraphrase"},
    {"a": "sub01_alpha.pdf",  "b": "pitch01_kilo.mp4",   "relation": "cross_modal"},
    {"a": "sub08_hotel.docx", "b": "quick_commerce.txt", "relation": "external"}
  ],
  "template_files": ["sub01_alpha.pdf", "sub02_bravo.pdf", "sub03_charlie.docx", "sub04_delta.docx", "sub05_echo.docx"],
  "note": "Overlap is flagged on BOTH sides of a copy pair; deciding who copied is a human decision."
}
```

---

## 7. evaluate.py

```text
load manifest + reports/*.json
predicted flags per file:
  peer     = len(peer_matches) + len(multimodal_matches) > 0
  external = len(external_matches) > 0
  ai       = ai_generated_probability is not None and >= 50
  status   = report status
for signal in peer, external, ai: confusion matrix (TP FP FN TN) + precision + recall over ok files
pair recall: pair found if b appears as matched_file in a's report (or a in b's)
template check: for each template file, excluded_boilerplate non-empty? (count x/5)
status check: degraded files skipped? (x/2)
print everything as Markdown tables; also write reports/evaluation_results.md
```

---

## 8. Scaling brief (paste into README, adjust wording, keep numbers labelled "estimate")

- **Flow:** upload → object storage (S3) → one queue message per file (Redis + Celery, or SQS) → worker pools per stage: extract (CPU) → transcribe (GPU) → embed (GPU batch) → index upsert → score → report. Each job is keyed by the file's SHA-256, so retries are safe and exact duplicate files are caught before any ML runs.
- **Compute:** PDF/DOCX parsing is cheap CPU work (~0.2 s/file). Whisper is the main cost: 10k pitches × 3 min = 500 h of audio. faster-whisper `small` on one T4 GPU runs at roughly 30–60× real time, so about 10–15 GPU-hours (estimate), which is cheap on spot instances. Embeddings: ~40 chunks/doc × 10k = 400k chunks. MiniLM on a GPU handles thousands of chunks/s, so a few minutes in total.
- **Vector DB:** 400k × 384 floats ≈ 0.6 GB. A Flat (exact) index still fits, but all-vs-all search is O(n²). In production use HNSW (Qdrant or pgvector, M=16, ef_search≈128): ~95–99% recall at millisecond queries, and it supports adding files as they arrive. Partition by `challenge_id` so files are only compared within one competition plus the reference corpus. Use IVF-PQ only if it grows past ~10M chunks.
- **Boilerplate at scale:** organisers upload the official prompt, which is embedded once and auto-excluded. The "appears in ≥ N files" rule scales N with corpus size (e.g. 1%).
- **Bottlenecks:** (1) ASR GPU time, (2) perplexity model, which can move to a small fine-tuned classifier on GPU, (3) report generation is trivial. Latency target: report within ~2 min of upload; the full cross-check re-runs after the deadline.
- **Reliability:** dead-letter queue for corrupt files, per-stage retries, metrics on skip rate and queue depth.
- **Cost (rough):** tens of dollars of spot GPU per 10k-submission challenge; storage is negligible.

---

## 9. Known limitations to mention in EVALUATION.md (confirm with real results)

1. AI detector: distilgpt2 perplexity is a weak signal. Short text, spoken transcripts, and simple human writing can score "AI-like". Fix: use a fine-tuned detector and calibrate on labelled data.
2. Speech vs text: Whisper writes "nine hundred" while a PDF might say "900". Jaccard drops, but embeddings still match. Accent or noise would hurt transcription. Fix: a larger Whisper model plus number normalisation.
3. Template chunks that mix template and original sentences may fall just below the boilerplate threshold. Fix: sentence-level template matching.
4. Thresholds (0.80, 0.85, ≥3 files) were set on 13 files, so there is a risk of overfitting. Fix: tune on a bigger labelled set.
