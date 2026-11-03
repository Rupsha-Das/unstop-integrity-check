# GreenCart Integrity Check (Multimodal Plagiarism Detection Engine)

A CLI that takes a folder of submissions (PDF · DOCX · MP4), checks each one for
plagiarism against every other submission and against a small reference corpus,
scores AI-likelihood, and writes one JSON + one Markdown report per file plus a
summary — designed so a judge can review 13 files in ~3 minutes.

## What it does

- Extracts text (PyMuPDF / python-docx) and speech (faster-whisper, `tiny.en`,
  transcribes the audio track of each MP4 into time-stamped segments).
- Chops text into passages of ≤120 words without crossing paragraph boundaries;
  unusually long sentences are split into bounded pieces. Video chunks pack
  consecutive transcript segments and retain their start/end timestamps.
- Extracts DOCX paragraphs and table rows, with paragraph or table/row locations.
- Embeds everything with `sentence-transformers/all-MiniLM-L6-v2` (384 numbers),
  builds a FAISS `IndexFlatIP`, and for every chunk finds its nearest neighbours
  **in other files only**:
  - same file → ignored,
  - similar to chunks in ≥3 other files → "common template/boilerplate", ignored
    for score and shown under *Ignored*,
  - similarity ≥ 0.80 → a match (peer), or vs the reference corpus (external).
    A match between different modalities is a *cross-modal* match (speech vs text).
- Scores AI-likelihood with distilgpt2: mean warm-context sentence perplexity +
  burstiness (coefficient of variation) → 0–100 %.
- Writes `reports/<file>.json`, `reports/<file>.md` and `reports/summary.*`,
  plus `reports/evaluation_results.md`.
- Preserves every matched query/source passage pair in the audit reports, with
  its similarity score and page, paragraph, or timestamp locations.

## Setup

```bash
uv sync        # creates .venv and installs everything (CPU-only torch)
```

Model weights (MiniLM ~90 MB, whisper tiny.en ~75 MB, distilgpt2 ~350 MB) are
downloaded automatically on first run — no API keys needed.

## Run

```bash
uv sync          # once
uv run integrity-check
```

That single command extracts, transcribes, matches and scores everything in
`./sample_submissions`, writes `./reports/` (JSON + Markdown per file, plus
`summary.md`) and evaluates itself against the dataset manifest — no arguments
needed.

The explicit form from the assessment brief works the same way. Run it with
`uv run` so the project environment is used on Windows, macOS, and Linux:

```bash
uv run run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

## Use it on your own files

```bash
uv run integrity-check my_folder                 # a folder of submissions
uv run integrity-check my_draft.docx             # ONE file: checked against the
                                                 # other files in its folder
                                                 # + the reference corpus; the
                                                 # report is printed in the CLI
uv run integrity-check my_draft.docx --output_dir ./out
uv run integrity-check --input_dir ./my_submissions --output_dir ./out   # folders
uv run integrity-check --reference_dir ./my_web  # custom 'web' texts
uv run integrity-check --no-eval                 # skip evaluation
uv run integrity-check -v                        # show progress logs
```

Supported types: `.pdf`, `.docx`, `.mp4` (txt/docx/pdf text, mp4 audio).
Unsupported, empty or silent files are reported as `UNPROCESSED`, never crash.

Evaluation can also be run/printed separately with
`uv run evaluate.py [report_dir]`, and `uv run pytest -q` runs the core tests.

## How it works

```text
sample_submissions/*.pdf|docx|mp4
        │
        ▼
extract:  file ──► paragraph units (page no. / paragraph no. / speech timestamps)
          broken/empty/silent file ──► status "skipped" + reason (never crashes)
        ▼
chunking: units ──► ≤120-word chunks (1-sentence overlap, video segments packed)
        ▼
matching: chunks ──► 384-d embeddings ──► FAISS IndexFlatIP
          per chunk: neighbours in other files; ≥0.80 = match
          similar in ≥3 other files = boilerplate → ignored for scoring
        ▼
ai:       distilgpt2 perplexity + burstiness ──► AI likelihood %
        ▼
report:   reports/<file>.json + .md + summary
```

All thresholds and model names live in `integrity/config.py`.

## Scores

Each report has `peer_similarity_percentage` (% of scorable passages matching
another submission), `external_similarity_percentage` (% matching the reference
corpus) and `ai_generated_probability`. Risk = HIGH/MEDIUM/LOW from the
configured cut-offs: peer/external 40 = HIGH, 15 = MEDIUM, ai 80 = HIGH,
50 = MEDIUM (see `integrity/config.py`).

## Evaluation

Every run re-scores itself against the ground truth in
`sample_submissions/dataset_manifest.json` (see `evaluate.py`; run manually
with `uv run evaluate.py [report_dir]`). On the 13-file dataset:

| signal | precision | recall | FP | FN |
|---|---|---|---|---|
| peer | 1.0 | 1.0 | 0 | 0 |
| external | 1.0 | 1.0 | 0 | 0 |
| ai | 0.5 | 1.0 | 1 | 0 |

All 4 ground-truth pairs found (exact copy, paraphrase, cross-modal, external),
5/5 template files recognised as boilerplate, 2/2 degraded files skipped.
Failure analysis and design trade-offs: [EVALUATION.md](EVALUATION.md).

## Assumptions

- Chunks are capped at 120 words (roughly 160 tokens) for precise citations.
  The assignment's 200–400-token chunk size is an example rather than a hard
  requirement; this smaller setting trades some context for more precise
  passage-level references. Long sentences are split to preserve the cap.
- Overlap is flagged on **both** sides of a copy pair (sub02 and sub03 both get
  reports); deciding who copied from whom is a human decision. The manifest in
  `sample_submissions/dataset_manifest.json` carries the ground-truth labels.
- DOCX has no real pages, so citations use paragraph numbers or table/row
  numbers. Table rows are extracted as separate units.
- `reference_corpus/` is a small simulated "web": 6 neutral encyclopedia-style
  texts on quick-commerce topics.
- The AI score is a **heuristic indicator**, not a calibrated probability or
  proof of AI authorship (perplexity + burstiness); see EVALUATION.md §3 for
  where it fails. It needs ≥5 sentences of
  ≥5 words, so a very short pitch transcript can report
  `ai_generated_probability: null` with the reason in `ai_details.note`; when it
  does compute, speech scores very low (it looks nothing like AI prose).
- Matching is chunk-level (nearest neighbours); near-duplicate paragraphs
  spanning a chunk boundary are caught by the 1-sentence overlap.

## Scaling (rough estimates)

- **Flow:** upload → object storage (S3) → one queue message per file (Redis +
  Celery, or SQS) → worker pools per stage: extract (CPU) → transcribe (GPU) →
  embed (GPU batch) → index upsert → score → report. Each job is keyed by the
  file's SHA-256 so retries are safe and exact duplicates are caught before any
  ML runs.
- **Compute:** the heaviest stage is ASR: 10k pitches × 3 min ≈ 500 h of audio;
  faster-whisper `small` on a T4 runs at roughly 30–60× real time ≈ 10–15
  GPU-hours (estimate; spot instances are cheap). 10k docs × ~40 chunks =
  ~400k chunks; MiniLM on GPU embeds thousands/s → a few minutes.
- **Vector DB:** 400k × 384 floats ≈ 0.6 GB — a flat index still fits, but
  all-vs-all search is O(n²); production should use HNSW (Qdrant or pgvector,
  M=16, ef_search≈128) for ~95–99% recall at millisecond queries, partitioned by
  `challenge_id` (files compared only within one competition + the reference
  corpus). IVF-PQ only past ~10M chunks.
- **Boilerplate at scale:** organisers upload the official prompt once; it is
  embedded and auto-excluded. The "appears in ≥ N files" rule scales N with
  corpus size (e.g. 1%).
- **Bottlenecks:** (1) ASR GPU time, (2) the perplexity model (could become a
  small fine-tuned classifier), (3) report generation is trivial. Target:
  report within ~2 min of upload; full cross-check re-runs after the deadline.
- **Reliability:** dead-letter queue for corrupt files, per-stage retries,
  metrics on skip rate and queue depth.
- **Cost (rough):** tens of dollars of spot GPU per 10k-submission challenge;
  storage negligible.

## AI-Assisted Development & Disclosure

AI tools were incorporated into my development workflow at different stages of this project. I used ChatGPT, GitHub Copilot, Claude, and GLM (where applicable) to assist with architectural exploration, implementation planning, code suggestions, boilerplate generation, debugging, troubleshooting, and documentation.

I believe AI-assisted development is an important part of modern software engineering. These tools helped me explore alternative approaches, accelerate repetitive tasks, and work through technical challenges more efficiently.

However, AI-generated suggestions were not treated as final implementations. I reviewed and adapted the suggestions, made the final engineering decisions, integrated the components, and validated the system through testing against the assessment scenarios.

I take responsibility for the overall implementation, system integration, evaluation methodology, technical decisions, and final submission.
