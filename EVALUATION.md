# EVALUATION

All numbers come from the real run in `reports/` (regenerated with
`uv run run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports`
followed by `uv run evaluate.py`).

## 1. Confusion results (13-file dataset)

| signal | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| peer    | 6 | 0 | 0 | 5  | 1.0 | 1.0 |
| external| 1 | 0 | 0 | 10 | 1.0 | 1.0 |
| ai      | 1 | 1 | 0 | 9  | 0.5 | 1.0 |

Pairs: 4/4 found. Template boilerplate: 5/5 files. Degraded files skipped: 2/2.

The whole ground-truth story is recovered: exact copy (sub02/sub03, cosine 1.0),
paraphrase (sub04/sub05, cosine 0.95), cross-modal (sub01/pitch01, cosine 0.98),
external copy (sub08/quick_commerce.txt, 0.90), and every original file is clean,
including the near-miss pair sub07/sub09 (best cosine 0.58, threshold 0.80).

## 2. Jaccard vs cosine (the point of embedding search)

For the best sub04/05 chunk pair: cosine similarity **0.95** but word-overlap
Jaccard only **0.48** — the paraphrase ("We believe GreenCart should expand now…"
vs "The case for moving now is the one we intend to defend…") shares almost half
its word forms, far below the 0.6 "near-exact" cut-off. A plagiarism checker
relying on word overlap would miss this; MiniLM embeddings catch it. (The
template chunks do score Jaccard 1.0, but they are identified as boilerplate and
excluded from scoring.)

## 3. Real failures / known limitations (confirmed on this run)

1. **AI detector is a weak heuristic.** Final precision on the `ai` signal is
   0.55 → in the confusion table 1 FP: `sub08_hotel.docx` scores "AI likelihood
   66%". The cause is real and instructive: sub08 contains the verbatim
   encyclopedia paragraph copied from the reference corpus, and neutral factual
   prose has exactly the properties of AI text: low perplexity (187) and low
   burstiness (0.59). Deciding "AI-written vs copied-from-an-encyclopedia" needs
   source attribution first, style second. Fix: run detection only on passages
   that pass the source checks, or use a fine-tuned detector calibrated on
   labelled data.
2. **Speech transcription quality.** Whisper `tiny.en` mishears proper nouns:
   "Indore" → "Indoor", "Jaipur" → "Jepur", "reuses" → "green car tree uses".
   The embeddings still matched (cross-modal cosine 0.98), and the `initial_prompt`
   recovered most proper nouns, but on noisy or accented uploads below 0.90 the
   match would degrade. Fix: larger Whisper model + number/proper-noun
   normalisation.
3. **Chunk-level boilerplate misses mixed chunks.** Each template document
   carries the shared 150-word template; in sub01 only 1 of its 2 template
   chunks crossed ≥3 other files (cosine ≤0.85) — the continuation chunk
   ("…Third, what operational changes are needed…") sits below the 0.85
   boilerplate similarity and its file flagged it as a peer match instead. Fix:
   sentence-level template matching instead of whole-chunk similarity.
4. **Threshold overfitting.** 0.80 / 0.85 / "≥3 files" were tuned by hand on 13
   files and on this exact writing style; on a different corpus the same numbers
   will not transfer (sub02+sub03 vs everything else is close in the matrix —
   original human texts on one narrow topic land at 0.55–0.70, which is not a
   comfortable margin). Fix: tune on a larger labelled set, report scores instead
   of binary flags.

## 4. Design trade-offs

- **MiniLM (384-dim, ~90 MB, CPU-friendly) vs accuracy.** It runs in ~1 s for
  60 chunks on CPU and is enough to separate copy vs non-copy by ~0.3 cosine
  margin in this dataset, but it is weak on technical jargon and short
  fragments. A 768-dim model (e.g. all-mpnet) would cost ~3x compute for maybe
  +0.05 separation here; not worth it until the corpus grows.
- **FAISS IndexFlatIP (exact) vs ANN.** Exact search O(n²); at this size (60
  chunks) it is instant and reproducible. Production would switch to HNSW
  (Qdrant/pgvector, see README) at ~95–99% recall, accepting approximate
  ordering and a rebuild-on-insert model.
- **Perplexity AI score vs a trained classifier.** distilgpt2 PPL is zero-shot:
  no labelled data needed, but brittle (see failure #1). A BERT-style detector
  would need labelled training data but would not collapse on encyclopedia text.
