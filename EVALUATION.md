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
including the near-miss pair sub07/sub09 (best cosine 0.53, threshold 0.80).

## 2. Jaccard vs cosine (the point of embedding search)

For the best sub04/05 chunk pair: cosine similarity **0.95** but word-overlap
Jaccard only **0.48** — the paraphrase ("Tier-2 expansion changes the cost line
rather than the revenue line…" vs "What expansion into tier-2 cities alters is
the cost line, not the revenue line…") shares under half its word forms, below
the 0.6 "near-exact" cut-off. A plagiarism checker relying on word overlap would
miss this; MiniLM embeddings catch it. (The template chunks do score Jaccard
1.0, but they are identified as boilerplate and excluded from scoring.)

## 3. Real failures / known limitations (confirmed on this run)

1. **AI detector is a weak heuristic.** Final precision on the `ai` signal is
   0.5 → in the confusion table 1 FP: `sub08_hotel.docx` scores "AI likelihood
   66%". The cause is real and instructive: sub08 contains the verbatim
   encyclopedia paragraph copied from the reference corpus, and neutral factual
   prose has exactly the properties of AI text: low perplexity (187) and low
   burstiness (0.59). Deciding "AI-written vs copied-from-an-encyclopedia" needs
   source attribution first, style second. Fix: run detection only on passages
   that pass the source checks, or use a fine-tuned detector calibrated on
   labelled data.
2. **Speech transcription quality.** Whisper `tiny.en` garbles phrases even with
   the domain `initial_prompt`: "GreenCart reuses" is heard as "GreenCartry
   uses" (Indore/Coimbatore/Jaipur *are* recovered by the prompt). It also
   flips commas into sentence breaks between runs, which is why the pitch01 AI
   score comes back `null` (3 sentences) on one run and `0.0` (5 sentences) on
   the next — an ambiguous result, though peer/external scores and every
   confusion-table number are unaffected. The embeddings still matched
   (cross-modal cosine 0.98), but on noisy or accented uploads a match sitting
   near the 0.80 cut-off could degrade below it. Fix: larger Whisper model +
   number/proper-noun normalisation and pinned decoding settings for
   reproducible transcripts.
3. **Chunk-level miss inside a detected paraphrase pair.** sub04/sub05 are
   correctly flagged as a pair (best cosine 0.95), but their opening paragraphs
   ("We believe GreenCart should expand now…" vs "The case for moving now is the
   one we intend to defend…") embed at only **0.67** — below the 0.80 cut-off, so
   sub04 scores 80% peer instead of 100%, and a checker judging per-chunk would
   report a false negative on that paragraph (its highest neighbour is actually
   the shared template at 0.70). Fix: align chunks to paragraph pairs before
   embedding, or lower the threshold with a second-stage re-rank to hold
   precision.
4. **Threshold overfitting.** 0.80 / 0.85 / "≥3 files" were tuned by hand on 13
   files and on this exact writing style; on a different corpus the same numbers
   will not transfer (sub02+sub03 vs everything else is close in the matrix —
   original human texts on one narrow topic land at 0.52–0.70, which is not a
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

## 5. Explainability update

The JSON and Markdown reports now retain every detected query/source passage
pair rather than only the highest-scoring example for each matched file. Each
entry includes the passage text, similarity score, and available page,
paragraph, or video timestamp metadata. This makes multi-passage matches
reviewable directly from the report. The detector can still miss individual
paraphrased passages below the configured similarity threshold (failure #3).
