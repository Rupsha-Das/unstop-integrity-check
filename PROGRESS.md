# PROGRESS

STATUS: next=6.1 | blocked=no

Rules: one step → run V → matches E? tick + 1 log line : fix (3 fails → STOP-STUCK). Commit after each phase. `R` = `uv run`.

## Steps

### 0 Setup
- [x] 0.1 `uv --version` (missing → STOP-UV). `git init`, create `pyproject.toml` (SPEC §2), `integrity/config.py` (SPEC §3), `.gitignore` (`.venv/ __pycache__/ *.mp3`). `uv sync`.
  V: `R python -c "import fitz,docx,reportlab,faster_whisper,sentence_transformers,faiss,transformers,gtts,imageio_ffmpeg,av,torch;print('ok',torch.__version__)"`  E: `ok 2.x+cpu`

### 1 Dataset (SPEC §6)
- [x] 1.1 `reference_corpus/` 6 txt.  V: `wc -w reference_corpus/*`  E: 6 files, 120–250 words each
- [x] 1.2 `build_dataset.py` (texts inside + builder + manifest). `R build_dataset.py` (gTTS error → STOP-NET).
  V: `ls -la sample_submissions; R python -c "import av,glob;[print(f,round(av.open(f).duration/1e6),bool(av.open(f).streams.audio)) for f in sorted(glob.glob('sample_submissions/*.mp4'))]"`
  E: 13 files + manifest; sub10 = 0 bytes; pitch01/02 10–35 s True; pitch03 ~10 s False
- [x] 1.3 V: `R python -c "import fitz;d=fitz.open('sample_submissions/sub01_alpha.pdf');print(d.page_count,'Jaipur' in d[1].get_text())"`  E: `2 True` (or more pages)

### 2 Extract + chunk (SPEC §5.1–5.2)
- [x] 2.1 extract part.  V: `R python -m integrity.extract sample_submissions`  E: 11 ok; sub10 `empty file`; pitch03 `no audio track`; pitch01 text ≈ CROSSMODAL; no "Confidential"/"Page N" in sub01 text
- [x] 2.2 V: `printf x > /tmp/bad.pdf; R python -c "from integrity.extract import extract_file;print(extract_file('/tmp/bad.pdf')['status'])"`  E: `skipped`, no traceback
- [x] 2.3 chunk part.  V: same as 2.1  E: chunk lines: each ok file ≥1 chunk, max ≤130 words, sample chunk shows page/paragraph/start-end

### 3 Match (SPEC §5.3–5.4)
- [x] 3.1 embed + index.  V: `R python -c "from integrity.match import embed;v=embed(['the cat sat on the mat','a cat was sitting on a rug','stock prices fell']);print(v.shape,round(float(v[0]@v[1]),2),round(float(v[0]@v[2]),2))"`  E: `(3, 384)`, 1st number ≫ 2nd
- [x] 3.2 peer matching + boilerplate.  V: `R python -m integrity.match sample_submissions`
  E: sub02↔03 ≥0.95; sub04↔05 ≥0.80; sub01↔pitch01 ≥0.80; sub07↔09 <0.80; boilerplate ≥1 in sub01–05, 0 elsewhere.
  sub04↔05 low → reword sub05 closer to sub04's arguments, rebuild. Never lower thresholds.
- [x] 3.3 print jaccard of best sub04/05 pair.  E: <0.5. Log the number (goes in EVALUATION.md).
- [x] 3.4 external matching.  V: same as 3.2  E: sub08 ↔ quick_commerce.txt ≥0.9; others none (if not, log it, don't hide it)

### 4 AI (SPEC §5.5)
- [x] 4.1 V: `R python -m integrity.ai_detect sample_submissions`  E: sub06 highest and ≥50; most others <50.
  Fails → you may change only AI_PPL_*/AI_CV_* once; log old→new. Still failing → keep it, it becomes a failure case.

### 5 Report + CLI (SPEC §5.6–5.7)
- [x] 5.1 V: `rm -rf reports; R run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports && ls reports|wc -l`  E: 28, no traceback
- [x] 5.2 V: `R python -c "import json,glob;k={'document_id','total_integrity_risk','scores','peer_matches','multimodal_matches','explanations'};[print(f,k-set(json.load(open(f)))) for f in glob.glob('reports/*.json') if 'summary' not in f]"`  E: all `set()`
- [x] 5.3 V: `cat reports/pitch01_kilo.mp4.md reports/sub03_charlie.docx.md reports/sub07_golf.pdf.md`  E: pitch01 → sub01 with `00:00:..` + page 2; sub03 HIGH near-exact; sub07 LOW, no matches; the template sits under "Ignored" in sub01–05
- [x] 5.4 `tests/test_core.py`: no same-doc match; 3-doc chunk = boilerplate; 0-byte file → skipped.  V: `R pytest -q`  E: `3 passed`

### 6 Eval + docs
- [ ] 6.1 `evaluate.py` (SPEC §7).  V: `R evaluate.py`  E: 3 confusion tables, pairs x/4, template x/5, degraded x/2
- [ ] 6.2 EVALUATION.md: tables, the jaccard-vs-cosine number, ≥2 REAL failures (ideas in SPEC §9), trade-offs (MiniLM small/CPU vs accuracy; Flat now, HNSW at scale)
- [ ] 6.3 README.md: what it does · setup `uv sync` · run (1 cmd) · how it works (SPEC §0) · scores · Assumptions (120-word chunks; both sides of a copy flagged; DOCX uses paragraph no.; reference_corpus = simulated web; AI = heuristic) · Scaling (SPEC §8) · AI-use disclosure ("Planning, boilerplate and debugging helped by AI assistants (Claude, GLM); I reviewed all design decisions.")
- [ ] 6.4 V: `rm -rf reports && R run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports && R evaluate.py && R pytest -q`  E: all pass. Commit → STOP-DONE.

## Decisions
- reportlab 5.x: PageHeader/Footer callbacks go on `doc.build()`, not the constructor (constructor kwarg silently ignored).
- Header/footer rule: drop when key appears on strictly MORE than half the pages (literal ">=50%" would drop every block in 2-page docs).
- av 19 removed `metadata_errors`; faster-whisper 1.2.1 fails to open audio -> we decode to 16 kHz float32 numpy with plain av and pass to transcribe (kept av 19; av<15 has no Windows wheel).
- Sub08 built without TEMPLATE (spec row 8 => body only); QUICK_PARA kept as its own paragraph.
- Whisper transcribe uses beam_size=5 + domain initial_prompt (helps proper nouns); thresholds untouched.
- ai_detect: sentences scored with warm context (previous sentence prompts the LM, prefix masked); cold-start scoring inflated ppl ~10x (250-1400 vs intended 20-80).
- AI calibration (once, old->new): AI_PPL_LOW/HIGH 20/80 -> 120/400; AI_CV_LOW/HIGH 0.2/0.8 -> 0.3/0.9. Result: sub06 76.3 highest; 9 of 10 others <50 (sub08 66.2, realistic: copied encyclopedia paragraph is uniform).
- PDF paragraphs: PyMuPDF 2.0 "blocks" merges whole pages; re-group lines by y-gap (paragraph spaceAfter=12 in generated PDFs, gap threshold 0.5*lineheight+2) to get paragraph units.
- 1-page PDFs keep drawn header as a unit (spec filter only for pages>=2); dropped later by CHUNK_MIN_WORDS.
- pytest needs pythonpath=["."] in pyproject (test module import).
- PyMuPDF 2.0 merges whole reportlab pages into one text block -> PDF granularity = page-level paragraphs; chunking still ≤120 words.

## Log
<!-- one line per step, ≤15 words: `<step> ok|fail: <key result>`. e.g. `3.2 ok: 02-03 .99, 04-05 .84, 01-pitch01 .88` -->
0.1 ok: imports ok, torch 2.15.0+cpu
1.1 ok: 6 txt files, 221-249 words
1.2 ok: 14 files; sub10 0B; pitch01/02 35s audio, pitch03 10s no audio
1.3 ok: sub01 2 pages, Jaipur on page 2
2.1 ok: 11 ok; sub10 empty; pitch03 no audio; whisper text ~CROSSMODAL; headers stripped in sub01
2.2 ok: bad pdf -> skipped, warning only
2.3 ok: chunks 1-7/doc, max 119 words, locales page/para/start-end
3.1 ok: (3,384), cos 0.74 vs 0.09
3.2 ok: 02-03 1.00, 04-05 1.00, 01-pitch01 0.98, 07-09 0.58; boilerplate 1-2 only in sub01-05
3.3 ok: best 04-05 body pair .95 cos / jaccard 0.48
3.4 ok: sub08-quick_commerce 0.90, no other external
4.1 ok: sub06 76.3 top + >=50; 9/10 others <50
5.1 ok: 28 files, no traceback
5.2 ok: all json report keys set()
5.3 ok: pitch01->sub01 page2+ts; sub03 HIGH exact-copy; sub07 LOW; template Ignored
5.4 ok: pytest 3 passed

## STOP messages (send to the user, then wait)
- **STOP-UV:** "Please install uv, then reply 'done': `curl -LsSf https://astral.sh/uv/install.sh | sh` (Arch: `sudo pacman -S uv`)"
- **STOP-NET:** "Making the videos needs internet (Google text-to-speech). Error: `<err>`. Check your connection and reply 'retry'."
- **STOP-STUCK:** "Stuck on `<step>`: `<problem>`. Tried `<what>`. Can you check, or reply 'skip'?"
- **STOP-DONE:** "All built and checked. 1) Create a private GitHub repo. 2) `git remote add origin <url> && git push -u origin main`. 3) Give the evaluator access and submit the URL + `git rev-parse HEAD`. 4) Read README.md and EVALUATION.md before the interview."
