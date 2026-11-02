# PROGRESS

STATUS: next=1.1 | blocked=no

Rules: one step → run V → matches E? tick + 1 log line : fix (3 fails → STOP-STUCK). Commit after each phase. `R` = `uv run`.

## Steps

### 0 Setup
- [x] 0.1 `uv --version` (missing → STOP-UV). `git init`, create `pyproject.toml` (SPEC §2), `integrity/config.py` (SPEC §3), `.gitignore` (`.venv/ __pycache__/ *.mp3`). `uv sync`.
  V: `R python -c "import fitz,docx,reportlab,faster_whisper,sentence_transformers,faiss,transformers,gtts,imageio_ffmpeg,av,torch;print('ok',torch.__version__)"`  E: `ok 2.x+cpu`

### 1 Dataset (SPEC §6)
- [ ] 1.1 `reference_corpus/` 6 txt.  V: `wc -w reference_corpus/*`  E: 6 files, 120–250 words each
- [ ] 1.2 `build_dataset.py` (texts inside + builder + manifest). `R build_dataset.py` (gTTS error → STOP-NET).
  V: `ls -la sample_submissions; R python -c "import av,glob;[print(f,round(av.open(f).duration/1e6),bool(av.open(f).streams.audio)) for f in sorted(glob.glob('sample_submissions/*.mp4'))]"`
  E: 13 files + manifest; sub10 = 0 bytes; pitch01/02 10–35 s True; pitch03 ~10 s False
- [ ] 1.3 V: `R python -c "import fitz;d=fitz.open('sample_submissions/sub01_alpha.pdf');print(d.page_count,'Jaipur' in d[1].get_text())"`  E: `2 True` (or more pages)

### 2 Extract + chunk (SPEC §5.1–5.2)
- [ ] 2.1 extract part.  V: `R python -m integrity.extract sample_submissions`  E: 11 ok; sub10 `empty file`; pitch03 `no audio track`; pitch01 text ≈ CROSSMODAL; no "Confidential"/"Page N" in sub01 text
- [ ] 2.2 V: `printf x > /tmp/bad.pdf; R python -c "from integrity.extract import extract_file;print(extract_file('/tmp/bad.pdf')['status'])"`  E: `skipped`, no traceback
- [ ] 2.3 chunk part.  V: same as 2.1  E: chunk lines: each ok file ≥1 chunk, max ≤130 words, sample chunk shows page/paragraph/start-end

### 3 Match (SPEC §5.3–5.4)
- [ ] 3.1 embed + index.  V: `R python -c "from integrity.match import embed;v=embed(['the cat sat on the mat','a cat was sitting on a rug','stock prices fell']);print(v.shape,round(float(v[0]@v[1]),2),round(float(v[0]@v[2]),2))"`  E: `(3, 384)`, 1st number ≫ 2nd
- [ ] 3.2 peer matching + boilerplate.  V: `R python -m integrity.match sample_submissions`
  E: sub02↔03 ≥0.95; sub04↔05 ≥0.80; sub01↔pitch01 ≥0.80; sub07↔09 <0.80; boilerplate ≥1 in sub01–05, 0 elsewhere.
  sub04↔05 low → reword sub05 closer to sub04's arguments, rebuild. Never lower thresholds.
- [ ] 3.3 print jaccard of best sub04/05 pair.  E: <0.5. Log the number (goes in EVALUATION.md).
- [ ] 3.4 external matching.  V: same as 3.2  E: sub08 ↔ quick_commerce.txt ≥0.9; others none (if not, log it, don't hide it)

### 4 AI (SPEC §5.5)
- [ ] 4.1 V: `R python -m integrity.ai_detect sample_submissions`  E: sub06 highest and ≥50; most others <50.
  Fails → you may change only AI_PPL_*/AI_CV_* once; log old→new. Still failing → keep it, it becomes a failure case.

### 5 Report + CLI (SPEC §5.6–5.7)
- [ ] 5.1 V: `rm -rf reports; R run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports && ls reports|wc -l`  E: 28, no traceback
- [ ] 5.2 V: `R python -c "import json,glob;k={'document_id','total_integrity_risk','scores','peer_matches','multimodal_matches','explanations'};[print(f,k-set(json.load(open(f)))) for f in glob.glob('reports/*.json') if 'summary' not in f]"`  E: all `set()`
- [ ] 5.3 V: `cat reports/pitch01_kilo.mp4.md reports/sub03_charlie.docx.md reports/sub07_golf.pdf.md`  E: pitch01 → sub01 with `00:00:..` + page 2; sub03 HIGH near-exact; sub07 LOW, no matches; the template sits under "Ignored" in sub01–05
- [ ] 5.4 `tests/test_core.py`: no same-doc match; 3-doc chunk = boilerplate; 0-byte file → skipped.  V: `R pytest -q`  E: `3 passed`

### 6 Eval + docs
- [ ] 6.1 `evaluate.py` (SPEC §7).  V: `R evaluate.py`  E: 3 confusion tables, pairs x/4, template x/5, degraded x/2
- [ ] 6.2 EVALUATION.md: tables, the jaccard-vs-cosine number, ≥2 REAL failures (ideas in SPEC §9), trade-offs (MiniLM small/CPU vs accuracy; Flat now, HNSW at scale)
- [ ] 6.3 README.md: what it does · setup `uv sync` · run (1 cmd) · how it works (SPEC §0) · scores · Assumptions (120-word chunks; both sides of a copy flagged; DOCX uses paragraph no.; reference_corpus = simulated web; AI = heuristic) · Scaling (SPEC §8) · AI-use disclosure ("Planning, boilerplate and debugging helped by AI assistants (Claude, GLM); I reviewed all design decisions.")
- [ ] 6.4 V: `rm -rf reports && R run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports && R evaluate.py && R pytest -q`  E: all pass. Commit → STOP-DONE.

## Decisions
-

## Log
<!-- one line per step, ≤15 words: `<step> ok|fail: <key result>`. e.g. `3.2 ok: 02-03 .99, 04-05 .84, 01-pitch01 .88` -->
0.1 ok: imports ok, torch 2.15.0+cpu

## STOP messages (send to the user, then wait)
- **STOP-UV:** "Please install uv, then reply 'done': `curl -LsSf https://astral.sh/uv/install.sh | sh` (Arch: `sudo pacman -S uv`)"
- **STOP-NET:** "Making the videos needs internet (Google text-to-speech). Error: `<err>`. Check your connection and reply 'retry'."
- **STOP-STUCK:** "Stuck on `<step>`: `<problem>`. Tried `<what>`. Can you check, or reply 'skip'?"
- **STOP-DONE:** "All built and checked. 1) Create a private GitHub repo. 2) `git remote add origin <url> && git push -u origin main`. 3) Give the evaluator access and submit the URL + `git rev-parse HEAD`. 4) Read README.md and EVALUATION.md before the interview."
