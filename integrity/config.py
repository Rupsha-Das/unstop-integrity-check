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
AI_PPL_LOW, AI_PPL_HIGH = 120.0, 400.0   # calibrated on warm-context sentence ppl (was 20/80)
AI_CV_LOW, AI_CV_HIGH = 0.3, 0.9         # cold-start-free sentence ppl CV (was 0.2/0.8)
AI_MIN_SENTENCES = 5

RISK_HIGH = {"peer": 40.0, "external": 40.0, "ai": 80.0}
RISK_MEDIUM = {"peer": 15.0, "external": 15.0, "ai": 50.0}
