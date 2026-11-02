"""Perplexity / burstiness AI-likelihood score (SPEC §5.5)."""
import logging
import math
import re

import integrity.config as C

logger = logging.getLogger(__name__)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_lm = None


def _get_lm():
    global _lm
    if _lm is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(C.LM_MODEL)
        lm = AutoModelForCausalLM.from_pretrained(C.LM_MODEL)
        _lm = (tok, lm.eval())
    return _lm


def _sentence_ppls(sentences):
    """exp(loss) per sentence, conditioned on the previous sentence.

    Cold-start scoring (each sentence alone) inflates ppl 10x because the
    first tokens of every sentence are unpredictable with no context.
    We therefore prompt distilgpt2 with the previous sentence and mask the
    prefix out of the loss.
    """
    import torch
    tok, lm = _get_lm()
    ppls, ctx = [], ""
    for s in sentences:
        full = (ctx + " " + s) if ctx else s
        ids = tok(full, return_tensors="pt").input_ids
        labels = ids.clone()
        if ctx:
            prefix = tok(ctx, add_special_tokens=False).input_ids
            L = 0
            a = ids[0].tolist()
            while L < len(prefix) and L < len(a) and prefix[L] == a[L]:
                L += 1
            labels[0, :max(L, 1)] = -100
        with torch.no_grad():
            out = lm(input_ids=ids, labels=labels)
        ppls.append(math.exp(out.loss.item()))
        ctx = s
    return ppls


def ai_score(text):
    sentences = [s for s in SENT_SPLIT_RE.split(text) if len(s.split()) >= 5][:40]
    if len(sentences) < C.AI_MIN_SENTENCES:
        return {"probability": None, "perplexity": None, "burstiness": None,
                "note": "too little text"}
    try:
        ppls = _sentence_ppls(sentences)
    except Exception as e:
        logger.warning("ai_score: %s", e)
        return {"probability": None, "perplexity": None, "burstiness": None,
                "note": f"scoring failed: {str(e)[:120]}"}
    P = sum(ppls) / len(ppls)
    mean = P
    CV = (sum((p - mean) ** 2 for p in ppls) / len(ppls)) ** 0.5 / mean
    ppl_part = max(0.0, min(1.0, (C.AI_PPL_HIGH - P) / (C.AI_PPL_HIGH - C.AI_PPL_LOW)))
    burst_part = max(0.0, min(1.0, (C.AI_CV_HIGH - CV) / (C.AI_CV_HIGH - C.AI_CV_LOW)))
    probability = round(100 * (0.6 * ppl_part + 0.4 * burst_part), 1)
    return {"probability": probability, "perplexity": round(P, 1),
            "burstiness": round(CV, 2), "note": ""}


def doc_text(doc):
    return " ".join(u["text"] for u in doc["units"])


if __name__ == "__main__":
    from integrity.extract import extract_folder
    import json
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    docs = extract_folder("sample_submissions")
    for d in docs:
        s = ai_score(doc_text(d))
        print(f"{d['doc_id']} | {s['perplexity']} | {s['burstiness']} | {s['probability']} | {s['note']}")
