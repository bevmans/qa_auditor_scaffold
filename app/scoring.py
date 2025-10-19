import os, json, re
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Very compact instruction so we always get clean JSON back.
_SYSTEM = (
    "You are a strict QA rater for customer support messages. "
    "Return ONLY a single JSON object with keys empathy, accuracy, tone, resolution (0-10 floats) "
    "and reasoning (short string). No extra text."
)

def _extract_json(text: str) -> dict:
    """Best-effort: find the first {...} block and parse it."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    chunk = m.group(0) if m else text
    try:
        return json.loads(chunk)
    except Exception:
        # Try to coerce common mistakes (single quotes, trailing commas)
        chunk = chunk.replace("'", '"')
        chunk = re.sub(r",\s*([}\]])", r"\1", chunk)
        try:
            return json.loads(chunk)
        except Exception:
            return {}

def _safe_score(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0: return 0.0
    if v > 10: return 10.0
    return v

def score_message(message_text: str) -> dict:
    """
    Returns: { empathy, accuracy, tone, resolution, reasoning }
    All scores are floats 0..10. On failure, returns zeros with a brief reasoning.
    """
    if not OPENAI_API_KEY:
        return {
            "empathy": 0.0, "accuracy": 0.0, "tone": 0.0, "resolution": 0.0,
            "reasoning": "No OPENAI_API_KEY set."
        }

    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": message_text or ""},
        ],
        "temperature": 0,
        "max_tokens": 200,
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        if r.status_code >= 400:
            # Surface why it failed in worker logs but don’t crash the job
            print(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
            return {
                "empathy": 0.0, "accuracy": 0.0, "tone": 0.0, "resolution": 0.0,
                "reasoning": f"OpenAI error {r.status_code}"
            }
        data = r.json()
        content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
        parsed = _extract_json(content)

        e = _safe_score(parsed.get("empathy"))
        a = _safe_score(parsed.get("accuracy"))
        t = _safe_score(parsed.get("tone"))
        rr = _safe_score(parsed.get("resolution"))
        why = parsed.get("reasoning") or "Scored."

        return {"empathy": e, "accuracy": a, "tone": t, "resolution": rr, "reasoning": why}

    except Exception as ex:
        print("OpenAI scoring exception:", repr(ex))
        return {
            "empathy": 0.0, "accuracy": 0.0, "tone": 0.0, "resolution": 0.0,
            "reasoning": "Scoring failed (exception)."
        }
