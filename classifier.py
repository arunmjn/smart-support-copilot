"""
classifier.py – Classify query into troubleshooting | comparison | general.
Uses keyword rules first; falls back to Gemini if ambiguous.
"""
import re
import google.generativeai as genai

TROUBLESHOOTING_KW = [
    r"\bfix\b", r"\brepair\b", r"\berror\b", r"\bissue\b", r"\bproblem\b",
    r"\bnot working\b", r"\boverheating\b", r"\bbattery drain\b", r"\bcrash\b",
    r"\blag\b", r"\bfreez", r"\bslow\b", r"\brestart\b", r"\breset\b",
    r"\bconnect", r"\bwifi\b", r"\bbluetooth\b", r"\bsim\b", r"\bcharge\b",
    r"\bhow (do|to|can)\b", r"\bwhy (is|does|isn't|doesn't)\b",
    r"\bscreen\b", r"\bupdate\b", r"\bswelling\b", r"\bboot loop\b",
]

COMPARISON_KW = [
    r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b",
    r"\bbetter\b", r"\bwhich (one|phone|device|is)\b",
    r"\bs23.+s24\b", r"\bs24.+s23\b", r"\bspec\b",
]


def _rule_classify(query: str):
    q = query.lower()
    if sum(1 for p in COMPARISON_KW     if re.search(p, q)) >= 1:
        return "comparison"
    if sum(1 for p in TROUBLESHOOTING_KW if re.search(p, q)) >= 1:
        return "troubleshooting"
    return None


def _gemini_classify(query: str, chat_history: list, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-2.0-flash")
        history_snippet = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in (chat_history or [])[-3:]
        )
        prompt = f"""Classify this Samsung support query into exactly one category:
- troubleshooting
- comparison
- general

Chat history:
{history_snippet or 'None'}

Query: {query}

Reply with only one word."""
        resp = model.generate_content(prompt)
        result = resp.text.strip().lower()
        if result in ("troubleshooting", "comparison", "general"):
            return result
    except Exception:
        pass
    return "general"


def classify_query(query: str, chat_history: list, api_key: str) -> str:
    return _rule_classify(query) or _gemini_classify(query, chat_history, api_key)
