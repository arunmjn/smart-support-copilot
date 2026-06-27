import re
from groq import Groq

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

def _rule_classify(query):
    q = query.lower()
    if any(re.search(p, q) for p in COMPARISON_KW):     return "comparison"
    if any(re.search(p, q) for p in TROUBLESHOOTING_KW): return "troubleshooting"
    return None

def _llm_classify(query, chat_history, api_key):
    try:
        client  = Groq(api_key=api_key)
        history = "\n".join(f"{m['role'].upper()}: {m['content']}"
                            for m in (chat_history or [])[-3:])
        prompt  = f"""Classify this Samsung support query into exactly one word:
troubleshooting, comparison, or general.

Chat history: {history or 'None'}
Query: {query}

Reply with one word only."""
        resp = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        result = resp.choices[0].message.content.strip().lower()
        if result in ("troubleshooting", "comparison", "general"):
            return result
    except Exception:
        pass
    return "general"

def classify_query(query, chat_history, api_key):
    return _rule_classify(query) or _llm_classify(query, chat_history, api_key)
