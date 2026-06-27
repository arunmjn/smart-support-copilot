"""
responder.py – Generate structured responses using Google Gemini.
"""
import google.generativeai as genai

SYSTEM_NOTE = """You are a Smart Support Copilot for Samsung consumer electronics.
Answer based on the provided document context. Use Markdown formatting."""


def _troubleshooting_prompt(query, context, history):
    return f"""{SYSTEM_NOTE}

DOCUMENT CONTEXT:
{context or 'No specific document context available.'}

CHAT HISTORY:
{history or 'None'}

USER QUERY: {query}

Respond STRICTLY in this Markdown format:

## 🔍 Possible Causes
- List the likely causes

## 🛠️ Step-by-Step Solution
1. Step one
2. Step two

## ⚠️ When to Escalate
- Conditions requiring professional support

> 📌 Based on available documentation."""


def _comparison_prompt(query, context, history):
    return f"""{SYSTEM_NOTE}

DOCUMENT CONTEXT:
{context or 'No specific document context available.'}

CHAT HISTORY:
{history or 'None'}

USER QUERY: {query}

Respond STRICTLY in this Markdown format:

## 📊 Feature Comparison

| Feature | Model A | Model B |
|---------|---------|---------|
| ...     | ...     | ...     |

## 🔑 Key Differences
- Difference 1
- Difference 2

## ✅ Recommendation
Which device suits which user type.

> 📌 Based on available documentation."""


def _general_prompt(query, context, history):
    return f"""{SYSTEM_NOTE}

DOCUMENT CONTEXT:
{context or 'No specific document context available.'}

CHAT HISTORY:
{history or 'None'}

USER QUERY: {query}

Respond STRICTLY in this Markdown format:

## 💡 Answer
Clear, direct answer.

## 📖 Explanation
Brief explanation.

## 📝 Additional Notes
- Extra tips or caveats

> 📌 Based on available documentation."""


PROMPT_MAP = {
    "troubleshooting": _troubleshooting_prompt,
    "comparison":      _comparison_prompt,
    "general":         _general_prompt,
}


def generate_response(query, query_type, context, chat_history, api_key):
    prompt_fn = PROMPT_MAP.get(query_type, _general_prompt)
    prompt    = prompt_fn(query, context, chat_history)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp  = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"
