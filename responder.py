from groq import Groq

SYSTEM = "You are a Smart Support Copilot for Samsung electronics. Answer using the document context provided. Use Markdown formatting."

def _troubleshooting_prompt(query, context, history):
    return f"""DOCUMENT CONTEXT:\n{context or 'None'}\n\nCHAT HISTORY:\n{history or 'None'}\n\nUSER QUERY: {query}

Respond in this EXACT Markdown format:

## 🔍 Possible Causes
- cause 1
- cause 2

## 🛠️ Step-by-Step Solution
1. Step one
2. Step two

## ⚠️ When to Escalate
- escalation condition

> 📌 Based on available documentation."""

def _comparison_prompt(query, context, history):
    return f"""DOCUMENT CONTEXT:\n{context or 'None'}\n\nCHAT HISTORY:\n{history or 'None'}\n\nUSER QUERY: {query}

Respond in this EXACT Markdown format:

## 📊 Feature Comparison

| Feature | Model A | Model B |
|---------|---------|---------|
| row     | val     | val     |

## 🔑 Key Differences
- difference 1

## ✅ Recommendation
Who should buy which model.

> 📌 Based on available documentation."""

def _general_prompt(query, context, history):
    return f"""DOCUMENT CONTEXT:\n{context or 'None'}\n\nCHAT HISTORY:\n{history or 'None'}\n\nUSER QUERY: {query}

Respond in this EXACT Markdown format:

## 💡 Answer
Direct answer here.

## 📖 Explanation
Brief explanation.

## 📝 Additional Notes
- extra tip

> 📌 Based on available documentation."""

PROMPT_MAP = {
    "troubleshooting": _troubleshooting_prompt,
    "comparison":      _comparison_prompt,
    "general":         _general_prompt,
}

def generate_response(query, query_type, context, chat_history, api_key):
    prompt = PROMPT_MAP.get(query_type, _general_prompt)(query, context, chat_history)
    try:
        client = Groq(api_key=api_key)
        resp   = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"
