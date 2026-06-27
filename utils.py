def format_chat_history(history, max_turns=6):
    if not history: return ""
    lines = []
    for msg in history[-max_turns:]:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:400] + "…" if msg["role"] == "assistant" and len(msg["content"]) > 400 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
