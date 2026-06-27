def format_chat_history(history: list, max_turns: int = 6) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-max_turns:]:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if role == "Assistant" and len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
