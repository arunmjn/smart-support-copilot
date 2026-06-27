import streamlit as st
from rag import build_vectorstore, retrieve_context
from classifier import classify_query
from responder import generate_response
from utils import format_chat_history

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Support Copilot",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .query-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 6px;
    }
    .badge-troubleshooting { background: #ffe0e0; color: #c0392b; }
    .badge-comparison      { background: #e0f0ff; color: #1a6eb5; }
    .badge-general         { background: #e0ffe0; color: #1e8c45; }
    .source-box {
        background: #f8f9fa;
        border-left: 3px solid #6c757d;
        padding: 6px 12px;
        font-size: 12px;
        color: #555;
        margin-top: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "vectorstore"   not in st.session_state: st.session_state.vectorstore   = None
if "doc_loaded"    not in st.session_state: st.session_state.doc_loaded    = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/24/Samsung_Logo.svg", width=160)
    st.title("⚙️ Configuration")

    api_key = st.text_input("🔑 Gemini API Key", type="password",
                            help="Get free key at aistudio.google.com")
    if api_key:
        st.session_state["GEMINI_API_KEY"] = api_key
        st.success("✅ API Key set!")

    st.divider()
    st.markdown("### 📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF / TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("🔄 Processing documents…"):
            vs = build_vectorstore(uploaded_files)
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.doc_loaded  = True
                st.success(f"✅ {len(uploaded_files)} document(s) loaded!")

    # Auto-load bundled galaxy.txt
    if not st.session_state.doc_loaded:
        try:
            import io
            with open("galaxy.txt", "rb") as f:
                fake = io.BytesIO(f.read())
                fake.name = "galaxy.txt"
                vs = build_vectorstore([fake])
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.doc_loaded  = True
        except FileNotFoundError:
            pass

    st.divider()
    st.markdown("### 🏷️ Query Types")
    st.markdown("🔴 **Troubleshooting** — Step-by-step fix")
    st.markdown("🔵 **Comparison** — Feature table")
    st.markdown("🟢 **General** — Direct answer")
    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.doc_loaded:
        st.success("📄 Knowledge base ready")
    else:
        st.warning("⚠️ Upload a document to enable RAG")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 Smart Support Copilot")
st.caption("Powered by Google Gemini · RAG · Dynamic Response Routing")

if "GEMINI_API_KEY" not in st.session_state:
    st.warning("👈 Enter your **free** Gemini API key in the sidebar to get started.")
    st.markdown("""
    **How to get a free Gemini API key (2 minutes):**
    1. Go to [aistudio.google.com](https://aistudio.google.com)
    2. Sign in with your Gmail
    3. Click **Get API Key** → **Create API Key**
    4. Paste it in the sidebar ☝️
    """)

st.divider()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            qtype = msg.get("query_type", "general")
            badge = {"troubleshooting":"badge-troubleshooting",
                     "comparison":"badge-comparison",
                     "general":"badge-general"}.get(qtype,"badge-general")
            st.markdown(f'<span class="query-badge {badge}">🏷 {qtype.upper()}</span>',
                        unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("source"):
            st.markdown(f'<div class="source-box">📌 {msg["source"]}</div>',
                        unsafe_allow_html=True)

# ── Sample query buttons ──────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
samples = ["Why is my Galaxy phone overheating?",
           "Compare Galaxy S23 vs S24",
           "How to reset Samsung Smart TV?"]
for col, q in zip([c1, c2, c3], samples):
    if col.button(q, use_container_width=True):
        st.session_state["prefill"] = q
        st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask anything about your Samsung device…") or prefill

if user_input:
    if "GEMINI_API_KEY" not in st.session_state:
        st.error("Please enter your Gemini API key in the sidebar first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            query_type = classify_query(user_input, st.session_state.chat_history,
                                        st.session_state.get("GEMINI_API_KEY",""))

            context, source = "", "General Knowledge"
            if st.session_state.vectorstore:
                context, source = retrieve_context(st.session_state.vectorstore, user_input)

            history_text = format_chat_history(st.session_state.chat_history[:-1])
            response = generate_response(
                query      = user_input,
                query_type = query_type,
                context    = context,
                chat_history = history_text,
                api_key    = st.session_state.get("GEMINI_API_KEY",""),
            )

        badge = {"troubleshooting":"badge-troubleshooting",
                 "comparison":"badge-comparison",
                 "general":"badge-general"}.get(query_type,"badge-general")
        st.markdown(f'<span class="query-badge {badge}">🏷 {query_type.upper()}</span>',
                    unsafe_allow_html=True)
        st.markdown(response)
        st.markdown(f'<div class="source-box">📌 {source}</div>', unsafe_allow_html=True)

    st.session_state.chat_history.append({
        "role": "assistant", "content": response,
        "query_type": query_type, "source": source,
    })
