import streamlit as st
import requests
import uuid
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Mini-RAG",
    page_icon="📚",
    layout="wide",
)

# ── session state ──────────────────────────────────────────────────────────────
# session_id doubles as project_id — each session is an isolated workspace.
# The user never sees or types a project_id.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Mini-RAG")
    st.caption("Multi-Agent · Multimodal · Memory")
    st.divider()

    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    st.caption(f"Session `{st.session_state.session_id[:8]}...`")

# project_id is derived from session — never exposed to the user
SESSION_ID = st.session_state.session_id
PROJECT_ID = SESSION_ID          # backend uses this as the vector-DB project namespace

# ── file uploader ──────────────────────────────────────────────────────────────
with st.expander("📂 Upload lectures", expanded=True):
    uploaded_files = st.file_uploader(
        "Drop files here — PDFs, images, or text. Multiple files allowed.",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff",
              "txt", "md", "csv", "json", "html"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 5))
        for i, f in enumerate(uploaded_files):
            with cols[i % 5]:
                if f.type == "application/pdf":
                    icon = "📄"
                elif f.type.startswith("image/"):
                    icon = "🖼️"
                else:
                    icon = "📝"
                st.caption(f"{icon} {f.name}")

# ── helpers ────────────────────────────────────────────────────────────────────

def build_files_payload(files):
    """Convert Streamlit UploadedFile list → multipart files list."""
    if not files:
        return []
    return [("files", (f.name, f.getvalue(), f.type)) for f in files]


def stream_query(query: str, files) -> str:
    """
    POST to /api/v1/nlp/multimodal-query/stream.
    Renders streaming tokens into the current Streamlit context.
    Returns the full assembled response text.
    """
    form_data = {
        "query":      query,
        "project_id": PROJECT_ID,
        "session_id": SESSION_ID,
    }
    files_payload = build_files_payload(files)

    placeholder = st.empty()
    full_text   = ""

    try:
        with requests.post(
            f"{API_URL}/v1/nlp/multimodal-query/stream",
            data=form_data,
            files=files_payload or None,
            stream=True,
            timeout=180,
        ) as resp:
            if not resp.ok:
                st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
                return ""
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if line.startswith("data: "):
                    token = line[6:]
                    if token == "[DONE]":
                        break

                    # التعديل هنا: استرجاع الـ Newlines المخبأة عشان التنسيق يظهر صح
                    token = token.replace("\\n", "\n")

                    full_text += token
                    placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as exc:
        st.error(f"Request failed: {exc}")

    return full_text


def fetch_trace_and_sources(query: str, files) -> tuple[list, list]:
    """
    Fire a non-streaming request to get agent_trace and sources_used.
    Returns (agent_trace, sources_used) — both empty lists on failure.
    """
    try:
        resp = requests.post(
            f"{API_URL}/v1/nlp/multimodal-query",
            data={
                "query":      query,
                "project_id": PROJECT_ID,
                "session_id": SESSION_ID,
            },
            files=build_files_payload(files) or None,
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            return data.get("agent_trace", []), data.get("sources_used", [])
    except Exception:
        pass
    return [], []

# ── chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources_used"):
            with st.expander("Sources used", expanded=False):
                for s in msg["sources_used"]:
                    st.markdown(f"- {s}")
        if msg.get("agent_trace"):
            with st.expander("Agent trace", expanded=False):
                for step in msg["agent_trace"]:
                    st.markdown(f"- `{step}`")

# ── chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your documents...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = stream_query(user_input, uploaded_files)

    trace, sources = fetch_trace_and_sources(user_input, uploaded_files)

    st.session_state.chat_history.append({
        "role":        "assistant",
        "content":     answer,
        "sources_used": sources,
        "agent_trace": trace,
    })

# ── document tools ─────────────────────────────────────────────────────────────
st.divider()
col_sum, col_quiz = st.columns(2)

with col_sum:
    if st.button("📋 Summarize lectures", use_container_width=True):
        SUMMARY_QUERY = "summarize"
        st.session_state.chat_history.append({"role": "user", "content": "Summarize lectures"})
        with st.chat_message("user"):
            st.markdown("Summarize lectures")
        with st.chat_message("assistant"):
            with st.spinner("Summarizing..."):
                answer = stream_query(SUMMARY_QUERY, uploaded_files)
        if answer:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Summary**\n\n{answer}",
            })

with col_quiz:
    if st.button("🧠 Generate quiz", use_container_width=True):
        QUIZ_QUERY = "generate quiz"
        st.session_state.chat_history.append({"role": "user", "content": "Generate quiz"})
        with st.chat_message("user"):
            st.markdown("Generate quiz")
        with st.chat_message("assistant"):
            with st.spinner("Generating quiz..."):
                answer = stream_query(QUIZ_QUERY, uploaded_files)
        if answer:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Quiz**\n\n{answer}",
            })
