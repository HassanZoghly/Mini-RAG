"""
Mini-RAG Streamlit Frontend — Phase 7 complete rewrite.

Key changes (items 3, 7, 8):

1. Upload-once → process → index → READY flow (item 3)
   - Files are uploaded via POST /v1/data/upload/{project_id}
   - A background job is kicked off via POST /v1/data/process/{project_id}
   - The UI polls GET /v1/data/status/{project_id} and shows a live
     progress indicator through UPLOADED → EXTRACTING → CHUNKING →
     EMBEDDING → INDEXING → READY (or FAILED)
   - chat_input is disabled (hidden) until status == READY
   - Uploaded lectures are listed from GET /v1/data/assets/{project_id}
     so the user can filter which lecture(s) to chat about

2. Chat uses /v1/nlp/agent-query/stream (item 3)
   - No more file re-upload with every message
   - asset_ids passed from the indexed assets list

3. Teaching modes sidebar selector (item 7)
   - quick_review / full_explanation / exam_prep / step_by_step
   - Passed as form field on every query

4. Sources / citations display (item 8)
   - /agent-query/stream doesn't return structured citations (streaming),
     so the non-streaming /agent-query is called in parallel to get them
   - A collapsible "📌 Sources" expander is shown under each answer

5. Summary & Quiz use the indexed pipeline, not file re-upload
   - Summary uses POST /v1/nlp/summarize/{project_id} with asset_ids + language
   - Quiz uses /v1/nlp/agent-query/stream with a quiz query

Architecture note
-----------------
project_id == session_id (UUID).  The user never sees the project_id.
Each browser session is a fully isolated project namespace in the backend.
"""

import time
import uuid

import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

POLL_INTERVAL_S  = 2      # seconds between status polls
POLL_TIMEOUT_S   = 600    # give up after 10 minutes

ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff",
                 "txt", "md", "csv", "json", "html"]

PROCESSING_STATUS_LABELS = {
    "UPLOADED":    "📤 Uploaded — queued for processing",
    "PROCESSING":  "⚙️  Starting pipeline…",
    "EXTRACTING":  "🔍 Extracting text…",
    "CHUNKING":    "✂️  Chunking…",
    "EMBEDDING":   "🧮 Generating embeddings…",
    "INDEXING":    "📥 Indexing into vector DB…",
    "READY":       "✅ Ready — you can start chatting!",
    "FAILED":      "❌ Processing failed",
}

TEACHING_MODE_OPTIONS = {
    "Default (auto)":       "",
    "🏃 Quick Review":      "quick_review",
    "📖 Full Explanation":  "full_explanation",
    "📝 Exam Preparation":  "exam_prep",
    "🐢 Step-by-Step":      "step_by_step",
}

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini-RAG AI Tutor",
    page_icon="📚",
    layout="wide",
)

# ── Session state init ──────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "session_id":        str(uuid.uuid4()),
        "chat_history":      [],
        # Upload / processing
        "uploaded_asset_ids":  [],      # list of {asset_id, asset_name} from backend
        "processing_status":   None,    # last polled status string
        "processing_detail":   "",
        "is_processing":       False,
        "is_ready":            False,
        # Teaching mode
        "teaching_mode":       "",
        # Interactive quiz state
        "quiz_active":         False,
        "quiz_questions":      [],
        "quiz_index":          0,
        "quiz_score":          0,
        "quiz_answers":        {},
        # Diagram state
        "diagram_data":        None,
        "show_diagram":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

SESSION_ID = st.session_state.session_id
PROJECT_ID = SESSION_ID   # backend project namespace


# ══════════════════════════════════════════════════════════════════════════════
# Backend helpers
# ══════════════════════════════════════════════════════════════════════════════

def _upload_file(file_obj) -> dict | None:
    """Upload a single file to the backend; returns response JSON or None."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/data/upload/{PROJECT_ID}",
            files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)},
            timeout=120,
        )
        if resp.ok:
            return resp.json()
        st.error(f"Upload failed for {file_obj.name}: {resp.text[:200]}")
    except Exception as exc:
        st.error(f"Upload error: {exc}")
    return None


def _trigger_processing(asset_ids: list) -> bool:
    """Kick off the background process+index job for the project."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/data/process/{PROJECT_ID}",
            json={"chunk_size": 800, "overlap_size": 125, "do_reset": 0},
            timeout=30,
        )
        return resp.status_code in (200, 202)
    except Exception as exc:
        st.error(f"Processing trigger failed: {exc}")
    return False


def _poll_status() -> dict:
    """Poll GET /v1/data/status/{PROJECT_ID} and return the JSON."""
    try:
        resp = requests.get(f"{API_URL}/v1/data/status/{PROJECT_ID}", timeout=10)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {}


def _fetch_assets() -> list:
    """Fetch the list of uploaded/indexed assets for this project."""
    try:
        resp = requests.get(f"{API_URL}/v1/data/assets/{PROJECT_ID}", timeout=10)
        if resp.ok:
            return resp.json().get("assets", [])
    except Exception:
        pass
    return []


def _stream_agent_query(query: str, asset_ids: list, teaching_mode: str) -> str:
    """
    Call /v1/nlp/agent-query/stream and stream tokens into a Streamlit
    placeholder.  Returns the full accumulated text.
    """
    payload = {
        "query":         query,
        "project_id":    PROJECT_ID,
        "session_id":    SESSION_ID,
        "asset_ids":     asset_ids,
        "teaching_mode": teaching_mode,
        "image_paths":   [],
    }

    placeholder = st.empty()
    full_text = ""

    try:
        with requests.post(
            f"{API_URL}/v1/nlp/agent-query/stream",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            if not resp.ok:
                st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
                return ""
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                token = line[6:]
                if token == "[DONE]":
                    break
                token = token.replace("\\n", "\n")
                full_text += token
                placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as exc:
        st.error(f"Stream request failed: {exc}")

    return full_text


def _fetch_citations(query: str, asset_ids: list, teaching_mode: str) -> list:
    """
    Call /v1/nlp/agent-query (non-streaming) in a lightweight way to
    get structured citations.  Only runs in the background after the
    stream completes; failure is silent.
    """
    try:
        resp = requests.post(
            f"{API_URL}/v1/nlp/agent-query",
            json={
                "query":         query,
                "project_id":    PROJECT_ID,
                "session_id":    SESSION_ID,
                "asset_ids":     asset_ids,
                "teaching_mode": teaching_mode,
                "image_paths":   [],
            },
            timeout=60,
        )
        if resp.ok:
            return resp.json().get("citations") or []
    except Exception:
        pass
    return []


def _generate_quiz(asset_ids: list, language: str, num_questions: int, difficulty: str = "MEDIUM") -> dict | None:
    """Call POST /v1/quiz/generate/{PROJECT_ID} and return the parsed JSON."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/quiz/generate/{PROJECT_ID}",
            json={
                "num_questions": num_questions,
                "asset_ids":     asset_ids,
                "language":      language,
                "difficulty":    difficulty,
            },
            timeout=180,
        )
        if resp.ok:
            return resp.json()
        st.error(f"Quiz generation failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as exc:
        st.error(f"Quiz request error: {exc}")
    return None


def _check_answer(question_data: dict, user_answer: str) -> dict | None:
    """Call POST /v1/quiz/answer to get feedback."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/quiz/answer",
            json={
                "question_id":    question_data["id"],
                "question":       question_data["question"],
                "correct_answer": question_data["correct_answer"],
                "user_answer":    user_answer,
                "hint":           question_data.get("hint", ""),
                "explanation":    question_data.get("explanation", ""),
            },
            timeout=15,
        )
        if resp.ok:
            return resp.json()
    except Exception as exc:
        st.error(f"Answer check error: {exc}")
    return None


def _generate_diagram(asset_ids: list, language: str) -> dict | None:
    """Call POST /v1/diagram/generate/{PROJECT_ID} and return the parsed JSON."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/diagram/generate/{PROJECT_ID}",
            json={"asset_ids": asset_ids, "language": language},
            timeout=180,
        )
        if resp.ok:
            return resp.json()
        st.error(f"Diagram generation failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as exc:
        st.error(f"Diagram request error: {exc}")
    return None


def _stream_summary(asset_ids: list, language: str) -> str:
    """
    Call /v1/nlp/summarize/{PROJECT_ID} with the map-reduce streaming path.
    Returns the full accumulated summary text.
    """
    payload = {"asset_ids": asset_ids, "language": language}
    placeholder = st.empty()
    full_text = ""

    try:
        with requests.post(
            f"{API_URL}/v1/nlp/summarize/{PROJECT_ID}",
            json=payload,
            stream=True,
            timeout=600,
        ) as resp:
            if not resp.ok:
                st.error(f"Summary error {resp.status_code}: {resp.text[:200]}")
                return ""
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                token = line[6:]
                if token == "[DONE]":
                    break
                token = token.replace("\\n", "\n")
                full_text += token
                placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as exc:
        st.error(f"Summary request failed: {exc}")

    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# Processing status poller
# ══════════════════════════════════════════════════════════════════════════════

def _run_processing_with_progress(asset_ids: list):
    """
    Show a live progress UI while the backend processes and indexes files.
    Blocks (via st.rerun loop) until status is READY or FAILED.
    """
    st.session_state.is_processing = True
    st.session_state.processing_status = "PROCESSING"

    triggered = _trigger_processing(asset_ids)
    if not triggered:
        st.error("Failed to start processing job.")
        st.session_state.is_processing = False
        return

    status_box   = st.empty()
    progress_bar = st.progress(0)
    detail_box   = st.empty()

    STATUS_ORDER = [
        "UPLOADED", "PROCESSING", "EXTRACTING",
        "CHUNKING", "EMBEDDING", "INDEXING", "READY",
    ]

    deadline = time.time() + POLL_TIMEOUT_S

    while time.time() < deadline:
        data = _poll_status()
        current = data.get("status") or "PROCESSING"
        detail  = data.get("detail", "")

        st.session_state.processing_status = current
        st.session_state.processing_detail = detail

        label = PROCESSING_STATUS_LABELS.get(current, current)
        status_box.info(f"**{label}**")
        detail_box.caption(detail)

        # Update progress bar
        try:
            idx = STATUS_ORDER.index(current)
            pct = int((idx / (len(STATUS_ORDER) - 1)) * 100)
        except ValueError:
            pct = 50
        progress_bar.progress(pct)

        if current == "READY":
            st.session_state.is_ready       = True
            st.session_state.is_processing  = False
            # Refresh asset list now that indexing is done
            st.session_state.uploaded_asset_ids = _fetch_assets()
            status_box.success("✅ **Lecture indexed and ready! Start chatting below.**")
            progress_bar.progress(100)
            detail_box.empty()
            time.sleep(1)
            st.rerun()
            return

        if current == "FAILED":
            st.session_state.is_processing = False
            status_box.error(f"❌ Processing failed: {detail}")
            progress_bar.empty()
            return

        time.sleep(POLL_INTERVAL_S)

    # Timeout
    st.session_state.is_processing = False
    st.error("⏱️ Processing timed out. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 Mini-RAG")
    st.caption("AI Tutor · Multi-Agent · RAG")
    st.divider()

    if st.button("🔄 New Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()

    st.caption(f"Session `{SESSION_ID[:8]}…`")
    st.divider()

    # ── Teaching mode selector (item 7) ─────────────────────────────────
    st.subheader("🎓 Teaching Mode")
    selected_mode_label = st.selectbox(
        "How should the tutor explain?",
        list(TEACHING_MODE_OPTIONS.keys()),
        index=0,
        key="mode_selector",
        help=(
            "Quick Review — brief refresher\n"
            "Full Explanation — deep teaching\n"
            "Exam Prep — definitions & comparisons\n"
            "Step-by-Step — slow, guided learning"
        ),
    )
    teaching_mode = TEACHING_MODE_OPTIONS[selected_mode_label]
    st.session_state.teaching_mode = teaching_mode

    st.divider()

    # ── Lecture selector ─────────────────────────────────────────────────
    st.subheader("📄 Lectures")
    assets = st.session_state.get("uploaded_asset_ids", [])

    if assets:
        asset_options = {"All lectures": None}
        for a in assets:
            asset_options[a["asset_name"]] = a["asset_id"]

        selected_lecture = st.selectbox(
            "Target lecture:",
            list(asset_options.keys()),
            key="lecture_selector",
        )
        selected_asset_ids = (
            [asset_options[selected_lecture]]
            if asset_options[selected_lecture] is not None
            else [a["asset_id"] for a in assets]
        )
    else:
        st.caption("No lectures indexed yet.")
        selected_asset_ids = []

    st.divider()

    # ── Output language ──────────────────────────────────────────────────
    st.subheader("🌍 Output Language")
    tool_lang = st.radio("Language:", ["English", "العربية"], horizontal=True)
    lang_code = "en" if tool_lang == "English" else "ar"

    st.divider()

    # ── Summary ──────────────────────────────────────────────────────────
    st.subheader("📋 Summary")
    st.caption("Full map-reduce summary of the selected lecture(s).")
    sum_btn = st.button(
        "Generate Summary",
        use_container_width=True,
        disabled=not st.session_state.is_ready,
    )

    st.divider()

    # ── Quiz ─────────────────────────────────────────────────────────────
    st.subheader("🧠 Quiz")
    num_questions = st.number_input(
        "Number of questions:", min_value=1, max_value=50, value=5, step=1
    )
    quiz_btn = st.button(
        "Generate Quiz",
        use_container_width=True,
        disabled=not st.session_state.is_ready,
    )

    st.divider()

    # ── Interactive Quiz (new Feature 1) ─────────────────────────────────
    st.subheader("🎮 Interactive Quiz")
    st.caption("Answer questions one-by-one with hints and explanations.")
    iq_num_questions = st.number_input(
        "Questions:", min_value=1, max_value=20, value=5, step=1,
        key="iq_num_q",
    )
    iq_difficulty = st.selectbox(
        "Difficulty:",
        ["EASY", "MEDIUM", "HARD"],
        index=1,
        key="iq_diff",
    )
    interactive_quiz_btn = st.button(
        "▶ Start Interactive Quiz",
        use_container_width=True,
        disabled=not st.session_state.is_ready,
        type="primary",
    )
    if st.session_state.quiz_active:
        if st.button("✖ Exit Quiz", key="exit_quiz_sidebar_btn", use_container_width=True):
            st.session_state.quiz_active    = False
            st.session_state.quiz_questions = []
            st.session_state.quiz_index     = 0
            st.session_state.quiz_score     = 0
            st.session_state.quiz_answers   = {}
            st.rerun()

    st.divider()

    # ── Diagram (new Feature 2) ──────────────────────────────────────────
    st.subheader("🗺️ Lecture Diagram")
    st.caption("Visual concept map of the entire lecture.")
    diagram_btn = st.button(
        "Generate Diagram",
        use_container_width=True,
        disabled=not st.session_state.is_ready,
    )

    # ── Napkin visualisation ──────────────────────────────────────────────
    st.subheader("🎨 Visualize (Napkin AI)")
    vis_options = {
        "Mind Map":              "Mind Map",
        "Flowchart":             "Flowchart",
        "Hierarchy diagram":     "Hierarchy diagram",
        "Comparison table":      "Comparison table or diagram",
        "Cycle diagram":         "Cycle diagram",
    }
    selected_vis = st.selectbox("Diagram type:", list(vis_options.keys()))
    if st.button("Draw last answer", use_container_width=True, disabled=not st.session_state.is_ready):
        last_msg = next(
            (m["content"] for m in reversed(st.session_state.chat_history)
             if m["role"] == "assistant" and "<img" not in m["content"]),
            None,
        )
        if not last_msg:
            st.toast("No previous answer to visualize.", icon="⚠️")
        else:
            with st.spinner("Generating visualization…"):
                try:
                    vis_prompt = (
                        f"Please strictly generate a {vis_options[selected_vis]} "
                        f"for the following content:\n\n{last_msg}"
                    )
                    res = requests.post(
                        f"{API_URL}/v1/nlp/visualize",
                        json={"text": vis_prompt},
                        timeout=120,
                    )
                    if res.ok:
                        b64_list = res.json().get("images_base64", [])
                        if b64_list:
                            html_imgs = "".join(
                                f'<div style="text-align:center">'
                                f'<img src="data:image/png;base64,{b64}" '
                                f'width="65%" style="border-radius:8px; '
                                f'box-shadow:0 4px 8px rgba(0,0,0,0.1); margin-bottom:20px"/>'
                                f'</div>'
                                for b64 in b64_list
                            )
                            st.session_state.chat_history.append({
                                "role":    "assistant",
                                "content": f"**{selected_vis}**\n\n{html_imgs}",
                            })
                            st.rerun()
                        else:
                            st.error("No image returned by Napkin AI.")
                    else:
                        st.error(f"Napkin API error: {res.text[:200]}")
                except Exception as exc:
                    st.error(f"Visualization error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Main area — Upload + Processing gate
# ══════════════════════════════════════════════════════════════════════════════

st.title("📚 Mini-RAG AI Tutor")

# ── Upload section ───────────────────────────────────────────────────────────
with st.expander(
    "📂 Upload Lecture Files",
    expanded=not st.session_state.is_ready,
):
    st.caption(
        "Upload your lecture PDF(s) or text files. Once uploaded, the system "
        "will automatically extract, chunk, embed and index them. "
        "**You can only chat after processing is complete.**"
    )

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files and not st.session_state.is_processing:
        if st.button("⚡ Upload & Process", type="primary", use_container_width=True):
            # Step 1 — upload each file
            upload_progress = st.progress(0)
            newly_uploaded_ids = []

            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Uploading {f.name}…"):
                    result = _upload_file(f)
                    if result and result.get("file_id"):
                        newly_uploaded_ids.append(result["file_id"])
                upload_progress.progress(int((i + 1) / len(uploaded_files) * 100))

            upload_progress.empty()

            if not newly_uploaded_ids:
                st.error("No files were uploaded successfully.")
            else:
                st.success(f"✅ {len(newly_uploaded_ids)} file(s) uploaded. Starting processing…")
                # Step 2 — run the background pipeline with live progress UI
                _run_processing_with_progress(newly_uploaded_ids)

    # Show current status if already processed/processing
    if st.session_state.processing_status:
        current = st.session_state.processing_status
        label   = PROCESSING_STATUS_LABELS.get(current, current)
        if current == "READY":
            st.success(f"**{label}**")
        elif current == "FAILED":
            st.error(f"**{label}** — {st.session_state.processing_detail}")
        elif current:
            st.info(f"**{label}**")


# ── If still processing, show a blocking spinner and poll ───────────────────
if st.session_state.is_processing:
    st.warning("⏳ Processing in progress — please wait…")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Chat interface (only shown after READY)
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.is_ready:
    st.info(
        "👆 Upload a lecture file above and click **Upload & Process** "
        "to get started. Chat will be enabled once indexing is complete."
    )
    st.stop()

# ── Display chat history ─────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

        # Citations / Sources (item 8)
        citations = msg.get("citations") or []
        if citations:
            with st.expander("📌 Sources", expanded=False):
                for cit in citations:
                    parts = [f"**{cit.get('source', '')}**"]
                    if cit.get("page"):
                        parts.append(f"Page {cit['page']}")
                    if cit.get("section"):
                        parts.append(f"*{cit['section']}*")
                    st.markdown("- " + " · ".join(parts))

        # Agent trace (debug)
        if msg.get("agent_trace"):
            with st.expander("🔍 Agent trace", expanded=False):
                for step in msg["agent_trace"]:
                    st.markdown(f"- `{step}`")


# ── Sidebar actions (summary / quiz) ────────────────────────────────────────
if sum_btn:
    lang_label = "English" if lang_code == "en" else "العربية"
    display_text = f"Summarize lecture(s) [{lang_label}]"
    st.session_state.chat_history.append({"role": "user", "content": display_text})

    with st.chat_message("user"):
        st.markdown(display_text)

    with st.chat_message("assistant"):
        with st.spinner("Generating full lecture summary (map-reduce)… ⏳"):
            answer = _stream_summary(
                asset_ids=selected_asset_ids,
                language=lang_code,
            )

    if answer:
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": f"**📋 Lecture Summary**\n\n{answer}",
        })
    st.toast("✅ Summary complete!", icon="✅")
    st.rerun()

if quiz_btn:
    lang_inst = "in English" if lang_code == "en" else "باللغة العربية"
    quiz_query = f"Generate quiz [{num_questions}] {lang_inst}"
    display_text = f"Quiz — {num_questions} question(s) [{lang_label if 'lang_label' in dir() else tool_lang}]"

    st.session_state.chat_history.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)

    with st.chat_message("assistant"):
        with st.spinner("Generating quiz questions… ⏳"):
            answer = _stream_agent_query(
                query=quiz_query,
                asset_ids=selected_asset_ids,
                teaching_mode=teaching_mode,
            )

    if answer:
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": f"**🧠 Quiz**\n\n{answer}",
        })
    st.toast("✅ Quiz ready!", icon="✅")
    st.rerun()


# ── Sidebar: Interactive Quiz button trigger ─────────────────────────────────
if interactive_quiz_btn:
    with st.spinner(f"Generating {iq_num_questions} quiz questions from lecture ({iq_difficulty.lower()} difficulty)… ⏳"):
        quiz_data = _generate_quiz(
            asset_ids=selected_asset_ids,
            language=lang_code,
            num_questions=iq_num_questions,
            difficulty=iq_difficulty,
        )
    if quiz_data and quiz_data.get("questions"):
        st.session_state.quiz_questions = quiz_data["questions"]
        st.session_state.quiz_active    = True
        st.session_state.quiz_index     = 0
        st.session_state.quiz_score     = 0
        st.session_state.quiz_answers   = {}
        st.toast(f"✅ {len(quiz_data['questions'])} questions ready!", icon="🎮")
        st.rerun()
    else:
        st.error("Failed to generate quiz questions. Please try again.")

# ── Sidebar: Diagram button trigger ─────────────────────────────────────────
if diagram_btn:
    with st.spinner("Generating lecture concept diagram… ⏳"):
        diagram_data = _generate_diagram(
            asset_ids=selected_asset_ids,
            language=lang_code,
        )
    if diagram_data and diagram_data.get("content"):
        st.session_state.diagram_data  = diagram_data
        st.session_state.show_diagram  = True
        st.rerun()
    else:
        st.error("Failed to generate diagram. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive Quiz view (replaces chat when active)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.quiz_active:
    questions = st.session_state.quiz_questions
    idx       = st.session_state.quiz_index
    total     = len(questions)
    score     = st.session_state.quiz_score

    st.markdown("---")
    st.subheader(f"🎮 Interactive Quiz  —  Question {idx + 1} of {total}  |  Score: {score}/{total}")

    # Progress bar
    st.progress(int(idx / total * 100))

    if idx >= total:
        # ── Final result screen ──────────────────────────────────────────
        pct = int(score / total * 100)
        if pct >= 80:
            st.success(f"🎉 Excellent! You scored **{score}/{total}** ({pct}%)")
        elif pct >= 50:
            st.warning(f"👍 Good effort! You scored **{score}/{total}** ({pct}%)")
        else:
            st.error(f"📚 Keep studying! You scored **{score}/{total}** ({pct}%)")

        st.markdown("### 📋 Review your answers")
        for q in questions:
            qid      = q["id"]
            ans_data = st.session_state.quiz_answers.get(qid, {})
            if ans_data.get("correct"):
                icon = "✅"
            elif qid in st.session_state.quiz_answers:
                icon = "❌"
            else:
                icon = "⬜"

            with st.expander(f"{icon} {q['question']}", expanded=False):
                st.markdown(f"**Correct answer:** {q['correct_answer']}")
                if ans_data.get("user_answer"):
                    st.markdown(f"**Your answer:** {ans_data['user_answer']}")
                if q.get("explanation"):
                    st.info(f"💡 {q['explanation']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Quiz", use_container_width=True):
                st.session_state.quiz_index   = 0
                st.session_state.quiz_score   = 0
                st.session_state.quiz_answers = {}
                st.rerun()
        with col2:
            if st.button("✖ Exit Quiz", key="exit_quiz_results_btn", use_container_width=True):
                st.session_state.quiz_active    = False
                st.session_state.quiz_questions = []
                st.session_state.quiz_index     = 0
                st.session_state.quiz_score     = 0
                st.session_state.quiz_answers   = {}
                st.rerun()

    else:
        # ── Current question ─────────────────────────────────────────────
        q   = questions[idx]
        qid = q["id"]

        st.markdown(f"### ❓ {q['question']}")
        st.markdown("")

        already_answered = qid in st.session_state.quiz_answers

        if not already_answered:
            if q.get("hint"):
                with st.expander("💡 Show Hint", expanded=False):
                    st.info(q["hint"])

            # Show options as clickable buttons
            option_labels = ["A", "B", "C", "D"]
            for i, opt in enumerate(q["options"][:4]):
                label = f"**{option_labels[i]}.**  {opt}"
                if st.button(label, key=f"opt_{qid}_{i}", use_container_width=True):
                    # Check answer
                    feedback = _check_answer(q, opt)
                    if feedback:
                        is_correct = feedback.get("correct", False)
                        st.session_state.quiz_answers[qid] = {
                            "user_answer": opt,
                            "correct":     is_correct,
                            "feedback":    feedback,
                        }
                        if is_correct:
                            st.session_state.quiz_score += 1
                        st.rerun()
        else:
            # Show result for this question
            ans_data = st.session_state.quiz_answers[qid]
            feedback = ans_data.get("feedback", {})

            if ans_data["correct"]:
                st.success(f"✅ **{feedback.get('message', 'Correct!')}**")
                st.info(f"💡 **Explanation:** {feedback.get('explanation', q.get('explanation', ''))}")
            else:
                st.error(f"❌ **{feedback.get('message', 'Wrong!')}**")
                user_ans = ans_data["user_answer"]
                st.markdown(f"Your answer: ~~{user_ans}~~")
                st.markdown(f"✅ Correct answer: **{q['correct_answer']}**")
                st.info(f"📖 **Explanation:** {feedback.get('explanation', q.get('explanation', ''))}")

            st.markdown("")
            nav_cols = st.columns([1, 1])
            with nav_cols[0]:
                if idx > 0 and st.button("← Previous", use_container_width=True):
                    st.session_state.quiz_index -= 1
                    st.rerun()
            with nav_cols[1]:
                next_label = "Next →" if idx < total - 1 else "See Results 🏁"
                if st.button(next_label, use_container_width=True, type="primary"):
                    st.session_state.quiz_index += 1
                    st.rerun()

    # Block the rest of the page while quiz is active
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Diagram view (shown as a collapsible section above chat)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.show_diagram and st.session_state.diagram_data:
    diag = st.session_state.diagram_data
    with st.expander(f"🗺️ Lecture Diagram: **{diag.get('title', 'Concept Map')}**", expanded=True):
        mermaid_code = diag.get("content", "")

        # Render Mermaid via an HTML component with Mermaid.js CDN
        mermaid_html = f"""
        <div class="mermaid" style="background:#fff; padding:16px; border-radius:8px;">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
        """
        st.components.v1.html(mermaid_html, height=500, scrolling=True)

        # Show raw Mermaid code toggle — use a checkbox instead of a nested
        # expander because Streamlit does not allow nested st.expander calls.
        if st.checkbox("📋 Show raw Mermaid code", key="show_raw_mermaid"):
            st.code(mermaid_code, language="text")

        if st.button("✖ Close Diagram", key="close_diag"):
            st.session_state.show_diagram = False
            st.rerun()

    st.markdown("---")

# ── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask anything about your lecture…",
)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = _stream_agent_query(
            query=user_input,
            asset_ids=selected_asset_ids,
            teaching_mode=st.session_state.teaching_mode,
        )

    # Fetch citations from the non-streaming endpoint in the background
    citations = _fetch_citations(
        query=user_input,
        asset_ids=selected_asset_ids,
        teaching_mode=st.session_state.teaching_mode,
    )

    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   answer,
        "citations": citations,
    })

    st.toast("✅ Response complete!", icon="✅")
    st.rerun()