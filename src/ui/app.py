"""
Mini-RAG Streamlit Frontend — Clean Synchronous Flow.
"""

import time
import uuid
import requests
import streamlit as st
import os

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff",
                 "txt", "md", "csv", "json", "html"]

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini-RAG AI Tutor",
    page_icon="📚",
    layout="wide",
)

# ── Session state init ──────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "session_id":          str(uuid.uuid4()),
        "chat_history":        [],
        # Upload / processing
        "uploaded_asset_ids":  [],      # list of {asset_id, asset_name}
        "is_ready":            False,
        # Interactive quiz state
        "quiz_active":         False,
        "quiz_questions":      [],
        "quiz_index":          0,
        "quiz_score":          0,
        "quiz_answers":        {},
        # Diagram state
        "diagram_data":        None,
        "show_diagram":        False,
        # Imagine state
        "imagine_data":        None,
        "show_imagine":        False,
        # Walkthrough state
        "wt_active":           False,
        "wt_slide_index":      1,
        "wt_state":            "IDLE", # States: IDLE, LECTURING, WAITING_FOR_ACTION
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

def _process_file(file_id: str) -> bool:
    try:
        resp = requests.post(
            f"{API_URL}/v1/data/process/{PROJECT_ID}",
            json={"file_id": file_id, "chunk_size": 800, "overlap_size": 125, "do_reset": 0},
            timeout=180,
        )
        if resp.ok:
            return True
        st.error(f"Processing failed for file ID {file_id}: {resp.text[:200]}")
    except Exception as exc:
        st.error(f"Process error: {exc}")
    return False

def _index_project() -> bool:
    try:
        resp = requests.post(
            f"{API_URL}/v1/nlp/index/push/{PROJECT_ID}",
            json={"do_reset": 1},
            timeout=300,
        )
        if resp.ok:
            return True
        st.error(f"Indexing failed: {resp.text[:200]}")
    except Exception as exc:
        st.error(f"Indexing error: {exc}")
    return False

def _stream_agent_query(query: str, asset_ids: list) -> str:
    payload = {
        "query":         query,
        "project_id":    PROJECT_ID,
        "session_id":    SESSION_ID,
        "asset_ids":     asset_ids,
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

def _fetch_citations(query: str, asset_ids: list) -> list:
    try:
        resp = requests.post(
            f"{API_URL}/v1/nlp/agent-query",
            json={
                "query":         query,
                "project_id":    PROJECT_ID,
                "session_id":    SESSION_ID,
                "asset_ids":     asset_ids,
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

def _generate_imagine(asset_ids: list, language: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/v1/imagine/generate/{PROJECT_ID}",
            json={"asset_ids": asset_ids, "language": language},
            timeout=180,
        )
        if resp.ok:
            return resp.json()
        st.error(f"Imagine generation failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as exc:
        st.error(f"Imagine request error: {exc}")
    return None

def _stream_summary(asset_ids: list, language: str) -> str:
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

    # ── Interactive Quiz ─────────────────────────────────
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

    # ── Diagram ──────────────────────────────────────────
    st.subheader("🗺️ Lecture Diagram")
    st.caption("Visual concept map of the entire lecture.")
    diagram_btn = st.button(
        "Generate Diagram",
        use_container_width=True,
        disabled=not st.session_state.is_ready,
    )

    # ── Imagine ──────────────────────────────────────────
    st.subheader("🎨 Imagine")
    st.caption("Generate a visual educational poster.")
    imagine_btn = st.button(
        "Generate Imagine",
        use_container_width=True,
        disabled=not st.session_state.is_ready,
    )

# ── Auto Walkthrough ──────────────────────────────────────────
    st.subheader("▶️ Auto Walkthrough")
    st.caption("Let the AI explain the lecture slide-by-slide.")
    
    wt_cols = st.columns([4, 1])
    with wt_cols[0]:
        btn_label = "Resume Walkthrough" if st.session_state.wt_slide_index > 1 else "Start Walkthrough"
        walkthrough_btn = st.button(
            btn_label,
            use_container_width=True,
            disabled=not st.session_state.is_ready,
            type="primary"
        )
    with wt_cols[1]:
        reset_wt_btn = st.button(
            "🔄", 
            help="Restart from Slide 1",
            use_container_width=True, 
            disabled=not st.session_state.is_ready
        )
    st.divider()

    # ── Napkin visualisation (تم إرجاعه هنا) ──────────────────────────────
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

    if uploaded_files and not st.session_state.is_ready:
        if st.button("⚡ Upload & Process", type="primary", use_container_width=True):
            
            status_text = st.empty()
            progress_bar = st.progress(0)

            all_success = True
            new_assets = []
            total_files = len(uploaded_files)

            for i, f in enumerate(uploaded_files):
                status_text.info(f"📤 Uploading: {f.name}...")
                upload_res = _upload_file(f)

                if upload_res and upload_res.get("file_id"):
                    file_id = upload_res["file_id"]
                    new_assets.append({"asset_name": f.name, "asset_id": file_id})

                    status_text.warning(f"✂️ Chunking: {f.name}...")
                    progress_bar.progress(int(((i + 0.5) / total_files) * 50))

                    if not _process_file(file_id):
                        all_success = False
                        break
                else:
                    all_success = False
                    break

            if all_success:
                status_text.info("🧮 Embedding & 📥 Indexing into Vector DB...")
                progress_bar.progress(75)

                if _index_project():
                    progress_bar.progress(100)
                    status_text.success("✅ **Lectures processed and indexed successfully! You can now start chatting.**")
                    st.session_state.uploaded_asset_ids = new_assets
                    st.session_state.is_ready = True
                    time.sleep(1.5)
                    st.rerun()
                else:
                    status_text.error("❌ Failed during the Embedding/Indexing phase.")
            else:
                status_text.error("❌ Failed during the Upload/Chunking phase.")

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

        # Citations / Sources
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

# ── Sidebar actions (summary / quiz / diagram) ───────────────────────────────
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

if imagine_btn:
    with st.spinner("Generating visual educational poster… ⏳"):
        imagine_data = _generate_imagine(
            asset_ids=selected_asset_ids,
            language=lang_code,
        )
    if imagine_data and imagine_data.get("content"):
        st.session_state.imagine_data  = imagine_data
        st.session_state.show_imagine  = True
        st.rerun()
    else:
        st.error("Failed to generate infographic. Please try again.")


# ==============================================================================
# Lecture Walkthrough - State Machine (CLEANED UP & FIXED)
# ==============================================================================

# 1. User clicks Start/Resume from sidebar
if walkthrough_btn:
    st.session_state.wt_active = True
    # لاحظ أننا مسحنا السطر الذي يرجع الـ index إلى 1 هنا لكي يكمل من مكانه
    st.session_state.wt_state = "LECTURING"
    st.rerun()

# 1.B User clicks Restart (🔄) from sidebar
if reset_wt_btn:
    st.session_state.wt_active = True
    st.session_state.wt_slide_index = 1  # تصفير العداد فقط عند الضغط على زر الإعادة
    st.session_state.wt_state = "LECTURING"
    st.rerun()

# 2. State Engine Execution
if st.session_state.get("wt_active"):
    
    st.markdown("---")
    st.info(f"👨‍🏫 **Interactive Walkthrough Mode Active** - Slide {st.session_state.wt_slide_index}")

    # STATE A: LECTURING (Fetch from Backend)
    if st.session_state.wt_state == "LECTURING":
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""
            payload = {
                "query": "walkthrough", 
                "project_id": PROJECT_ID,
                "session_id": SESSION_ID,
                "asset_ids": selected_asset_ids,
                "image_paths": []
            }
            
            try:
                # Passing slide_index correctly
                url = f"{API_URL}/v1/nlp/walkthrough/stream/{PROJECT_ID}?slide_index={st.session_state.wt_slide_index}"
                with requests.post(url, json=payload, stream=True) as resp:
                    if resp.ok:
                        for raw_line in resp.iter_lines():
                            if raw_line:
                                line = raw_line.decode("utf-8")
                                if line.startswith("data: "):
                                    token = line[6:]
                                    if token == "[DONE]":
                                        break
                                    full_text += token.replace("\\n", "\n")
                                    placeholder.markdown(full_text + "▌")
                        placeholder.markdown(full_text)
                        
                        # Save response
                        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
                        
                        # Transition state
                        st.session_state.wt_state = "WAITING_FOR_ACTION"
                        st.rerun()
                    else:
                        st.error("Error fetching slide.")
                        st.session_state.wt_active = False
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.wt_active = False

    # STATE B: WAITING_FOR_ACTION (Show interactive buttons)
    elif st.session_state.wt_state == "WAITING_FOR_ACTION":
        st.markdown("### How would you like to proceed?")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        if col1.button("➡️ Continue", use_container_width=True, type="primary"):
            st.session_state.wt_slide_index += 1
            st.session_state.wt_state = "LECTURING"
            st.rerun()
            
        if col2.button("🛑 Stop Walkthrough", use_container_width=True):
            st.session_state.wt_active = False
            st.session_state.wt_state = "IDLE"
            st.rerun()
            
        st.caption("Or simply type your question in the chat box below to ask about this slide!")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive Quiz view
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.quiz_active:
    questions = st.session_state.quiz_questions
    idx       = st.session_state.quiz_index
    total     = len(questions)
    score     = st.session_state.quiz_score

    st.markdown("---")

    if idx >= total:
        st.subheader(f"🏁 Quiz Results  |  Final Score: {score}/{total}")
        st.progress(100)
        
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
        st.subheader(f"🎮 Interactive Quiz  —  Question {idx + 1} of {total}  |  Score: {score}/{total}")
        st.progress(int(idx / total * 100))
        
        q   = questions[idx]
        qid = q["id"]

        st.markdown(f"### ❓ {q['question']}")
        st.markdown("")

        already_answered = qid in st.session_state.quiz_answers

        if not already_answered:
            if q.get("hint"):
                with st.expander("💡 Show Hint", expanded=False):
                    st.info(q["hint"])

            option_labels = ["A", "B", "C", "D"]
            for i, opt in enumerate(q["options"][:4]):
                label = f"**{option_labels[i]}.** {opt}"
                if st.button(label, key=f"opt_{qid}_{i}", use_container_width=True):
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

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Diagram view
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.show_diagram and st.session_state.diagram_data:
    diag = st.session_state.diagram_data
    with st.expander(f"🗺️ Lecture Diagram: **{diag.get('title', 'Concept Map')}**", expanded=True):
        mermaid_code = diag.get("content", "")

        mermaid_html = f"""
        <div class="mermaid" style="background:#fff; padding:16px; border-radius:8px;">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
        """
        st.components.v1.html(mermaid_html, height=500, scrolling=True)

        if st.checkbox("📋 Show raw Mermaid code", key="show_raw_mermaid"):
            st.code(mermaid_code, language="text")

        if st.button("✖ Close Diagram", key="close_diag"):
            st.session_state.show_diagram = False
            st.rerun()

    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Imagine view
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.show_imagine and st.session_state.imagine_data:
    img_data = st.session_state.imagine_data
    with st.expander(f"🎨 Educational Poster: **{img_data.get('title', 'Lecture Poster')}**", expanded=True):
        html_code = img_data.get("content", "")
        st.components.v1.html(html_code, height=900, scrolling=True)

        if st.button("✖ Close Poster", key="close_imagine"):
            st.session_state.show_imagine = False
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
        )

    citations = _fetch_citations(
        query=user_input,
        asset_ids=selected_asset_ids,
    )

    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   answer,
        "citations": citations,
    })

    st.toast("✅ Response complete!", icon="✅")
    st.rerun()