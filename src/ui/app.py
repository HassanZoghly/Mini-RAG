import streamlit as st
import streamlit.components.v1 as components
import requests
import uuid
import os
import html as html_lib

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
if "active_view" not in st.session_state:
    st.session_state.active_view = "chat"
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_current" not in st.session_state:
    st.session_state.quiz_current = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False
if "quiz_last_correct" not in st.session_state:
    st.session_state.quiz_last_correct = None
if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = []
if "diagram_data" not in st.session_state:
    st.session_state.diagram_data = None

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Mini-RAG")
    st.caption("Multi-Agent · Multimodal · Memory")
    st.divider()

    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.active_view = "chat"
        st.session_state.quiz_data = None
        st.session_state.diagram_data = None
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

# ── Sidebar (القائمة الجانبية) ────────────────────────────────────────────────
with st.sidebar:
    st.title("🛠️ أدوات المحاضرات")
    st.divider()

    # 1. القائمة المنسدلة لاختيار المحاضرة
    file_options = ["جميع المحاضرات"] + [f.name for f in uploaded_files] if uploaded_files else ["جميع المحاضرات"]
    selected_file = st.selectbox("📄 المحاضرة المستهدفة:", file_options)

    # 2. لغة المخرجات
    tool_lang = st.radio("🌍 لغة المخرجات:", ["English", "العربية"], horizontal=True)

    st.divider()
    st.subheader("🎨 الرسوم التوضيحية (Napkin AI)")
    st.caption("اختر نوع الرسم واضغط على الزر لتحويل آخر إجابة إلى رسمة:")

    # قائمة بأنواع الرسومات لتوجيه Napkin
    vis_options = {
        "خريطة ذهنية (Mind Map)": "Mind Map",
        "مخطط انسيابي (Flowchart)": "Flowchart",
        "هيكل تنظيمي (Hierarchy)": "Hierarchy diagram",
        "مقارنة (Comparison)": "Comparison table or diagram",
        "دورة حياة (Cycle)": "Cycle diagram"
    }
    selected_vis_ar = st.selectbox("نوع الرسم:", list(vis_options.keys()))
    selected_vis_en = vis_options[selected_vis_ar]

    if st.button("🎨 ارسم الإجابة الأخيرة", use_container_width=True):
        # البحث عن آخر إجابة للموديل في الشات
        last_assistant_msg = None
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "assistant" and "<img" not in msg["content"]:
                last_assistant_msg = msg["content"]
                break

        if not last_assistant_msg:
            st.toast("⚠️ لا توجد إجابة سابقة لرسمها!", icon="⚠️")
        else:
            with st.spinner(f"جاري إنشاء {selected_vis_ar}... ⏳"):
                try:
                    # توجيه Napkin بشكل صريح لنوع الرسمة المطلوبة
                    vis_prompt = f"Please strictly generate a {selected_vis_en} for the following content:\n\n{last_assistant_msg}"

                    res = requests.post(
                        f"{API_URL}/v1/nlp/visualize",
                        json={"text": vis_prompt},
                        timeout=120
                    )

                    if res.ok:
                        data = res.json()
                        b64_list = data.get("images_base64", [])
                        if b64_list:
                            html_images = ""
                            for b64 in b64_list:
                                # 🔥 السر هنا: استخدام HTML لتصغير الحجم (width="60%") وعمل توسيط للصورة
                                html_images += f'<div style="text-align: center;"><img src="data:image/png;base64,{b64}" width="65%" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;"/></div>'

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**تم توليد: {selected_vis_ar}**\n\n{html_images}"
                            })
                            st.rerun()
                        else:
                            st.error("لم يتم إرجاع أي رسمة من السيرفر.")
                    else:
                        st.error(f"خطأ في الاتصال: {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # 3. قسم التلخيص
    st.subheader("📋 التلخيص")
    sum_btn = st.button("إنشاء ملخص", use_container_width=True)

    st.divider()

    # 4. المخطط التفاعلي للمحاضرة
    st.subheader("🧩 Lecture Diagram")
    diagram_btn = st.button("Generate Diagram", use_container_width=True)

    st.divider()

    # 5. الاختبار التفاعلي
    st.subheader("🧠 Interactive Quiz")
    num_questions = st.number_input("🔢 عدد الأسئلة:", min_value=1, max_value=20, value=5, step=1)
    quiz_btn = st.button("Interactive Quiz", use_container_width=True)

# 💡 الفلترة الذكية (تجهيز الملفات للباك-إند)
if selected_file == "جميع المحاضرات":
    target_files = uploaded_files
else:
    target_files = [f for f in uploaded_files if f.name == selected_file]

# ── helpers ────────────────────────────────────────────────────────────────────
def build_files_payload(files):
    if not files:
        return []
    return [("files", (f.name, f.getvalue(), f.type)) for f in files]

def stream_query(query: str, files, visualize: bool = False) -> str:
    form_data = {
        "query":      query,
        "project_id": PROJECT_ID,
        "session_id": SESSION_ID,
        "visualize":  str(visualize).lower()
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

                    token = token.replace("\\n", "\n")
                    full_text += token
                    placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as exc:
        st.error(f"Request failed: {exc}")

    return full_text

def fetch_trace_and_sources(query: str, files) -> tuple[list, list]:
    try:
        resp = requests.post(
            f"{API_URL}/v1/nlp/multimodal-query",
            data={"query": query, "project_id": PROJECT_ID, "session_id": SESSION_ID},
            files=build_files_payload(files) or None,
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            return data.get("agent_trace", []), data.get("sources_used", [])
    except Exception:
        pass
    return [], []


def generate_interactive_quiz(files, num_questions: int, language: str):
    resp = requests.post(
        f"{API_URL}/v1/nlp/quiz/generate/{PROJECT_ID}",
        data={"num_questions": str(num_questions), "language": language},
        files=build_files_payload(files) or None,
        timeout=180,
    )
    if not resp.ok:
        raise RuntimeError(resp.text[:500])
    return resp.json()


def generate_lecture_diagram(files, language: str):
    resp = requests.post(
        f"{API_URL}/v1/nlp/diagram/generate/{PROJECT_ID}",
        data={"language": language, "diagram_type": "flowchart"},
        files=build_files_payload(files) or None,
        timeout=180,
    )
    if not resp.ok:
        raise RuntimeError(resp.text[:500])
    return resp.json()


def _normalize_answer(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _correct_answer_text(question: dict) -> str:
    correct = str(question.get("correct_answer", "")).strip()
    options = question.get("options", []) or []
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "أ": 0, "ب": 1, "ج": 2, "د": 3}
    if correct.upper() in letter_map and letter_map[correct.upper()] < len(options):
        return options[letter_map[correct.upper()]]
    if correct in letter_map and letter_map[correct] < len(options):
        return options[letter_map[correct]]
    return correct


def render_mermaid_diagram(mermaid_code: str, height: int = 620):
    safe_code = html_lib.escape(mermaid_code or "")
    html = f"""
    <div style="font-family: system-ui, -apple-system, Segoe UI, sans-serif;">
      <pre class="mermaid" style="background: white; padding: 16px; border-radius: 12px;">
{safe_code}
      </pre>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


def reset_quiz_runtime():
    st.session_state.quiz_current = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answered = False
    st.session_state.quiz_last_correct = None
    st.session_state.quiz_completed = []


def render_interactive_quiz_view():
    quiz = st.session_state.quiz_data or {}
    questions = quiz.get("questions", [])

    st.title("🧠 Interactive Quiz")

    if not questions:
        st.info("No quiz has been generated yet.")
        if st.button("Back to Chat"):
            st.session_state.active_view = "chat"
            st.rerun()
        return

    total = len(questions)
    idx = min(st.session_state.quiz_current, total)

    if idx >= total:
        st.success("🎉 Quiz completed!")
        st.metric("Final Score", f"{st.session_state.quiz_score} / {total}")
        percent = int((st.session_state.quiz_score / total) * 100) if total else 0
        st.progress(percent / 100)
        st.write(f"**Result:** {percent}%")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retake Quiz", use_container_width=True):
                reset_quiz_runtime()
                st.rerun()
        with col2:
            if st.button("Back to Chat", use_container_width=True):
                st.session_state.active_view = "chat"
                st.rerun()
        return

    question = questions[idx]
    st.caption(f"Question {idx + 1} of {total}")
    st.progress(idx / total)
    st.metric("Score", f"{st.session_state.quiz_score} / {total}")

    st.subheader(question.get("question", ""))
    options = question.get("options", [])
    labels = ["A", "B", "C", "D"]
    display_options = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]
    option_lookup = {display_options[i]: options[i] for i in range(len(options))}

    selected_display = st.radio(
        "Choose one answer:",
        display_options,
        key=f"quiz_choice_{quiz.get('quiz_id', 'quiz')}_{idx}",
        disabled=st.session_state.quiz_answered,
    )
    selected_answer = option_lookup.get(selected_display, selected_display)
    correct_answer = _correct_answer_text(question)

    if not st.session_state.quiz_answered:
        if st.button("Submit Answer", type="primary", use_container_width=True):
            is_correct = _normalize_answer(selected_answer) == _normalize_answer(correct_answer)
            st.session_state.quiz_answered = True
            st.session_state.quiz_last_correct = is_correct
            if is_correct and idx not in st.session_state.quiz_completed:
                st.session_state.quiz_score += 1
                st.session_state.quiz_completed.append(idx)
            st.rerun()
    else:
        if st.session_state.quiz_last_correct:
            st.success("✅ Correct!")
            st.write(question.get("explanation", ""))
        else:
            st.error("❌ Wrong answer.")
            st.info(f"Hint: {question.get('hint', 'Review the related lecture concept.')}")

        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.quiz_last_correct and st.button("Retry", use_container_width=True):
                st.session_state.quiz_answered = False
                st.session_state.quiz_last_correct = None
                st.rerun()
        with col2:
            next_label = "Finish Quiz" if idx == total - 1 else "Next Question"
            if st.button(next_label, use_container_width=True):
                st.session_state.quiz_current += 1
                st.session_state.quiz_answered = False
                st.session_state.quiz_last_correct = None
                st.rerun()

        with st.expander("Show explanation and correct answer", expanded=bool(st.session_state.quiz_last_correct)):
            st.write(f"**Correct answer:** {correct_answer}")
            st.write(question.get("explanation", ""))

    st.divider()
    if st.button("Exit Quiz and return to Chat"):
        st.session_state.active_view = "chat"
        st.rerun()


def render_diagram_view():
    diagram = st.session_state.diagram_data or {}
    st.title("🧩 Lecture Diagram")

    if not diagram:
        st.info("No diagram has been generated yet.")
        if st.button("Back to Chat"):
            st.session_state.active_view = "chat"
            st.rerun()
        return

    st.subheader(diagram.get("title", "Lecture Diagram"))
    mermaid_code = diagram.get("content", "")
    render_mermaid_diagram(mermaid_code)

    with st.expander("Mermaid source", expanded=False):
        st.code(mermaid_code, language="mermaid")

    if st.button("Back to Chat"):
        st.session_state.active_view = "chat"
        st.rerun()

# ── dedicated tool views ───────────────────────────────────────────────────────
# When a tool is active, render it as its own page/view instead of mixing it into chat.
if st.session_state.active_view == "quiz":
    render_interactive_quiz_view()
    st.stop()

if st.session_state.active_view == "diagram":
    render_diagram_view()
    st.stop()

# ── chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg.get("sources_used"):
            with st.expander("Sources used", expanded=False):
                for s in msg["sources_used"]:
                    st.markdown(f"- {s}")
        if msg.get("agent_trace"):
            with st.expander("Agent trace", expanded=False):
                for step in msg["agent_trace"]:
                    st.markdown(f"- `{step}`")

# ── Sidebar Actions Processing ─────────────────────────────────────────────────
lang_instruction = "in English" if tool_lang == "English" else "باللغة العربية"
target_instruction = "all the provided lectures" if selected_file == "جميع المحاضرات" else f"ONLY the lecture titled '{selected_file}'"

if sum_btn:
    SUMMARY_QUERY = f"Summarize {target_instruction} {lang_instruction}"
    display_text = f"Summarize {selected_file} ({tool_lang})"
    st.session_state.chat_history.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)
    with st.chat_message("assistant"):
        with st.spinner("جاري إعداد الملخص... ⏳"):
            answer = stream_query(SUMMARY_QUERY, target_files)
    st.toast("✅ اكتمل الملخص!", icon="✅")
    if answer:
        st.session_state.chat_history.append({"role": "assistant", "content": f"**Summary**\n\n{answer}"})
        st.rerun()

if diagram_btn:
    if selected_file == "جميع المحاضرات" and uploaded_files and len(uploaded_files) > 1:
        st.error("⚠️ يرجى تحديد محاضرة واحدة فقط من القائمة المنسدلة (المحاضرة المستهدفة) لإنشاء الرسم التوضيحي. دمج عدة محاضرات يسبب خطأ في حجم البيانات.")
    else:
        with st.spinner("Generating lecture diagram... ⏳"):
            try:
                st.session_state.diagram_data = generate_lecture_diagram(target_files, tool_lang)
                st.session_state.active_view = "diagram"
                st.toast("✅ Diagram generated!", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Diagram generation failed: {e}")

if quiz_btn:
    with st.spinner("Generating interactive quiz... ⏳"):
        try:
            st.session_state.quiz_data = generate_interactive_quiz(target_files, num_questions, tool_lang)
            reset_quiz_runtime()
            st.session_state.active_view = "quiz"
            st.toast("✅ Interactive quiz ready!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Quiz generation failed: {e}")

# ── chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your documents...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري التفكير وصياغة الرد... ⏳"):
            answer = stream_query(user_input, target_files)

    st.toast("✅ اكتمل الرد! النظام جاهز لسؤالك التالي.", icon="✅")
    trace, sources = fetch_trace_and_sources(user_input, target_files)

    st.session_state.chat_history.append({
        "role":        "assistant",
        "content":     answer,
        "sources_used": sources,
        "agent_trace": trace,
    })
    st.rerun()
