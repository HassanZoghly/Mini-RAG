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

    # 4. قسم الاختبار (مع تحديد عدد الأسئلة)
    st.subheader("🧠 الاختبار (Quiz)")
    num_questions = st.number_input("🔢 عدد الأسئلة:", min_value=1, max_value=50, value=5, step=1)
    quiz_btn = st.button("توليد أسئلة", use_container_width=True)

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

if quiz_btn:
    # تضمين عدد الأسئلة بين قوسين ليتمكن الباك-إند من قراءته
    QUIZ_QUERY = f"Generate quiz [{num_questions}] for {target_instruction} {lang_instruction}"
    display_text = f"Quiz ({num_questions} questions) on {selected_file} ({tool_lang})"
    st.session_state.chat_history.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)
    with st.chat_message("assistant"):
        with st.spinner("جاري إعداد الأسئلة... ⏳"):
            answer = stream_query(QUIZ_QUERY, target_files)
    st.toast("✅ اكتملت الأسئلة!", icon="✅")
    if answer:
        st.session_state.chat_history.append({"role": "assistant", "content": f"**Quiz**\n\n{answer}"})
        st.rerun()

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
