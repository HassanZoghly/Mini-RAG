import os
import base64
from typing import Any, Dict, Optional
import requests
import streamlit as st

# ==========================================
# ⚙️ System Configuration (Hidden from User)
# ==========================================
API_URL = os.getenv("MINI_RAG_API_URL", "http://localhost:8000")
PROJECT_ID = 1
CHUNK_SIZE = 800
OVERLAP_SIZE = 150
DO_RESET = 1
RETRIEVAL_LIMIT = 5
REQUEST_TIMEOUT = 180
# ==========================================

def stream_answer(query: str):
    payload = {"text": query, "limit": RETRIEVAL_LIMIT}
    try:
        response = requests.post(f"{API_URL}/v1/nlp/index/answer_stream/{PROJECT_ID}", json=payload, stream=True, timeout=REQUEST_TIMEOUT)

        if response.status_code >= 400:
            yield f"- **Error**: HTTP {response.status_code} - {response.text}"
            return

        # قراءة الداتا قطعة قطعة وتمريرها للواجهة
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                yield chunk
    except Exception as e:
        yield f"- **Connection Error**: {str(e)}"


def post_json(route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{API_URL}{route}", json=payload, timeout=REQUEST_TIMEOUT)
    try:
        body = response.json()
    except ValueError:
        body = {"signal": "invalid_json", "response_text": response.text}
    if response.status_code >= 400:
        detail = body.get("signal") or body.get("detail") or response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return body

def upload_pdf(uploaded_file) -> Dict[str, Any]:
    content_type = uploaded_file.type or "application/pdf"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), content_type)}
    response = requests.post(f"{API_URL}/v1/data/upload/{PROJECT_ID}", files=files, timeout=REQUEST_TIMEOUT)
    try:
        body = response.json()
    except ValueError:
        body = {"signal": "invalid_json", "response_text": response.text}
    if response.status_code >= 400:
        detail = body.get("signal") or body.get("detail") or response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return body

def run_processing_pipeline(file_id: str) -> Dict[str, Any]:
    process_body = post_json(
        route=f"/v1/data/process/{PROJECT_ID}",
        payload={
            "file_id": file_id,
            "chunk_size": CHUNK_SIZE,
            "overlap_size": OVERLAP_SIZE,
            "do_reset": DO_RESET,
        },
    )
    index_body = post_json(
        route=f"/v1/nlp/index/push/{PROJECT_ID}",
        payload={"do_reset": DO_RESET},
    )
    return {"process": process_body, "index": index_body}

def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_ready" not in st.session_state:
        st.session_state.is_ready = False

def render_chat_history() -> None:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=False)

            if message["role"] == "assistant":
                if "image_base64" in message:
                    img_bytes = base64.b64decode(message["image_base64"])
                    st.image(img_bytes, caption="Generated intelligently by Napkin AI", use_container_width=True)
                else:
                    if st.button("🎨 Generate Visual Diagram", key=f"napkin_btn_{idx}"):
                        with st.spinner("Napkin AI is drawing... Please wait (might take 10-20 seconds)"):
                            try:
                                res = requests.post(
                                    f"{API_URL}/v1/nlp/visualize",
                                    json={"text": message["content"]},
                                    timeout=REQUEST_TIMEOUT
                                )
                                if res.status_code == 200:
                                    data = res.json()
                                    if "image_base64" in data:
                                        st.session_state.messages[idx]["image_base64"] = data["image_base64"]
                                        st.rerun()
                                    else:
                                        st.warning("Napkin AI returned an empty image.")
                                else:
                                    st.error(f"Failed to generate diagram: {res.text}")
                            except Exception as exc:
                                st.error(f"Error: {exc}")

def main() -> None:
    st.set_page_config(page_title="AI Tutor", layout="centered")
    init_chat_state()

    # ==========================================
    # Sidebar: Control Panel & Tools
    # ==========================================
    with st.sidebar:
        st.title("🛠️ Control Panel")

        if st.session_state.is_ready:
            st.success("Document Loaded and Ready!")
            if st.button("⬅️ Upload New Document", use_container_width=True):
                st.session_state.is_ready = False
                st.session_state.messages = []
                st.rerun()

            st.divider()

            st.subheader("Learning Tools")
            if st.button("📝 Generate Quiz", use_container_width=True, type="primary"):
                with st.spinner("Creating a quiz from your lecture..."):
                    try:
                        res = requests.post(f"{API_URL}/v1/nlp/quiz/{PROJECT_ID}", timeout=REQUEST_TIMEOUT)
                        if res.status_code == 200:
                            quiz_data = res.json().get("quiz")
                            if quiz_data:
                                st.session_state.messages.append({"role": "assistant", "content": f"**Here is your Quiz!**\n\n{quiz_data}"})
                                st.rerun()
                            else:
                                st.error("Received empty response from the server.")
                        else:
                            st.error(f"Could not generate quiz. Server returned: {res.status_code}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

    # ==========================================
    # Screen 1: Document Upload & Setup
    # ==========================================
    if not st.session_state.is_ready:
        st.title("📚 AI Tutor")
        st.write("Upload your lecture PDF, and I will prepare it for your questions.")

        uploaded_file = st.file_uploader("Upload Document", type=["pdf"], label_visibility="collapsed")

        if uploaded_file:
            if st.button("Process Document & Start Learning 🚀", use_container_width=True, type="primary"):
                with st.spinner("Reading and preparing the document... (This might take a few seconds)"):
                    try:
                        upload_response = upload_pdf(uploaded_file)
                        file_id = upload_response.get("file_id")

                        if not file_id:
                            st.error("Missing ID! Please check the API response format below:")
                            st.json(upload_response)
                            st.stop()

                        run_processing_pipeline(file_id=file_id)

                        st.session_state.is_ready = True
                        st.rerun()
                    except Exception as exc:
                        st.error(f"An error occurred: {exc}")

    # ==========================================
    # Screen 2: Interactive Chat
    # ==========================================
    else:
        st.title("💬 Ask about the lecture")
        st.divider()

        render_chat_history()

        user_prompt = st.chat_input("Type your question here...")

        if user_prompt:
            # 1. إضافة وعرض رسالة المستخدم
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt, unsafe_allow_html=False)

            # 2. عرض رسالة الموديل بنظام الـ Streaming 🚀
            with st.chat_message("assistant"):
                try:
                    # st.write_stream هتقرأ من الـ generator وتطبع حرف بحرف فوراً
                    answer = st.write_stream(stream_answer(user_prompt))
                except Exception as exc:
                    answer = f"- **Error**: {exc}"
                    st.markdown(answer)

                # لو الرد رجع فاضي لأي سبب
                if not answer:
                    answer = "Sorry, I could not find an answer in this document."
                    st.markdown(answer)

            # 3. حفظ الرد النهائي في الجلسة وعمل ريفرش عشان زرار الـ Napkin يظهر
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

if __name__ == "__main__":
    main()
