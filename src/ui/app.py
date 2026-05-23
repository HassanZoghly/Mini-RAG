import os
import base64
from typing import Any, Dict, Optional
import requests
import streamlit as st
import uuid
import json

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
    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = str(uuid.uuid4())
    if "agent_chat_history" not in st.session_state:
        st.session_state.agent_chat_history = []

def agent_query_stream(query: str, project_id: str, session_id: str):
    """
    Calls POST /v1/nlp/agent-query/stream
    Yields text chunks as they arrive.
    Falls back to non-streaming if stream endpoint not available.
    """
    try:
        payload = {
            "query": query,
            "project_id": str(project_id),
            "asset_ids": [],
            "session_id": session_id,
            "image_paths": []
        }
        with requests.post(
            f"{API_URL}/v1/nlp/agent-query/stream",
            json=payload,
            stream=True,
            timeout=120
        ) as r:
            for line in r.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        token = decoded[6:]
                        if token != "[DONE]":
                            yield token
    except Exception as e:
        yield f"[Error: {e}]"

def show_agent_trace(trace: list):
    """Shows the agent execution trace in an expander."""
    with st.expander("Agent execution trace", expanded=False):
        for step in trace:
            st.markdown(f"- `{step}`")

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

        st.sidebar.markdown("---")
        mode = st.sidebar.radio(
            "Query mode",
            ["Classic RAG", "Agent Mode"],
            index=0
        )

        if st.session_state.is_ready:
            st.success("Document Loaded and Ready!")
            if st.button("⬅️ Upload New Document", use_container_width=True):
                st.session_state.is_ready = False
                st.session_state.messages = []
                st.rerun()

            st.divider()

            st.subheader("Learning Tools")

            # زر الكويز
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

            # زر التلخيص — Streaming عشان منحصلش timeout مع المحاضرات الكبيرة أو الـ OCR
            if st.button("📄 Summarize Lecture", use_container_width=True, type="secondary"):
                with st.spinner("Starting summary generation..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/v1/nlp/summarize/{PROJECT_ID}",
                            stream=True,
                            timeout=REQUEST_TIMEOUT,
                        )
                        if response.status_code >= 400:
                            st.error(f"Could not generate summary. Server returned: {response.status_code}")
                        else:
                            # نعرض الـ summary وهي بتتكتب word by word زي الـ chat
                            with st.chat_message("assistant"):
                                full_summary = st.write_stream(
                                    chunk
                                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True)
                                    if chunk
                                )
                            if full_summary:
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"**Lecture Summary:**\n\n{full_summary}",
                                })
                                st.rerun()
                            else:
                                st.error("Received empty response from the server.")
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

        if mode == "Classic RAG":
            render_chat_history()

            user_prompt = st.chat_input("Type your question here...")

            if user_prompt:
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt, unsafe_allow_html=False)

                with st.chat_message("assistant"):
                    try:
                        answer = st.write_stream(stream_answer(user_prompt))
                    except Exception as exc:
                        answer = f"- **Error**: {exc}"
                        st.markdown(answer)

                    if not answer:
                        answer = "Sorry, I could not find an answer in this document."
                        st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

        elif mode == "Agent Mode":
            st.markdown("#### Agent mode")
            st.caption(f"Session: {st.session_state.agent_session_id[:8]}...")

            # Show chat history
            for msg in st.session_state.agent_chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg.get("trace"):
                        show_agent_trace(msg["trace"])

            user_input = st.chat_input("Ask the agents...")
            if user_input:
                st.session_state.agent_chat_history.append({
                    "role": "user", "content": user_input
                })
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    response_container = st.empty()
                    full_response = ""
                    for token in agent_query_stream(
                        query=user_input,
                        project_id=st.session_state.get("project_id", str(PROJECT_ID)),
                        session_id=st.session_state.agent_session_id
                    ):
                        full_response += token
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)

                # Fetch trace from non-streaming endpoint for display
                try:
                    trace_resp = requests.get(
                        f"{API_URL}/v1/nlp/agent-trace/{st.session_state.agent_session_id}",
                        timeout=10
                    )
                    trace_data = trace_resp.json() if trace_resp.ok else []
                except:
                    trace_data = []
                
                if trace_data and isinstance(trace_data, list):
                    # Fallback mapping if backend returned full memory records
                    mapped_trace = []
                    for record in trace_data:
                        if isinstance(record, dict) and "content" in record:
                            mapped_trace.append(record["content"][:150] + "...")
                        else:
                            mapped_trace.append(str(record))
                    trace_data = mapped_trace

                st.session_state.agent_chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "trace": trace_data
                })

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("New session"):
                    st.session_state.agent_session_id = str(uuid.uuid4())
                    st.session_state.agent_chat_history = []
                    st.rerun()

if __name__ == "__main__":
    main()
