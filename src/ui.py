import streamlit as st
import requests
import os
import json

# Set up backend URL
BACKEND_URL = "http://localhost:5000/v1"

st.set_page_config(page_title="Mini RAG Tutor", layout="wide")

st.title("🎓 Mini RAG Tutor")
st.markdown("An AI avatar tutor powered by your documents.")

# Sidebar for project management and file uploading
with st.sidebar:
    st.header("Project Setup")
    project_id = st.text_input("Project ID", value="default_project")

    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "png", "jpg", "jpeg"]
    )

    if st.button("Upload & Process"):
        if uploaded_file is not None and project_id:
            with st.spinner("Uploading file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                upload_res = requests.post(f"{BACKEND_URL}/data/upload/{project_id}", files=files)

                if upload_res.status_code == 200:
                    data = upload_res.json()
                    st.success(f"File uploaded successfully! ID: {data.get('File_id')}")

                    with st.spinner("Processing file..."):
                        process_payload = {
                            "file_id": data.get('File_id'),
                            "chunk_size": 512,
                            "overlap_size": 50,
                            "do_reset": 0
                        }
                        process_res = requests.post(f"{BACKEND_URL}/data/process/{project_id}", json=process_payload)
                        if process_res.status_code == 200:
                            st.success("File processed into chunks!")

                            with st.spinner("Indexing into Vector DB..."):
                                push_payload = {"do_reset": 0}
                                push_res = requests.post(f"{BACKEND_URL}/nlp/index/push/{project_id}", json=push_payload)
                                if push_res.status_code == 200:
                                    st.success("Indexing complete! You can now ask questions.")
                                else:
                                    st.error(f"Indexing failed: {push_res.text}")
                        else:
                            st.error(f"Processing failed: {process_res.text}")
                else:
                    st.error(f"Upload failed: {upload_res.text}")
        else:
            st.warning("Please provide a Project ID and select a file.")

# Main interaction area
st.header("Chat & Actions")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Ask Questions")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Ask the backend
        with st.spinner("Thinking..."):
            try:
                search_payload = {
                    "text": prompt,
                    "limit": 5
                }
                res = requests.post(f"{BACKEND_URL}/nlp/index/answer/{project_id}", json=search_payload)
                if res.status_code == 200:
                    answer = res.json().get("answer", "I could not find an answer.")
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error getting answer: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

with col2:
    st.subheader("Tools")

    if st.button("Summarize Lecture", use_container_width=True):
        if project_id:
            with st.spinner("Generating summary..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/nlp/index/summarize/{project_id}")
                    if res.status_code == 200:
                        summary = res.json().get("summary")
                        st.session_state.messages.append({"role": "assistant", "content": f"**Lecture Summary:**\n\n{summary}"})
                        st.rerun()
                    else:
                        st.error(f"Error generating summary: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.warning("Project ID is required.")

    st.markdown("---")
    st.markdown("### Visualize with Napkin AI")
    st.markdown("1. Copy a markdown response from the chat.")
    st.markdown("2. Click the button below to go to Napkin AI.")
    st.markdown("3. Paste your text to instantly generate diagrams and infographics!")
    st.link_button("Open Napkin AI", "https://app.napkin.ai/", use_container_width=True)
