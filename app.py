import streamlit as st
import os
from utils import (
    download_all_files_from_drive_folder,
    extract_text_from_file
)
from vector_store import (
    create_chunks,
    create_vector_store,
    get_top_chunks,
    ask_question_with_context,
    deduplicate_chunks  # ✅ new import
)

st.set_page_config(page_title="📚 Summarizer Chatbot", layout="wide")
st.title("📁 Google Drive Folder RAG Chatbot")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 📂 Google Drive folder link input
folder_link = st.text_input("🔗 Paste your Google Drive folder link (shared with service account):")

# Store vector store state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Load and process folder
if st.button("📥 Load Folder"):
    if "folders/" in folder_link:
        folder_id = folder_link.split("folders/")[1].split("?")[0]

        with st.spinner("📂 Downloading all files from folder..."):
            file_paths = download_all_files_from_drive_folder(folder_id)

        combined_text = ""
        for path in file_paths:
            with open(path, "rb") as f:
                combined_text += extract_text_from_file(f) + "\n"

        if combined_text.strip():
            with st.spinner("🔍 Creating vector index..."):
                chunks = create_chunks(combined_text)
                st.session_state.vector_store = create_vector_store(chunks)
            st.success("✅ Folder indexed successfully!")
        else:
            st.warning("⚠️ No readable documents found in the folder.")
    else:
        st.error("❌ Please paste a valid Google Drive folder link.")

# Q&A interface
if st.session_state.vector_store:
    st.markdown("---")
    question = st.text_input("❓ Ask something about the documents:")
    if question:
        with st.spinner("🤖 Thinking..."):
            top_chunks = get_top_chunks(st.session_state.vector_store, question, k=5)
            top_chunks = deduplicate_chunks(top_chunks)  # ✅ Remove repeated content
            answer = ask_question_with_context(top_chunks, question)
            st.markdown("### 💬 Answer")
            st.write(answer)
