import streamlit as st
import os
from utils import (
    download_all_files_from_drive_folder,
    extract_text_from_file
)
from vector_store import (
    create_chunks,
    create_vector_store,
    deduplicate_chunks,
    multi_query_search,
    ask_question_with_context
)

st.set_page_config(page_title="📚 Summarizer Chatbot", layout="wide")
st.title("📁 Google Drive Folder RAG Chatbot")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 📂 Google Drive folder link input
folder_link = st.text_input("🔗 Paste your Google Drive folder link (shared with service account):")

# Store vector store and raw text state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

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
                chunks = deduplicate_chunks(chunks)
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.raw_text = combined_text
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
            relevant_chunks = multi_query_search(st.session_state.vector_store, question, k=6)
            relevant_chunks = deduplicate_chunks(relevant_chunks)
            answer = ask_question_with_context(relevant_chunks, question)

            # Fallback if no answer found
            if "no information found" in answer.lower() or "no person found" in answer.lower():
                from vector_store import fallback_keyword_search
                fallback_answer = fallback_keyword_search(st.session_state.raw_text, question)
                if "no information found" not in fallback_answer.lower():
                    answer = fallback_answer

            st.markdown("### 💬 Answer")
            st.write(answer)
