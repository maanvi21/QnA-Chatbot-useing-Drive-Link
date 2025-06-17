# app.py - Fixed version with better error handling and type safety
import streamlit as st
import os
import tempfile
import traceback
from utils import extract_drive_file_id, download_all_files_from_drive_folder, extract_text_from_file
from vector_store import process_documents_and_create_database, query_system
import fitz  # PyMuPDF
from docx import Document

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with better error handling"""
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text.strip())
        doc.close()
        
        full_text = "\n".join(text_parts)
        return full_text.strip()
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX with better error handling"""
    if not docx_path or not os.path.exists(docx_path):
        return ""
    
    try:
        doc = Document(docx_path)
        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())
        
        full_text = "\n".join(text_parts)
        return full_text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX {docx_path}: {e}")
        return ""

def extract_text_from_txt(txt_content):
    """Extract text from TXT content"""
    try:
        if isinstance(txt_content, bytes):
            return txt_content.decode("utf-8", errors="ignore").strip()
        elif isinstance(txt_content, str):
            return txt_content.strip()
        else:
            return str(txt_content).strip()
    except Exception as e:
        st.error(f"Error reading TXT content: {e}")
        return ""

def process_uploaded_files(uploaded_files):
    """Process uploaded files with improved error handling"""
    if not uploaded_files:
        return [], []
    
    documents_text, filenames = [], []
    
    for uploaded_file in uploaded_files:
        if not uploaded_file:
            continue
            
        filename = uploaded_file.name
        st.write(f"📄 Processing: {filename}")
        
        try:
            # Create temporary file
            file_extension = os.path.splitext(filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            text = ""
            
            # Extract text based on file type
            if file_extension == ".pdf":
                text = extract_text_from_pdf(tmp_path)
            elif file_extension == ".docx":
                text = extract_text_from_docx(tmp_path)
            elif file_extension == ".txt":
                text = extract_text_from_txt(uploaded_file.getvalue())
            else:
                st.warning(f"⚠️ Unsupported file type: {filename}")
                os.unlink(tmp_path)
                continue

            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Validate extracted text
            if text and isinstance(text, str) and text.strip():
                documents_text.append(text.strip())
                filenames.append(filename)
                st.success(f"✅ Extracted {len(text)} characters from {filename}")
            else:
                st.warning(f"⚠️ No text extracted from {filename}")
                
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {str(e)}")
            print(f"Error processing {filename}: {traceback.format_exc()}")
            continue
    
    return documents_text, filenames

def process_drive_links(drive_links):
    """Process Google Drive links with improved error handling"""
    if not drive_links:
        return [], []
    
    documents_text, filenames = [], []
    
    for i, drive_url in enumerate(drive_links):
        if not drive_url or not drive_url.strip():
            continue
            
        st.write(f"🔗 Processing Drive link {i+1}")
        
        try:
            folder_id = extract_drive_file_id(drive_url.strip())
            if not folder_id:
                st.error(f"❌ Could not extract folder ID from link {i+1}")
                continue
            
            st.write(f"📁 Folder ID: {folder_id}")
            file_tuples = download_all_files_from_drive_folder(folder_id)
            
            if not file_tuples:
                st.warning(f"⚠️ No supported files found in folder link {i+1}")
                continue
            
            st.write(f"📋 Found {len(file_tuples)} files")
            
            for file_name, file_path in file_tuples:
                try:
                    if not os.path.exists(file_path):
                        st.error(f"❌ Downloaded file not found: {file_path}")
                        continue
                    
                    with open(file_path, "rb") as f:
                        text = extract_text_from_file(f)
                    
                    # Clean up the downloaded file
                    os.unlink(file_path)
                    
                    if text and isinstance(text, str) and text.strip():
                        documents_text.append(text.strip())
                        filenames.append(file_name)
                        st.success(f"✅ Processed {file_name}: {len(text)} characters")
                    else:
                        st.warning(f"⚠️ No text extracted from {file_name}")
                        
                except Exception as e:
                    st.error(f"❌ Error processing {file_name}: {str(e)}")
                    # Clean up file if it exists
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                    continue
                    
        except Exception as e:
            st.error(f"❌ Error processing drive link {i+1}: {str(e)}")
            print(f"Drive link error: {traceback.format_exc()}")
            continue
    
    return documents_text, filenames

def validate_inputs(documents_text, filenames):
    """Validate and clean input data"""
    if not documents_text or not filenames:
        return [], []
    
    clean_docs, clean_files = [], []
    
    for doc, fname in zip(documents_text, filenames):
        # Type validation
        if not isinstance(doc, str) or not isinstance(fname, str):
            st.warning(f"⚠️ Skipping invalid types - File: {fname} (doc: {type(doc)}, name: {type(fname)})")
            continue
        
        # Content validation
        doc_clean = doc.strip()
        fname_clean = fname.strip()
        
        if not doc_clean:
            st.warning(f"⚠️ Skipping empty document: {fname_clean}")
            continue
        
        if not fname_clean:
            st.warning(f"⚠️ Skipping document with empty filename")
            continue
        
        # Length validation
        if len(doc_clean) < 10:
            st.warning(f"⚠️ Skipping very short document: {fname_clean} ({len(doc_clean)} chars)")
            continue
        
        clean_docs.append(doc_clean)
        clean_files.append(fname_clean)
    
    return clean_docs, clean_files

def main():
    st.set_page_config(
        page_title="Smart Resume Chatbot", 
        page_icon="🤖", 
        layout="wide"
    )
    
   # Session state init
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'person_skills' not in st.session_state:
        st.session_state.person_skills = None
    if 'person_info' not in st.session_state:
        st.session_state.person_info = None
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose resume files", 
            type=['pdf', 'docx', 'txt'], 
            accept_multiple_files=True
        )
        
        st.markdown("---")
        st.header("🔗 Google Drive Links")
        drive_links_text = st.text_area(
            "Paste Google Drive sharing links (one per line)", 
            height=100
        )
        
        process_button = st.button("🚀 Process Documents", type="primary")

    if process_button:
        st.session_state.processing_complete = False
        documents_text, filenames = [], []

        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                up_texts, up_files = process_uploaded_files(uploaded_files)
                documents_text.extend(up_texts)
                filenames.extend(up_files)

        if drive_links_text.strip():
            links = [link.strip() for link in drive_links_text.split('\n') if link.strip()]
            with st.spinner("Processing Google Drive links..."):
                gd_texts, gd_files = process_drive_links(links)
                documents_text.extend(gd_texts)
                filenames.extend(gd_files)

        clean_docs, clean_files = validate_inputs(documents_text, filenames)

        if clean_docs:
            with st.spinner("🧠 Building knowledge base with LLM..."):
                try:
                    vs, ps, pi, chunks = process_documents_and_create_database(clean_docs, clean_files)
                    st.session_state.vector_store = vs
                    st.session_state.person_skills = ps
                    st.session_state.person_info = pi
                    st.session_state.all_chunks = chunks
                    st.session_state.processing_complete = True
                    st.success(f"✅ Processed {len(clean_docs)} documents and built knowledge base")
                except Exception as e:
                    st.error(f"Error building knowledge base: {str(e)}")
        else:
            st.warning("⚠️ No valid documents to process.")

    if st.session_state.processing_complete:
        st.header("💬 Ask a Question")
        query = st.text_input("Type your question (e.g. 'Who knows CNN?')")
        if st.button("🔍 Get Answer") and query.strip():
            with st.spinner("Asking the LLM..."):
                answer = query_system(
                    st.session_state.vector_store,
                    st.session_state.person_skills,
                    st.session_state.person_info,
                    st.session_state.all_chunks,
                    query
                )
                st.subheader("💡 Answer")
                st.write(answer)

if __name__ == "__main__":
    main()
