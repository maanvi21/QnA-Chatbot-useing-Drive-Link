import os
import re
import torch
import streamlit as st
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from collections import defaultdict


# -----------------------------
# ✅ EMBEDDING LOADER (CACHED)
# -----------------------------
@st.cache_resource(show_spinner="🔄 Loading embedding model...")
def get_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        raise


# -----------------------------
# ✅ FLAN-T5 PIPELINE (CACHED)
# -----------------------------
@st.cache_resource(show_spinner="🔄 Loading FLAN-T5 model...")
def get_flant5_pipeline():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "finetune", "flant5-checkpoints", "checkpoint-90")

        if not os.path.exists(model_path):
            st.warning(f"⚠️ Fine-tuned model not found at {model_path}, using base model")
            model_path = "google/flan-t5-base"
            local_files_only = False
        else:
            local_files_only = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_files_only).to(device)

        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            device=0 if torch.cuda.is_available() else -1,
            do_sample=False,
            temperature=0.3
        )

    except Exception as e:
        st.error(f"❌ Error loading FLAN-T5: {e}")
        traceback.print_exc()
        return None


# -----------------------------
# ✅ CHUNKING
# -----------------------------
def create_chunks(text, filename=None):
    if not text or not isinstance(text, str):
        st.warning("⚠️ Invalid text input for chunking")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text.strip())
    return [
        Document(
            page_content=chunk.strip(),
            metadata={"source": filename.lower().replace(" ", "_") if filename else "unknown", "chunk_id": i}
        ) for i, chunk in enumerate(chunks) if chunk.strip()
    ]


# -----------------------------
# ✅ NAME + SKILLS EXTRACTION
# -----------------------------
def extract_names_and_skills_with_llm(text):
    if not text or not isinstance(text, str):
        return []

    text = text.strip()[:2000]  # truncate
    prompt = f"""Extract the full name and technical skills from this resume/CV text.\nOnly extract real names and actual technical skills.\n\nText:\n{text}\n\nFormat your response exactly as:\nName: [Full Name]; Skills: [Skill1, Skill2, Skill3]\n\nOnly include actual names and skills.\nIf none found, respond with 'None found'.\n\nResponse:"""

    pipe = get_flant5_pipeline()
    if not pipe:
        return []

    try:
        result = pipe(prompt)
        generated = result[0].get("generated_text", "").strip()
        if "None found" in generated:
            return []

        match = re.match(r"Name:\s*(.+?);\s*Skills:\s*(.+)", generated, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            skills = [s.strip() for s in match.group(2).split(",") if s.strip()]
            return [{"name": name, "skills": skills}] if name else []
        return []

    except Exception as e:
        st.error(f"❌ Error during name/skill extraction: {e}")
        return []


# -----------------------------
# ✅ PERSON-SKILL DATABASE
# -----------------------------
def create_person_skill_database(chunks):
    people = defaultdict(set)
    info = {}
    for chunk in chunks:
        if not chunk.page_content.strip():
            continue
        extracted = extract_names_and_skills_with_llm(chunk.page_content)
        for entry in extracted:
            name = entry.get("name")
            skills = entry.get("skills", [])
            if name:
                people[name].update(skills)
                if name not in info:
                    info[name] = {
                        "source": chunk.metadata.get("source", "unknown"),
                        "text_sample": chunk.page_content[:300]
                    }
    return {k: list(v) for k, v in people.items()}, info


# -----------------------------
# ✅ VECTOR STORE
# -----------------------------
def create_vector_store(chunks):
    valid = [c for c in chunks if c.page_content.strip()]
    model = get_embedding_model()
    return FAISS.from_documents(valid, model)


# -----------------------------
# ✅ QUERY INTERFACE
# -----------------------------
def get_top_chunks(vector_store, query, k=6):
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"❌ Error getting top chunks: {e}")
        return []


def ask_question_with_context(chunks, query, person_skills, person_info):
    if not query:
        return "Please provide a valid question."

    skill_patterns = [
        r"who knows\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"who has experience with\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"who uses\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"find people with\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"who worked with\s+([a-zA-Z0-9\s\.\#\+\-]+)"
    ]

    query_lower = query.lower()
    for pattern in skill_patterns:
        match = re.search(pattern, query_lower)
        if match:
            skill = match.group(1).strip()
            if not person_skills:
                break
            matches = [name for name, skills in person_skills.items() if any(skill.lower() in s.lower() for s in skills)]
            return f"People who know {skill}: {', '.join(matches)}" if matches else f"No one has '{skill}' listed."

    # fallback to LLM
    if not chunks:
        return "No relevant information found."

    context = "\n\n".join([c.page_content.strip() for c in chunks[:5] if c.page_content.strip()])[:1500]
    prompt = f"""Answer the question based on the following resume content. Mention names when available.\n\nResume:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

    pipe = get_flant5_pipeline()
    if not pipe:
        return "Error: LLM pipeline not available."

    try:
        result = pipe(prompt)
        return result[0].get("generated_text", "").strip()
    except Exception as e:
        st.error(f"❌ Error generating answer: {e}")
        return "Error generating answer."


# -----------------------------
# ✅ MAIN PROCESSING FUNCTION
# -----------------------------
def process_documents_and_create_database(doc_texts, filenames):
    all_chunks = []
    for text, fname in zip(doc_texts, filenames):
        all_chunks.extend(create_chunks(text, fname))

    if not all_chunks:
        raise ValueError("No valid chunks found")

    vector_store = create_vector_store(all_chunks)
    person_skills, person_info = create_person_skill_database(all_chunks)
    return vector_store, person_skills, person_info, all_chunks


# -----------------------------
# ✅ QUERY SYSTEM ENTRY POINT
# -----------------------------
def query_system(vector_store, person_skills, person_info, all_chunks, query):
    if not query or not isinstance(query, str):
        return "Please provide a valid question."

    query = query.strip()
    if not query:
        return "Please provide a valid question."

    try:
        top_chunks = get_top_chunks(vector_store, query, k=6)
        answer = ask_question_with_context(top_chunks, query, person_skills, person_info)
        return answer
    except Exception as e:
        st.error(f"❌ Error processing query: {e}")
        return f"Error processing query: {str(e)}"
