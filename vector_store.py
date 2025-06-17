import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from collections import defaultdict
import traceback

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Global pipeline variable
flant5_pipeline = None

def get_flant5_pipeline():
    """Load FLAN-T5 pipeline with better error handling"""
    global flant5_pipeline
    
    if flant5_pipeline is not None:
        return flant5_pipeline
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), "finetune", "flant5-checkpoints","checkpoint-90")
        
        # Check if fine-tuned model exists
        if not os.path.exists(model_path):
            print(f"⚠️ Fine-tuned model not found at {model_path}, using base model")
            model_path = "google/flan-t5-base"
            local_files_only = False
        else:
            local_files_only = True
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Loading FLAN-T5 from: {model_path} on device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_files_only).to(device)
        
        flant5_pipeline = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=256, 
            device=0 if torch.cuda.is_available() else -1,
            do_sample=False,
            temperature=0.3
        )
        
        print("✅ FLAN-T5 pipeline loaded successfully")
        return flant5_pipeline
        
    except Exception as e:
        print(f"❌ Error loading FLAN-T5 pipeline: {e}")
        print(traceback.format_exc())
        return None

def create_chunks(text, filename=None):
    """Create chunks from text with better validation"""
    if not text or not isinstance(text, str):
        print(f"⚠️ Invalid text input for chunking: {type(text)}")
        return []
    
    text = text.strip()
    if not text:
        print("⚠️ Empty text after stripping")
        return []
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create documents manually to ensure proper format
        chunks = splitter.split_text(text)
        docs = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": filename.lower().replace(" ", "_") if filename else "unknown",
                        "chunk_id": i
                    }
                )
                docs.append(doc)
        
        print(f"✅ Created {len(docs)} chunks from {filename or 'unknown file'}")
        return docs
        
    except Exception as e:
        print(f"❌ Error creating chunks: {e}")
        return []

def extract_names_and_skills_with_llm(text):
    """Extract names and skills using LLM with improved prompting"""
    if not text or not isinstance(text, str):
        return []
    
    text = text.strip()
    if not text:
        return []
    
    # Truncate text if too long
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    prompt = f"""Extract the full name and technical skills from this resume/CV text. 
Only extract real names of people and actual technical skills/technologies.

Text:
{text}

Format your response exactly as:
Name: [Full Name]; Skills: [Skill1, Skill2, Skill3]

Only include actual names and skills found in the text. If no clear name or skills are found, respond with "None found".

Response:"""

    pipe = get_flant5_pipeline()
    if not pipe:
        print("❌ FLAN-T5 pipeline not available")
        return []
    
    try:
        result = pipe(prompt, max_length=256, do_sample=False)
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = str(result)
        
        print(f"🤖 LLM Response: {generated_text}")
        
        if "None found" in generated_text or not generated_text.strip():
            return []
        
        lines = generated_text.strip().split('\n')
        parsed = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to match the expected format
            match = re.match(r"Name:\s*(.+?);\s*Skills:\s*(.+)", line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                skills_text = match.group(2).strip()
                
                # Parse skills
                skills = [s.strip() for s in skills_text.split(",") if s.strip()]
                
                if name and name.lower() not in ["none", "unknown", "not found"]:
                    parsed.append({"name": name, "skills": skills})
                    print(f"✅ Extracted: {name} -> {skills}")
        
        return parsed
        
    except Exception as e:
        print(f"❌ Error extracting with LLM: {e}")
        print(traceback.format_exc())
        return []

def create_person_skill_database(chunks):
    """Create person-skill database from chunks"""
    if not chunks:
        print("⚠️ No chunks provided for person-skill database")
        return {}, {}
    
    person_skills = defaultdict(set)
    person_info = {}

    print(f"🔍 Processing {len(chunks)} chunks for name/skill extraction...")
    
    for i, chunk in enumerate(chunks):
        if not hasattr(chunk, 'page_content'):
            print(f"⚠️ Invalid chunk format at index {i}")
            continue
            
        text = chunk.page_content
        source = chunk.metadata.get('source', 'unknown')
        
        if not text or not isinstance(text, str):
            continue
        
        extracted = extract_names_and_skills_with_llm(text)
        
        for entry in extracted:
            name = entry.get("name", "").strip()
            skills = entry.get("skills", [])
            
            if name and isinstance(skills, list):
                person_skills[name].update(skills)
                if name not in person_info:
                    person_info[name] = {
                        "source": source,
                        "all_names": [name],
                        "text_sample": text[:300] + "..." if len(text) > 300 else text
                    }

    # Convert sets to lists
    person_skills_dict = {k: list(v) for k, v in person_skills.items()}
    
    print(f"✅ Created person-skill database with {len(person_skills_dict)} people")
    return person_skills_dict, person_info

def create_vector_store(chunks):
    """Create FAISS vector store from chunks"""
    if not chunks:
        raise ValueError("No chunks provided to create vector store")
    
    if not embedding_model:
        raise ValueError("Embedding model not loaded")
    
    # Filter out invalid chunks
    valid_chunks = []
    for chunk in chunks:
        if hasattr(chunk, 'page_content') and chunk.page_content.strip():
            valid_chunks.append(chunk)
    
    if not valid_chunks:
        raise ValueError("No valid chunks found")
    
    try:
        vector_store = FAISS.from_documents(valid_chunks, embedding_model)
        print(f"✅ Created vector store with {len(valid_chunks)} documents")
        return vector_store
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        raise

def ask_question_with_context(chunks, query, person_skills=None, person_info=None):
    """Answer questions using context and skill database"""
    if not query or not isinstance(query, str):
        return "Please provide a valid question."
    
    query = query.strip()
    if not query:
        return "Please provide a valid question."
    
    # Skill-based query patterns
    skill_query_patterns = [
        r"who knows?\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"who has experience with\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"who uses\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"find people with\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"list people who know\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"who worked with\s+([a-zA-Z0-9\s\.\#\+\-]+)",
        r"who has\s+([a-zA-Z0-9\s\.\#\+\-]+)\s+skills?",
        r"who can work with\s+([a-zA-Z0-9\s\.\#\+\-]+)"
    ]
    
    query_lower = query.lower().strip()
    
    # Try skill pattern matching first
    for pattern in skill_query_patterns:
        match = re.search(pattern, query_lower)
        if match and person_skills:
            skill = match.group(1).strip()
            matching_people = []
            
            for name, skills in person_skills.items():
                if any(skill.lower() in s.lower() for s in skills):
                    matching_people.append(name)
            
            if matching_people:
                return f"People who know {skill}: {', '.join(matching_people)}"
            else:
                return f"No one in the database has {skill} listed in their skills."

    # Fallback to LLM-based QA
    if not chunks:
        return "No relevant information found."
    
    # Get context from chunks
    context_parts = []
    for chunk in chunks[:5]:  # Limit to top 5 chunks
        if hasattr(chunk, 'page_content') and chunk.page_content.strip():
            context_parts.append(chunk.page_content.strip())
    
    if not context_parts:
        return "No relevant information found."
    
    context = "\n\n".join(context_parts)
    
    # Truncate context if too long
    pipe = get_flant5_pipeline()
    if not pipe:
        return "Error: Language model not available."
    
    try:
        # Limit context length
        max_context_length = 1500
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""Answer the question based on the resume/CV information provided below. Be specific and mention names when available.

Resume Information:
{context}

Question: {query}

Answer:"""

        result = pipe(prompt, max_length=200, do_sample=False)
        
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get("generated_text", "")
        else:
            answer = str(result)
        
        return answer.strip() if answer.strip() else "I couldn't find a clear answer to your question."
        
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

def process_documents_and_create_database(documents_text_list, filenames_list):
    """Process all documents and create both vector store and person-skill database"""
    if not documents_text_list or not filenames_list:
        raise ValueError("No documents provided")
    
    if len(documents_text_list) != len(filenames_list):
        raise ValueError("Mismatch between documents and filenames")
    
    all_chunks = []
    processed_count = 0
    
    for text, filename in zip(documents_text_list, filenames_list):
        if not isinstance(text, str) or not isinstance(filename, str):
            print(f"⚠️ Skipping invalid entry - Filename: {filename} | Text type: {type(text)}")
            continue
        
        text = text.strip()
        filename = filename.strip()
        
        if not text:
            print(f"⚠️ Skipping empty document: {filename}")
            continue
        
        try:
            chunks = create_chunks(text, filename)
            if chunks:
                all_chunks.extend(chunks)
                processed_count += 1
                print(f"✅ Processed {filename}: {len(chunks)} chunks")
            else:
                print(f"⚠️ No chunks created for {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    if not all_chunks:
        raise ValueError("No valid chunks created from documents")

    print(f"✅ Total: {len(all_chunks)} chunks from {processed_count} documents")

    # Create vector store
    try:
        vector_store = create_vector_store(all_chunks)
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        raise

    # Create person-skill database
    try:
        person_skills, person_info = create_person_skill_database(all_chunks)
    except Exception as e:
        print(f"❌ Error creating person-skill database: {e}")
        person_skills, person_info = {}, {}

    # Display results
    if person_skills:
        print("📊 Extracted People and Skills:")
        for person, skills in person_skills.items():
            skills_display = ', '.join(skills[:5])
            if len(skills) > 5:
                skills_display += f' (+{len(skills) - 5} more)'
            print(f"  👤 {person}: {skills_display}")
    else:
        print("⚠️ No people/skills extracted from documents")

    return vector_store, person_skills, person_info, all_chunks

def get_top_chunks(vector_store, query, k=6):
    """Get top similar chunks from vector store"""
    if not vector_store or not query:
        return []
    
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        print(f"❌ Error getting top chunks: {e}")
        return []

def query_system(vector_store, person_skills, person_info, all_chunks, query):
    """Main query processing function"""
    if not query or not isinstance(query, str):
        return "Please provide a valid question."
    
    query = query.strip()
    if not query:
        return "Please provide a valid question."
    
    try:
        # Get relevant chunks
        top_chunks = get_top_chunks(vector_store, query, k=6)
        
        # Generate answer
        answer = ask_question_with_context(top_chunks, query, person_skills, person_info)
        return answer
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return f"Error processing query: {str(e)}"
