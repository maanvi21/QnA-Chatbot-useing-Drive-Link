from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableMap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Step 1: Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Chunk text
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

# Step 3: Create FAISS vector store
def create_vector_store(chunks):
    return FAISS.from_documents(chunks, embedding_model)

# Step 4: Search relevant chunks
def get_top_chunks(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)

# Optional: Filter out duplicate/resume-like repetitions
def deduplicate_chunks(chunks):
    seen = set()
    unique = []
    for chunk in chunks:
        key = chunk.page_content[:100]  # crude heuristic
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique

# Step 5: Local FLAN-T5 based generation
def ask_question_with_context(chunks, query):
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""The following contains multiple resumes.
List the names of all people who have submitted their resume based on the content below:

{context}

Question: {question}"""
    )

    chain = prompt | llm

    context = "\n".join([doc.page_content for doc in chunks])
    return chain.invoke({"context": context, "question": query})
