from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Step 1: Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Split text into chunks
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

# Step 3: Create FAISS vector store
def create_vector_store(chunks):
    return FAISS.from_documents(chunks, embedding_model)

# Step 4: Retrieve relevant chunks from FAISS
def get_top_chunks(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)

# Optional: Remove duplicates (e.g., repeated resume data)
def deduplicate_chunks(chunks):
    seen = set()
    unique = []
    for chunk in chunks:
        key = chunk.page_content[:100]  # crude heuristic
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique

# Step 5: Ask question using local FLAN-T5 model
def ask_question_with_context(chunks, query):
    # Load model and tokenizer
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Create a HuggingFace pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Based only on the provided context, answer the user's question as accurately and specifically as possible.

Context:
{context}

Question:
{question}

Instructions:
- Extract relevant information ONLY for the person whose name is mentioned in the question.
- Look for details such as: full name, qualifications, grades, skills, projects, and professional experience.
- If the person is not mentioned in the context, reply with: "No information found for the specified person."
- Do not include information about any other individuals.
- Be factual and concise, and include only what is present in the context.
- Only include information related to the person named in the question.
- Ignore and exclude any data about other individuals.
- If no information is found for the named person, say: "No information found for the specified person."
- Include only factual data mentioned in the context: CGPA, grades, skills, projects, or experience.
- If asking for names of people who know a particular skill, experience , grade tec then give list of names of people associated with that particular skill, experience, grade.

Answer:"""
    )

    # Build and run the prompt + model chain
    chain = prompt | llm

    # Combine the retrieved chunks into a single context string
    context = "\n".join([doc.page_content for doc in chunks])

    # Run the chain with the context and query
    return chain.invoke({"context": context, "question": query})
