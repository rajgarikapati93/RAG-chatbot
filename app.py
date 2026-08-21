import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt
import chromadb
import tempfile
import os

st.set_page_config(page_title="Ask My Documents", page_icon="📄")
st.title("📄 Upload files you would like me to understand")
st.caption("Upload a PDF, DOCX, or TXT file and ask questions about it.")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    api_key = st.secrets["GOOGLE_API_KEY"]
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=api_key, temperature=0.2)

embedding_model = load_embedding_model()
llm = load_llm()

if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()
    st.session_state.collection = st.session_state.chroma_client.create_collection(
        name="session_docs"
    )
    st.session_state.processed_files = []
    st.session_state.chat_history = []  # NEW: stores the conversation so far


def load_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    elif file_path.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def ingest_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    text = load_text_from_file(tmp_path)
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks).tolist()

    start_id = st.session_state.collection.count()
    ids = [f"chunk_{start_id + i}" for i in range(len(chunks))]
    metadatas = [{"source": uploaded_file.name} for _ in chunks]

    st.session_state.collection.add(
        documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids
    )
    st.session_state.processed_files.append(uploaded_file.name)


def answer_question(question):
    """This is literally our proven Cell 8 logic, unchanged -- just moved
    into a reusable function so the chat UI can call it."""
    question_embedding = embedding_model.encode([question]).tolist()
    results = st.session_state.collection.query(
        query_embeddings=question_embedding, n_results=3
    )
    context = "\n\n---\n\n".join(results["documents"][0])
    sources = [m["source"] for m in results["metadatas"][0]]

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below.
You MAY combine and synthesize information from multiple parts of the context to answer questions
like comparisons, as long as every fact you state is grounded in the context.
If the answer is not in the context at all, say "I don't have that information in the documents I was given."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content[0]['text'] if isinstance(response.content, list) else response.content
    return answer, sources


# --- Upload UI ---
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md"])

if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
    with st.spinner(f"Processing {uploaded_file.name}..."):
        ingest_file(uploaded_file)
    st.success(f"Added {uploaded_file.name} to this session.")

if st.session_state.processed_files:
    st.info(f"Documents in this session: {', '.join(st.session_state.processed_files)}")

st.divider()

# --- Chat UI ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if st.session_state.collection.count() == 0:
        st.warning("Please upload at least one document before asking a question.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(question)
            st.markdown(answer)
            st.caption(f"Sources: {', '.join(set(sources))}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
