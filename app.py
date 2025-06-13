import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AskTheDoc", layout="wide")
st.title("AskTheDoc: AI-Powered PDF Assistant")

# Initialize state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []

# -------------------------------
# Top Control Bar: Reset + Export
# -------------------------------
col1, col2, col3 = st.columns([2, 2, 25])

with col1:
    if st.button("Reset Chat"):
        if not st.session_state.chat_history:
            st.toast("Nothing to reset — start a conversation first!")
        else:
            for key in ["chat_history", "qa_chain", "stats", "selected_files"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.toast("Session reset successfully.")

with col2:
    chat_text = ""
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "AI"
        chat_text += f"{role}: {msg['content']}\n\n"

    st.download_button("Export Chat", chat_text, file_name="chat_history.txt", key="download_button")

# -------------------------------
# Multi-PDF Upload & Vector Store per File
# -------------------------------
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if "vectorstores" not in st.session_state:
        st.session_state.vector_stores = {}

    total_chunks = 0
    total_chars = 0

    embeddings = OpenAIEmbeddings()

    for file in uploaded_files:
        if file.name not in st.session_state.vector_stores:
            # Load & embed once
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                pdf_path = tmp_file.name
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)

            total_chunks += len(chunks)
            total_chars += sum(len(doc.page_content) for doc in chunks)

            db = FAISS.from_documents(chunks, embeddings)
            st.session_state.vector_stores[file.name] = db

    avg_chunk_length = total_chars // total_chunks if total_chunks else 0
    st.session_state.stats = {
        "chunks": total_chunks,
        "avg_length": avg_chunk_length,
        "total_chars": total_chars
    }

# -------------------------------
# File Selection (Multi-select with Checkboxes)
# -------------------------------
if st.session_state.vector_stores:
    selected_file_names = st.multiselect("Select files to query", options=list(st.session_state.vector_stores.keys()))
    selected_dbs = [st.session_state.vector_stores[name] for name in selected_file_names]

    if selected_dbs:
        # Merge selected DBs
        all_docs = []
        for db in selected_dbs:
            all_docs.extend(db.docstore._dict.values())

        merged_index = FAISS.from_documents(all_docs, OpenAIEmbeddings())

        retriever = merged_index.as_retriever()
        llm = ChatOpenAI(temperature=0)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

# -------------------------------
# Stats Panel (On-Demand)
# -------------------------------
if "stats" in st.session_state:
    with st.expander("View Vector Store Stats"):
        stats = st.session_state.stats
        st.write(f"Total chunks: {stats['chunks']}")
        st.write(f"Avg chunk size (chars): {stats['avg_length']}")
        st.write(f"Total characters embedded: {stats['total_chars']}")

# -------------------------------
# Chat History Rendering
# -------------------------------
if "chat_history" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# -------------------------------
# Chat Input & Response
# -------------------------------
if st.session_state.qa_chain:
    user_query = st.chat_input("Ask something about your selected document(s)...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("ai"):
            response_text = ""
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain(user_query)
                answer = result["result"]
                sources = result.get("source_documents", [])

                for char in answer:
                    response_text += char
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)

                if sources:
                    with st.expander("Show source chunks"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i + 1}:**\n\n{doc.page_content}")

        st.session_state.chat_history.append({"role": "ai", "content": response_text})
