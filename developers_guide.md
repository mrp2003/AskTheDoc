# AskTheDoc – Developer Guide

This guide breaks down the full codebase of AskTheDoc (Streamlit + LangChain PDF QA system) into digestible sections. Each block is explained in detail to make the system easy to understand and extend.

---

## 1. Environment Setup

```python
import os
import tempfile
from dotenv import load_dotenv
```

**Explanation**: Imports modules for working with environment variables, temporary files, and loading `.env` securely.

---

## 2. LangChain & Streamlit Imports

```python
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
```

**Explanation**:

* Streamlit drives the UI.
* LangChain handles PDF ingestion (`PyPDFLoader`), chunking (`RecursiveCharacterTextSplitter`), embedding, and retrieval (`FAISS`).
* `RetrievalQA` connects LLM + retriever into one chain.
* `ChatOpenAI` is the LLM backend (you can swap this with other LLMs).

---

## 3. API Key Loading

```python
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

**Explanation**: Loads your OpenAI API key from a `.env` file and sets it as an environment variable so LangChain can access it.

---

## 4. Streamlit App Configuration

```python
st.set_page_config(page_title="AskTheDoc", layout="wide")
st.title("AskTheDoc: AI-Powered PDF Assistant")
```

**Explanation**: Sets up the Streamlit app title and layout format.

---

## 5. Session State Initialization

```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []
```

**Explanation**: Initializes memory to persist chat, vector DBs, selected files, and the QA chain between reruns.

---

## 6. Control Bar (Reset & Export)

```python
col1, col2, col3 = st.columns([2, 2, 25])
```

**Explanation**: Three columns layout for reset and export buttons.

```python
with col1:
    if st.button("Reset Chat"):
        # Wipe session state if chat history exists
        if not st.session_state.chat_history:
            st.toast("Nothing to reset — start a conversation first!")
        else:
            for key in ["chat_history", "qa_chain", "stats", "selected_files"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.toast("Session reset successfully.")
```

```python
with col2:
    chat_text = ""
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "AI"
        chat_text += f"{role}: {msg['content']}\n\n"

    st.download_button("Export Chat", chat_text, file_name="chat_history.txt", key="download_button")
```

**Explanation**: Lets users reset the session or export conversation history to a `.txt` file.

---

## 7. PDF Upload & Vectorization

```python
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
```

**Explanation**: UI for selecting multiple PDFs.

```python
if uploaded_files:
    ...
    embeddings = OpenAIEmbeddings()

    for file in uploaded_files:
        if file.name not in st.session_state.vector_stores:
            # Save to temp, process it, chunk & embed
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
```

**Explanation**: Stores each uploaded PDF in memory only once. Uses LangChain to:

* Load pages from PDF
* Chunk them into overlapping parts
* Embed them via OpenAI
* Store them in FAISS vector DB

```python
    avg_chunk_length = total_chars // total_chunks if total_chunks else 0
    st.session_state.stats = {
        "chunks": total_chunks,
        "avg_length": avg_chunk_length,
        "total_chars": total_chars
    }
```

**Explanation**: Adds summary stats for viewing later.

---

## 8. Multi-file Selection & Index Merging

```python
selected_file_names = st.multiselect("Select files to query", options=list(st.session_state.vector_stores.keys()))
```

**Explanation**: Allows selecting one or more PDFs for querying.

```python
selected_dbs = [st.session_state.vector_stores[name] for name in selected_file_names]

if selected_dbs:
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
```

**Explanation**: Dynamically combines selected vector stores into a new index. Rebuilds the retrieval chain.

---

## 9. Stats Panel

```python
if "stats" in st.session_state:
    with st.expander("View Vector Store Stats"):
        stats = st.session_state.stats
        st.write(f"Total chunks: {stats['chunks']}")
        st.write(f"Avg chunk size (chars): {stats['avg_length']}")
        st.write(f"Total characters embedded: {stats['total_chars']}")
```

**Explanation**: Lets users inspect how many chunks are being embedded per PDF.

---

## 10. Chat Display & Input

```python
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

**Explanation**: Re-renders previous conversation history in the chat window.

```python
if st.session_state.qa_chain:
    user_query = st.chat_input("Ask something about your selected document(s)...")
```

**Explanation**: Displays input box for user queries.

```python
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
```

**Explanation**:

* Handles user input.
* Calls the LangChain QA chain.
* Streams the answer live.
* Displays sources.
* Saves the AI's response.

---

## Final Notes

* Everything runs client-side using Streamlit.
* Each PDF is embedded only once per session.
* Allows selection of multiple documents dynamically.
* Chat memory persists between questions.

You now have a full breakdown of how `AskTheDoc` works. Modify components (like retrievers, LLMs, or UIs) to fit your use case.
