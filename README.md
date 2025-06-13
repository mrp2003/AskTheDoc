# AskTheDoc: AI-Powered PDF Assistant

AskTheDoc is a powerful AI assistant that lets users chat with one or more PDF documents via a Streamlit web interface or CLI. It uses LangChain's RAG (Retrieval-Augmented Generation) pipeline with OpenAI LLMs, FAISS vector search, and chunked document embeddings to deliver accurate, source-backed answers.

## Features
- Multi-PDF upload and selection
- Per-file vector store creation and merging
- Real-time chat with source chunk display
- Exportable chat history
- Stats panel for embedded document analytics
- Persistent vector store (saved to disk)

## Tech Stack
- LangChain: Document loading, splitting, QA chains
- PyPDFLoader: PDF parsing
- FAISS: Fast Approximate Nearest Neighbor vector DB
- OpenAIEmbeddings: Embedding generation
- ChatOpenAI: Language model (GPT-style)
- Streamlit: Interactive web UI
- dotenv: Environment variable management

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/mrp2003/askthedoc
cd askthedoc
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set your OpenAI key**
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_key
```

## Usage

```bash
streamlit run app.py
```

- Upload one or more PDFs.
- Select the PDFs you want to query.
- Ask questions and see source-backed answers.
- View stats, reset session, and export chat logs.

## Folder Structure

```
.
├── app.py
├── requirments.txt
└── README.md
```

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or improve.
If you found this helpful, give it a ⭐ on GitHub!
