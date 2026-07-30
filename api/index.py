import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# LangChain Loaders & Utilities
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings

# Handle import depending on execution context
try:
    from .embedder import RawFixedEmbedder
except ImportError:
    from embedder import RawFixedEmbedder

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static')
)

UPLOAD_FOLDER = BASE_DIR / 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)


# -------------------------------------------------------------
# Custom Embeddings Wrapper
# -------------------------------------------------------------
class CustomEmbeddings(Embeddings):
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_fn(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_fn(text)


# -------------------------------------------------------------
# INITIALIZE GLOBALS & HELPER FUNCTIONS
# -------------------------------------------------------------
embedders = RawFixedEmbedder(dimensions=384)
embedder = CustomEmbeddings(embedders)

# Global in-memory Vector Store reference
vector_store = None

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate.from_template("""
you are a helpfull assistant and provide answerd based on the context for user question. and
if you dont know the answer then you can say 'I DONT KNOW'.
"context": {context}
"question": {question}
""")


def load_document(file_path: str, filename: str):
    """Loads text based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif ext in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return loader.load()


def process_and_add_docs(file_path: str, filename: str):
    """
    Loads custom document, splits it using RecursiveCharacterTextSplitter,
    and indexes it into the global vector store.
    """
    global vector_store

    # 1. Load Document
    docs = load_document(file_path, filename)

    # 2. Split Document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_data = splitter.split_documents(docs)

    if not splitted_data:
        return 0

    # 3. Add to FAISS Vector Store
    if vector_store is None:
        vector_store = FAISS.from_documents(documents=splitted_data, embedding=embedder)
    else:
        vector_store.add_documents(documents=splitted_data)

    return len(splitted_data)


# -------------------------------------------------------------
# UPLOAD ROUTE HANDLER (FOR CUSTOM DOCS & PDFS)
# -------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file parameter provided'}), 400

    uploaded_files = request.files.getlist('file')
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400

    total_chunks = 0

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            chunks_created = process_and_add_docs(save_path, filename)
            total_chunks += chunks_created
        except Exception as e:
            return jsonify({'error': f"Failed to process {filename}: {str(e)}"}), 500
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    return jsonify({
        'message': f'Successfully processed {len(uploaded_files)} file(s)!',
        'total_chunks': total_chunks
    }), 200


# -------------------------------------------------------------
# FAST CHAT ROUTE HANDLER
# -------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("user_query")

        if not user_query:
            return render_template("index.html")

        # Check if vector store contains documents
        if vector_store is None:
            answer = "No document uploaded yet. Please upload a PDF or document first."
        else:
            # 1. Similarity Search on loaded vector store
            docs = vector_store.similarity_search(query=user_query, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            # 2. Invoke RAG Chain
            formatted_prompt = prompt.format(context=context, question=user_query)
            res = llm.invoke(formatted_prompt)
            answer = res.content if hasattr(res, 'content') else str(res)

        # Return JSON for AJAX/Fetch requests or render template
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json or 'application/json' in request.headers.get('Accept', ''):
            return jsonify({"response": answer})

        return render_template("index.html", responce=answer, response=answer)

    return render_template("index.html")

