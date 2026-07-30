import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
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

# Safe Path Resolution pointing to project root
BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static')
)

app.secret_key = 'your_secret_key_here' 

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
# INITIALIZE RAG & VECTOR STORE ONCE AT STARTUP (GLOBAL SCOPE)
# -------------------------------------------------------------
def initialize_vector_store():
    # # 1. Locate PDF file
    # pdf_path = BASE_DIR / "python_rag_reference.pdf"
    # if not os.path.exists(pdf_path):
    #     pdf_path = BASE_DIR / "api" / "python_rag_reference.pdf"

    # # 2. Load & Split PDF
    # loader = PyPDFLoader(str(pdf_path))
    # docs = loader.load()
    docs = session.get('docs_file')

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_data = splitter.split_documents(docs)

    # 3. Create Embeddings & Vector Store in Memory
    embedders = RawFixedEmbedder(dimensions=384)
    embedder = CustomEmbeddings(embedders)

    vector_store = FAISS.from_documents(documents=splitted_data, embedding=embedder)
    return vector_store


# Initialize globally during container warm-up
vector_store = initialize_vector_store()

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate.from_template("""
you are a helpfull assistant and provide answerd based on the context for user question. and
if you dont know the answer then you can say 'I DONT KNOW'.
"context": {context}
"question": {question}
""")


# -------------------------------------------------------------
# FAST ROUTE HANDLER (Execution < 2 seconds)
# -------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("user_query")
        file = request.form.get("file")
        session['docs_file'] = file
        if not user_query:
            return render_template("index.html")
        
        # 1. Similarity Search on pre-loaded vector store
        docs = vector_store.similarity_search(query=user_query, k=3)
        context = "\n".join([doc.page_content for doc in docs])


        # 2. Invoke RAG Chain
        formatted_prompt = prompt.format(context=context, question=user_query)
        res = llm.invoke(formatted_prompt)
        answer = res.content if hasattr(res, 'content') else str(res)
            
            # 1. IF JS FETCH REQUEST: Return JSON (No Reload)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json or 'application/json' in request.headers.get('Accept', ''):
            return jsonify({"response": answer})
        return render_template("index.html", responce=res.content, response=res.content)

    return render_template("index.html")



