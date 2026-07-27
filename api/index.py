import os
from flask import Flask, render_template, request, session
from pathlib import Path
from .embedder import RawFixedEmbedder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates')),
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../static')))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("user_query")

        # chat_history = ("chat_history", [])

        # 1. Safe Path Resolution
        BASE_DIR = Path(__file__).resolve().parent
        pdf_path = BASE_DIR / "python_rag_reference.pdf"
        
        if not os.path.exists(pdf_path):
            # Backward compatibility: agar parent folder me ho
            pdf_path = BASE_DIR.parent / "python_rag_reference.pdf"
        
        # print(f"Loading PDF from: {pdf_path}")
        
        # 2. PDF Load & Split
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_data = splitter.split_documents(docs)
        
        embedders = RawFixedEmbedder(dimensions=384)
        # print(f"Total Chunks Created: {len(splitted_data)}")
        class CustomEmbeddings(Embeddings):
            def __init__(self, embed_fn):
                self.embed_fn = embed_fn
        
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # Handle batching multiple text chunks
                return [self.embed_fn(t) for t in texts]
        
            def embed_query(self, text: str) -> list[float]:
                # Handle single search query
                return self.embed_fn(text)
        
        # Wrap your function
        embedder = CustomEmbeddings(embedders)
        # 3. Vector Embeddings & Store
        
        
        
        # FAISS automatically embeds all chunks using embedder.embed_documents()
        vector_store = FAISS.from_documents(documents=splitted_data, embedding=embedder)
        
        # 4. Save to Disk
        db_path = BASE_DIR / "vector_db"
        vector_store.save_local(str(db_path))
        
        # print(f"✅ Vector DB successfully created and saved in '{db_path}'!")
        query = ""
        data = vector_store.similarity_search(query=query)
        
        context = ""
        for doc in data:
            context += doc.page_content + "\n"
        
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile"
        )
        
        
        # res = llm.invoke(f"can you provide me the answer based on provided context for my question: {context} and question: {query}. and ask question out of the conext then you say I DONT KNOW.")
        def get_context(query:str):
            data = vector_store.similarity_search(query=query)
            context = ""
            for doc in data:
                context += doc.page_content + "\n"
        
            return {
                "context": context,
                "question": query
            }
        
        prompt = PromptTemplate.from_template("""
              you are a helpfull assistant and provide answerd based on the context for user question. and
              if you dont know the answer then you can say 'I DONT KNOW'.
              "context": {context}
              "question": {question}
        """)
        
        rag_chain = get_context | prompt | llm
        res = rag_chain.invoke(user_query)
        return render_template("index.html", responce=res.content)

    return render_template("index.html")

