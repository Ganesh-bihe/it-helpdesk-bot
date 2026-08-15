import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Step 1: Loading documents from data/ folder...")
docs = []
for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        filepath = os.path.join("data", filename)
        loader = TextLoader(filepath, encoding="utf-8")
        docs.extend(loader.load())
print(f"Loaded {len(docs)} documents.")

print("Step 2: Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

print("Step 3: Loading embedding model (first run downloads it, may take a few minutes)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Step 4: Creating and saving the vector database...")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("Done! Vector database saved in the 'chroma_db' folder.")