from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

query = "my VPN keeps disconnecting"
results = vectordb.similarity_search(query, k=2)

print(f"\nQuery: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(doc.page_content)
    print()