from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

print("Loading knowledge base...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

print("Connecting to Llama 3.2...")
llm = Ollama(model="llama3.2")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("\nIT Helpdesk Assistant ready! Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    response = qa_chain.invoke({"query": question})
    print(f"\nAssistant: {response['result']}")
    sources = [doc.metadata.get("source", "unknown") for doc in response["source_documents"]]
    print(f"(Sources: {sources})\n")