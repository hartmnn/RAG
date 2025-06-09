import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from PyPDF2 import PdfReader
from llama_index.core.query_engine import RetrieverQueryEngine


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Load PDFs & Index
def create_index_from_pdfs(pdf_dir):
    documents = load_all_pdfs_as_documents(pdf_dir)
    print(f"📄 Loaded {len(documents)} documents")
    return VectorStoreIndex.from_documents(documents)

def load_all_pdfs_as_documents(pdf_dir):
    documents = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_dir, file)
            reader = PdfReader(full_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            documents.append(Document(text=text, metadata={"file_name": file}))
    return documents
    
# Step 2: Create a query engine
def create_query_engine(index):
    retriever = index.as_retriever(similarity_top_k=10) 
    return RetrieverQueryEngine(retriever=retriever)

# Step 3: Run a query
def run_query(engine, user_query):
    response = engine.query(user_query)
    return response.response

# Main chain
if __name__ == "__main__":

    pdf_dir = "./data"
    user_question = "What are the hidden messages in the PDFs?" 

    Settings.llm = OpenAI(model="gpt-4o")
    
    index = create_index_from_pdfs(pdf_dir)
    engine = create_query_engine(index)
    answer = run_query(engine, user_question)

    print("\n💬 Answer:")
    print(answer)