from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()


pdf_files=['data/apple_diseases.pdf', 'data/Blueberry_diseases.pdf', 'data/disease_flower.pdf', 'data/Strawberry_diseases.pdf', 'data/Tomato_diseases.pdf']
all_pages =[]
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    pages = loader.load()
    all_pages.extend(pages)
print(len(all_pages))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_pages)
print(len(chunks))

client = chromadb.PersistentClient(path="./chroma_db")


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

collection = client.get_or_create_collection(name = 'crop_diseases')
for i , chunk in enumerate(chunks):
    response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input = chunk.page_content) 
    embedding = response.data[0].embedding
    collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[chunk.page_content],
    metadatas=[{"source": chunk.metadata.get("source", "unknown")}]
)