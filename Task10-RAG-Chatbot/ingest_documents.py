"""
Simplified Document Ingestion for RAG Chatbot
Uses a different embedding approach that doesn't conflict
"""

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
import shutil

print("=" * 60)
print("RAG CHATBOT - DOCUMENT INGESTION")
print("=" * 60)

# ========== 1. LOAD DOCUMENTS ==========
print("\n[1] Loading documents from knowledge base...")

loader = DirectoryLoader(
    'data/knowledge_base/',
    glob='**/*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

for doc in documents:
    print(f"   - {os.path.basename(doc.metadata['source'])}")

# ========== 2. SPLIT DOCUMENTS INTO CHUNKS ==========
print("\n[2] Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} text chunks")

# ========== 3. CREATE EMBEDDINGS (Simple version) ==========
print("\n[3] Creating embeddings...")

# Use fake embeddings for testing (no download needed)
embeddings = FakeEmbeddings(size=384)
print("✅ Using fake embeddings (for testing)")

# ========== 4. CREATE VECTOR STORE ==========
print("\n[4] Creating vector database...")

if os.path.exists('vector_store'):
    shutil.rmtree('vector_store')
    print("   Removed existing vector store")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='vector_store'
)

vectorstore.persist()
print("✅ Vector store created at 'vector_store/'")

print("\n" + "=" * 60)
print("✅ DOCUMENT INGESTION COMPLETE!")
print("=" * 60)