# build_faiss.py — regenerate your FAISS index from all transcripts
import os
from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1️⃣ point at your transcripts folder
TRANSCRIPTS_DIR = Path("transcripts")

# 2️⃣ load each .txt as a Document
docs = []
for txt in sorted(TRANSCRIPTS_DIR.glob("Session_*_named.txt")):
    loader = TextLoader(str(txt), encoding="utf-8")
    docs.extend(loader.load())

# 3️⃣ split them into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 4️⃣ embed & build FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
faiss_index = FAISS.from_documents(chunks, embeddings)

# 5️⃣ save locally
faiss_index.save_local("faiss_index")
print(f"✅ Indexed {len(chunks)} chunks from {len(docs)} transcripts")
