# app.py  – Streamlit chat for your Curse-of-Strahd knowledge base
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ── configuration ────────────────────────────────────────────────
INDEX_DIR   = "faiss_index"             # folder you’ll upload
EMBED_MODEL = "text-embedding-3-small"  # cheap, accurate embeddings
CHAT_MODEL  = "gpt-4o-mini"             # fast, low-cost chat model
# -----------------------------------------------------------------

# ask Streamlit Secrets for your key (you’ll add it in the Cloud UI)
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Add your OPENAI_API_KEY in Settings → Secrets")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# cache the Retrieval-QA chain so it loads once per server session
@st.cache_resource(show_spinner=True)
def load_qa_chain():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True   # we trust our own pickle
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

qa = load_qa_chain()

st.title("🧛‍♂️  Curse-of-Strahd Session QA")
query = st.text_input("Ask the archive…", placeholder="e.g. Summarise session 28 in 3 sentences")
if query:
    with st.spinner("Consulting Barovia…"):
        answer = qa.invoke(query)["result"]
    st.markdown("**Answer:**\n\n" + answer)
