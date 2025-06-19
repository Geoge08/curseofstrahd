# app.py — minimal Streamlit + FAISS + OpenAI retrieval assistant
import os
from pathlib import Path

import streamlit as st
import faiss
import pickle
from openai import OpenAI
from openai.error import OpenAIError

# ── CONFIG ────────────────────────────────────────────────
INDEX_DIR    = "faiss_index"
TRANSCRIPTS  = "transcripts"
SYSTEM_PROMPT = (
    "You are the party’s seasoned bard, recounting past adventures with flair. "
    "Answer vividly but accurately, and cite your memories when asked."
)
# ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Barovian Bardic Archive")

# 1️⃣ Load your OpenAI key from Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Add OPENAI_API_KEY in Settings → Secrets")
    st.stop()
API_KEY = st.secrets["OPENAI_API_KEY"]

# 2️⃣ Load FAISS index + metadata
@st.cache_resource
def load_index():
    # load index
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    # load id→ text mapping
    with open(f"{INDEX_DIR}/index.pkl", "rb") as f:
        id2text = pickle.load(f)
    return index, id2text

index, id2text = load_index()

st.title("🧛‍♂️ Barovian Bardic Archive")

# Optional character block
chars = Path("docs/CHARACTERS.md")
if chars.exists():
    st.markdown("---")
    st.markdown("## Character Introductions")
    st.markdown(chars.read_text())

# 3️⃣ Ask the user
question = st.text_input("Ask the archive…")

if question:
    # 4️⃣ VECTOR RETRIEVAL
    # embed the question
    client = OpenAI(api_key=API_KEY)
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )
    except OpenAIError as e:
        st.error(f"Error embedding: {e}")
        st.stop()

    q_vec = emb_resp["data"][0]["embedding"]
    # FAISS search
    D, I = index.search([q_vec], k=4)
    contexts = "\n\n---\n\n".join(id2text[i] for i in I[0])

    # 5️⃣ BUILD THE CHAT PROMPT
    system = SYSTEM_PROMPT
    prompt = [
        {"role":"system", "content": system},
        {"role":"user",   "content":
            "Here are the relevant excerpts from past sessions:\n\n"
            f"{contexts}\n\n"
            f"Question: {question}\n\n"
            "Answer in bullet points, citing which excerpt you used."}
    ]

    # 6️⃣ CALL OpenAI CHAT COMPLETION
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.3
        )
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

    answer = chat.choices[0].message.content
    st.markdown(answer)

