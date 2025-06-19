# app.py — minimal Streamlit + FAISS + OpenAI retrieval assistant
import os
import pickle
from pathlib import Path

import streamlit as st
import faiss
import openai
from openai.error import OpenAIError

# ── CONFIG ────────────────────────────────────────────────
INDEX_DIR    = "faiss_index"
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
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 2️⃣ Load FAISS index + metadata
@st.cache_resource
def load_index():
    idx = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/index.pkl", "rb") as f:
        id2text = pickle.load(f)
    return idx, id2text

index, id2text = load_index()

st.title("🧛‍♂️ Barovian Bardic Archive")

# Optional Character block
chars = Path("docs/CHARACTERS.md")
if chars.exists():
    st.markdown("---")
    st.markdown("## Character Introductions")
    st.markdown(chars.read_text())

# 3️⃣ Ask the user
question = st.text_input("Ask the archive…")
if not question:
    st.stop()

# 4️⃣ VECTOR RETRIEVAL
try:
    emb_resp = openai.Embedding.create(
        model="text-embedding-3-small",
        input=question
    )
    q_vec = emb_resp["data"][0]["embedding"]
except OpenAIError as e:
    st.error(f"Embedding error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unknown error in embedding: {e}")
    st.stop()

# FAISS search
D, I = index.search([q_vec], k=4)
contexts = "\n\n---\n\n".join(id2text[i] for i in I[0])

# 5️⃣ BUILD THE CHAT PROMPT
prompt = [
    {"role":"system",  "content": SYSTEM_PROMPT},
    {"role":"user",    "content":
        f"Here are the relevant excerpts:\n\n{contexts}\n\n"
        f"Question: {question}\n\n"
        "Please answer as bullet points, and cite which excerpt each point came from."}
]

# 6️⃣ CALL OpenAI CHAT COMPLETION
try:
    chat = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.3
    )
    answer = chat.choices[0].message.content
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unknown error in chat completion: {e}")
    st.stop()

# 7️⃣ Render the answer
st.markdown(answer)
