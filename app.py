st.write("Secrets loaded:", "OPENAI_API_KEY" in st.secrets)

# app.py – conversational, bard-flavored Curse-of-Strahd assistant 
import os
from pathlib import Path

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ── config ───────────────────────────────────────────────
INDEX_DIR    = "faiss_index"
EMBED_MODEL  = "text-embedding-3-small"
CHAT_MODEL   = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are the party’s seasoned bard, recounting past adventures with flair. "
    "Answer vividly but accurately, and cite your memories when asked."
)
# ─────────────────────────────────────────────────────────

st.set_page_config(page_title="Barovian Bardic Archive")

# ensure your key is in Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Add OPENAI_API_KEY in Settings → Secrets")
    st.stop()

# export so downstream libs can see it
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource(show_spinner=True)
def load_chain():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.3,       # a touch more creativity
        streaming=True,
        system_message=SYSTEM_PROMPT,
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_k=4),
        return_source_documents=True,  # so we can expand citations
    )

chain = load_chain()

if "history" not in st.session_state:
    # each entry is (role, message, [source_docs])
    st.session_state.history = []

# ──────────────────────────────────────────────────────────
st.title("🧛‍♂️  Barovian Bardic Archive")

# ── Character Introductions block ─────────────────────────
characters_path = Path("docs/CHARACTERS.md")
if characters_path.exists():
    st.markdown("---")
    st.markdown("## Character Introductions")
    st.markdown(characters_path.read_text())

# ───── display chat so far ────────────────────────────────
for role, msg, src in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)
        if src:
            with st.expander("Show sources", expanded=False):
                for doc in src:
                    st.markdown(f"> *…{doc.page_content.strip()}*")

# ───── user input ─────────────────────────────────────────
user_msg = st.chat_input("Ask the archive…")
if user_msg:
    # echo user
    with st.chat_message("user"):
        st.markdown(user_msg)

    # rebuild simple (user,assistant) history
    history = [(u, a) for u, a, _ in st.session_state.history]

    # assistant turn
    with st.chat_message("assistant"):
        inputs = {
            "question":     user_msg,
            "chat_history": history,
        }
        result  = chain(inputs)
        answer  = result["answer"]
        sources = result.get("source_documents", [])

        st.markdown(answer)

    # save both turns
    st.session_state.history.append(("user",      user_msg, []))
    st.session_state.history.append(("assistant", answer,  sources))



