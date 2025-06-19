# app.py – conversational, bard-flavored Curse-of-Strahd assistant 
import os
from pathlib import Path

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ── config ────────────────────────────────────────────────
INDEX_DIR    = "faiss_index"
EMBED_MODEL  = "text-embedding-3-small"
CHAT_MODEL   = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are the party’s seasoned bard, recounting past adventures with flair. "
    "Answer vividly but accurately, and cite your memories when asked."
)
# ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Barovian Bardic Archive")

# 1️⃣ Load your OpenAI key from Streamlit Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("🔑 Please set OPENAI_API_KEY under Settings → Secrets")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource(show_spinner=True)
def load_chain():
    # load FAISS
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    # init LLM (no system_message here—LangChain will use the chain’s prompt template)
    llm = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=0.3,
        streaming=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_k=4),
        return_source_documents=True,
        # if your LangChain version supports it you can still pass:
        # system_prompt=SYSTEM_PROMPT
    )

chain = load_chain()

# ── initialize chat history ──────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, message, [docs])

st.title("🧛‍♂️ Barovian Bardic Archive")

# ── Character sheet from docs/CHARACTERS.md ─────────────
char_file = Path("docs/CHARACTERS.md")
if char_file.exists():
    st.markdown("---")
    st.markdown("## Character Introductions")
    st.markdown(char_file.read_text())

# ── replay past chat ─────────────────────────────────────
for role, text, docs in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if docs:
            with st.expander("Show sources"):
                for d in docs:
                    st.markdown(f"> *…{d.page_content.strip()}*")

# ── user input ───────────────────────────────────────────
user_msg = st.chat_input("Ask the archive…")
if user_msg:
    # echo user
    with st.chat_message("user"):
        st.markdown(user_msg)

    # build simple history for chain
    history = [(u, a) for u, a, _ in st.session_state.history]

    # assistant responds
    with st.chat_message("assistant"):
        try:
            result = chain({
                "question":     user_msg,
                "chat_history": history
            })
        except Exception as e:
            st.error(f"❌ Chain error: {type(e).__name__}: {e}")
            import traceback; st.text(traceback.format_exc())
            st.stop()

        answer  = result["answer"]
        sources = result.get("source_documents", [])
        st.markdown(answer)

    # append to history
    st.session_state.history.append(("user",      user_msg, []))
    st.session_state.history.append(("assistant", answer,  sources))

