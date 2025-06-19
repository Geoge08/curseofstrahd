# ── app.py – conversational, bard-flavored Curse-of-Strahd assistant ──────────

# ── 0️⃣ LOGGING BOILERPLATE: capture everything to streamlit.log ─────────────
import logging, traceback, sys
from pathlib import Path

LOG_PATH = Path(__file__).parent / "streamlit.log"
if LOG_PATH.exists():
    LOG_PATH.unlink()

logging.basicConfig(
    filename=str(LOG_PATH),
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
)

def log_uncaught_exceptions(exc_type, exc_value, exc_tb):
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

sys.excepthook = log_uncaught_exceptions
# ────────────────────────────────────────────────────────────────────────────


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
    logging.info("Loading FAISS index and LLM chain")
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
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_k=4),
        return_source_documents=True,  # so we can expand citations
    )
    logging.info("Chain loaded successfully")
    return chain

chain = load_chain()

if "history" not in st.session_state:
    st.session_state.history = []  # each entry is (role, message, [source_docs])

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

    # rebuild simple (user, assistant) history
    history = [(u, a) for u, a, _ in st.session_state.history]

    # assistant turn
    with st.chat_message("assistant"):
        try:
            result = chain({"question": user_msg, "chat_history": history})
        except Exception as e:
            st.error(f"⚠️ Chain error: {type(e).__name__}: {e}")
            st.text(traceback.format_exc())
            logging.error("Chain invocation failed", exc_info=True)
            st.stop()

        answer  = result["answer"]
        sources = result.get("source_documents", [])

        st.markdown(answer)

    # save both turns
    st.session_state.history.append(("user",      user_msg, []))
    st.session_state.history.append(("assistant", answer,  sources))


# ── 〆 VIEW LOG in UI ───────────────────────────────────────────────
if LOG_PATH.exists():
    with st.expander("📄 View app log", expanded=False):
        lines = LOG_PATH.read_text().splitlines()
        st.text("\n".join(lines[-100:]))
