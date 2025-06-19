# app.py – conversational, bard-flavored Curse-of-Strahd assistant 
import os
from pathlib import Path
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ── config ───────────────────────────────────────────────
INDEX_DIR   = "faiss_index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are the party’s seasoned bard, recounting past adventures with flair. "
    "Answer vividly but accurately, and cite your memories when asked."
)
# ─────────────────────────────────────────────────────────

st.set_page_config(page_title="Barovian Bardic Archive")

# make sure your key is stored in Streamlit → Settings → Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("🔑 Please add your `OPENAI_API_KEY` under Settings → Secrets")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource(show_spinner=True)
def load_chain():
    # load your FAISS index
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    # create a ChatOpenAI model (note: no `system_message` here!)
    llm = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=0.3,
        streaming=True
    )
    # build the conversational‐retrieval chain
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_k=4),
        return_source_documents=True,
        # you can still pass your system prompt here if your LangChain version supports it:
        # system_prompt=SYSTEM_PROMPT
    )

chain = load_chain()

# initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []  # each entry: (role, text, [docs])

st.title("🧛‍♂️ Barovian Bardic Archive")

# show your CHARACTER sheet from docs/CHARACTERS.md
char_file = Path("docs/CHARACTERS.md")
if char_file.exists():
    st.markdown("---")
    st.markdown("## Character Introductions")
    st.markdown(char_file.read_text())

# replay the conversation so far
for role, text, docs in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if docs:
            with st.expander("Show sources"):
                for d in docs:
                    st.markdown(f"> *…{d.page_content.strip()}*")

# user prompt
user_msg = st.chat_input("Ask the archive…")
if user_msg:
    # show user message
    with st.chat_message("user"):
        st.markdown(user_msg)

    # build simple history for the chain
    history = [(u, a) for u, a, _ in st.session_state.history]

    # assistant response
    with st.chat_message("assistant"):
        try:
            result = chain({
                "question":      user_msg,
                "chat_history":  history
            })
        except Exception as e:
            st.error(f"❌ Chain error: {type(e).__name__}: {e}")
            import traceback; st.text(traceback.format_exc())
            st.stop()

        answer  = result["answer"]
        sources = result.get("source_documents", [])

        st.markdown(answer)

    # append to our session history
    st.session_state.history.append(("user",      user_msg, []))
    st.session_state.history.append(("assistant", answer,  sources))

