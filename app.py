# app.py – conversational, bard-flavored Curse-of-Strahd assistant
import os, streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
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

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Add OPENAI_API_KEY in Settings → Secrets")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource(show_spinner=True)
def load_chain():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.3,          # a touch more creativity
        streaming=True,
        system_message=SYSTEM_PROMPT,
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_k=4),
        return_source_documents=True,          # enable citations
    )

chain = load_chain()

if "history" not in st.session_state:
    st.session_state.history = []   # stores (role, msg, sources)

st.title("🧛‍♂️  Barovian Bardic Archive")

# ───── display chat so far ──────────────────────────────
for role, msg, src in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)
        if src:
            with st.expander("Show sources", expanded=False):
                for s in src:
                    st.markdown(f"• *…{s.page_content.strip()}*")

# ───── input box ────────────────────────────────────────
user_msg = st.chat_input("Ask the archive…")
if user_msg:
    # show user bubble
    with st.chat_message("user"):
        st.markdown(user_msg)

    # assistant bubble (streaming)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""

        for chunk in chain.stream(
            {
                "question": user_msg,
                "chat_history": [
                    (h, a) for h, a, _ in st.session_state.history
                    if h == "user"
                ],
            }
        ):
            text = chunk["answer"]
            partial = text
            placeholder.markdown(partial + "▌")

        placeholder.markdown(p
