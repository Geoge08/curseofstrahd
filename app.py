    # ───── user input ─────────────────────────────────────────
    user_msg = st.chat_input("Ask the archive…")
    if user_msg:
        # echo the user
        with st.chat_message("user"):
            st.markdown(user_msg)

        # rebuild simple (user, assistant) history
        history = [(u, a) for u, a, _ in st.session_state.history]

        # assistant turn
        with st.chat_message("assistant"):
            # DEBUG WRAPPER: catch & display the real exception
            try:
                result = chain(
                    question=user_msg,
                    chat_history=history,
                )
            except Exception as e:
                st.error(f"⚠️ Chain error: {type(e).__name__}: {e}")
                import traceback
                st.text(traceback.format_exc())
                st.stop()

            # if we get here, chain succeeded
            answer  = result["answer"]
            sources = result.get("source_documents", [])

            st.markdown(answer)

        # save both turns
        st.session_state.history.append(("user",      user_msg, []))
        st.session_state.history.append(("assistant", answer,  sources))

