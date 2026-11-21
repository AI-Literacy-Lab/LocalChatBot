"""
streamlit_app.py - Streamlit front end for rag_chatbot.py (keep both files in the same folder).

Run:
    pip install streamlit numpy requests chromadb
    ollama pull nomic-embed-text
    streamlit run streamlit_app.py

Everything runs locally: Ollama embeddings + ChromaDB + Ollama LLM. No API keys, no cost.
If Ollama isn't running, the app falls back to extractive answers.
"""

from pathlib import Path

import streamlit as st

from rag_chatbot import (
    LLM_MODEL,
    LocalLLM,
    is_embed_model,
    list_ollama_models,
    log,
    NaiveBot,
    RAGChat,
    ChromaVectorStore,
    build_store,
    load_docs,
    make_embedder,
)

st.set_page_config(page_title="Local RAG Chatbot", page_icon="💬", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources: load the embedding model once, re-index only when files change
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding backend...")
def get_embedder():
    return make_embedder()  # Ollama nomic-embed-text by default (set EMBED_BACKEND to change)


def folder_signature(folder: str) -> tuple:
    p = Path(folder)
    return tuple(
        (f.name, f.stat().st_mtime_ns)
        for f in sorted(p.glob("*"))
        if f.suffix.lower() in {".txt", ".md"}
    )


@st.cache_resource(show_spinner="Chunking and embedding documents...")
def get_store(folder: str, signature: tuple, max_words: int, overlap: int, store_type: str):
    return build_store(folder, store_type=store_type, max_words=max_words,
                       overlap=overlap, embedder=get_embedder())


# ---------------------------------------------------------------------------
# Sidebar: knowledge base + settings
# ---------------------------------------------------------------------------
try:
    embedder = get_embedder()
except Exception as e:  # e.g. Ollama busy/not running, or nomic-embed-text not pulled
    log.error(f"[ui] embedding backend failed: {e}")
    st.error(f"Could not start the embedding backend.\n\n`{e}`\n\n"
             "Check that Ollama is running and the model is pulled (`ollama pull nomic-embed-text`), "
             "then reload this page.")
    st.stop()

with st.sidebar:
    st.title("⚙️ Settings")
    rag_off = st.toggle("🚫 RAG off (plain chatbot, no documents)", value=False, key="rag_off",
                        help="Skips retrieval entirely: the LLM answers from its own memory, like the hook demo. "
                             "Turn it back on to show the fix.")
    st.divider()
    st.caption(f"🧬 Embeddings: {embedder.name}")

    st.subheader("Knowledge base")
    docs_folder = st.text_input("Documents folder", value="docs")
    load_docs(docs_folder)  # creates sample docs on first run

    uploads = st.file_uploader("Add .txt / .md files", type=["txt", "md"], accept_multiple_files=True)
    if uploads and st.button("Add to knowledge base", use_container_width=True):
        for f in uploads:
            (Path(docs_folder) / f.name).write_bytes(f.getvalue())
        st.success(f"Added {len(uploads)} file(s)")
        st.rerun()

    store_type = st.radio("Vector store", ["chroma", "numpy"], horizontal=True,
                          format_func=lambda x: {"chroma": "ChromaDB", "numpy": "In-memory NumPy"}[x],
                          help="ChromaDB persists the index in ./chroma_db and reuses it between runs.")

    with st.expander("Chunking"):
        max_words = st.slider("Max words per chunk", 30, 250, 80, 10)
        overlap = st.slider("Overlap (sentences)", 0, 3, 1)

    st.subheader("Model")
    chat_models = [m for m in list_ollama_models() if not is_embed_model(m)]
    if chat_models:
        model_name = st.selectbox("Ollama model", chat_models,
                                  index=chat_models.index(LLM_MODEL) if LLM_MODEL in chat_models else 0)
    else:
        model_name = st.text_input("Ollama model", value=LLM_MODEL)

    st.subheader("Retrieval")
    k = st.slider("Top-k chunks", 1, 8, 3)
    min_score = st.slider("Similarity threshold", 0.0, 0.9, float(embedder.default_min_score), 0.01,
                          help="Chunks scoring below this are dropped. If nothing passes, the bot says it doesn't know.")

    st.subheader("Conversation fixes")
    rewrite = st.toggle("Rewrite follow-up questions", value=True,
                        help="Turns 'what about its storage?' into a standalone question before searching.")
    use_memory = st.toggle("Use chat memory", value=True)
    keep_turns = st.slider("Recent turns kept verbatim", 1, 8, 3,
                           help="Older turns are folded into a rolling summary.")

    st.subheader("Demo helpers")
    compare = st.toggle("Also show plain chatbot (no retrieval)", value=False,
                        help="Shows what the LLM says WITHOUT your documents - i.e. the hallucination.")


# ---------------------------------------------------------------------------
# Build index + bot
# ---------------------------------------------------------------------------
store = get_store(docs_folder, folder_signature(docs_folder), max_words, overlap, store_type)
llm = LocalLLM(model_name)

if "bot" not in st.session_state:
    st.session_state.bot = RAGChat(store, llm)
    st.session_state.messages = []
    st.session_state.pending = None

bot: RAGChat = st.session_state.bot
# Settings are plain attributes, so changes apply instantly without losing memory
bot.store, bot.llm = store, llm
bot.k, bot.min_score, bot.keep_turns = k, min_score, keep_turns
bot.rewrite_followups, bot.use_memory = rewrite, use_memory

with st.sidebar:
    st.divider()
    if isinstance(store, ChromaVectorStore):
        st.caption("🗄️ ChromaDB (persisted in ./chroma_db)" + (" - reused existing index" if store.reused else ""))
    else:
        st.caption("🗄️ In-memory NumPy store" + (" (ChromaDB unavailable - see terminal)" if store_type == "chroma" else ""))
    st.caption(f"📚 {len(store.chunks)} chunks from {len(set(c.source for c in store.chunks))} documents")
    if llm.available:
        st.caption(f"🟢 Ollama: {llm.model}")
        if llm.note:
            st.warning(llm.note)
    else:
        st.caption("🟡 Ollama not detected - using extractive fallback (returns best passage)")

    with st.expander("🧠 Memory"):
        st.write(f"**Recent messages:** {len(bot.history)}")
        st.write("**Rolling summary:**", bot.summary or "_(none yet)_")

    if st.button("🗑️ Reset conversation", use_container_width=True):
        bot.reset()
        st.session_state.messages = []
        st.rerun()

    if docs_folder == "docs":
        st.subheader("Try a conversation")
        for example in [
            "How much does the Pro plan cost?",
            "What about its storage limit?",
            "How long do I have to get a refund on it?",
            "Does Acme Cloud integrate with Kubernetes?",
        ]:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example

        st.subheader("Try the guardrail")
        for example in ["What is the capital of France?", "How do I bake sourdough bread?"]:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example

    with st.expander("🛡️ Guardrail lab"):
        st.caption("See how a question scores against the threshold - no LLM call, instant.")
        probe = st.text_input("Test a question", key="probe", placeholder="e.g. What is the capital of France?")
        if probe.strip():
            found = store.search(probe.strip(), k=1, min_score=-1.0)
            if found:
                top_c, top_s = found[0]
                st.progress(min(max(top_s, 0.0), 1.0))
                if top_s >= min_score:
                    st.success(f"Passes: best score {top_s:.2f} >= threshold {min_score:.2f} ({top_c.source})")
                else:
                    st.error(f"Guardrail fires: best score {top_s:.2f} < threshold {min_score:.2f}")
        if docs_folder == "docs" and st.button("Score table: in-scope vs out-of-scope", key="score_table"):
            rows = ["| type | score | result | question |", "|---|---|---|---|"]
            for label, qs in (
                ("in scope", ["How much does the Pro plan cost?",
                              "How long do I have to request a refund on the Pro plan?",
                              "What languages does support offer?"]),
                ("OUT of scope", ["What is the capital of France?",
                                  "How do I bake sourdough bread?",
                                  "Does Acme Cloud integrate with Kubernetes?"]),
            ):
                for q in qs:
                    top = store.search(q, k=1, min_score=-1.0)[0][1]
                    rows.append(f"| {label} | {top:.2f} | {'pass' if top >= min_score else '**BLOCK**'} | {q} |")
            st.markdown("\n".join(rows))
            st.caption("Set the threshold between the highest OUT-of-scope score and the lowest in-scope score.")


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
st.title("💬 Chat with your documents")
st.caption("Local RAG: chunking → embeddings → vector search → grounded answer. Zero API cost.")
if rag_off:
    st.warning("🚫 **RAG is OFF.** Answers come from the model's own memory only: no documents, no sources.")


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        meta = msg.get("meta") if msg["role"] == "assistant" else None

        # The question that was actually used for the search, shown right on the main screen
        if meta:
            if meta["query"].strip() != meta["original"].strip():
                st.info(f"🔄 **Follow-up rewritten to:** {meta['query']}")
            elif meta.get("followup") and not (meta.get("rewrite_on") and meta.get("memory_on")):
                why = "rewriting is off" if not meta.get("rewrite_on") else "memory is off"
                st.caption(f"↪ Searched with your raw message because {why}, so words like 'it' or 'its' are not resolved.")

        st.markdown(msg["content"])
        if msg.get("rag_off"):
            st.caption("🚫 RAG off: answered from the model's memory only, no documents searched.")
            return
        if not meta:
            return

        thr, cands = meta["threshold"], meta["candidates"]
        best = cands[0][2] if cands else 0.0
        used = [c for c in cands if c[2] >= thr]
        refused = msg["content"].replace("\u2019", "'").strip().lower().startswith("i don't know")

        # Guardrail status: layer 1 = similarity threshold, layer 2 = the prompt's "say I don't know" rule
        if meta["guard"]:
            st.warning(f"🛡️ **Guardrail fired.** The best chunk scored `{best:.2f}`, below the threshold "
                       f"`{thr:.2f}`, so the LLM was never called.")
        elif refused:
            st.info(f"🛡️ Retrieval passed (best score `{best:.2f}` ≥ `{thr:.2f}`), but the model judged that "
                    "the chunks don't answer the question (prompt rule).")
        else:
            st.caption(f"✅ Passed guardrail: best score {best:.2f} ≥ threshold {thr:.2f}")

        with st.expander(f"🔎 Sources & retrieval ({len(used)} used, {len(cands) - len(used)} dropped)"):
            if meta["query"] != meta["original"]:
                st.markdown(f"**Follow-up rewritten to:** `{meta['query']}`")
            else:
                st.markdown(f"**Search query:** `{meta['query']}`")
            for i, (source, text, score) in enumerate(cands, 1):
                status = "✅ used" if score >= thr else "⛔ dropped (below threshold)"
                st.markdown(f"**[{i}] {source}**  ·  similarity `{score:.2f}`  ·  {status}")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption(text.replace("#", "\\#"))
        if meta.get("naive"):
            with st.expander("⚠️ Plain chatbot answer (no retrieval)"):
                st.markdown(meta["naive"])


for m in st.session_state.messages:
    render_message(m)

prompt = st.chat_input("Ask about your documents...")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    log.info(f"[ui] prompt received | k={k}, threshold={min_score:.2f}, rewrite={rewrite}, "
             f"memory={use_memory}, store={store_type}, model={llm.model}")
    is_followup = any(m["role"] == "user" for m in st.session_state.messages)
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    if rag_off:
        # Plain chatbot path: no retrieval, no rewrite, and RAGChat memory is left untouched
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking (no retrieval)..."):
                    plain = NaiveBot(llm).ask(prompt)
            except Exception as e:
                log.error(f"[ui] plain request failed: {e}")
                st.error(f"LLM call failed: {e}")
                st.stop()
        st.session_state.messages.append({"role": "assistant", "content": plain, "rag_off": True})
        st.rerun()

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and thinking..."):
                result = bot.ask(prompt)
                naive = NaiveBot(llm).ask(prompt) if compare else None
        except Exception as e:
            log.error(f"[ui] request failed: {e}")
            st.error(f"LLM call failed: {e}")
            st.stop()

    assistant_msg = {
        "role": "assistant",
        "content": result["answer"],
        "meta": {
            "original": prompt,
            "query": result["query"],
            "candidates": [(c.source, c.text, s) for c, s in result["candidates"]],
            "guard": result["guard"],
            "threshold": bot.min_score,
            "followup": is_followup,
            "rewrite_on": rewrite,
            "memory_on": use_memory,
            "naive": naive,
        },
    }
    st.session_state.messages.append(assistant_msg)
    st.rerun()  # re-render history in one place so ordering stays consistent
