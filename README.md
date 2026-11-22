# Local RAG Chatbot (Streamlit + Ollama + ChromaDB)

A conversational Retrieval-Augmented Generation (RAG) chatbot that runs **entirely on your machine with zero API cost**. Ask questions about your own `.txt` / `.md` documents, hold a multi-turn conversation, and watch every step of the pipeline: chunking, embeddings, vector search, guardrails, follow-up rewriting, and memory.

It is built as a teaching project. The code and UI are designed to show **why a plain chatbot hallucinates**, **how RAG fixes it**, and **what breaks between turn 1 and turn 3** of a conversation.

---

## Features

- **Fully local and free:** embeddings and answers come from [Ollama](https://ollama.com); the vector index lives in [ChromaDB](https://www.trychroma.com/) on disk.
- **End-to-end RAG pipeline:** sentence-aware chunking with overlap, `nomic-embed-text` embeddings, cosine search, grounded answer generation with `[1]`-style citations.
- **Real conversations:** chat history, a rolling summary of older turns, and **follow-up rewriting** ("what about its storage limit?" becomes a standalone question before searching).
- **Two-layer guardrail:** a similarity threshold blocks weak matches before the LLM is called, and a prompt rule tells the model to say "I don't know" when chunks don't answer the question.
- **Plain chatbot comparison:** one toggle shows what an LLM says *without* your documents (the hallucination).
- **Streamlit front end:** settings, source panel with scores, rewritten question shown on screen, memory panel, guardrail lab, file upload.
- **Terminal modes:** a scripted demo and an interactive chat with live commands.
- **Terminal logs:** every step is logged (rewrite, scores, prompts, timings).
- **Resilient:** falls back gracefully if ChromaDB, Ollama or a model is unavailable.

---

## How it works

```
INDEXING (once, or when documents change)
  docs/*.md|txt  ->  chunk_text  ->  Ollama embeddings  ->  ChromaDB (./chroma_db)

ASKING (every turn)
  question
    -> rewrite follow-up into a standalone question   (needs chat memory)
    -> embed the query and search the vector store    (top-k chunks)
    -> guardrail: drop chunks below the threshold; if none are left, answer "I don't know"
    -> build prompt: chunks + recent turns + rolling summary
    -> local LLM generates a cited answer             (one retry if it wrongly refuses)
    -> save the clean Q/A to memory, fold old turns into a summary
```

---

## Project structure

```
.
├── rag_chatbot.py      # the whole pipeline + terminal demo/chat (CLI)
├── streamlit_app.py    # Streamlit front end that imports rag_chatbot.py
├── docs/               # your documents (sample Acme Cloud docs are created on first run)
├── chroma_db/          # persisted vector index (created automatically)
└── README.md
```

Keep `rag_chatbot.py` and `streamlit_app.py` in the **same folder**.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Uses modern type hints (`list[str]`, `str \| None`) |
| [Ollama](https://ollama.com) | Runs the chat model and embeddings locally |
| ~4 GB free RAM | For a 3B model; larger models need more |

### 1. Install the Python packages

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate           # macOS / Linux

pip install streamlit numpy requests chromadb
```

### 2. Pull the Ollama models

```powershell
ollama pull nomic-embed-text         # embeddings
ollama pull llama3.2:3b              # chat model (any chat model works)
```

Check that Ollama is running with `ollama list`.

---

## Quick start

### Streamlit app

```powershell
python -m streamlit run .\streamlit_app.py
```

> Use `python -m streamlit` rather than `streamlit`. On locked-down Windows machines the `streamlit.exe` launcher can be blocked by an Application Control policy, while the Python interpreter is allowed.

The first run creates `./docs` with five sample documents (a fictional company, **Acme Cloud**), embeds them, and opens the app in your browser.

### Terminal demo (scripted tour)

```powershell
python .\rag_chatbot.py demo --log-level ERROR
```

Runs three parts:

1. **Plain chatbot vs RAG** on a question only your documents can answer.
2. **Turn 1 to turn 4**: a BROKEN bot (stateless, raw query) vs a FIXED bot (rewrite + memory).
3. **Guardrails**: score table of in-scope vs out-of-scope questions, then out-of-scope questions asked for real.

### Terminal chat

```powershell
python .\rag_chatbot.py chat
python .\rag_chatbot.py chat --docs my_folder --store numpy --log-level DEBUG
```

| Command | What it does |
|---|---|
| `/reset` | Clear conversation memory |
| `/debug` | Show the search query and all candidate chunks with scores |
| `/threshold 0.6` | Change the similarity threshold live (no number = show it) |
| `/log` | Toggle verbose logs (full prompts + raw LLM replies) |
| `/quit` | Exit |

---

## Using the Streamlit UI

**Main screen**

- Each answer can show a blue **🔄 Follow-up rewritten to: ...** box with the standalone question used for the search.
- A **guardrail status** line: fired (LLM never called), passed, or retrieval passed but the model declined.
- A **Sources & retrieval** expander lists every candidate chunk with its similarity score, marked used or dropped.
- With the compare toggle on, an extra expander shows the **plain chatbot answer** (no retrieval).

**Sidebar**

| Section | Control | Purpose |
|---|---|---|
| Knowledge base | Documents folder, file upload | Add `.txt` / `.md` files; the index rebuilds automatically |
| Chunking | Max words (30-250), overlap (0-3 sentences) | Change chunk size; re-embeds when changed |
| Model | Ollama model dropdown | Lists models installed in Ollama (embedding models are hidden) |
| Vector store | ChromaDB / In-memory NumPy | Chroma persists the index in `./chroma_db` |
| Retrieval | Top-k, similarity threshold | How many chunks to fetch, and the guardrail cut-off |
| Conversation fixes | Rewrite follow-ups, chat memory, turns kept | Toggle the fixes on and off live to see the failures |
| Demo helpers | Show plain chatbot | Side-by-side hallucination demo |
| Guardrail lab | Test a question, score table | Score any question against the threshold without calling the LLM |
| Memory | Message count, rolling summary | See older turns being compressed |
| Examples | One-click question buttons | Run the sample conversation in order |

Settings are applied live; changing a slider does **not** clear the conversation.

---

## Configuration

Set these as environment variables (PowerShell: `$env:LOG_LEVEL="DEBUG"`).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is listening |
| `LLM_MODEL` | `llama3.2:3b` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_BACKEND` | `ollama` | `ollama`, `st` (sentence-transformers), `hash` (pure NumPy), or `auto` |
| `VECTOR_STORE` | `chroma` | `chroma` or `numpy` |
| `CHROMA_PATH` | `chroma_db` | Where the Chroma index is saved |
| `LOG_LEVEL` | `INFO` | `DEBUG` also prints full prompts and raw LLM replies |

CLI flags for `rag_chatbot.py`: `--docs`, `--model`, `--store chroma|numpy`, `--log-level DEBUG|INFO|WARNING|ERROR`.

---

## Questions to try (sample Acme Cloud docs)

**Hallucination vs RAG** (turn on "Also show plain chatbot")
- How long do I have to request a refund on the Pro plan? *(14 days)*
- Which regions can I store my data in? *(Mumbai or Frankfurt)*

**Follow-up chain** (ask in order, in one conversation)
1. How much does the Pro plan cost?
2. What about its storage limit?
3. How long do I have to get a refund on it?
4. And what support do I get?

Then turn off **Rewrite follow-up questions** and **Use chat memory** and repeat. The follow-ups should now retrieve the wrong chunks.

**Guardrail** (not in the documents)
- What is the capital of France?
- Does Acme Cloud integrate with Kubernetes?

---

## Code walkthrough

### `rag_chatbot.py`

| Piece | What it does |
|---|---|
| `chunk_text` | Converts markdown headings (`# Title` becomes `Title:`), splits into sentences, packs them into chunks of up to `max_words` (default 80), and repeats the last sentence of each chunk at the start of the next (overlap). Chunks keep their source filename. |
| `OllamaEmbedder` | Calls Ollama's `/api/embed` and normalizes vectors so a dot product equals cosine similarity. Adds nomic's `search_document:` / `search_query:` prefixes. |
| `HashingEmbedder`, `SentenceTransformerEmbedder`, `make_embedder` | Alternative backends and the selector. The hashing one is a pure-NumPy lexical fallback; sentence-transformers is optional. |
| `ChromaVectorStore` | Persistent store. Uses cosine space; `score = 1 - distance`. A **signature** (documents + chunk settings + chunker version + embedding model) decides whether to reuse the saved index or rebuild it. |
| `VectorStore` | Same interface in plain NumPy, used as a fallback if Chroma can't load. |
| `LocalLLM` | Thin Ollama client. Resolves the requested model against what is installed, retries slow calls, and raises readable errors. |
| `NaiveBot` | The control: asks the LLM directly with no retrieval. |
| `RAGChat.ask` | The core loop: rewrite, search, guardrail, generate, update memory. |
| `RAGChat._generate` | Builds the prompt (chunks in the current user message, recent turns, rolling summary), calls the LLM, and retries once with a minimal prompt if it refuses despite good chunks. |
| `RAGChat._compress_memory` | Keeps the last N turns verbatim and folds older ones into a rolling summary. |
| `build_store`, `docs_signature` | Load, chunk, embed, and index; skip re-embedding when nothing changed. |
| `demo`, `chat`, `main` | Terminal demo, interactive chat, and the CLI. |

### `streamlit_app.py`

- `@st.cache_resource` loads the embedder and builds the index once; it only re-indexes when files, chunk settings, or the store type change.
- Sidebar values are written onto the `RAGChat` object on each run, so changes apply instantly without losing memory.
- `st.session_state` keeps the bot and message list between reruns.
- `render_message` draws the rewritten question, guardrail status, answer, and source panel for each message.
- Errors from Ollama are shown as readable messages instead of stack traces.

---

## What breaks between turn 1 and turn 3 (and the fix)

| # | Problem | Fix in this project |
|---|---|---|
| 1 | Follow-ups aren't searchable. "What about its storage limit?" has no plan name, so it matches every plan equally. | `_condense` rewrites it into a standalone question before retrieval. |
| 2 | The bot forgets, so "it" can't be resolved. | Chat history is passed to the rewriter and the LLM. |
| 3 | History grows until the context window overflows. | `_compress_memory`: last N turns verbatim plus a rolling summary. |
| 4 | Retrieved chunks pollute history. | Only the clean question and answer are stored; fresh chunks go only in the current turn. |
| 5 | Out-of-scope questions still get an answer, because top-k always returns something. | Similarity threshold plus a strict "say I don't know" prompt rule. |
| 6 | Small models copy earlier refusals from history and keep refusing. | Refused turns are left out of the prompt; one retry with a minimal prompt. |

---

## Logging

Logs print to the terminal that started the app.

```
[turn 2] user: 'what about its storage limit?' | history=2 msg(s), rewrite=on, memory=on
[rewrite] 'what about its storage limit?' -> 'What is the storage limit of the Pro plan?'
[search]   #1 score=0.810 KEEP storage.md: Storage Limits: The Basic plan includes 50 GB...
[generate] prompt = 3 chunk(s) (~120 words) + 1 prior turn(s) + summary=no
[llm] -> llama3.2:3b: 4 message(s), ~260 words, temp=0.1
[llm] <- 31 words in 4.2s
```

| Tag | Meaning |
|---|---|
| `[docs]` `[chunk]` `[index]` `[chroma]` | Loading, chunking and indexing; whether the saved index was reused |
| `[rewrite]` | Follow-up and its standalone rewrite |
| `[search]` | Each candidate chunk with score, marked KEEP or DROP |
| `[guard]` | Guardrail fired; the LLM was not called |
| `[generate]` | Prompt composition, refusal warnings, retry result |
| `[llm]` | Every Ollama call with size and duration (full text at `DEBUG`) |
| `[memory]` | Turns kept, folded into the summary, or left out of the prompt |
| `[ollama]` | Retries when Ollama is slow to respond |

---

## Using your own documents

1. Put `.txt` or `.md` files in a folder (or upload them in the Streamlit sidebar).
2. Point the app at the folder: set **Documents folder** in the sidebar, or run `python .\rag_chatbot.py chat --docs my_folder`.
3. Tune chunk size, top-k and the threshold using the sources panel and the Guardrail lab.

Tips:
- The sample documents are tiny (one chunk each). For real documents, try a larger chunk size (about 150 to 250 words).
- **Tune the threshold** between the highest out-of-scope score and the lowest in-scope score. Words shared with your documents (like a brand name) raise a question's score, so out-of-scope questions that mention them can slip past the threshold. The prompt rule then has to catch them.
- Default thresholds per embedder: nomic `0.50`, MiniLM `0.30`, hashing `0.35`. These are starting points, so check your own scores.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Application Control policy has blocked this file` on `streamlit.exe` | Run `python -m streamlit run .\streamlit_app.py` instead. |
| `DLL load failed ... Application Control policy` when importing scikit-learn / torch | The sentence-transformers path is blocked. The default Ollama embeddings avoid it. |
| ChromaDB won't import | The app logs the reason and falls back to the in-memory NumPy store. |
| `404 Not Found` for `/api/chat` | The model isn't installed under that exact tag. Run `ollama list` and `ollama pull <model>`; the app now picks an installed chat model and warns you. |
| `ReadTimeout` talking to Ollama | Ollama is busy loading a model. Calls retry automatically; check `ollama ps` and avoid running heavy requests in parallel. |
| Answers are raw chunks instead of sentences | No chat model was detected (extractive fallback). Check `ollama list` and the `[llm]` log lines. |
| "I don't know" although the right chunk was retrieved | Small models can be too literal. Check `[generate]` logs, use `LOG_LEVEL=DEBUG` to see the prompt, or try a larger model (for example `llama3.1:8b` or `qwen2.5:7b`). |
| Old chunks still showing after changing the code or documents | Stop the app and delete the `chroma_db` folder to force a rebuild. |
| Everything is blocked as "out of scope" | The threshold is too high for your embedder. Check scores in the Guardrail lab and lower it. |

---

## Limitations and ideas

- **Chunking** is word-based and sentence-aware, not heading- or structure-aware. Possible upgrades: split by markdown sections first, semantic chunking, parent-document retrieval.
- Only `.txt` and `.md` files are loaded. PDF or DOCX support would need extra parsing.
- Answers appear all at once; streaming would require using Ollama's `stream: true`.
- Retrieval is dense-only; hybrid search (BM25 + vectors) and a re-ranker would improve precision.
- Memory is per session and in-process; it is not saved across restarts.

---

## License

Use and adapt freely for learning and internal projects.
