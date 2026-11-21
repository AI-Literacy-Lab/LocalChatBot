#!/usr/bin/env python3
"""
rag_chatbot.py - A conversational RAG chatbot that runs with ZERO API costs.

Stack (all local, all free):
  * Embeddings : Ollama nomic-embed-text (local HTTP, no native Python deps)
  * Vector DB  : ChromaDB, persisted in ./chroma_db (falls back to in-memory NumPy)
  * LLM        : Ollama running a small open model (llama3.2:3b by default)
                 -> if Ollama isn't running, the bot falls back to extractive
                    answers (returns the best-matching passage) so it still works.

Setup:
  pip install numpy requests chromadb
  ollama pull nomic-embed-text     # embeddings
  # If Windows blocks torch/scikit-learn DLLs, the code auto-falls back to Ollama embeddings
  # (ollama pull nomic-embed-text) or a pure-NumPy backend. Force one: set EMBED_BACKEND=ollama|hash
  # optional but recommended (for real generated answers):
  #   install Ollama from https://ollama.com, then:  ollama pull llama3.2:3b

Usage:
  python rag_chatbot.py demo                 # guided tour of all 5 topics
  python rag_chatbot.py chat                 # interactive chat over ./docs
  python rag_chatbot.py chat --docs my_dir   # chat over your own .txt/.md files
  Chat commands: /reset  /debug  /quit

What this file covers
  1. Why a plain chatbot hallucinates on YOUR documents      -> NaiveBot, demo part 1
  2. The RAG pipeline: chunk -> embed -> search -> generate   -> chunk_text, VectorStore, RAGChat._generate
  3. Turning it into a conversation (history, memory, follow-ups) -> RAGChat.ask / _condense / _compress_memory
  4. What breaks between turn 1 and turn 3, and the fixes     -> demo part 2 (BROKEN vs FIXED)
  5. Zero API cost                                            -> everything above is local
"""

import argparse
import hashlib
import logging
import os
import re
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

CHUNK_VERSION = 2   # bump when chunking changes, so persisted Chroma indexes get rebuilt
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# Embedding backend: auto | st | ollama | hash
#   st     = sentence-transformers (needs torch/scikit-learn DLLs)
#   ollama = embeddings served by Ollama (no native Python deps)  <- good on locked-down PCs
#   hash   = pure-NumPy lexical embeddings (always works, lowest quality)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama")
# Vector store: chroma (persistent ChromaDB) | numpy (in-memory)
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

# ---------------------------------------------------------------------------
# Terminal logging.  LOG_LEVEL=INFO (default) prints every pipeline step;
# LOG_LEVEL=DEBUG also prints the full prompts sent to the LLM and its raw replies.
# ---------------------------------------------------------------------------
log = logging.getLogger("rag")


def setup_logging(level: str | None = None):
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    if not log.handlers:
        try:  # Windows consoles can choke on unusual characters; never crash because of a log line
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
        log.propagate = False   # avoid duplicate lines when Streamlit configures the root logger
    log.setLevel(getattr(logging, lvl, logging.INFO))


def _preview(text: str, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


setup_logging()


def ollama_get(path: str, timeout: float = 10, tries: int = 3):
    """GET against Ollama with retries. Ollama handles requests one at a time, so while it is busy
    loading a model even a tiny call like /api/tags can stall for a few seconds - that is not an error."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(f"{OLLAMA_URL}{path}", timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            log.warning(f"[ollama] GET {path} failed (attempt {attempt}/{tries}): {type(e).__name__}")
            if attempt < tries:
                time.sleep(attempt)  # 1s, 2s, ...
    raise last


# ---------------------------------------------------------------------------
# Sample corpus (fictional company, so the base LLM cannot possibly know it)
# ---------------------------------------------------------------------------
SAMPLE_DOCS = {
    "pricing.md": """# Acme Cloud Pricing
Acme Cloud offers three plans: Basic, Pro and Enterprise.
The Basic plan costs $9 per month and is meant for individuals.
The Pro plan costs $29 per month and is meant for small teams of up to 25 users.
The Enterprise plan has custom pricing and is negotiated with our sales team.
Annual billing gives a 20 percent discount on Basic and Pro.""",
    "storage.md": """# Storage Limits
The Basic plan includes 50 GB of storage per account.
The Pro plan includes 1 TB of shared team storage.
The Enterprise plan includes unlimited storage with a fair-use policy.
Files larger than 5 GB must be uploaded using the Acme desktop sync client.""",
    "refunds.md": """# Refund Policy
Basic plan customers can request a refund within 7 days of purchase.
Pro plan customers can request a full refund within 14 days of purchase.
Enterprise refunds are governed by the individual customer contract.
Refunds are returned to the original payment method within 5 to 10 business days.
Annual subscriptions cancelled after the refund window are not refunded but remain active until the term ends.""",
    "support.md": """# Support
Basic plan customers get email support with a 72 hour response time.
Pro plan customers get email and chat support with a 24 hour response time.
Enterprise customers get 24x7 phone support and a dedicated account manager.
Support is available in English, Hindi and Malayalam.""",
    "security.md": """# Security and Compliance
Acme Cloud is SOC 2 Type II certified.
All data is encrypted at rest with AES-256 and in transit with TLS 1.3.
Customers can choose to store data in the Mumbai or Frankfurt region.
Single sign-on via SAML is available on the Pro and Enterprise plans.""",
}


# ---------------------------------------------------------------------------
# 1) Loading + chunking
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    source: str
    idx: int


def load_docs(folder: str) -> dict[str, str]:
    """Read every .txt/.md file in a folder. Creates the sample corpus if missing."""
    p = Path(folder)
    if not p.exists():
        p.mkdir(parents=True)
        for name, body in SAMPLE_DOCS.items():
            (p / name).write_text(body, encoding="utf-8")
        log.info(f"[setup] created sample documents in ./{folder}")
    docs = {f.name: f.read_text(encoding="utf-8", errors="ignore")
            for f in sorted(p.glob("*")) if f.suffix.lower() in {".txt", ".md"}}
    if not docs:
        sys.exit(f"No .txt/.md files found in {folder}")
    log.info(f"[docs] loaded {len(docs)} file(s) from ./{folder}: {', '.join(docs)}")
    return docs


def chunk_text(text: str, source: str, max_words: int = 80, overlap: int = 1) -> list[Chunk]:
    """
    Sentence-aware chunking with overlap.
    - Splitting on sentences avoids cutting facts in half (the #1 chunking bug).
    - Overlap keeps context that straddles a boundary retrievable from both sides.
    """
    # "# Storage Limits" -> "Storage Limits:" so headings read naturally inside a chunk
    text = re.sub(r"^[ \t]*#+[ \t]*(.+?)[ \t]*$", lambda m: m.group(1).rstrip(":.") + ":", text, flags=re.M)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return [Chunk(c, source, i) for i, c in enumerate(chunks)]


# ---------------------------------------------------------------------------
# 2) Embeddings + vector search
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    name = "sentence-transformers (MiniLM)"
    default_min_score = 0.30

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # may fail on locked-down machines
        self._m = SentenceTransformer(EMBED_MODEL)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return self._m.encode(texts, normalize_embeddings=normalize_embeddings,
                              show_progress_bar=show_progress_bar)


class OllamaEmbedder:
    """Embeddings from Ollama's HTTP API. Needs: ollama pull nomic-embed-text"""
    default_min_score = 0.50   # nomic scores run higher than MiniLM; tune with /debug

    def __init__(self, model: str = OLLAMA_EMBED_MODEL):
        self.model = model
        self.name = f"Ollama ({model})"
        r = ollama_get("/api/tags")
        names = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        if model.split(":")[0] not in names:
            raise RuntimeError(f"model not pulled - run: ollama pull {model}")
        self.encode(["test"])  # fail fast if the endpoint doesn't work

    # nomic-embed-text was trained with task prefixes; they noticeably improve retrieval.
    def _pre(self, kind: str) -> str:
        return f"{kind}: " if self.model.startswith("nomic") else ""

    def embed_documents(self, texts):
        return self.encode([self._pre("search_document") + t for t in texts])

    def embed_query(self, text):
        return self.encode([self._pre("search_query") + text])

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        t0 = time.time()
        r = requests.post(f"{OLLAMA_URL}/api/embed",
                          json={"model": self.model, "input": list(texts)}, timeout=180)
        r.raise_for_status()
        v = np.array(r.json()["embeddings"], dtype=np.float32)
        log.debug(f"[embed] {len(texts)} text(s) -> {v.shape} via {self.model} in {time.time() - t0:.2f}s")
        if normalize_embeddings:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v


_STOP = set("a an the of to in on for and or is are was were be do does did i you we it its my your "
            "can how what when where which who with at by from as that this".split())


class HashingEmbedder:
    """Pure-NumPy lexical embeddings (hashed bag-of-words, sublinear TF).
    No downloads, no native deps beyond NumPy. Matches words, not meaning."""
    name = "hashing bag-of-words (pure NumPy, lexical only)"
    default_min_score = 0.35

    def __init__(self, dims: int = 1024):
        self.dims = dims

    def _tok(self, text):
        out = []
        for t in re.findall(r"[a-z0-9]+", text.lower()):
            if t in _STOP:
                continue
            if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]
            out.append(t)
        return out

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        v = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for t in self._tok(text):
                v[i, zlib.crc32(t.encode()) % self.dims] += 1.0
        v = np.where(v > 0, 1.0 + np.log(np.maximum(v, 1.0)), 0.0).astype(np.float32)
        if normalize_embeddings:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v


def embed_docs(emb, texts):
    """Embed passages (uses nomic's document prefix when the backend supports it)."""
    fn = getattr(emb, "embed_documents", None)
    return fn(texts) if fn else emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(emb, text):
    """Embed one query -> array of shape (1, dim)."""
    fn = getattr(emb, "embed_query", None)
    return fn(text) if fn else emb.encode([text], normalize_embeddings=True, show_progress_bar=False)


def make_embedder(backend: str | None = None):
    """Pick an embedding backend; in 'auto' mode fall through st -> ollama -> hash."""
    backend = (backend or EMBED_BACKEND).lower()
    makers = {"st": SentenceTransformerEmbedder, "ollama": OllamaEmbedder, "hash": HashingEmbedder}
    order = ["st", "ollama", "hash"] if backend == "auto" else [backend]
    errors = []
    for b in order:
        try:
            emb = makers[b]()
            log.info(f"[embeddings] backend: {emb.name}")
            return emb
        except Exception as e:  # ImportError (blocked DLL), connection errors, etc.
            errors.append(f"{b}: {type(e).__name__}: {str(e)[:140]}")
            log.warning(f"[embeddings] '{b}' unavailable -> {errors[-1]}")
    raise RuntimeError("No embedding backend could be loaded. " + " | ".join(errors))


class VectorStore:
    def __init__(self, embedder=None):
        self.model = embedder or make_embedder()
        self.default_min_score = getattr(self.model, "default_min_score", 0.30)
        self.chunks: list[Chunk] = []
        self.vecs: np.ndarray | None = None

    def add(self, chunks: list[Chunk]):
        # Prefix the source name: "refunds.md: ..." helps match topic-style queries.
        texts = [f"{c.source}: {c.text}" for c in chunks]
        v = embed_docs(self.model, texts)
        self.chunks += chunks
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])

    def search(self, query: str, k: int = 3, min_score: float = 0.0):
        """Cosine similarity (vectors are normalised, so dot product == cosine)."""
        q = embed_query(self.model, query)[0]
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]


class ChromaVectorStore:
    """
    Same interface as VectorStore (search / chunks / default_min_score) but backed by ChromaDB.
      * Embeddings come from OUR embedder (Ollama), so Chroma never downloads its own model.
      * Persistent: the index lives in ./chroma_db and is reused on the next run if the
        documents, chunk settings and embedding model are unchanged.
      * Cosine space: Chroma returns distance = 1 - cosine, so score = 1 - distance.
    """

    def __init__(self, embedder=None, path: str = CHROMA_PATH):
        import chromadb  # may fail on locked-down machines -> build_store falls back to NumPy
        self.model = embedder or make_embedder()
        self.default_min_score = getattr(self.model, "default_min_score", 0.30)
        self.client = chromadb.PersistentClient(path=path)
        tag = re.sub(r"[^a-zA-Z0-9]+", "_", getattr(self.model, "name", "emb")).strip("_").lower()[:40]
        self.cname = f"docs_{tag}"          # valid Chroma name: starts/ends alphanumeric
        self.col = None
        self.reused = False
        self._chunks: list[Chunk] | None = None

    def index(self, chunks: list[Chunk], signature: str):
        """Build the collection, or reuse the persisted one if the signature matches."""
        self._chunks, self.reused = None, False
        existing = None
        try:
            existing = self.client.get_collection(self.cname, embedding_function=None)
        except Exception:
            pass
        if existing is not None:
            if (existing.metadata or {}).get("signature") == signature and existing.count() == len(chunks):
                self.col, self.reused = existing, True
                log.info(f"[chroma] collection '{self.cname}' is up to date -> reusing "
                         f"{existing.count()} stored chunks (no re-embedding)")
                return
            log.info(f"[chroma] collection '{self.cname}' is out of date (documents, chunking or "
                     "embedding model changed) -> rebuilding")
            self.client.delete_collection(self.cname)

        try:
            self.col = self.client.create_collection(
                self.cname, metadata={"hnsw:space": "cosine", "signature": signature},
                embedding_function=None)
        except Exception:  # newer Chroma versions prefer the 'configuration' argument
            self.col = self.client.create_collection(
                self.cname, metadata={"signature": signature},
                configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

        B = 32
        t0 = time.time()
        log.info(f"[chroma] embedding {len(chunks)} chunks in batches of {B} "
                 "(only happens when documents change)...")
        for i in range(0, len(chunks), B):
            batch = chunks[i:i + B]
            vecs = embed_docs(self.model, [f"{c.source}: {c.text}" for c in batch])
            self.col.add(
                ids=[f"{c.source}::{c.idx}" for c in batch],
                embeddings=vecs.tolist(),
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "idx": c.idx} for c in batch],
            )
        log.info(f"[chroma] stored {self.col.count()} chunks in {time.time() - t0:.1f}s")

    @property
    def chunks(self) -> list[Chunk]:
        if self._chunks is None:
            r = self.col.get(include=["documents", "metadatas"])
            self._chunks = [Chunk(d, m["source"], int(m["idx"]))
                            for d, m in zip(r["documents"], r["metadatas"])]
        return self._chunks

    def search(self, query: str, k: int = 3, min_score: float = 0.0):
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=embed_query(self.model, query).tolist(),
                             n_results=min(k, n),
                             include=["documents", "metadatas", "distances"])
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1.0 - float(dist)
            if score >= min_score:
                out.append((Chunk(doc, meta["source"], int(meta["idx"])), score))
        return out


# ---------------------------------------------------------------------------
# Local LLM (Ollama) - free, no API key
# ---------------------------------------------------------------------------
_EMBED_HINTS = ("embed", "nomic", "bge", "minilm", "e5-", "gte-")


def is_embed_model(name: str) -> bool:
    return any(h in name.lower() for h in _EMBED_HINTS)


def list_ollama_models() -> list[str]:
    """Names of models installed in Ollama, e.g. ['llama3.2:3b', 'nomic-embed-text:latest']."""
    try:
        return [m["name"] for m in ollama_get("/api/tags", timeout=8, tries=2).json().get("models", [])]
    except Exception:
        return []


class LocalLLM:
    """Thin Ollama client. Resolves the requested model against what is ACTUALLY installed,
    so a missing tag (llama3.2:3b vs llama3.2:1b) never turns into a cryptic 404."""

    def __init__(self, model: str = LLM_MODEL):
        self.requested = model
        self.installed = list_ollama_models()
        if not self.installed:
            log.warning("[llm] could not read the model list from Ollama (not running, or too slow) "
                        "-> using the extractive fallback until it responds")
        chat_models = [m for m in self.installed if not is_embed_model(m)]
        self.note = ""
        self.model = model
        if model in self.installed:
            pass
        elif ":" not in model and f"{model}:latest" in self.installed:
            self.model = f"{model}:latest"
        elif chat_models:
            self.model = chat_models[0]
            self.note = (f"'{model}' is not installed; using '{self.model}' instead "
                         f"(or run: ollama pull {model})")
        self.available = bool(chat_models) and (self.model in self.installed)
        if self.note:
            log.warning(f"[llm] {self.note}")
        if self.installed and not chat_models:
            log.warning("[llm] Ollama is running but only embedding models are installed. "
                  f"Run: ollama pull {model}")

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        t0 = time.time()
        words = sum(len(m["content"].split()) for m in messages)
        log.info(f"[llm] -> {self.model}: {len(messages)} message(s), ~{words} words, temp={temperature}")
        if log.isEnabledFor(logging.DEBUG):
            for m in messages:
                log.debug(f"[llm]   {m['role']:>9}: " + m["content"].replace("\n", "\n" + " " * 38))
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature, "num_ctx": 4096}},
            timeout=180,
        )
        if r.status_code != 200:
            try:
                detail = r.json().get("error", r.text)
            except Exception:
                detail = r.text
            hint = f" -> run: ollama pull {self.model}" if r.status_code == 404 else ""
            log.error(f"[llm] Ollama returned HTTP {r.status_code} for '{self.model}': {detail}{hint}")
            raise RuntimeError(f"Ollama error {r.status_code} for model '{self.model}': {detail}{hint}")
        out = r.json()["message"]["content"].strip()
        log.info(f"[llm] <- {len(out.split())} words in {time.time() - t0:.1f}s")
        log.debug("[llm] reply: " + out.replace("\n", "\n" + " " * 38))
        return out


# ---------------------------------------------------------------------------
# 1) The plain chatbot: no retrieval, no knowledge of your documents
# ---------------------------------------------------------------------------
class NaiveBot:
    """Asks the LLM directly. It has never seen your docs, so it makes things up."""

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def ask(self, question: str) -> str:
        if not self.llm.available:
            return "(Ollama not running - start it to see the hallucination live)"
        return self.llm.chat([
            {"role": "system", "content": "You are the customer support assistant for Acme Cloud."},
            {"role": "user", "content": question},
        ], temperature=0.7)


# ---------------------------------------------------------------------------
# 2-4) The RAG chatbot: retrieval + generation + conversation handling
# ---------------------------------------------------------------------------
RAG_SYSTEM = (
    "You are a helpful support assistant. You get numbered context passages and a question.\n"
    "1. If the passages contain information that answers the question - even if it is worded "
    "differently - answer directly and concisely from it, and cite the passage numbers like [1]. "
    "Treat synonyms and paraphrases as matches (e.g. 'storage limit' matches 'includes 1 TB of storage').\n"
    "2. ONLY if the passages contain nothing relevant to the question, reply: "
    "I don't know based on the provided documents.\n"
    "3. Never use outside knowledge."
)

# Second attempt when the first answer was a refusal: simpler task, no chat history to confuse a small model.
RAG_RETRY_SYSTEM = (
    "Read the numbered passages and find the sentence(s) most relevant to the question. "
    "Answer in one or two sentences using only those, and cite like [1]. "
    "If no sentence is relevant at all, reply: I don't know based on the provided documents."
)

CONDENSE_PROMPT = """Rewrite the follow-up message as a single standalone question that can be
understood WITHOUT the conversation.
Rules:
- Resolve pronouns and implied topics (it, its, that, "what about X") using the conversation.
- Keep the ORIGINAL question type. If the follow-up asks what / how much / how long, the rewrite must
  too. Do NOT turn it into a yes/no question.
- Keep it short. Return ONLY the rewritten question.

Example
Conversation:
user: How much does the Pro plan cost?
assistant: The Pro plan costs $29 per month.
Follow-up: what about storage limit
Standalone question: What is the storage limit of the Pro plan?

Conversation:
{history}

Follow-up: {question}
Standalone question:"""

SUMMARY_PROMPT = """Summarise this conversation in 2-3 sentences. Keep names, plans, numbers
and the topic currently being discussed.

Previous summary: {prev}
New messages:
{new}

Summary:"""

PRONOUNS = re.compile(r"\b(it|its|that|this|they|them|their|those|these|one|there)\b", re.I)
NO_ANSWER = "I don't know based on the provided documents."


class RAGChat:
    def __init__(
        self,
        store: VectorStore,
        llm: LocalLLM,
        k: int = 3,
        min_score: float | None = None,   # None -> use the embedder's default
        keep_turns: int = 3,
        rewrite_followups: bool = True,   # FIX for "follow-up questions retrieve garbage"
        use_memory: bool = True,          # FIX for "bot forgets what we were talking about"
    ):
        self.store, self.llm = store, llm
        self.k, self.keep_turns = k, keep_turns
        self.min_score = min_score if min_score is not None else store.default_min_score
        self.rewrite_followups, self.use_memory = rewrite_followups, use_memory
        self.history: list[dict] = []     # recent turns, verbatim
        self.summary: str = ""            # compressed older turns
        self.turn = 0

    def reset(self):
        self.history, self.summary, self.turn = [], "", 0
        log.info("[memory] conversation reset")

    # --- follow-up handling -------------------------------------------------
    def _condense(self, question: str) -> str:
        """Turn 'what about its storage?' into 'What is the storage limit of the Pro plan?'"""
        if not self.history:
            return question
        if self.llm.available:
            hist = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-4:])
            if self.summary:
                hist = f"(summary: {self.summary})\n{hist}"
            out = self.llm.chat([{"role": "user",
                                  "content": CONDENSE_PROMPT.format(history=hist, question=question)}])
            rewritten = out.strip().strip('"') or question
            log.info(f"[rewrite] {question!r} -> {rewritten!r}")
            return rewritten
        # Heuristic fallback when no LLM is available: glue the last user question on.
        if len(question.split()) <= 7 or PRONOUNS.search(question):
            last_q = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
            log.info(f"[rewrite] no LLM available, heuristic: {question!r} + previous question {last_q!r}")
            return f"{last_q} {question}"
        return question

    # --- memory management --------------------------------------------------
    def _compress_memory(self):
        """Keep the last N turns verbatim; fold older ones into a rolling summary.
        Without this the prompt grows every turn until it overflows the context window."""
        max_msgs = 2 * self.keep_turns
        if len(self.history) <= max_msgs:
            return
        old, self.history = self.history[:-max_msgs], self.history[-max_msgs:]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        if self.llm.available:
            self.summary = self.llm.chat([{"role": "user",
                                           "content": SUMMARY_PROMPT.format(prev=self.summary or "none", new=text)}])
        else:
            self.summary = (self.summary + " " + text)[-600:]
        log.info(f"[memory] folded {len(old) // 2} older turn(s) into the rolling summary: "
                 f"{_preview(self.summary, 120)!r}")

    # --- generation ---------------------------------------------------------
    @staticmethod
    def _is_refusal(text: str) -> bool:
        return text.replace("\u2019", "'").strip().lower().startswith("i don't know")

    def _history_for_prompt(self) -> list[dict]:
        """Recent turns, minus turns the bot refused. If past refusals stay in the prompt, small
        models copy the pattern and keep refusing ('history poisoning')."""
        msgs = self.history[-2 * self.keep_turns:]
        out = []
        for i in range(0, len(msgs) - 1, 2):
            if not self._is_refusal(msgs[i + 1]["content"]):
                out += [msgs[i], msgs[i + 1]]
            else:
                log.info(f"[memory] left refused turn out of the prompt: {msgs[i]['content']!r}")
        return out

    def _generate(self, question: str, hits) -> str:
        if not self.llm.available:  # extractive fallback: still grounded, still free
            log.warning("[generate] no LLM available -> returning the best chunk verbatim (extractive)")
            best, _ = hits[0]
            return f"{best.text}  [source: {best.source}]"

        context = "\n\n".join(f"[{i + 1}] ({c.source}) {c.text}" for i, (c, _) in enumerate(hits))
        msgs = [{"role": "system", "content": RAG_SYSTEM}]
        hist: list[dict] = []
        if self.use_memory:
            if self.summary:
                msgs.append({"role": "system", "content": f"Earlier conversation summary: {self.summary}"})
            hist = self._history_for_prompt()
            msgs += hist
        log.info(f"[generate] prompt = {len(hits)} chunk(s) (~{len(context.split())} words) + "
                 f"{len(hist) // 2} prior turn(s) + summary={'yes' if (self.use_memory and self.summary) else 'no'}")
        # Retrieved context goes ONLY into the current user message (see "what breaks" #4).
        # The task is restated AFTER the context: small models weight the end of the prompt most.
        msgs.append({"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {question}\n\n"
                                "Answer using the context above."})
        answer = self.llm.chat(msgs)

        if self._is_refusal(answer):  # retry once with a minimal prompt before giving up
            log.warning("[generate] model refused although relevant chunks were retrieved "
                        "-> retrying once with a minimal prompt")
            retry = self.llm.chat([
                {"role": "system", "content": RAG_RETRY_SYSTEM},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
            ])
            if not self._is_refusal(retry):
                log.info("[generate] retry succeeded")
                answer = retry
            else:
                log.warning("[generate] retry also refused -> returning the refusal "
                            "(try a larger model, or check the prompt with LOG_LEVEL=DEBUG)")
        return answer

    # --- the main entry point -----------------------------------------------
    def ask(self, question: str) -> dict:
        self.turn += 1
        t_start = time.time()
        log.info("=" * 72)
        log.info(f"[turn {self.turn}] user: {question!r} | history={len(self.history)} msg(s), "
                 f"rewrite={'on' if self.rewrite_followups else 'off'}, memory={'on' if self.use_memory else 'off'}")

        # 1. Make the question searchable on its own
        if self.rewrite_followups and self.history:
            query = self._condense(question)
        else:
            query = question
            log.info("[rewrite] skipped (first turn or rewriting is off) -> searching with the raw question")

        # 2. Retrieve; show every candidate, then apply the similarity floor
        t0 = time.time()
        candidates = self.store.search(query, k=self.k, min_score=-1.0)
        hits = [(c, sc) for c, sc in candidates if sc >= self.min_score]
        log.info(f"[search] top-{self.k} for {query!r} in {time.time() - t0:.2f}s "
                 f"(threshold {self.min_score:.2f})")
        for rank, (c, sc) in enumerate(candidates, 1):
            verdict = "KEEP" if sc >= self.min_score else "DROP"
            log.info(f"[search]   #{rank} score={sc:.3f} {verdict} {c.source}: {_preview(c.text)}")

        # 3. Guardrail: nothing relevant found -> don't let the LLM improvise
        if not hits:
            log.warning("[guard] no chunk passed the threshold -> answering 'I don't know' without calling the LLM")
            answer = NO_ANSWER
        else:
            answer = self._generate(query, hits)

        # 4. Update memory. Store the clean Q/A only (NOT the retrieved chunks).
        if self.use_memory:
            self.history += [{"role": "user", "content": question},
                             {"role": "assistant", "content": answer}]
            self._compress_memory()
            log.info(f"[memory] now holding {len(self.history)} message(s), "
                     f"summary={'yes' if self.summary else 'no'}")

        log.info(f"[turn {self.turn}] answer: {_preview(answer, 140)!r} (total {time.time() - t_start:.1f}s)")
        return {"answer": answer, "query": query, "hits": hits,
                "candidates": candidates, "guard": not hits}


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------
def docs_signature(docs: dict, max_words: int, overlap: int, embedder) -> str:
    """Fingerprint of everything that affects the index, so Chroma knows when to re-embed."""
    h = hashlib.sha256(f"v{CHUNK_VERSION}|{getattr(embedder, 'name', '')}|{max_words}|{overlap}".encode())
    for name in sorted(docs):
        h.update(name.encode())
        h.update(docs[name].encode("utf-8", "ignore"))
    return h.hexdigest()[:32]


def build_store(folder: str, store_type: str | None = None, max_words: int = 80,
                overlap: int = 1, embedder=None, db_path: str = CHROMA_PATH):
    store_type = (store_type or VECTOR_STORE).lower()
    docs = load_docs(folder)
    embedder = embedder or make_embedder()
    chunks = []
    for name, text in docs.items():
        chunks += chunk_text(text, name, max_words=max_words, overlap=overlap)

    log.info(f"[chunk] {len(docs)} document(s) -> {len(chunks)} chunk(s) "
             f"(max_words={max_words}, overlap={overlap}, store={store_type})")
    for ch in chunks:
        log.debug(f"[chunk]   {ch.source}#{ch.idx}: {_preview(ch.text, 110)}")

    if store_type == "chroma":
        try:
            store = ChromaVectorStore(embedder, path=db_path)
            store.index(chunks, docs_signature(docs, max_words, overlap, embedder))
            state = "reused persisted index" if store.reused else "embedded and stored"
            log.info(f"[index] ChromaDB ({state}): {len(docs)} documents, {len(chunks)} chunks -> ./{db_path}")
            return store
        except Exception as e:
            log.warning(f"[index] ChromaDB unavailable ({type(e).__name__}: {str(e)[:160]}) "
                        "-> falling back to the in-memory NumPy store")

    t0 = time.time()
    store = VectorStore(embedder)
    store.add(chunks)
    log.info(f"[index] NumPy in-memory store: {len(docs)} documents, {len(chunks)} chunks "
             f"embedded in {time.time() - t0:.1f}s")
    return store


# ---------------------------------------------------------------------------
# Demo: walks through all five topics
# ---------------------------------------------------------------------------
def banner(t: str):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def show_hits(hits, indent="      "):
    for c, s in hits:
        print(f"{indent}{s:.2f}  {c.source}: {c.text[:70]}...")


def demo(store: VectorStore, llm: LocalLLM):
    print(f"[llm] Ollama model '{llm.model}':",
          "available" if llm.available else "NOT running -> extractive fallback mode")

    # ---- Part 1: hallucination ------------------------------------------------
    banner("PART 1 - Plain chatbot vs RAG on a question only YOUR docs can answer")
    q = "How long do I have to request a refund on the Pro plan?"
    print(f"Q: {q}\n")
    print("NaiveBot (no retrieval):\n  ", NaiveBot(llm).ask(q), "\n")
    r = RAGChat(store, llm).ask(q)
    print("RAG bot (retrieval):\n  ", r["answer"])
    print("   retrieved:")
    show_hits(r["hits"])
    print("\n   Truth is in refunds.md: Pro = 14 days. The naive bot has never seen that file,\n"
          "   so it produces a confident, plausible-sounding guess. That's hallucination.")

    # ---- Part 2: multi-turn failures -------------------------------------------
    banner("PART 2 - Turn 1 works. Turn 2 and 3 break. Then we fix them.")
    turns = [
        "How much does the Pro plan cost?",
        "What about its storage limit?",
        "And how long do I have to get a refund on it?",
        "Does Acme Cloud integrate with Kubernetes?",   # not in the docs at all
    ]
    broken = RAGChat(store, llm, rewrite_followups=False, use_memory=False)
    fixed = RAGChat(store, llm)

    for i, t in enumerate(turns, 1):
        print(f"\n--- Turn {i}: {t!r}")
        for label, bot in (("BROKEN (stateless, raw query)", broken), ("FIXED  (rewrite + memory)", fixed)):
            r = bot.ask(t)
            print(f"  [{label}]")
            print(f"    search query : {r['query']!r}")
            print(f"    top chunks   :")
            show_hits(r["hits"], indent="      ")
            if r["guard"]:
                print("    guardrail    : FIRED - no chunk passed the threshold, LLM was not called")
            print(f"    answer       : {r['answer']}")

    # ---- Part 3: guardrails ------------------------------------------------------
    banner("PART 3 - Guardrails: what happens when the docs DON'T contain the answer")
    in_scope = ["How much does the Pro plan cost?",
                "How long do I have to request a refund on the Pro plan?",
                "What languages does support offer?"]
    out_scope = ["What is the capital of France?",
                 "How do I bake sourdough bread?",
                 "Does Acme Cloud integrate with Kubernetes?"]
    thr = fixed.min_score
    print(f"Layer 1 = similarity threshold (currently {thr:.2f}). Best chunk score for each question:\n")
    for label, qs in (("in scope", in_scope), ("OUT of scope", out_scope)):
        for q in qs:
            top = store.search(q, k=1, min_score=-1.0)[0][1]
            print(f"   {label:<13} {top:.2f}  {'pass ' if top >= thr else 'BLOCK'}  {q}")
    print("\nNow asking the out-of-scope questions for real:")
    for q in out_scope:
        r = RAGChat(store, llm).ask(q)
        top = r["candidates"][0][1]
        print(f"\n   Q: {q}")
        if r["guard"]:
            print(f"   GUARDRAIL FIRED: best score {top:.2f} < {thr:.2f}, so the LLM was never called.")
        else:
            print(f"   Layer 1 let it through (best score {top:.2f} >= {thr:.2f}); "
                  "layer 2, the 'say I don't know' prompt rule, has to catch it.")
        print(f"   A: {r['answer']}")
    print(textwrap_block("""

    Two layers of defence:
      1. Threshold  - cheap and deterministic: weak matches never reach the LLM.
      2. Prompt rule - the LLM is told to say "I don't know" if the chunks don't answer the question.
    Tune the threshold between the highest OUT-of-scope score and the lowest in-scope score above.
    Words shared with your docs (like the brand name "Acme Cloud") raise a question's score,
    so out-of-scope questions that mention them can slip past layer 1.
    """))

    banner("WHAT BROKE, AND THE FIX FOR EACH")
    print(textwrap_block("""
    1. Follow-ups are not searchable.  "What about its storage limit?" has no plan name in it,
       so the embedding matches ALL plans equally.
       FIX -> _condense(): rewrite the follow-up into a standalone question before retrieval.

    2. The bot forgets.  A stateless bot can't resolve "it".
       FIX -> keep chat history and pass it to the LLM.

    3. History grows forever.  By turn ~15 you overflow the context window and quality drops.
       FIX -> _compress_memory(): last N turns verbatim + rolling summary of older ones.

    4. Retrieved chunks pollute history.  If you store context in memory, turn 3's prompt holds
       turn 1 and 2's passages too: noisy, slow, and confusing for small models.
       FIX -> only store the clean question/answer; inject fresh context into the current turn only.

    5. Out-of-scope questions still get an answer.  Top-k ALWAYS returns something, even for
       "Do you support Kubernetes?".
       FIX -> similarity threshold (min_score) + a strict "say I don't know" prompt.
    """))


def textwrap_block(s: str) -> str:
    return "\n".join(line[4:] if line.startswith("    ") else line for line in s.strip("\n").splitlines())


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------
def chat(store: VectorStore, llm: LocalLLM):
    bot = RAGChat(store, llm)
    debug = False
    print("\nChat ready. Commands: /reset  /debug  /log  /threshold [value]  /quit\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            bot.reset()
            print("(memory cleared)")
            continue
        if q == "/debug":
            debug = not debug
            print(f"(debug {'on' if debug else 'off'})")
            continue
        if q.startswith("/threshold"):  # show or change the similarity threshold live
            parts = q.split()
            if len(parts) > 1:
                try:
                    bot.min_score = float(parts[1])
                except ValueError:
                    print("usage: /threshold 0.55")
                    continue
            print(f"(similarity threshold = {bot.min_score:.2f}; chunks scoring below it are dropped)")
            continue
        if q == "/log":  # toggle verbose logs (full prompts + raw replies) on the fly
            lvl = "INFO" if log.level == logging.DEBUG else "DEBUG"
            setup_logging(lvl)
            print(f"(log level {lvl})")
            continue
        r = bot.ask(q)
        if debug:
            print(f"   [query] {r['query']}")
            show_hits(r["candidates"], indent="   [candidate] ")
        if r["guard"]:
            best = r["candidates"][0][1] if r["candidates"] else 0.0
            print(f"   [guardrail] fired: best score {best:.2f} < threshold {bot.min_score:.2f}, LLM not called")
        print(f"bot > {r['answer']}\n")


def main():
    ap = argparse.ArgumentParser(description="Local, zero-cost conversational RAG chatbot")
    ap.add_argument("mode", choices=["demo", "chat"], nargs="?", default="demo")
    ap.add_argument("--docs", default="docs", help="folder of .txt/.md files (sample docs are created if missing)")
    ap.add_argument("--model", default=LLM_MODEL, help="Ollama model name")
    ap.add_argument("--store", choices=["chroma", "numpy"], default=VECTOR_STORE, help="vector store backend")
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
                    help="DEBUG also prints full prompts and raw LLM replies (or set env LOG_LEVEL)")
    args = ap.parse_args()
    if args.log_level:
        setup_logging(args.log_level)

    llm = LocalLLM(args.model)
    store = build_store(args.docs, store_type=args.store)
    demo(store, llm) if args.mode == "demo" else chat(store, llm)


if __name__ == "__main__":
    main()
