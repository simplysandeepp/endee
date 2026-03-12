"""Premium Streamlit RAG UI using Endee + Groq."""

import html
import json
import os
import time

import msgpack
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon=":material/smart_toy:",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


def resolve_endee_url() -> str:
    """Resolve Endee URL for local and Render deployments."""
    explicit_url = os.getenv("ENDEE_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    hostport = os.getenv("ENDEE_HOSTPORT")
    if hostport:
        if hostport.startswith("http://") or hostport.startswith("https://"):
            return hostport.rstrip("/")
        return f"http://{hostport}".rstrip("/")

    return "http://localhost:8080"


ENDEE_URL = resolve_endee_url()
INDEX_NAME = os.getenv("INDEX_NAME", "RAGSYS")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def inject_theme() -> None:
    """Inject premium dark styling."""
    st.markdown(
        """
<style>
:root {
  --bg: #0b1220;
  --surface: #0f1b31;
  --card: #16243a;
  --border: #2b3b57;
  --text: #f1f5f9;
  --muted: #cbd5e1;
  --accent: #22c55e;
  --accent-alt: #3b82f6;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: radial-gradient(1100px 520px at 0% 0%, #1e3a8a 0%, var(--bg) 42%) no-repeat,
              linear-gradient(180deg, #0a1120 0%, var(--bg) 100%);
  color: var(--text);
  font-family: "Inter", "Poppins", "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] * {
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1427 0%, #0c172b 100%);
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(22, 36, 58, 0.92);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.45);
}

[data-testid="stMetric"] {
  background: rgba(15, 27, 49, 0.9);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px;
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
  color: var(--text) !important;
}

.stMarkdown, [data-testid="stMarkdownContainer"], p, li, label, span {
  color: var(--text);
}

[data-testid="stCaptionContainer"], .stCaption {
  color: var(--muted) !important;
}

.stTextInput label, .stTextArea label, .stFileUploader label, .stSelectbox label, .stSlider label {
  color: var(--text) !important;
  font-weight: 600;
}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
  color: var(--muted) !important;
  opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(15, 27, 49, 0.8) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 12px !important;
}

[data-testid="stExpander"] details {
  background: rgba(15, 27, 49, 0.88);
  border: 1px solid var(--border);
  border-radius: 12px;
}

[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
  color: var(--text) !important;
}

[data-testid="stChatMessage"] {
  background: rgba(15, 27, 49, 0.75);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.5rem 0.75rem;
}

.stButton > button {
  width: 100%;
  border-radius: 12px;
  border: 0;
  color: #ffffff;
  background: linear-gradient(90deg, var(--accent-alt), var(--accent));
  box-shadow: 0 8px 20px rgba(34, 197, 94, 0.25);
  transition: all 0.2s ease;
  font-weight: 600;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 30px rgba(99, 102, 241, 0.35);
}

.header-wrap {
  padding: 8px 0 16px 0;
}

.header-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0;
  color: var(--text);
}

.header-subtitle {
  margin: 4px 0 12px 0;
  color: var(--muted);
  font-size: 0.95rem;
}

.header-divider {
  height: 2px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent-alt), var(--accent));
  opacity: 0.85;
}

.status-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 10px;
  margin-bottom: 8px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: rgba(15, 23, 42, 0.75);
}

.status-left {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text);
  font-size: 0.88rem;
}

.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}

.pill {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid transparent;
}

.pill-ok {
  color: #bbf7d0;
  background: rgba(22, 163, 74, 0.2);
  border-color: rgba(34, 197, 94, 0.5);
}

.pill-off {
  color: #fecaca;
  background: rgba(220, 38, 38, 0.2);
  border-color: rgba(239, 68, 68, 0.5);
}

.small-note {
  color: var(--muted);
  font-size: 0.82rem;
}

.source-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px;
  margin-bottom: 10px;
  background: rgba(15, 23, 42, 0.7);
  transition: all 0.18s ease;
}

.source-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.45);
}

.source-head {
  display: flex;
  justify-content: space-between;
  color: var(--text);
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 4px;
}

.source-doc {
  color: #cbd5e1;
  font-size: 0.82rem;
  margin-bottom: 4px;
}

.source-text {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.4;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    """Initialize session state keys."""
    defaults = {
        "docs_indexed": 0,
        "chunks_stored": 0,
        "last_query_time": None,
        "chat_history": [],
        "last_sources": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def endee_headers(content_type_json: bool = False):
    """Build request headers for Endee API calls."""
    headers = {}
    if ENDEE_AUTH_TOKEN:
        headers["Authorization"] = ENDEE_AUTH_TOKEN
    if content_type_json:
        headers["Content-Type"] = "application/json"
    return headers or None


def is_endee_available() -> bool:
    """Check whether Endee health endpoint is reachable."""
    try:
        response = requests.get(
            f"{ENDEE_URL}/api/v1/health",
            headers=endee_headers(),
            timeout=2,
        )
        return response.ok
    except Exception:
        return False


@st.cache_resource
def load_embedding_model():
    """Lazy-load sentence transformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def create_index() -> bool:
    """Create vector index in Endee."""
    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/create",
            json={
                "index_name": INDEX_NAME,
                "dim": 384,
                "space_type": "cosine",
                "precision": "float32",
                "M": 16,
                "ef_con": 200,
            },
            headers=endee_headers(content_type_json=True),
            timeout=10,
        )
        return response.ok
    except Exception:
        return False


def is_missing_index_files_error(response: requests.Response) -> bool:
    """Detect Endee index-file-missing error from API response."""
    if response.status_code != 400:
        return False
    body = response.text.lower()
    return "required files missing for index" in body


def delete_index() -> bool:
    """Delete index if it exists."""
    try:
        response = requests.delete(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/delete",
            headers=endee_headers(),
            timeout=10,
        )
        return response.status_code in (200, 404)
    except Exception:
        return False


def chunk_text(text: str, chunk_size: int = 500):
    """Split text into word chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest_document(text: str, model, doc_name: str, chunk_size: int) -> int:
    """Ingest document chunks and metadata into Endee."""
    chunks = chunk_text(text, chunk_size=chunk_size)
    if not chunks:
        st.error("No chunks created from this file.")
        return 0

    embeddings = model.encode(chunks, normalize_embeddings=True)
    seed = f"{doc_name}-{int(time.time())}"

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        meta = json.dumps({"doc": doc_name, "text": chunk})
        vectors.append(
            {
                "id": f"{seed}-{i}",
                "vector": embedding.astype(np.float32).tolist(),
                "meta": meta,
            }
        )

    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
            json=vectors,
            headers=endee_headers(content_type_json=True),
            timeout=45,
        )
        if response.ok:
            return len(chunks)

        # Endee on free/ephemeral environments can lose index files after restart.
        # Auto-recreate index and retry insert once.
        if is_missing_index_files_error(response):
            delete_index()
            if create_index():
                retry = requests.post(
                    f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
                    json=vectors,
                    headers=endee_headers(content_type_json=True),
                    timeout=45,
                )
                if retry.ok:
                    return len(chunks)
                st.error(f"Endee API error after index recovery: {retry.status_code} - {retry.text}")
                return 0
            st.error("Index files were missing and auto-recreate failed. Click 'Initialize Index' and retry.")
            return 0

        st.error(f"Endee API error: {response.status_code} - {response.text}")
        return 0
    except Exception as exc:
        st.error(f"Insert failed: {exc}")
        return 0


def parse_meta(meta_value):
    """Parse metadata returned from Endee."""
    if isinstance(meta_value, bytes):
        meta_value = meta_value.decode("utf-8", errors="replace")

    if isinstance(meta_value, dict):
        return str(meta_value.get("doc", "Document")), str(meta_value.get("text", ""))

    if isinstance(meta_value, str):
        try:
            parsed = json.loads(meta_value)
            if isinstance(parsed, dict):
                return str(parsed.get("doc", "Document")), str(parsed.get("text", ""))
        except Exception:
            return "Document", meta_value

    return "Document", str(meta_value)


def search_similar(question: str, model, top_k: int = 3):
    """Search Endee and return source objects."""
    query_embedding = model.encode([question])[0]

    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
            json={"vector": query_embedding.astype(np.float32).tolist(), "k": top_k},
            headers=endee_headers(content_type_json=True),
            timeout=20,
        )
        if not response.ok:
            return []

        raw_results = msgpack.unpackb(response.content, raw=False)
        sources = []

        if isinstance(raw_results, list):
            for item in raw_results:
                score = None
                meta_val = ""

                if isinstance(item, dict):
                    meta_val = item.get("meta", "")
                    score = item.get("score", item.get("distance"))
                elif isinstance(item, list) and len(item) > 2:
                    score = item[1]
                    meta_val = item[2]
                else:
                    continue

                doc_name, text = parse_meta(meta_val)
                sources.append({"doc": doc_name, "text": text, "score": score})

        return sources
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return []


def generate_answer(question: str, context: str, temperature: float) -> str:
    """Generate answer with Groq."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return "GROQ_API_KEY is missing."

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content


def status_badge(label: str, is_ok: bool) -> str:
    """Render one status row for sidebar."""
    color = "#22c55e" if is_ok else "#ef4444"
    pill_class = "pill-ok" if is_ok else "pill-off"
    pill_text = "OK" if is_ok else "OFF"

    return (
        "<div class='status-item'>"
        "  <div class='status-left'>"
        f"    <span class='dot' style='background:{color}'></span>"
        f"    <span>{html.escape(label)}</span>"
        "  </div>"
        f"  <span class='pill {pill_class}'>{pill_text}</span>"
        "</div>"
    )


def render_header() -> None:
    """Render project header."""
    st.markdown(
        """
<div class="header-wrap">
  <p class="header-title">AI Document Assistant</p>
  <p class="header-subtitle">Powered by Endee Vector Database and Groq</p>
  <div class="header-divider"></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(endee_available: bool):
    """Render professional sidebar with settings."""
    with st.sidebar:
        st.markdown("## System Status")
        st.markdown(status_badge("Vector DB Connected", endee_available), unsafe_allow_html=True)
        st.markdown(
            status_badge(
                "LLM API Loaded",
                bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here"),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            status_badge("Documents Indexed", st.session_state.docs_indexed > 0),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("## Retrieval Settings")
        top_k = st.slider("Top-K", min_value=1, max_value=8, value=3, step=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=900, value=500, step=100)

        st.markdown("---")
        st.markdown("## Runtime")
        st.markdown(f"<div class='small-note'>Index: <b>{html.escape(INDEX_NAME)}</b></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='small-note'>Endee: <code>{html.escape(ENDEE_URL)}</code></div>",
            unsafe_allow_html=True,
        )

    return {"top_k": top_k, "temperature": temperature, "chunk_size": chunk_size}


def render_metrics() -> None:
    """Render quick metrics row."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents Indexed", st.session_state.docs_indexed)
    with col2:
        st.metric("Chunks Stored", st.session_state.chunks_stored)
    with col3:
        qt = st.session_state.last_query_time
        st.metric("Last Query Time", f"{qt:.2f}s" if isinstance(qt, float) else "-")


def render_sources(sources) -> None:
    """Render source cards panel."""
    with st.container(border=True):
        st.subheader("Retrieved Sources")

        if not sources:
            st.caption("No sources yet. Ask a question after indexing a document.")
            return

        with st.expander("View Sources", expanded=True):
            for idx, src in enumerate(sources, 1):
                score = src.get("score")
                if isinstance(score, (int, float)):
                    score_text = f"{score:.4f}"
                else:
                    score_text = "N/A"

                doc = html.escape(str(src.get("doc", "Document")))
                snippet = html.escape(str(src.get("text", ""))[:360])

                st.markdown(
                    f"""
<div class="source-card">
  <div class="source-head">
    <span>Source {idx}</span>
    <span>Score: {score_text}</span>
  </div>
  <div class="source-doc">{doc}</div>
  <div class="source-text">{snippet}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )


def main() -> None:
    """App entry point."""
    inject_theme()
    init_state()

    endee_available = is_endee_available()
    settings = render_sidebar(endee_available)

    render_header()
    render_metrics()

    with st.container(border=True):
        st.subheader("Document Upload")
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

        if uploaded_file:
            file_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
            estimated_chunks = len(chunk_text(file_text, settings["chunk_size"]))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption(f"File: {uploaded_file.name}")
            with c2:
                st.caption(f"Characters: {len(file_text)}")
            with c3:
                st.caption(f"Estimated Chunks: {estimated_chunks}")

            if st.button("Ingest Document", type="primary", disabled=not endee_available):
                model = load_embedding_model()
                progress = st.progress(0)

                with st.status("Ingestion Pipeline", expanded=True) as status:
                    status.write("Reading uploaded document...")
                    progress.progress(20)

                    status.write("Generating embeddings...")
                    progress.progress(55)

                    count = ingest_document(
                        file_text,
                        model,
                        uploaded_file.name,
                        settings["chunk_size"],
                    )
                    progress.progress(90)

                    if count > 0:
                        st.session_state.docs_indexed += 1
                        st.session_state.chunks_stored += count
                        status.write("Writing vectors to Endee...")
                        progress.progress(100)
                        status.update(label="Ingestion complete", state="complete", expanded=False)
                        st.success(f"File: {uploaded_file.name} | Chunks created: {count} | Embeddings stored: {count}")
                    else:
                        status.update(label="Ingestion failed", state="error", expanded=True)

                progress.empty()
        else:
            st.caption("Upload a text document to start indexing.")

    with st.container(border=True):
        st.subheader("Ask Question")
        question = st.text_area("Type your question", height=110, placeholder="Ask about your uploaded documents")

        ask_disabled = (not question.strip()) or (not endee_available)
        if st.button("Ask Question", type="primary", disabled=ask_disabled):
            st.session_state.chat_history.append({"role": "user", "content": question.strip()})

            timer_start = time.perf_counter()
            with st.status("Query Pipeline", expanded=True) as status:
                status.write("Retrieving context from vector database...")
                model = load_embedding_model()
                sources = search_similar(question.strip(), model, settings["top_k"])

                status.write("Generating answer...")
                context_text = "\n\n".join(src["text"] for src in sources)
                answer = generate_answer(question.strip(), context_text, settings["temperature"])

                elapsed = time.perf_counter() - timer_start
                st.session_state.last_query_time = elapsed
                st.session_state.last_sources = sources

                status.update(label="Answer ready", state="complete", expanded=False)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        if not endee_available:
            st.info("Vector DB is currently unavailable. Wait for Endee to wake up.")

    with st.container(border=True):
        st.subheader("Answer")
        if not st.session_state.chat_history:
            st.caption("No conversation yet. Ingest a document and ask a question.")
        else:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    render_sources(st.session_state.last_sources)


if __name__ == "__main__":
    main()
