"""Premium Streamlit RAG UI using Endee + Groq."""

import html
import io
import json
import os
import re
import textwrap
import time

import msgpack
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(
    page_title="AI Document Assistant by Sandeep Prajapati",
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
    """Inject monochrome glassmorphism styling."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg-0: #040404;
  --bg-1: #0a0a0a;
  --glass: rgba(255, 255, 255, 0.08);
  --glass-strong: rgba(255, 255, 255, 0.13);
  --glass-stroke: rgba(255, 255, 255, 0.22);
  --text: #f4f4f4;
  --muted: #b7b7b7;
  --ink: #0a0a0a;
  --glow: rgba(255, 255, 255, 0.35);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:
    radial-gradient(1200px 700px at 12% 8%, #101010 0%, transparent 52%),
    radial-gradient(900px 500px at 90% 92%, #141414 0%, transparent 58%),
    linear-gradient(160deg, var(--bg-1) 0%, var(--bg-0) 42%, #000000 100%);
  color: var(--text);
  font-family: "Manrope", "Segoe UI", sans-serif;
}

body {
  line-height: 1.45;
}

[data-testid="stAppViewContainer"] {
  position: relative;
  isolation: isolate;
  overflow-x: hidden;
}

[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
  content: "";
  position: fixed;
  pointer-events: none;
  z-index: 0;
  width: clamp(260px, 34vw, 520px);
  aspect-ratio: 1 / 1;
  border-radius: 53% 47% 62% 38% / 44% 53% 47% 56%;
  background: radial-gradient(
    circle at 32% 26%,
    rgba(255, 255, 255, 0.42) 0%,
    rgba(255, 255, 255, 0.15) 36%,
    rgba(255, 255, 255, 0.05) 58%,
    rgba(255, 255, 255, 0) 74%
  );
  filter: blur(26px);
  opacity: 0.34;
}

[data-testid="stAppViewContainer"]::before {
  left: -8vw;
  top: -16vh;
  animation: liquidFloatA 18s ease-in-out infinite;
}

[data-testid="stAppViewContainer"]::after {
  right: -10vw;
  bottom: -20vh;
  border-radius: 39% 61% 43% 57% / 55% 38% 62% 45%;
  animation: liquidFloatB 22s ease-in-out infinite;
}

[data-testid="stMainBlockContainer"] {
  position: relative;
  z-index: 2;
  padding-top: 1rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(
    180deg,
    rgba(14, 14, 14, 0.9) 0%,
    rgba(8, 8, 8, 0.86) 100%
  );
  border-right: 1px solid var(--glass-stroke);
  backdrop-filter: blur(16px) saturate(115%);
}

[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"] {
  background: linear-gradient(
    135deg,
    rgba(255, 255, 255, 0.1) 0%,
    rgba(255, 255, 255, 0.03) 65%
  );
  border: 1px solid var(--glass-stroke);
  border-radius: 18px;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 28px 68px rgba(0, 0, 0, 0.56);
  backdrop-filter: blur(18px) saturate(110%);
  transition: transform 0.2s ease, box-shadow 0.25s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-1px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 32px 76px rgba(0, 0, 0, 0.6);
}

[data-testid="stMetric"] {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 14px;
  padding: 10px;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.18),
    0 16px 36px rgba(0, 0, 0, 0.4);
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
  color: var(--text) !important;
}

[data-testid="stMetricLabel"] {
  line-height: 1.35 !important;
}

[data-testid="stMetricValue"] {
  line-height: 1.15 !important;
}

.stMarkdown, [data-testid="stMarkdownContainer"], p, li, label, span {
  color: var(--text);
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
  line-height: 1.5 !important;
  overflow-wrap: anywhere;
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
  background: rgba(8, 8, 8, 0.82) !important;
  color: #f5f5f5 !important;
  -webkit-text-fill-color: #f5f5f5 !important;
  border: 1px solid rgba(255, 255, 255, 0.3) !important;
  border-radius: 12px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14) !important;
  line-height: 1.45 !important;
  caret-color: #ffffff !important;
}

div[data-baseweb="input"],
div[data-baseweb="textarea"],
[data-testid="stTextArea"] > div,
[data-testid="stTextInput"] > div {
  background: transparent !important;
}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: rgba(8, 8, 8, 0.82) !important;
  color: #f5f5f5 !important;
  -webkit-text-fill-color: #f5f5f5 !important;
  caret-color: #ffffff !important;
  border: 1px solid rgba(255, 255, 255, 0.3) !important;
  border-radius: 12px !important;
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
  border-color: rgba(255, 255, 255, 0.58) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.16),
    0 0 0 1px rgba(255, 255, 255, 0.24) !important;
}

div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
  color: var(--muted) !important;
  opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px dashed rgba(255, 255, 255, 0.28) !important;
  border-radius: 14px !important;
  backdrop-filter: blur(16px);
}

[data-testid="stExpander"] details {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 12px;
  backdrop-filter: blur(14px);
}

[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
  color: var(--text) !important;
}

[data-testid="stChatMessage"] {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 14px;
  padding: 0.5rem 0.75rem;
  backdrop-filter: blur(14px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14);
}

.stButton > button {
  width: 100%;
  border-radius: 13px;
  border: 1px solid rgba(255, 255, 255, 0.55);
  color: var(--ink);
  background: linear-gradient(
    145deg,
    #ffffff 0%,
    #e6e6e6 45%,
    #cbcbcb 100%
  );
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.8),
    0 10px 22px rgba(0, 0, 0, 0.42);
  transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
  font-weight: 700;
  letter-spacing: 0.01em;
}

.stButton > button[kind="primary"],
.stButton > button[kind="primary"] * {
  color: #101010 !important;
  -webkit-text-fill-color: #101010 !important;
}

.stButton > button[kind="secondary"] {
  color: #f2f2f2;
  background: linear-gradient(
    145deg,
    rgba(255, 255, 255, 0.1) 0%,
    rgba(255, 255, 255, 0.04) 100%
  );
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.stButton > button[kind="secondary"],
.stButton > button[kind="secondary"] * {
  color: #f2f2f2 !important;
  -webkit-text-fill-color: #f2f2f2 !important;
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.8),
    0 16px 30px rgba(0, 0, 0, 0.5);
  filter: brightness(1.04);
}

.stButton > button:disabled {
  opacity: 0.62;
  color: #f0f0f0;
  background: linear-gradient(
    145deg,
    rgba(255, 255, 255, 0.2) 0%,
    rgba(255, 255, 255, 0.12) 100%
  );
  border-color: rgba(255, 255, 255, 0.35);
}

.header-wrap {
  padding: 12px 0 18px 0;
}

.header-title {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 800;
  margin: 0;
  line-height: 1.05;
  letter-spacing: 0.015em;
  background: linear-gradient(180deg, #ffffff 0%, #d6d6d6 52%, #f4f4f4 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 10px 38px rgba(255, 255, 255, 0.16);
}

.header-subtitle {
  margin: 6px 0 14px 0;
  color: var(--muted);
  font-size: 0.95rem;
  letter-spacing: 0.01em;
}

.header-divider {
  height: 1px;
  border-radius: 999px;
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0.02),
    rgba(255, 255, 255, 0.92),
    rgba(255, 255, 255, 0.02)
  );
  opacity: 0.9;
  box-shadow: 0 0 24px rgba(255, 255, 255, 0.28);
}

.status-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  padding: 8px 10px;
  margin-bottom: 8px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(12px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.status-left {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text);
  font-size: 0.88rem;
  min-width: 0;
  flex: 1 1 auto;
}

.status-left span:last-child {
  white-space: normal;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}

.pill {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.24);
  color: #ededed;
  background: rgba(255, 255, 255, 0.09);
  flex-shrink: 0;
  margin-left: auto;
}

.pill-ok {
  color: #bbf7d0;
  background: rgba(34, 197, 94, 0.2);
  border-color: rgba(34, 197, 94, 0.55);
}

.pill-off {
  color: #fecaca;
  background: rgba(239, 68, 68, 0.2);
  border-color: rgba(239, 68, 68, 0.55);
}

.small-note {
  color: var(--muted);
  font-size: 0.82rem;
}

.source-card {
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 12px;
  padding: 10px 12px;
  margin-bottom: 10px;
  background: rgba(255, 255, 255, 0.04);
  transition: all 0.18s ease;
  backdrop-filter: blur(12px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.source-card:hover {
  transform: translateY(-2px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.14),
    0 16px 28px rgba(0, 0, 0, 0.45);
}

.source-head {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 6px 10px;
  color: var(--text);
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 4px;
  line-height: 1.35;
}

.source-doc {
  color: #d2d2d2;
  font-size: 0.82rem;
  margin-bottom: 4px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.source-text {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.route-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 6px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.04);
  backdrop-filter: blur(14px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.route-nav-sidebar {
  gap: 8px;
}

.route-nav-sidebar .route-btn {
  flex: 1 1 100%;
}

.route-btn {
  flex: 1 1 170px;
  text-align: center;
  text-decoration: none !important;
  color: #f2f2f2 !important;
  border: 1px solid rgba(255, 255, 255, 0.26);
  border-radius: 11px;
  padding: 8px 12px;
  font-size: 0.88rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  background: linear-gradient(
    140deg,
    rgba(255, 255, 255, 0.08) 0%,
    rgba(255, 255, 255, 0.03) 100%
  );
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.2),
    0 10px 24px rgba(0, 0, 0, 0.38);
  transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

.route-btn:hover {
  transform: translateY(-1px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.24),
    0 14px 28px rgba(0, 0, 0, 0.44);
  filter: brightness(1.04);
}

.route-btn-active {
  color: #0d0d0d !important;
  border-color: rgba(255, 255, 255, 0.56);
  background: linear-gradient(145deg, #ffffff 0%, #dbdbdb 52%, #bdbdbd 100%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.8),
    0 14px 30px rgba(0, 0, 0, 0.42);
}

.landing-hero {
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 22px;
  padding: clamp(18px, 4vw, 40px);
  background: linear-gradient(
    135deg,
    rgba(255, 255, 255, 0.12) 0%,
    rgba(255, 255, 255, 0.03) 68%
  );
  backdrop-filter: blur(20px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 32px 72px rgba(0, 0, 0, 0.5);
}

.landing-kicker {
  display: inline-block;
  font-size: 0.74rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #ececec;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.4);
  background: rgba(255, 255, 255, 0.08);
  margin-bottom: 10px;
}

.landing-title {
  margin: 0;
  font-size: clamp(1.7rem, 5vw, 3.25rem);
  line-height: 1.08;
  color: #ffffff;
}

.landing-copy {
  margin: 10px 0 0 0;
  max-width: 900px;
  color: #c8c8c8;
  font-size: 1rem;
  line-height: 1.65;
}

.landing-chips {
  margin-top: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.landing-chip {
  font-size: 0.78rem;
  color: #f0f0f0;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.24);
  background: rgba(255, 255, 255, 0.05);
}

.landing-card {
  height: 100%;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  background: rgba(255, 255, 255, 0.04);
  backdrop-filter: blur(12px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.landing-card-title {
  margin: 0;
  font-size: 1rem;
  color: #ffffff;
  font-weight: 700;
}

.landing-card-copy {
  margin: 6px 0 0 0;
  color: #c7c7c7;
  font-size: 0.9rem;
  line-height: 1.55;
}

.landing-section-title {
  margin: 0 0 8px 0;
  font-size: 1.25rem;
  color: #fafafa;
  font-weight: 700;
}

.landing-section-copy {
  margin: 0;
  color: #c9c9c9;
  line-height: 1.7;
  font-size: 0.96rem;
}

.landing-list {
  margin: 0;
  padding-left: 1.15rem;
  color: #cdcdcd;
}

.landing-list li {
  margin-bottom: 0.42rem;
  line-height: 1.62;
}

[data-testid="stList"] li {
  line-height: 1.6;
}

[data-testid="stAlert"] {
  border-radius: 12px;
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.22);
  background: rgba(255, 255, 255, 0.06);
}

::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.04);
}

::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: linear-gradient(180deg, #dadada, #8e8e8e);
  border: 2px solid rgba(0, 0, 0, 0.42);
}

@keyframes liquidFloatA {
  0% { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
  50% { transform: translate3d(3vw, 5vh, 0) rotate(18deg) scale(1.08); }
  100% { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
}

@keyframes liquidFloatB {
  0% { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
  50% { transform: translate3d(-3vw, -4vh, 0) rotate(-16deg) scale(1.06); }
  100% { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
}

@media (max-width: 900px) {
  .status-item {
    align-items: flex-start;
  }

  .pill {
    margin-top: 2px;
  }
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

Answer format rules:
- Write between 5 and 10 lines.
- Each line should be a complete, useful sentence.
- Keep the answer grounded in the provided context.
- If context is insufficient, mention that clearly in one line.

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    answer = response.choices[0].message.content or ""
    return enforce_answer_length(answer, min_lines=5, max_lines=10)


def enforce_answer_length(answer: str, min_lines: int = 5, max_lines: int = 10) -> str:
    """Ensure answer stays within a readable 5-10 line range."""
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if min_lines <= len(lines) <= max_lines:
        return "\n".join(lines)

    source_text = " ".join(lines) if lines else answer.strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        if sentence.strip()
    ]

    if not sentences:
        return answer

    if len(sentences) < min_lines:
        filler = "This answer is based on retrieved context and can be expanded with more source detail."
        while len(sentences) < min_lines:
            sentences.append(filler)

    sentences = sentences[:max_lines]
    return "\n".join(f"{idx}. {text}" for idx, text in enumerate(sentences, start=1))


def build_answer_screenshot(question: str, answer: str):
    """Create a downloadable PNG screenshot for the latest answer."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    title = "RAG Answer Snapshot"
    question_text = f"Question: {question.strip()}" if question.strip() else "Question: -"
    body = answer.strip() or "-"

    width = 1300
    padding = 56
    inner_pad = 34

    try:
        title_font = ImageFont.truetype("arial.ttf", 44)
        body_font = ImageFont.truetype("arial.ttf", 27)
        small_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    wrap_width = 68
    body_lines = []
    for paragraph in body.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            body_lines.append("")
            continue
        body_lines.extend(textwrap.wrap(paragraph, width=wrap_width) or [""])

    if not body_lines:
        body_lines = ["-"]

    line_height = 40
    card_height = max(360, inner_pad * 2 + line_height * (len(body_lines) + 4))
    height = padding * 2 + card_height

    image = Image.new("RGB", (width, height), (6, 6, 6))
    draw = ImageDraw.Draw(image)

    # Outer glow panel
    draw.rounded_rectangle(
        [(padding, padding), (width - padding, height - padding)],
        radius=36,
        fill=(16, 16, 16),
        outline=(210, 210, 210),
        width=2,
    )

    x = padding + inner_pad
    y = padding + inner_pad
    draw.text((x, y), title, fill=(245, 245, 245), font=title_font)
    y += 58
    draw.text((x, y), question_text, fill=(199, 199, 199), font=small_font)
    y += 50

    for line in body_lines[:26]:
        draw.text((x, y), line, fill=(236, 236, 236), font=body_font)
        y += line_height

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def status_badge(label: str, is_ok: bool) -> str:
    """Render one status row for sidebar."""
    color = "#22c55e" if is_ok else "#ef4444"
    pill_class = "pill-ok" if is_ok else "pill-off"
    pill_text = "OK" if is_ok else "OFF"

    return (
        "<div class='status-item'>"
        "  <div class='status-left'>"
        f"    <span class='dot' style='background:{color};box-shadow:0 0 10px {color}'></span>"
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


def normalize_page_route(raw_page) -> str:
    """Normalize route names to supported page keys."""
    if raw_page is None:
        return "home"

    page = str(raw_page).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if page in {"home", "landing"}:
        return "home"
    if page in {"rag", "workspace", "ragworkspace"}:
        return "rag"
    return "home"


def get_current_page() -> str:
    """Get active page from URL query params using canonical values."""
    raw_page = st.query_params.get("page")
    if isinstance(raw_page, list):
        raw_page = raw_page[0] if raw_page else None

    page = normalize_page_route(raw_page)
    raw_text = "" if raw_page is None else str(raw_page).strip().lower()
    if raw_text not in {"home", "rag"}:
        st.query_params["page"] = page

    return page


def render_route_nav(current_page: str, sidebar_mode: bool = False) -> None:
    """Render in-page route controls (no new tab)."""
    key_prefix = "side" if sidebar_mode else "main"
    col1, col2 = st.columns(2)

    home_label = "Home (Active)" if current_page == "home" else "Home"
    rag_label = "RAG Workspace (Active)" if current_page == "rag" else "RAG Workspace"

    with col1:
        if st.button(
            home_label,
            key=f"{key_prefix}_route_home",
            use_container_width=True,
            type="primary" if current_page == "home" else "secondary",
        ) and current_page != "home":
            st.query_params["page"] = "home"
            st.rerun()

    with col2:
        if st.button(
            rag_label,
            key=f"{key_prefix}_route_rag",
            use_container_width=True,
            type="primary" if current_page == "rag" else "secondary",
        ) and current_page != "rag":
            st.query_params["page"] = "rag"
            st.rerun()


def render_sidebar(endee_available: bool, current_page: str):
    """Render sidebar status and workspace controls."""
    with st.sidebar:
        st.markdown("## Navigation")
        render_route_nav(current_page, sidebar_mode=True)

        st.markdown("---")
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

        # Keep retrieval controls fixed (UI removed as requested).
        top_k = 3
        temperature = 0.3
        chunk_size = 500

        if current_page != "rag":
            st.markdown("---")
            st.markdown("## Quick Start")
            st.markdown(
                "<div class='small-note'>Use Home to understand architecture and value. "
                "Switch to <b>RAG Workspace</b> to ingest docs and ask questions.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("## Runtime")
        st.markdown(f"<div class='small-note'>Index: <b>{html.escape(INDEX_NAME)}</b></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='small-note'>Endee: <code>{html.escape(ENDEE_URL)}</code></div>",
            unsafe_allow_html=True,
        )

    return {"top_k": top_k, "temperature": temperature, "chunk_size": chunk_size}


def render_landing_page() -> None:
    """Render a detailed home landing page for the RAG project."""
    st.markdown(
        """
<div class="landing-hero">
  <div class="landing-kicker">Home / Landing</div>
  <h1 class="landing-title">Why This RAG Project Matters</h1>
  <p class="landing-copy">
    Classic LLM chat is powerful but often generic, stale, or disconnected from your real data.
    Retrieval-Augmented Generation (RAG) solves that by grounding answers in your own documents.
    This project combines a responsive Streamlit interface, semantic retrieval, and Groq generation
    to deliver answers that are fast, relevant, and explainable.
  </p>
  <div class="landing-chips">
    <span class="landing-chip">Grounded Answers</span>
    <span class="landing-chip">Lower Hallucination Risk</span>
    <span class="landing-chip">Traceable Sources</span>
    <span class="landing-chip">Production-Ready Workflow</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="landing-card">
  <p class="landing-card-title">RAG Importance</p>
  <p class="landing-card-copy">
    RAG injects real context at query time, so model output is based on your corpus, not only
    pretraining memory. This improves trust and factual alignment.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="landing-card">
  <p class="landing-card-title">Why Endee Vector DB</p>
  <p class="landing-card-copy">
    Endee gives efficient nearest-neighbor retrieval over embeddings, enabling low-latency
    search even as document volume grows.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="landing-card">
  <p class="landing-card-title">Business Advantage</p>
  <p class="landing-card-copy">
    Teams answer internal questions faster, reduce manual lookup time, and preserve knowledge
    quality with source-backed responses.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.container(border=True):
        st.markdown("<p class='landing-section-title'>Detailed Advantages of Using RAG with Endee</p>", unsafe_allow_html=True)
        st.markdown(
            """
<ol class="landing-list">
  <li><b>Better factual reliability:</b> semantic retrieval fetches the most relevant chunks before generation, reducing unsupported answers.</li>
  <li><b>Fresh knowledge path:</b> new documents can be ingested without retraining a foundation model, so the system keeps pace with changing internal information.</li>
  <li><b>Source transparency:</b> retrieved chunks are displayed for verification, improving confidence and auditability.</li>
  <li><b>Scalable architecture:</b> vector indexing in Endee enables efficient similarity search across growing content sets.</li>
  <li><b>Cost efficiency:</b> focused context means smaller, targeted prompts to the LLM.</li>
</ol>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown("<p class='landing-section-title'>How This Project Delivers Value</p>", unsafe_allow_html=True)
        st.markdown(
            """
<p class="landing-section-copy">
The workflow is simple and practical: upload document, chunk text, generate embeddings, store vectors in Endee,
retrieve top-K context for each user query, and generate a final response with Groq. This design creates a strong
balance between speed, answer quality, and operational simplicity.
</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
- Useful for policy assistants, support knowledge bots, research copilots, and internal documentation Q&A.
- Separates retrieval and generation cleanly, making tuning easier (`chunk_size`, `top_k`, `temperature`).
- Works as a strong baseline for future features like reranking, citations, feedback loops, and evaluation metrics.
            """
        )


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

    current_page = get_current_page()
    endee_available = is_endee_available()
    settings = render_sidebar(endee_available, current_page)

    render_header()
    render_route_nav(current_page)

    if current_page == "home":
        render_landing_page()
        return

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

            latest_user = next(
                (m["content"] for m in reversed(st.session_state.chat_history) if m.get("role") == "user"),
                "",
            )
            latest_answer = next(
                (m["content"] for m in reversed(st.session_state.chat_history) if m.get("role") == "assistant"),
                "",
            )

            if latest_answer:
                screenshot_bytes = build_answer_screenshot(latest_user, latest_answer)
                if screenshot_bytes:
                    st.download_button(
                        "Download Answer Screenshot (PNG)",
                        data=screenshot_bytes,
                        file_name="rag-answer-screenshot.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    st.caption("Screenshot export requires Pillow: `pip install pillow`.")

    render_sources(st.session_state.last_sources)


if __name__ == "__main__":
    main()
