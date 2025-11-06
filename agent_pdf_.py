import io
import os
import re
import json
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pdfplumber
import httpx

# Optional deps
try:
    import fitz  # PyMuPDF
    _PYMUPDF_OK = True
except Exception:
    _PYMUPDF_OK = False

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False


# =========================
# Configs
# =========================
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MAX_CONTEXT_CHARS = 16000
TOP_K = 6


# =========================
# Ollama client
# =========================
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_HOST):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=180)

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2, json_mode: bool = False) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **({"format": "json"} if json_mode else {}),
            "stream": False,
        }
        r = self.client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    def embed(self, model: str, texts: List[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            r = self.client.post(f"{self.base_url}/api/embeddings", json={"model": model, "prompt": t})
            r.raise_for_status()
            vec = r.json().get("embedding", [])
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


# =========================
# PDF utilities
# =========================
@dataclass
class PageChunk:
    page_num: int
    text: str


HEADER_FOOTER_RATIO = 0.07  # ignore top/bot 7%


def _clean_text(txt: str) -> str:
    txt = re.sub(r"\\(textbf|emph|cite|ref|eqref|label|mathbf|mathrm|begin|end)\{[^}]*\}", " ", txt)
    txt = re.sub(r"\$[^$]*\$", " ", txt)
    txt = re.sub(r"\\\[[\\s\\S]*?\\\]", " ", txt)
    txt = re.sub(r"\\\([^)]*\\\)", " ", txt)
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[\t\x0b\x0c\r]", " ", txt)
    txt = re.sub(r" {2,}", " ", txt)
    return txt.strip()


def _extract_with_pymupdf(file_bytes: bytes) -> List[PageChunk]:
    if not _PYMUPDF_OK:
        return []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: List[PageChunk] = []
    for i in range(len(doc)):
        page = doc[i]
        blocks = page.get_text("blocks")
        height = page.rect.height
        y_top = HEADER_FOOTER_RATIO * height
        y_bot = (1 - HEADER_FOOTER_RATIO) * height
        filtered = [b for b in blocks if b[1] >= y_top and b[3] <= y_bot]
        filtered.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        text = "\n".join([b[4] for b in filtered if isinstance(b[4], str)])
        text = _clean_text(text)
        if text:
            pages.append(PageChunk(page_num=i + 1, text=text))
    return pages


def _extract_with_pdfplumber(file_bytes: bytes) -> List[PageChunk]:
    pages: List[PageChunk] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            width = page.width or 1
            height = page.height or 1
            y_top = HEADER_FOOTER_RATIO * height
            y_bot = (1 - HEADER_FOOTER_RATIO) * height
            crop = page.within_bbox((0, y_top, width, y_bot))
            txt = crop.extract_text(layout=True) or ""
            txt = _clean_text(txt)
            if txt:
                pages.append(PageChunk(page_num=i, text=txt))
    return pages


def pdf_to_pages(file_bytes: bytes) -> List[PageChunk]:
    pages = _extract_with_pymupdf(file_bytes)
    if not pages:
        pages = _extract_with_pdfplumber(file_bytes)
    return pages


# =========================
# Hybrid Figure Extraction
# =========================
def extract_figures_or_pages(file_bytes: bytes, output_dir: str = "figures") -> List[str]:
    """Extract bitmap figures, fallback to rendering full pages as images."""
    if not _PYMUPDF_OK:
        return []
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    figures = []

    # Try extracting embedded images first
    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for j, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:
                fig_path = os.path.join(output_dir, f"fig_{i+1}_{j+1}.png")
                pix.save(fig_path)
                figures.append(fig_path)
            pix = None

    # If no embedded images, render pages
    if not figures:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            fig_path = os.path.join(output_dir, f"page_{i+1}.png")
            pix.save(fig_path)
            figures.append(fig_path)

    return figures


# =========================
# Indexing & Retrieval (batched embeddings)
# =========================
def build_index(chunks: List[str], embedder: OllamaClient, embed_model: str, batch_size: int = 6):
    """Embed text chunks in small batches to avoid large HTTP payloads."""
    all_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            vecs = embedder.embed(embed_model, batch)
            all_vecs.append(vecs)
        except Exception as e:
            print(f"Embedding batch {i // batch_size + 1} failed: {e}")
    if not all_vecs:
        return ("none", None, None)
    mat = np.vstack(all_vecs)
    if _FAISS_OK and len(chunks) > 2 and mat.size > 0:
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(mat)
        index.add(mat)
        return ("faiss", index, mat)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat_norm = mat / norms
    return ("numpy", mat_norm, mat)


def search_index(query: str, embedder: OllamaClient, embed_model: str, index_tuple, chunks: List[str], top_k: int = TOP_K) -> List[int]:
    backend, store, mat = index_tuple
    if backend == "none" or store is None or mat is None:
        return list(range(min(top_k, len(chunks))))
    q = embedder.embed(embed_model, [query])[0]
    q = q / (np.linalg.norm(q) + 1e-12)
    if backend == "faiss":
        D, I = store.search(np.array([q], dtype=np.float32), top_k)
        return [int(i) for i in I[0] if 0 <= int(i) < len(chunks)]
    sims = (store @ q.reshape(-1, 1)).ravel()
    idxs = np.argsort(-sims)[:top_k].tolist()
    return idxs


# =========================
# Web search
# =========================
def web_search(query: str, max_results: int = 3) -> str:
    try:
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for g in soup.select("div.tF2Cxc"):
            title = g.select_one("h3")
            snippet = g.select_one(".VwiC3b")
            if title and snippet:
                results.append(f"{title.text}\n{snippet.text}")
            if len(results) >= max_results:
                break
        return "\n\n".join(results) if results else "Nenhum resultado encontrado."
    except Exception as e:
        return f"Erro na busca web: {e}"


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Agente PDF (Texto + Figuras + Web + Memória)", layout="wide")
st.title("Agente para leitura de artigos do arXiv")

chat_model = st.sidebar.text_input("Modelo (Ollama)", value=DEFAULT_CHAT_MODEL)
embed_model = st.sidebar.text_input("Modelo de embeddings", value=DEFAULT_EMBED_MODEL)

persona = st.sidebar.selectbox(
    "Personalidade do agente",
    ["Consultor Jurídico", "Professor Universitário", "Cientista", "Advogado Corporativo", "Analista Técnico", "Personalidade Customizada"]
)

custom_persona = ""
if persona == "Personalidade Customizada":
    custom_persona = st.sidebar.text_area("Descreva a personalidade", placeholder="Ex: Um cosmologo quebrando a cabeca para calcular o melhor valor de H0.")

if 'memory_context' not in st.session_state:
    st.session_state.memory_context = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

PERSONA_STYLES = {
    "Consultor Jurídico": "Você é um consultor jurídico experiente e técnico.",
    "Professor Universitário": "Você é um professor didático e detalhista.",
    "Cientista": "Você é um cientista lógico e analítico.",
    "Advogado Corporativo": "Você é um advogado corporativo, objetivo e estratégico.",
    "Analista Técnico": "Você é um analista técnico, claro e sistemático."
}

uploaded_pdf = st.file_uploader("PDF (contrato ou artigo)", type=["pdf"], accept_multiple_files=False)

if uploaded_pdf:
    client = OllamaClient()
    pdf_bytes = uploaded_pdf.read()

    with st.spinner("Extraindo texto e figuras do PDF..."):
        pages = pdf_to_pages(pdf_bytes)
        figures = extract_figures_or_pages(pdf_bytes)
        if not pages:
            st.error("Nenhum texto extraído.")
            st.stop()

    chunks = [p.text[:15000] for p in pages]
    with st.spinner("Gerando embeddings em lotes..."):
        index_tuple = build_index(chunks, client, embed_model, batch_size=6)

    st.success(f"PDF processado ({len(chunks)} blocos de texto e {len(figures)} figuras/páginas renderizadas).")

    if figures:
        st.subheader("🖼️ Figuras ou páginas renderizadas")
        for f in figures:
            st.image(f, use_column_width=True)
    else:
        st.info("Nenhuma figura detectada ou renderizada.")

    st.subheader("💬 Chat com o PDF (e Web opcional)")
    for m in st.session_state.chat_history:
        with st.chat_message(m['role']):
            st.write(m['content'])

    user_msg = st.chat_input("Pergunte sobre o PDF; use 'pesquise ...' para buscar na web.")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            if user_msg.lower().startswith("pesquise ") or "busque na web" in user_msg.lower():
                query = user_msg.replace("pesquise", "").replace("busque na web", "").strip()
                reply = web_search(query)
            else:
                idxs = search_index(user_msg, client, embed_model, index_tuple, chunks, top_k=TOP_K)
                selected = [chunks[i] for i in idxs]
                context_text = "\n\n".join(selected)[:MAX_CONTEXT_CHARS]
                persona_desc = custom_persona if persona == "Personalidade Customizada" else PERSONA_STYLES.get(persona, "")
                system = (
                    f"{persona_desc} Responda apenas com base no texto fornecido. "
                    "Cite páginas se possível. Responda em português."
                )
                user = f"TRECHOS RELEVANTES:\n{context_text}\n\nPERGUNTA:{user_msg}"
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                try:
                    reply = client.chat(chat_model, msgs, temperature=0.2, json_mode=False)
                    st.session_state.memory_context += f"Usuário: {user_msg}\nAssistente: {reply}\n"
                except Exception as e:
                    reply = f"Erro ao consultar modelo: {e}"
            st.write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    st.markdown("---")
    st.text_area("🧠 Chat Memory", value=st.session_state.memory_context, height=250)
else:
    st.info("Envie um PDF para iniciar a análise e o chat.")
