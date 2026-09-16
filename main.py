import os
import json
import time
import hashlib
import asyncio
import logging
import re
from pathlib import Path

from typing import List, Dict, Any
from collections import OrderedDict

import faiss
import httpx
import numpy as np

from datasets import load_dataset
from google import genai

from sentence_transformers import SentenceTransformer
# CONFIG
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash-lite"
DATASET_NAME = "hellotayssir/FinQA_TAT-QA_financial_finetuning_dataset"
DATASET_SPLIT = "train[:800]"

INDEX_PATH = "finance.index"
DOCS_PATH = "finance_docs.json"
EMBED_MODEL_NAME = ("sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K_RETRIEVAL = 8
TOP_K_FINAL = 4
CACHE_LIMIT = 500
MAX_RESPONSE_CHARS = 6000
REQUEST_TIMEOUT = 30
EMBED_BATCH_SIZE = 32
MAX_QUERY_LENGTH = 500
# LOGGING
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("financial-rag")

BASE_DIR = Path(__file__).resolve().parent

# HTTP CLIENT
http_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)

gemini_client = None


def validate_configuration() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")


def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        validate_configuration()
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return gemini_client
# BASIC INPUT FILTERING
async def classify_query_intent(query: str) -> bool:
    """
    Returns True if the query is safe to process.
    Returns False if it should be blocked.
    """
    classification_prompt = f"""You are a content safety classifier for a financial education assistant.

Classify the following user query as SAFE or BLOCKED.

BLOCKED queries include:
- Requests to commit financial fraud or theft
- Requests to hack, exploit, or illegally access financial systems
- Requests for malware, phishing tools, or attack infrastructure
- Requests to launder money or evade financial regulations illegally

SAFE queries include:
- Questions about investing, budgeting, markets, or financial planning
- Questions about how fraud works for educational or protective awareness
- Questions about cybersecurity in finance from a defensive perspective
- Any general financial education question

Reply with ONLY the word SAFE or BLOCKED. Nothing else.

Query: {query}"""

    try:
        response = await asyncio.to_thread(
            get_gemini_client().models.generate_content,
            model=MODEL_NAME,
            contents=classification_prompt,
        )
        result = response.text.strip().upper()
        return result == "SAFE"

    except Exception:
        logger.warning("Intent classification failed — defaulting to safe")
        return True 

# HELPERS
def normalize_text(text: str) -> str:
    return re.sub( r"\s+", " ", text.strip().lower())

def sanitize_text(text: str) -> str:
   return re.sub(r"\s+", " ", text).strip()

def make_cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()

# CHUNKING
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks
     

# RAG SYSTEM
class RAGSystem:
    def __init__(self):
        self.faiss_index = None
        self.documents: List[Dict[str, Any]] = []
        self.embedding_model = None
        self.embedding_lock = asyncio.Lock()
        self.cache: OrderedDict = OrderedDict()

    #  Cache 
    def cache_get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def cache_set(self, key: str, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > CACHE_LIMIT:
            self.cache.popitem(last=False)

    # Models 
    async def get_embedding_model(self):
        async with self.embedding_lock:
            if self.embedding_model is None:
                logger.info("Loading embedding model...")
                self.embedding_model = SentenceTransformer(
                    EMBED_MODEL_NAME, device="cpu"
                )
        return self.embedding_model
    # Embeddings
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        model = await self.get_embedding_model()
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    # Index
    async def build_index_if_needed(self):
        index_path = BASE_DIR / INDEX_PATH
        docs_path = BASE_DIR / DOCS_PATH
        if index_path.exists() and docs_path.exists():
            logger.info("Using existing index")
            return

        logger.info("Building FAISS index...")
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

        processed_docs = []
        seen_hashes = set()
        doc_id = 0

        for row in dataset:
            text = row.get("context") or ""
            if not text:
                continue
            text = sanitize_text(text)
            for chunk in chunk_text(text):
                normalized = normalize_text(chunk)
                if not normalized:
                    continue
                chunk_hash = hashlib.sha256(normalized.encode()).hexdigest()
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)
                processed_docs.append({"doc_id": doc_id, "text": chunk, "source": DATASET_NAME,})
                doc_id += 1

        logger.info(f"Chunks created: {len(processed_docs)}")

        texts = [doc["text"] for doc in processed_docs]
        if not texts:
            raise RuntimeError("The finance dataset did not contain any usable documents")
        embeddings = await self.generate_embeddings(texts)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        logger.info("Adding vectors...")
        index.add(embeddings)
        faiss.write_index(index, str(index_path))

        with docs_path.open("w", encoding="utf-8") as f:
            json.dump(processed_docs, f, ensure_ascii=False)
        logger.info("Index build complete")

    async def initialize(self):
        await self.build_index_if_needed()
        logger.info("Loading index...")
        self.faiss_index = faiss.read_index(str(BASE_DIR / INDEX_PATH))
        with (BASE_DIR / DOCS_PATH).open("r", encoding="utf-8") as f:
            self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} docs")
    
    #  Retrieval 
    async def retrieve(self, query: str) -> List[Dict]:
        query = query[:MAX_QUERY_LENGTH]
        cache_key = make_cache_key(query)
        cached = self.cache_get(cache_key)
        if cached:
            return cached
        query_embedding = await self.generate_embeddings([query])
        scores, indices = await asyncio.to_thread(self.faiss_index.search, query_embedding, TOP_K_RETRIEVAL)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            candidates.append({"score": float(score), "doc": self.documents[idx]})

        final_docs = sorted(candidates, key=lambda x: x["score"], reverse=True)[:TOP_K_FINAL]
        return final_docs
        reranked = sorted(
            [
                {"rerank_score": float(s), "doc": item["doc"]}
                for item, s in zip(candidates, rerank_scores)
            ],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        final_docs = reranked[:TOP_K_FINAL]
        self.cache_set(cache_key, final_docs)
        return final_docs
async def initialize_system() -> RAGSystem:
    rag = RAGSystem()
    await rag.initialize()
    return rag 

# MARKET SUMMARY
async def fetch_market_summary() -> str:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        data = response.json()
        meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose")
        return f"S&P 500 ETF price: {price}. Previous close: {prev}."
    except Exception as e:
        logger.warning(f"Market API failed: {e}")
        return "Market data unavailable."
        
# PROMPTING
SYSTEM_PROMPT = """
You are a financial education assistant.

Rules:
- Never guarantee profits
- Never fabricate financial facts
- Be uncertainty-aware
- Use retrieved context first
- Focus on long-term investing principles
- Be concise and practical
"""

async def build_prompt(query: str, retrieved_docs) -> str:
    market_summary = await fetch_market_summary()
    context_block = "\n\n".join(
        f"[Source {idx}]\n{item['doc']['text']}"
        for idx, item in enumerate(retrieved_docs, start=1)
    )
    return f"""
{SYSTEM_PROMPT}

Market Summary:
{market_summary}

Retrieved Context:
{context_block}

Question:
{query}

Instructions:
- Cite relevant sources
- If uncertain, say uncertainty exists
"""

# STREAMING
async def stream_financial_response(query: str, user: str, rag: RAGSystem):
    is_safe = await classify_query_intent(query)
    if not is_safe:
        yield "This query falls outside the scope of this financial assistant."
        return

    start_time = time.time()

    try:
        retrieved_docs = await rag.retrieve(query)
        prompt = await build_prompt(query=query, retrieved_docs=retrieved_docs)

        response_size = 0

        def run_stream() -> List[str]:
            """Run the blocking stream in a thread, collect all chunks."""
            chunks = []
            for chunk in get_gemini_client().models.generate_content_stream(
                model=MODEL_NAME, contents=prompt
            ):
                text = getattr(chunk, "text", "") or ""
                if text:
                    chunks.append(text)
            return chunks

        chunks = await asyncio.to_thread(run_stream)

        for text in chunks:
            response_size += len(text)

            if response_size >= MAX_RESPONSE_CHARS:
                yield "\n\nResponse truncated."
                break

            yield text
            await asyncio.sleep(0)

        latency = round(time.time() - start_time, 2)
        logger.info(json.dumps({
            "user": user,
            "query": query[:100],
            "latency": latency,
            "retrieved_docs": len(retrieved_docs),
            "response_size": response_size,
        }))

    except httpx.TimeoutException:
        logger.error("Market data request timed out")
        yield "Request timed out. Please try again."

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during market fetch: {e.response.status_code}")
        yield "Temporary service issue."

    except Exception:
        logger.exception("Streaming failed")
        yield "Temporary AI service issue." 

# SHUTDOWN
async def shutdown_system():
    if not http_client.is_closed:
        await http_client.aclose()
        logger.info("HTTP client closed")

# TEST
async def main():
    rag = RAGSystem()
    await rag.initialize()
    query = "How should Nigerians approach investing during inflation?"
    print("\n")
    async for chunk in stream_financial_response(
        query=query, user="demo-user", rag=rag
    ):
        print(chunk, end="", flush=True)
    print("\n")
    await shutdown_system() 
