import os
import re
import time
import logging
import hashlib
import numpy as np
from typing import List, Dict

import wikipedia
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wikipedia.set_user_agent(
    "WikiRAG-Chatbot/1.0 (https://github.com/kartikgaur; contact: kartik.gaur@shorthills.ai)"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
MAX_CONTEXT_CHARS = 3000
GROQ_MODEL = "llama-3.1-8b-instant"

BAD_IMAGE_HINTS = (
    "commons-logo", "edit-icon", "question_book", "ambox",
    "padlock", "disambig", "loudspeaker", "stub",
    "wikimedia-logo", "poweredby", "protect-shackle",
)
GOOD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def simple_embed(text: str, dim: int = 512) -> np.ndarray:
    """Lightweight TF-IDF-style bag-of-words embedding. Zero dependencies."""
    vec = np.zeros(dim, dtype=np.float32)
    words = re.findall(r"[a-z]+", text.lower())
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class WikipediaRAG:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.groq = Groq(api_key=api_key)
        # In-memory store: title -> {"chunks": [...], "embeddings": [...]}
        self._store: Dict[str, dict] = {}
        self._answer_cache: Dict[str, dict] = {}

    @staticmethod
    def _chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        chunks, start = [], 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            start = end - overlap
        return chunks

    def _get_or_build_index(self, page):
        key = page.title.lower()
        if key in self._store:
            logger.info(f"Using cached index for '{page.title}'")
            return self._store[key]

        logger.info(f"Building index for '{page.title}'")
        chunks = self._chunk_text(page.content) or [page.summary]
        embeddings = [simple_embed(c) for c in chunks]
        self._store[key] = {"chunks": chunks, "embeddings": embeddings}
        logger.info(f"Indexed {len(chunks)} chunks for '{page.title}'")
        return self._store[key]

    def _retrieve(self, index: dict, question: str, top_k=TOP_K) -> List[str]:
        q_vec = simple_embed(question)
        scored = [
            (cosine_similarity(q_vec, emb), chunk)
            for emb, chunk in zip(index["embeddings"], index["chunks"])
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored[:top_k]]

    def _pick_relevant_image(self, page, query: str):
        try:
            images = page.images or []
        except Exception as e:
            logger.warning(f"Could not fetch images: {e}")
            return None

        candidates = [
            url for url in images
            if any(url.lower().endswith(ext) for ext in GOOD_EXTENSIONS)
            and not any(bad in url.lower() for bad in BAD_IMAGE_HINTS)
        ]
        if not candidates:
            return None

        query_words = set(re.findall(r"[a-z]+", query.lower()))
        title_words = set(re.findall(r"[a-z]+", page.title.lower()))
        relevant_words = {w for w in (query_words | title_words) if len(w) > 2}

        best_url, best_score = candidates[0], -1
        for url in candidates:
            filename = url.lower().rsplit("/", 1)[-1]
            score = sum(1 for w in relevant_words if w in filename)
            if score > best_score:
                best_score, best_url = score, url

        logger.info(f"Selected image: {best_url}")
        return best_url

    def _resolve_page(self, question: str):
        results = self._retry(lambda: wikipedia.search(question, results=5))
        if not results:
            raise wikipedia.PageError(question)
        last_err = None
        for title in results:
            try:
                return self._retry(
                    lambda t=title: wikipedia.page(t, auto_suggest=False, redirect=True)
                )
            except wikipedia.DisambiguationError as e:
                if e.options:
                    try:
                        return self._retry(
                            lambda o=e.options[0]: wikipedia.page(o, auto_suggest=False, redirect=True)
                        )
                    except Exception as inner:
                        last_err = inner
                        continue
            except wikipedia.PageError as e:
                last_err = e
                continue
        raise last_err or wikipedia.PageError(question)

    @staticmethod
    def _retry(fn, attempts=3, delay=0.6):
        last_exc = None
        for i in range(attempts):
            try:
                return fn()
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                raise
            except Exception as e:
                last_exc = e
                logger.warning(f"Wikipedia API hiccup (attempt {i+1}/{attempts}): {e}")
                time.sleep(delay * (i + 1))
        raise last_exc

    @staticmethod
    def _format_history(chat_history: list) -> str:
        if not chat_history:
            return "No previous conversation."
        return "\n".join(
            f"{m.get('role','user').capitalize()}: {m.get('content','')}"
            for m in chat_history
        )

    def _ask_groq(self, context: str, question: str, history_str: str) -> str:
        system = (
            "You are a helpful assistant. Answer the question using ONLY "
            "the Wikipedia context provided. If the context does not contain "
            "the answer, say you don't have enough information rather than guessing."
        )
        user_msg = (
            f"Chat history:\n{history_str}\n\n"
            f"Context from Wikipedia:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and accurately:"
        )
        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def query(self, question: str, chat_history=None):
        if not chat_history:
            cache_key = question.strip().lower()
            if cache_key in self._answer_cache:
                logger.info(f"Cache hit: {question}")
                return self._answer_cache[cache_key]
        else:
            cache_key = None

        try:
            logger.info(f"Query: {question}")
            page = self._resolve_page(question)
            index = self._get_or_build_index(page)
            chunks = self._retrieve(index, question)

            context = "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]
            if not context.strip():
                context = page.summary[:MAX_CONTEXT_CHARS]

            answer = self._ask_groq(context, question, self._format_history(chat_history))
            image_url = self._pick_relevant_image(page, question)

            result = {"answer": answer, "sources": [page.url], "image": image_url}
            if cache_key:
                self._answer_cache[cache_key] = result
            return result

        except wikipedia.DisambiguationError as e:
            return {
                "answer": f"'{question}' is ambiguous. Did you mean: {', '.join(e.options[:5])}?",
                "sources": [], "image": None,
            }
        except wikipedia.PageError:
            return {
                "answer": f"Sorry, I couldn't find a Wikipedia page for '{question}'.",
                "sources": [], "image": None,
            }
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return {
                "answer": "Sorry, something went wrong. Please try again.",
                "sources": [], "image": None,
            }
