import os
import re
import time
import logging
import hashlib

import wikipedia
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wikipedia.set_user_agent(
    "WikiRAG-Chatbot/1.0 (https://github.com/kartikgaur; contact: kartik.gaur@shorthills.ai)"
)

CHROMA_DIR   = os.getenv("CHROMA_DIR", "/tmp/chroma_store")
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 100
TOP_K        = 4
MAX_CONTEXT_CHARS = 3000
GROQ_MODEL   = "llama3-8b-8192"

BAD_IMAGE_HINTS = (
    "commons-logo", "edit-icon", "question_book", "ambox",
    "padlock", "disambig", "loudspeaker", "stub",
    "wikimedia-logo", "poweredby", "protect-shackle",
)
GOOD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


class WikipediaRAG:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.groq = Groq(api_key=api_key)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._answer_cache = {}

    @staticmethod
    def _collection_name(title: str) -> str:
        h = hashlib.md5(title.lower().encode()).hexdigest()[:16]
        return f"wiki_{h}"

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

    def _get_or_build_collection(self, page):
        name = self._collection_name(page.title)
        collection = self.chroma_client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
        )
        if collection.count() > 0:
            logger.info(f"Using cached collection for '{page.title}'")
            return collection

        logger.info(f"Building new collection for '{page.title}'")
        chunks = self._chunk_text(page.content) or [page.summary]
        collection.add(
            documents=chunks,
            ids=[f"{name}_{i}" for i in range(len(chunks))],
        )
        logger.info(f"Stored {len(chunks)} chunks for '{page.title}'")
        return collection

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
        lines = []
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

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
            "Answer concisely and accurately based on the context above:"
        )
        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
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
            page       = self._resolve_page(question)
            collection = self._get_or_build_collection(page)

            retrieved        = collection.query(query_texts=[question], n_results=TOP_K)
            retrieved_chunks = retrieved.get("documents", [[]])[0]

            context = "\n\n".join(retrieved_chunks)[:MAX_CONTEXT_CHARS]
            if not context.strip():
                context = page.summary[:MAX_CONTEXT_CHARS]

            history_str = self._format_history(chat_history)
            answer      = self._ask_groq(context, question, history_str)
            image_url   = self._pick_relevant_image(page, question)

            result = {
                "answer":  answer,
                "sources": [page.url],
                "image":   image_url,
            }
            if cache_key:
                self._answer_cache[cache_key] = result
            return result

        except wikipedia.DisambiguationError as e:
            return {
                "answer":  f"'{question}' is ambiguous. Did you mean: {', '.join(e.options[:5])}?",
                "sources": [],
                "image":   None,
            }
        except wikipedia.PageError:
            return {
                "answer":  f"Sorry, I couldn't find a Wikipedia page for '{question}'.",
                "sources": [],
                "image":   None,
            }
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return {
                "answer":  "Sorry, something went wrong. Please try again.",
                "sources": [],
                "image":   None,
            }
