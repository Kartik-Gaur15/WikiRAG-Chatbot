import os
import re
import time
import logging
import hashlib
import numpy as np
from typing import List, Dict, Optional

import wikipedia
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wikipedia.set_user_agent(
    "WikiRAG-Chatbot/1.0 (https://github.com/kartikgaur; contact: kartik.gaur@shorthills.ai)"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6
MAX_CONTEXT_CHARS = 4000
GROQ_MODEL = "llama-3.1-8b-instant"

BAD_IMAGE_HINTS = (
    "commons-logo", "edit-icon", "question_book", "ambox", "padlock",
    "disambig", "loudspeaker", "stub", "wikimedia-logo", "poweredby",
    "protect-shackle", "anatomy", "diagram", "skeleton", "muscle",
    "medical", "flag", "coat_of_arms", "symbol", "icon", "logo",
    "seal", "stamp", "currency", "badge", "sign", "silhouette",
    "placeholder", "default", "unknown", "blank",
)

GOOD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def simple_embed(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    words = re.findall(r"[a-z]+", text.lower())
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def extract_topic_with_groq(question: str, groq_client, chat_history: list = None) -> str:
    history_str = ""
    if chat_history:
        history_str = "\n".join(
            f"{m.get('role','user').capitalize()}: {m.get('content','')}"
            for m in chat_history[-6:]
        )
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given the chat history and current question, extract the main Wikipedia search topic. "
                        "Resolve pronouns (his, her, their, it, he, she, they) using the chat history. "
                        "Return ONLY the topic name, nothing else. Be specific with product names.\n\n"
                        "Examples:\n"
                        "Q: iphone 14 pro → iPhone 14 Pro\n"
                        "Q: iphone 16 pro → iPhone 16 Pro\n"
                        "Q: who was babur → Babur\n"
                        "Q: What is the capital of Japan? → Japan\n"
                        "Q: Who is Lewis Hamilton? → Lewis Hamilton\n"
                        "Q: What are MS Dhoni stats? → MS Dhoni\n"
                        "History: 'User: who was babur / Assistant: Babur was...'\n"
                        "Q: who was his son → Humayun\n"
                        "Q: who was his sister → Khanzada Begum"
                    )
                },
                {
                    "role": "user",
                    "content": f"Chat history:\n{history_str}\n\nQuestion: {question}"
                }
            ],
            temperature=0,
            max_tokens=30,
        )
        topic = response.choices[0].message.content.strip()
        logger.info(f"Extracted topic: '{topic}' from: '{question}'")
        return topic
    except Exception as e:
        logger.warning(f"Topic extraction failed: {e}")
        return question


class WikipediaRAG:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.groq = Groq(api_key=api_key)
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

    @staticmethod
    def _extract_stats_sections(content: str) -> str:
        stat_keywords = [
            "statistics", "career statistics", "records", "achievements",
            "championships", "world championship", "season results",
            "batting", "bowling", "formula one career", "race wins",
            "pole positions", "fastest laps", "grand prix", "innings",
            "centuries", "half-centuries", "wickets", "averages"
        ]
        lines = content.split("\n")
        stat_sections = []
        capture = False
        buffer = []
        for line in lines:
            lower = line.lower().strip()
            if any(kw in lower for kw in stat_keywords):
                capture = True
                buffer = [line]
            elif capture:
                if line.strip() == "" and len(buffer) > 5:
                    stat_sections.append("\n".join(buffer))
                    buffer = []
                    capture = False
                else:
                    buffer.append(line)
        if buffer:
            stat_sections.append("\n".join(buffer))
        return "\n\n".join(stat_sections)

    def _get_or_build_index(self, page):
        key = page.title.lower()
        if key in self._store:
            logger.info(f"Using cached index for '{page.title}'")
            return self._store[key]
        logger.info(f"Building index for '{page.title}'")
        content = page.content
        chunks = self._chunk_text(content) or [page.summary]
        stats = self._extract_stats_sections(content)
        if stats:
            stat_chunks = self._chunk_text(stats)
            chunks = stat_chunks + chunks
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

    def _pick_relevant_image(self, page, topic: str) -> Optional[str]:
        try:
            images = page.images or []
        except Exception as e:
            logger.warning(f"Could not fetch images: {e}")
            return None

        # Step 1: extension + blocklist filter
        candidates = [
            url for url in images
            if any(url.lower().endswith(ext) for ext in GOOD_EXTENSIONS)
            and not any(bad in url.lower() for bad in BAD_IMAGE_HINTS)
        ]
        if not candidates:
            return None

        # Step 2: score by topic word match in filename
        topic_words = {w for w in re.findall(r"[a-z0-9]+", topic.lower()) if len(w) > 2}

        scored = []
        for url in candidates:
            filename = url.lower().rsplit("/", 1)[-1]
            score = sum(1 for w in topic_words if w in filename)
            scored.append((score, url))

        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_url = scored[0]

        # Step 3: if topic word matches, return it
        if best_score > 0:
            logger.info(f"Image selected (score={best_score}): {best_url}")
            return best_url

        # Step 4: fallback — return first clean candidate
        # (better to show something relevant than nothing)
        logger.info(f"No topic match in filename, using first clean candidate: {candidates[0]}")
        return candidates[0]

    def _resolve_page(self, topic: str):
        """
        FIX: Search Wikipedia with the exact topic first,
        then fall back to auto_suggest only if exact match fails.
        This prevents 'iPhone 14 Pro' from landing on 'iPhone 12 Pro'.
        """
        last_err = None

        # Try exact search first (most reliable for product names)
        results = self._retry(lambda: wikipedia.search(topic, results=8))
        if results:
            # Find exact or closest title match
            topic_lower = topic.lower()
            exact = [t for t in results if t.lower() == topic_lower]
            close = [t for t in results if topic_lower in t.lower()]
            ordered = exact + [t for t in close if t not in exact] + \
                      [t for t in results if t not in exact and t not in close]

            for title in ordered:
                try:
                    return self._retry(
                        lambda t=title: wikipedia.page(t, auto_suggest=False, redirect=True)
                    )
                except wikipedia.DisambiguationError as e:
                    if e.options:
                        # Pick option closest to topic
                        best = next(
                            (o for o in e.options if topic_lower in o.lower()),
                            e.options[0]
                        )
                        try:
                            return self._retry(
                                lambda o=best: wikipedia.page(o, auto_suggest=False, redirect=True)
                            )
                        except Exception as inner:
                            last_err = inner
                            continue
                except wikipedia.PageError as e:
                    last_err = e
                    continue

        # Last resort: auto_suggest
        try:
            return self._retry(
                lambda: wikipedia.page(topic, auto_suggest=True, redirect=True)
            )
        except Exception as e:
            last_err = e

        raise last_err or wikipedia.PageError(topic)

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
            "You are a knowledgeable assistant with access to Wikipedia data. "
            "Answer the question using the context provided. "
            "For statistical questions, extract and present relevant numbers, years, and records clearly. "
            "If the context does not contain enough information, say so honestly."
        )
        user_msg = (
            f"Chat history:\n{history_str}\n\n"
            f"Wikipedia context:\n{context}\n\n"
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
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    def query(self, question: str, chat_history=None):
        has_pronouns = bool(re.search(
            r'\b(his|her|their|its|he|she|they|him|them)\b', question.lower()
        ))
        if not chat_history and not has_pronouns:
            cache_key = question.strip().lower()
            if cache_key in self._answer_cache:
                logger.info(f"Cache hit: {question}")
                return self._answer_cache[cache_key]
        else:
            cache_key = None

        try:
            logger.info(f"Query: {question}")
            topic = extract_topic_with_groq(question, self.groq, chat_history)
            page = self._resolve_page(topic)
            logger.info(f"Resolved to page: '{page.title}'")

            index = self._get_or_build_index(page)
            chunks = self._retrieve(index, question)

            context = "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]
            if not context.strip():
                context = page.summary[:MAX_CONTEXT_CHARS]

            answer = self._ask_groq(context, question, self._format_history(chat_history))
            image_url = self._pick_relevant_image(page, topic)

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
