import logging
import os
import re
import requests
import chromadb
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "WikiRAG-Chatbot/1.0 (educational project; contact: kartikgaur0090@gmail.com)"
}

GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|yo|hii|hiii|heya|sup|good\s*(morning|afternoon|evening|night)|"
    r"how\s*are\s*you|what'?s\s*up|who\s*are\s*you|thanks|thank\s*you|bye|goodbye)\s*[!.?]*\s*$",
    re.IGNORECASE
)

GREETING_REPLIES = {
    "default": "Hey! I'm WikiRAG — ask me anything and I'll pull the answer straight from Wikipedia.",
    "who": "I'm WikiRAG, a chatbot that answers your questions using live Wikipedia content. Ask me about anything — people, places, events, stats.",
    "thanks": "You're welcome! Let me know if you have more questions.",
    "bye": "Goodbye! Come back anytime you want to look something up.",
}

MAX_HISTORY_TURNS = 6  # last N messages to keep for context


class WikipediaRAG:
    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.llm = ChatOllama(model="llama3.2", temperature=0.2)

        self.topic_prompt = ChatPromptTemplate.from_template(
            """Given the conversation so far and the new question, figure out what Wikipedia
topic the new question is REALLY about. If the question uses pronouns (he, she, it, they)
or refers to "that" / "this", resolve them using the conversation history.
Reply with ONLY the topic name, nothing else.

Conversation so far:
{history}

New question: "{question}"
Topic:"""
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            """Answer the new question directly and concisely, using the Wikipedia context
below and the conversation history for tone/continuity.

Rules:
- Get straight to the point. No preamble like "Based on the context" or "According to Wikipedia".
- Use 1-3 sentences for simple factual questions. Use more only if the question truly needs it (stats, lists).
- Sound like a natural reply in an ongoing conversation, not an isolated lookup.
- Never repeat the question back. Never pad with filler.
- If the context doesn't answer it, say so briefly.

Conversation so far:
{history}

Wikipedia context:
{context}

New question: {question}

Answer:"""
        )

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="wiki_chunks")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

    def _slugify(self, text: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

    def _format_history(self, chat_history: list) -> str:
        if not chat_history:
            return "(no prior messages)"
        recent = chat_history[-MAX_HISTORY_TURNS:]
        lines = []
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            label = "User" if role == "user" else "WikiRAG"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def _check_greeting(self, question: str):
        q = question.strip().lower()
        if not GREETING_PATTERNS.match(q):
            return None
        if "who" in q:
            return GREETING_REPLIES["who"]
        if "thank" in q:
            return GREETING_REPLIES["thanks"]
        if "bye" in q:
            return GREETING_REPLIES["bye"]
        return GREETING_REPLIES["default"]

    def _extract_topic(self, question: str, history_text: str) -> str:
        try:
            chain = self.topic_prompt | self.llm | StrOutputParser()
            topic = chain.invoke({"question": question, "history": history_text})
            topic = topic.strip().strip('"').strip("'").strip('.')
            topic = topic.split("\n")[0].strip()
            logging.info(f"Extracted topic: '{topic}' from question: '{question}'")
            return topic if topic else question
        except Exception as e:
            logging.warning(f"Topic extraction failed, using raw question: {e}")
            return question

    def _raw_search(self, topic: str):
        try:
            resp = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": topic,
                    "format": "json",
                    "srlimit": 3,
                },
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            if not results:
                return None
            return results[0]["title"]
        except Exception as e:
            logging.error(f"Wikipedia search failed: {e}")
            return None

    def _search_wikipedia(self, topic: str):
        title = self._raw_search(topic)
        if title:
            return title

        try:
            resp = requests.get(
                WIKI_API,
                params={
                    "action": "opensearch",
                    "search": topic,
                    "limit": 3,
                    "namespace": 0,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            titles = data[1] if len(data) > 1 else []
            if titles:
                logging.info(f"Fuzzy match found: '{titles[0]}' for '{topic}'")
                return titles[0]
        except Exception as e:
            logging.warning(f"Opensearch fallback failed: {e}")

        return None

    def _fetch_page_content(self, title: str):
        try:
            resp = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "prop": "extracts|info",
                    "exintro": False,
                    "explaintext": True,
                    "inprop": "url",
                    "titles": title,
                    "format": "json",
                    "redirects": 1,
                },
                headers=HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            page = next(iter(pages.values()))
            if "missing" in page:
                return None

            return {
                "title": page.get("title", title),
                "content": page.get("extract", ""),
                "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
            }
        except Exception as e:
            logging.error(f"Wikipedia page fetch failed: {e}")
            return None

    def _ingest_page(self, title: str):
        page_id = self._slugify(title)

        existing = self.collection.get(where={"page_id": page_id}, limit=1)
        if existing["ids"]:
            logging.info(f"Cache hit for page: {title}")
            return title

        logging.info(f"Cache miss, fetching from Wikipedia: {title}")
        page = self._fetch_page_content(title)
        if not page or not page["content"]:
            return None

        chunks = self.splitter.split_text(page["content"])
        if not chunks:
            return None

        ids = [f"{page_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"page_id": page_id, "title": page["title"], "url": page["url"]} for _ in chunks]

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        logging.info(f"Cached {len(chunks)} chunks for '{page['title']}'")
        return page["title"]

    def query(self, question: str, chat_history: list = None) -> dict:
        try:
            logging.info(f"Received query: {question}")

            greeting_reply = self._check_greeting(question)
            if greeting_reply:
                logging.info("Detected greeting/small talk — skipping Wikipedia.")
                return {"answer": greeting_reply, "sources": []}

            history_text = self._format_history(chat_history or [])

            topic = self._extract_topic(question, history_text)

            title = self._search_wikipedia(topic)
            if not title:
                return {
                    "answer": f"Sorry, I couldn't find a Wikipedia page for '{topic}'.",
                    "sources": []
                }

            resolved_title = self._ingest_page(title)
            if not resolved_title:
                return {
                    "answer": f"Sorry, I couldn't load the Wikipedia page for '{title}'.",
                    "sources": []
                }

            page_id = self._slugify(resolved_title)
            results = self.collection.query(
                query_texts=[question],
                n_results=6,
                where={"page_id": page_id},
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            if not docs:
                return {
                    "answer": "Sorry, I found the page but couldn't retrieve relevant content.",
                    "sources": []
                }

            context = "\n\n".join(docs)
            chain = self.answer_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question,
                "history": history_text
            })

            source_url = metas[0]["url"] if metas else None

            return {
                "answer": answer.strip(),
                "sources": [source_url] if source_url else [f"Wikipedia: {resolved_title}"]
            }

        except Exception as e:
            logging.error(f"Error in query: {str(e)}")
            return {
                "answer": f"Sorry, I had trouble answering '{question}'. Please try again.",
                "sources": []
            }
