"""
OpenAI-compatible API server for the Crawl4AI RAG pipeline.
"""
import os
import sys
import re
import json
import time
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional, Any
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_supabase_client, search_documents,
    add_documents_to_supabase, update_source_info, extract_source_summary
)

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
RAG_MODEL_ID = "crawl4ai-rag"

app = FastAPI(title="Crawl4AI RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ──

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


# ── Crawl helpers ──

URL_PATTERN = re.compile(r'https?://[^\s]+')

def extract_urls(text: str) -> List[str]:
    return URL_PATTERN.findall(text)

def is_crawl_intent(text: str) -> bool:
    keywords = ["crawl", "index", "scrape", "fetch", "store", "save", "add to knowledge"]
    text_lower = text.lower()
    return any(k in text_lower for k in keywords) and bool(extract_urls(text))

def extract_max_pages(text: str) -> int:
    """Extract max_pages value from message if present, otherwise use default."""
    match = re.search(r'max_pages=(\d+)', text)
    if match:
        return min(int(match.group(1)), 100)  # cap at 100
    return MAX_PAGES_DEFAULT

MAX_PAGES_DEFAULT = 20

SKIP_URL_PATTERNS = re.compile(
    r'(/page/\d+/?$)'           # pagination: /page/2/, /page/3/
    r'|(/tag/)'                 # tag listing pages
    r'|(/topic/)'               # topic listing pages
    r'|(/author/)'              # author listing pages
    r'|(/category/[^/]+/?$)'   # category index pages
    r'|(/section/[^/]+/?$)'    # section index pages (first level only)
    r'|(/(login|signup|register|account|cart|checkout|privacy|terms|contact|about|search|feed|rss)/?$)'
)

def should_skip_url(url: str) -> bool:
    return bool(SKIP_URL_PATTERNS.search(urlparse(url).path))


def url_priority(url: str) -> int:
    """Lower = crawled first. Articles before listings before nav."""
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    depth = len(segments)

    has_date = bool(re.search(r'/\d{4}/\d{2}/\d{2}/', path))
    # Slug-like endings: long hyphenated strings typical of article URLs
    last_seg = segments[-1] if segments else ""
    looks_like_article = len(last_seg) > 20 and last_seg.count("-") >= 2

    if has_date:
        return 1                    # highest priority
    if looks_like_article:
        return 5
    if depth >= 3:
        return 20
    if depth == 2:
        return 40
    return 80                       # shallow nav pages last


def extract_internal_links(result, base_url: str) -> List[tuple[str, str]]:
    """
    Extract internal links from a crawl result.
    Returns list of (url, link_text) tuples, filtered to same domain.
    """
    base_domain = urlparse(base_url).netloc
    seen = set()
    links = []

    if not (hasattr(result, "links") and result.links):
        return links

    for item in result.links.get("internal", []):
        href = item.get("href", "") if isinstance(item, dict) else str(item)
        text = (item.get("text", "") or "").strip() if isinstance(item, dict) else ""

        if not href or not href.startswith("http"):
            continue
        parsed = urlparse(href)
        clean = parsed._replace(fragment="", query="").geturl()
        if parsed.netloc == base_domain and clean not in seen:
            seen.add(clean)
            links.append((clean, text))

    return links


async def crawl_single_page(crawler, url: str) -> tuple[str, list[tuple[str, str]]]:
    """Crawl one page, return (markdown_content, [(url, link_text), ...])."""
    from crawl4ai import CrawlerRunConfig, CacheMode
    result = await crawler.arun(url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS))
    if not result.success:
        print(f"[crawl] Failed: {url} — {result.error_message}")
        return "", []
    content = result.markdown.raw_markdown if result.markdown else ""
    links = extract_internal_links(result, url)
    return content, links


async def crawl_and_store(url: str, max_pages: int = MAX_PAGES_DEFAULT) -> str:
    """Crawl a URL and its subpages, store everything in Supabase."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        supabase = get_supabase_client()
        source_id = urlparse(url).netloc

        visited: set[str] = set()
        # Two-section queue:
        #   homepage_queue — links found on the root page, kept in page order (top story first)
        #   deep_queue     — links found on subpages, sorted by priority
        homepage_queue: list[tuple[str, str]] = [(url, "")]
        deep_queue: list[tuple[str, str]] = []
        all_urls: list[str] = []
        all_chunks: list[str] = []
        all_chunk_numbers: list[int] = []
        all_metadatas: list[dict] = []
        url_to_full_doc: dict[str, str] = {}
        total_words = 0
        first_content = ""
        homepage_stories: list[tuple[str, str]] = []
        is_root = True

        def next_url() -> tuple[str, str] | None:
            """Always drain homepage queue first (page order), then deep queue."""
            if homepage_queue:
                return homepage_queue.pop(0)
            if deep_queue:
                return deep_queue.pop(0)
            return None

        def queued_urls() -> set[str]:
            return {u for u, _ in homepage_queue} | {u for u, _ in deep_queue}

        print(f"[crawl] Starting deep crawl of {url} (max {max_pages} pages)")

        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
            while len(visited) < max_pages:
                item = next_url()
                if item is None:
                    break
                current_url, _ = item
                if current_url in visited:
                    continue
                visited.add(current_url)

                print(f"[crawl] ({len(visited)}/{max_pages}) {current_url}")
                content, links = await crawl_single_page(crawler, current_url)

                # On the root page: article links (real headlines) go into homepage_queue
                # IN PAGE ORDER. Non-article links (short text, nav) go to deep_queue.
                if is_root:
                    is_root = False
                    for link_url, link_text in links:
                        if should_skip_url(link_url) or link_url in visited:
                            continue
                        clean_text = link_text.strip()
                        # A real article headline: long text with spaces (not a nav label)
                        is_article_link = len(clean_text) > 25 and " " in clean_text
                        if is_article_link:
                            homepage_queue.append((link_url, clean_text))
                            homepage_stories.append((link_url, clean_text))
                        else:
                            deep_queue.append((link_url, clean_text))

                # Skip low-content pages
                min_chars = 50 if current_url == url else 300
                if not content or len(content.strip()) < min_chars:
                    print(f"[crawl] Skipping low-content page: {current_url} ({len(content)} chars)")
                    continue

                if not first_content:
                    first_content = content

                # Links from subpages go into deep_queue, sorted by priority
                already_queued = queued_urls()
                new_links = [
                    (lu, lt) for lu, lt in links
                    if lu not in visited and lu not in already_queued and not should_skip_url(lu)
                ]
                deep_queue.extend(new_links)
                deep_queue.sort(key=lambda x: url_priority(x[0]))

                # Chunk the page content
                chunk_size = 5000
                page_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                url_to_full_doc[current_url] = content
                total_words += len(content.split())

                for i, chunk in enumerate(page_chunks):
                    all_urls.append(current_url)
                    all_chunk_numbers.append(i)
                    all_chunks.append(chunk)
                    all_metadatas.append({"source": current_url, "source_id": source_id})

        pages_crawled = len(visited)

        if not all_chunks:
            return f"No content extracted from {url}"

        # ── Site profile chunk ──────────────────────────────────────────────
        # Tells the agent what this source is so it can answer cross-source questions.
        site_summary = extract_source_summary(source_id, first_content[:5000])
        site_profile = (
            f"SITE PROFILE: {source_id}\n"
            f"Description: {site_summary}\n"
            f"Total articles indexed: {len(set(all_urls))}\n"
            f"Source URL: {url}"
        )
        all_urls.insert(0, url)
        all_chunk_numbers.insert(0, 9998)
        all_chunks.insert(0, site_profile)
        all_metadatas.insert(0, {"source": url, "source_id": source_id, "type": "site_profile"})

        # ── Homepage featured stories chunk ────────────────────────────────
        # Uses real headline text extracted from homepage links.
        if homepage_stories:
            seen_titles: set[str] = set()
            story_lines = [f"TOP STORIES currently featured on {source_id} homepage:\n"]
            for story_url, headline in homepage_stories:
                clean_headline = headline.strip()
                if clean_headline.lower() not in seen_titles and len(clean_headline) > 15:
                    seen_titles.add(clean_headline.lower())
                    story_lines.append(f"- {clean_headline}  ({story_url})")
            if len(story_lines) > 1:
                featured_text = "\n".join(story_lines)
                all_urls.insert(0, url)
                all_chunk_numbers.insert(0, 9999)
                all_chunks.insert(0, featured_text)
                all_metadatas.insert(0, {"source": url, "source_id": source_id, "type": "featured_stories"})
                url_to_full_doc[url] = url_to_full_doc.get(url, "") + "\n\n" + featured_text

        # Store everything in Supabase
        update_source_info(supabase, source_id, site_summary, total_words)
        add_documents_to_supabase(
            supabase, all_urls, all_chunk_numbers, all_chunks,
            all_metadatas, url_to_full_doc
        )

        print(f"[crawl] Done. {pages_crawled} pages, {len(all_chunks)} chunks stored.")
        return (
            f"Successfully crawled **{source_id}**.\n\n"
            f"- **{pages_crawled} page{'s' if pages_crawled != 1 else ''}** crawled"
            f"{' (limit reached)' if pages_crawled >= max_pages else ''}\n"
            f"- **{len(all_chunks)} chunks** stored\n"
            f"- **{total_words:,} words** indexed\n\n"
            f"You can now ask questions about this content."
        )

    except Exception as e:
        print(f"[crawl] Error: {e}")
        return f"Error crawling {url}: {str(e)}"


# ── RAG helpers ──

def get_indexed_sources() -> List[str]:
    """Return list of indexed source domains."""
    try:
        supabase = get_supabase_client()
        result = supabase.table("sources").select("source_id").execute()
        return [r["source_id"] for r in result.data]
    except Exception:
        return []


# Patterns that indicate the user wants a list of articles/pages rather than content search
_LISTING_PATTERNS = re.compile(
    r"\b(list|show|give me|what are|tell me).{0,30}\b(articles?|stories|pages?|posts?|news|headlines?|topics?)\b"
    r"|\b(top|latest|recent|all).{0,20}\b(articles?|stories|news|posts?)\b"
    r"|\bwhat.{0,20}(have you (crawled|indexed|stored)|do you have)\b"
    r"|\b(articles?|stories|news).{0,20}(available|indexed|stored|crawled)\b",
    re.IGNORECASE
)

def is_listing_intent(query: str) -> bool:
    return bool(_LISTING_PATTERNS.search(query))


def get_article_list_context(query: str, indexed_sources: List[str]) -> str:
    """
    For listing queries, fetch actual article URLs from the DB instead of using vector search.
    Tries to match a source domain mentioned in the query; falls back to all sources.
    """
    try:
        supabase = get_supabase_client()

        # Detect which source the user is asking about
        query_lower = query.lower()
        target_sources = [s for s in indexed_sources if s.lower() in query_lower]
        if not target_sources:
            target_sources = indexed_sources  # list everything if no specific source

        parts = []
        for source_id in target_sources:
            # Get distinct URLs for this source (chunk 0 = first/best chunk per page)
            rows = (
                supabase.table("crawled_pages")
                .select("url, content, metadata")
                .eq("source_id", source_id)
                .order("url")
                .execute()
                .data
            )

            # Deduplicate to one row per URL, skip internal profile chunks
            seen_urls: dict[str, str] = {}
            for row in rows:
                url = row["url"]
                meta_type = (row.get("metadata") or {}).get("type", "")
                if meta_type in ("site_profile",):
                    continue
                if url not in seen_urls:
                    # Derive a readable title from the slug
                    path = urlparse(url).path.rstrip("/")
                    slug = path.split("/")[-1] if path else ""
                    slug = re.sub(r'^\d+-', '', slug)
                    title = slug.replace("-", " ").replace("_", " ").strip().title() or url
                    seen_urls[url] = title

            if seen_urls:
                lines = [f"Articles indexed from {source_id}:\n"]
                for url, title in seen_urls.items():
                    lines.append(f"- {title}  ({url})")
                parts.append("\n".join(lines))

        return "\n\n---\n\n".join(parts) if parts else ""
    except Exception as e:
        print(f"Article list error: {e}")
        return ""


def _rewrite_query_for_embedding(query: str) -> str:
    """
    Convert a question into a declarative statement-style query for better vector similarity.
    Questions and their matching document passages often have lower cosine similarity
    than two similar statements.
    """
    # Strip question words at the start to make it more statement-like
    q = query.strip()
    rewrites = [
        (r"^what (alternative )?explanations? (exist |are there )?for\b", ""),
        (r"^how does .+ compare (to|with)\b", "comparison of "),
        (r"^what (is|are|was|were) (the )?(main |key |primary )?", ""),
        (r"^how (did|does|do|was|were)\b", ""),
        (r"^why (did|does|do|was|were|is|are)\b", ""),
        (r"^(can you |please )?(explain|describe|tell me about|summarize)\b", ""),
    ]
    for pattern, replacement in rewrites:
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE).strip()
    # Remove trailing question mark
    q = q.rstrip("?").strip()
    return q if q else query


def get_rag_context(query: str) -> tuple[str, bool]:
    """
    Search the knowledge base and return (context_text, found_results).
    Uses vector search + keyword fallback to handle analytical/comparative questions.
    """
    SIMILARITY_THRESHOLD = 0.3  # filter out chunks with very low relevance
    MATCH_COUNT = 15             # retrieve more candidates so threshold still leaves enough

    try:
        supabase = get_supabase_client()

        # 1. Primary vector search with rewritten query for better embedding match
        rewritten = _rewrite_query_for_embedding(query)
        results = search_documents(supabase, rewritten, match_count=MATCH_COUNT)

        # Also search with the original query and merge results
        if rewritten.lower() != query.lower():
            orig_results = search_documents(supabase, query, match_count=MATCH_COUNT)
            seen_ids = {r.get("id") for r in results}
            for r in orig_results:
                if r.get("id") not in seen_ids:
                    results.append(r)

        # 2. Keyword fallback — extract meaningful words (3+ chars, not stopwords)
        stopwords = {"what", "does", "this", "that", "with", "from", "have", "been",
                     "besides", "exist", "there", "compare", "other", "about", "which",
                     "would", "could", "should", "their", "these", "those", "than"}
        keywords = [
            w for w in re.findall(r"[a-zA-Z]{3,}", query)
            if w.lower() not in stopwords
        ]
        if keywords:
            # Search with the most distinctive keywords joined as a phrase
            keyword_query = " ".join(keywords[:6])
            kw_results = search_documents(supabase, keyword_query, match_count=MATCH_COUNT)
            seen_ids = {r.get("id") for r in results}
            for r in kw_results:
                if r.get("id") not in seen_ids:
                    results.append(r)

        # 3. Broad-query fallback for "top news / latest" style questions
        broad_keywords = {"top", "latest", "recent", "news", "headlines", "stories", "today"}
        query_words = set(query.lower().split())
        if query_words & broad_keywords:
            extra = search_documents(supabase, "latest news articles headlines", match_count=8)
            seen_ids = {r.get("id") for r in results}
            for r in extra:
                if r.get("id") not in seen_ids:
                    results.append(r)

        # 4. Filter by similarity threshold and sort best-first
        results = [r for r in results if (r.get("similarity") or 0) >= SIMILARITY_THRESHOLD]
        results.sort(key=lambda r: r.get("similarity") or 0, reverse=True)
        results = results[:MATCH_COUNT]  # cap final context size

        if not results:
            return "", False

        # Include source URL with each chunk so the model knows where content came from
        parts = []
        for r in results:
            url = r.get("url", "")
            content = r.get("content", "")
            if content:
                parts.append(f"[Source: {url}]\n{content}")

        return "\n\n---\n\n".join(parts), True
    except Exception as e:
        print(f"RAG search error: {e}")
        return "", False


def build_messages_with_rag(messages: List[Message]) -> List[dict]:
    """Inject RAG context into the message list."""
    user_query = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    indexed_sources = get_indexed_sources()
    sources_list = ", ".join(indexed_sources) if indexed_sources else "none"
    print(f"[rag] query={user_query!r}")
    print(f"[rag] indexed_sources={indexed_sources}")
    print(f"[rag] is_listing_intent={is_listing_intent(user_query)}")

    # For listing-intent queries bypass vector search and enumerate indexed URLs directly
    listing = user_query and is_listing_intent(user_query)
    if listing:
        context = get_article_list_context(user_query, indexed_sources)
        print(f"[rag] listing context length={len(context)}, preview={context[:200]!r}")
    else:
        context, _ = get_rag_context(user_query) if user_query else ("", False)

    if listing and context:
        system_content = (
            "You are a helpful assistant. The user is asking what articles or pages are available "
            "in the knowledge base. Below is the complete list of indexed articles.\n\n"
            "Present them clearly as a list. Use the title derived from the URL slug as the article name "
            "and include the URL. Do NOT say you lack information — the list below IS the answer.\n\n"
            f"Indexed sources: {sources_list}\n"
            f"\n## Indexed Articles\n\n{context}"
        )
    else:
        system_content = (
            "You are a helpful assistant that answers questions STRICTLY based on the content "
            "in the knowledge base provided below.\n\n"
            "RULES:\n"
            "- Only use information from the provided context. Do NOT add facts from your own training data.\n"
            "- If the context does not contain enough information to answer, say exactly: "
            "'I don't have enough information about this in the knowledge base.'\n"
            "- Never invent quotes, statistics, names, or facts not present in the context.\n"
            "- When referring to people or events, use only the descriptions in the context — "
            "do not apply your own knowledge of their current role or status.\n\n"
            f"Indexed sources: {sources_list}\n"
        )
        if context:
            system_content += f"\n## Knowledge Base Context\n\n{context}"
        else:
            system_content += (
                "\n## Knowledge Base Context\n\n"
                "No relevant content found for this query. "
                "Tell the user you don't have this information in the knowledge base "
                "and suggest they try a more specific question about the indexed sources."
            )

    result = [{"role": "system", "content": system_content}]
    result += [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    return result

def make_simple_response(chat_id: str, content: str, stream: bool):
    """Return a simple text response without calling OpenAI."""
    if stream:
        def generate():
            created = int(time.time())
            # Send content in one chunk
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": RAG_MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            done = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": RAG_MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": RAG_MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Endpoints ──

@app.get("/conversations")
def list_conversations():
    """Return all conversations ordered by most recent."""
    try:
        supabase = get_supabase_client()
        result = supabase.table("conversations").select("id, title, created_at, updated_at").order("updated_at", desc=True).execute()
        return {"conversations": result.data}
    except Exception as e:
        return {"conversations": [], "error": str(e)}


@app.post("/conversations")
def create_conversation():
    """Create a new empty conversation."""
    try:
        supabase = get_supabase_client()
        result = supabase.table("conversations").insert({
            "title": "New Chat",
            "messages": []
        }).execute()
        return {"conversation": result.data[0]}
    except Exception as e:
        return {"conversation": None, "error": str(e)}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Get a conversation with its full message history."""
    try:
        supabase = get_supabase_client()
        result = supabase.table("conversations").select("*").eq("id", conversation_id).execute()
        if not result.data:
            return {"conversation": None, "error": "Not found"}
        return {"conversation": result.data[0]}
    except Exception as e:
        return {"conversation": None, "error": str(e)}


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    messages: Optional[List[Any]] = None


@app.put("/conversations/{conversation_id}")
def update_conversation(conversation_id: str, body: UpdateConversationRequest):
    """Update conversation title and/or messages."""
    try:
        supabase = get_supabase_client()
        update_data: dict = {"updated_at": "now()"}
        if body.title is not None:
            update_data["title"] = body.title
        if body.messages is not None:
            update_data["messages"] = body.messages
        result = supabase.table("conversations").update(update_data).eq("id", conversation_id).execute()
        return {"conversation": result.data[0] if result.data else None}
    except Exception as e:
        return {"conversation": None, "error": str(e)}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        supabase = get_supabase_client()
        supabase.table("conversations").delete().eq("id", conversation_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/sources")
def list_sources():
    """Return all indexed sources with page counts from Supabase."""
    try:
        supabase = get_supabase_client()
        sources = supabase.table("sources").select("*").order("updated_at", desc=True).execute().data

        # Count distinct URLs (pages) per source_id from crawled_pages
        pages_result = supabase.table("crawled_pages").select("source_id, url").execute().data
        page_counts: dict[str, set] = {}
        for row in pages_result:
            sid = row["source_id"]
            page_counts.setdefault(sid, set()).add(row["url"])

        for source in sources:
            sid = source["source_id"]
            source["page_count"] = len(page_counts.get(sid, set()))

        return {"sources": sources}
    except Exception as e:
        return {"sources": [], "error": str(e)}


@app.delete("/sources/{source_id}")
def delete_source(source_id: str):
    """Delete a source and all its crawled pages from Supabase."""
    try:
        supabase = get_supabase_client()
        supabase.table("crawled_pages").delete().eq("source_id", source_id).execute()
        supabase.table("code_examples").delete().eq("source_id", source_id).execute()
        supabase.table("sources").delete().eq("source_id", source_id).execute()
        return {"success": True, "deleted": source_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/sources/{source_id}/pages")
def list_source_pages(source_id: str):
    """List all individual pages crawled for a source with preview text."""
    try:
        supabase = get_supabase_client()
        # Get chunk 0 for each URL (the first chunk = best preview)
        rows = (
            supabase.table("crawled_pages")
            .select("url, content, chunk_number, metadata")
            .eq("source_id", source_id)
            .order("url")
            .execute()
            .data
        )
        # Group by URL: keep first chunk as preview, count total chunks
        pages: dict[str, dict] = {}
        for row in rows:
            u = row["url"]
            if u not in pages:
                # Derive a readable title from the URL path
                path = urlparse(u).path.rstrip("/")
                slug = path.split("/")[-1] if path else u
                slug = re.sub(r'^\d+-', '', slug)   # strip leading IDs
                title = slug.replace("-", " ").replace("_", " ").strip().title() or u
                # Skip internal index/profile chunks
                page_type = (row.get("metadata") or {}).get("type", "page")
                pages[u] = {
                    "url": u,
                    "title": title,
                    "preview": row["content"][:200].strip(),
                    "chunk_count": 0,
                    "type": page_type,
                }
            pages[u]["chunk_count"] += 1

        result = sorted(pages.values(), key=lambda p: p["url"])
        return {"pages": result}
    except Exception as e:
        return {"pages": [], "error": str(e)}


@app.delete("/pages")
def delete_page(url: str):
    """Delete all chunks for a specific crawled page URL."""
    try:
        supabase = get_supabase_client()
        supabase.table("crawled_pages").delete().eq("url", url).execute()
        supabase.table("code_examples").delete().eq("url", url).execute()
        return {"success": True, "deleted": url}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": RAG_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "crawl4ai",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )

    # Handle crawl requests before passing to OpenAI
    if is_crawl_intent(user_message):
        urls = extract_urls(user_message)
        max_pages = extract_max_pages(user_message)
        results = await asyncio.gather(*[crawl_and_store(url, max_pages) for url in urls])
        response_text = "\n\n".join(results)
        return make_simple_response(chat_id, response_text, request.stream)

    # Normal RAG + OpenAI flow
    augmented_messages = build_messages_with_rag(request.messages)

    if request.stream:
        def generate():
            created = int(time.time())
            stream = openai_client.chat.completions.create(
                model=MODEL,
                messages=augmented_messages,
                stream=True,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                delta_dict = {}
                if delta.role:
                    delta_dict["role"] = delta.role
                if delta.content:
                    delta_dict["content"] = delta.content
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": RAG_MODEL_ID,
                    "choices": [{"index": 0, "delta": delta_dict, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=augmented_messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": RAG_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.choices[0].message.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8052"))
    print(f"Starting Crawl4AI RAG API on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
