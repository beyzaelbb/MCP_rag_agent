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
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Header
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

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "placeholder"))
MODEL = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
RAG_MODEL_ID = "crawl4ai-rag"
RECRAWL_INTERVAL_HOURS = 24

# Track which sources are currently being (re)crawled to avoid duplicate jobs
_crawling_sources: set[str] = set()


def set_next_crawl(supabase, source_id: str) -> None:
    """Set next_crawl_at to 24 hours from now for the given source."""
    next_time = (datetime.now(timezone.utc) + timedelta(hours=RECRAWL_INTERVAL_HOURS)).isoformat()
    try:
        supabase.table("sources").update({"next_crawl_at": next_time}).eq("source_id", source_id).execute()
        print(f"[scheduler] Next crawl for {source_id} scheduled at {next_time}")
    except Exception as e:
        print(f"[scheduler] Failed to set next_crawl_at for {source_id}: {e}")


async def auto_recrawl_scheduler() -> None:
    """Background task: sleeps until the next source is due, then fires its recrawl.

    Rather than polling on a fixed interval, we query the earliest next_crawl_at
    and sleep precisely until that moment. After waking we fire all sources whose
    deadline has arrived, then repeat.
    """
    print("[scheduler] Auto-recrawl scheduler started.")
    while True:
        try:
            supabase = get_supabase_client()

            # Enroll any sources that have no schedule yet (indexed before recrawl feature)
            unscheduled = (
                supabase.table("sources")
                .select("source_id")
                .is_("next_crawl_at", "null")
                .execute()
                .data
            )
            for r in unscheduled:
                sid = r["source_id"]
                print(f"[scheduler] Enrolling {sid} in auto-recrawl schedule")
                set_next_crawl(supabase, sid)

            # Find the soonest upcoming crawl
            row = (
                supabase.table("sources")
                .select("next_crawl_at")
                .not_.is_("next_crawl_at", "null")
                .order("next_crawl_at")
                .limit(1)
                .execute()
                .data
            )
            if not row:
                # No sources scheduled yet — check again in an hour
                await asyncio.sleep(3600)
                continue

            next_due = datetime.fromisoformat(row[0]["next_crawl_at"].replace("Z", "+00:00"))
            sleep_secs = max(0.0, (next_due - datetime.now(timezone.utc)).total_seconds())
            print(f"[scheduler] Next recrawl in {sleep_secs:.0f}s ({next_due.isoformat()})")
            await asyncio.sleep(sleep_secs)

            # Fire all sources that are now due (handles ties / near-simultaneous deadlines)
            now_iso = datetime.now(timezone.utc).isoformat()
            due_rows = (
                supabase.table("sources")
                .select("source_id, url")
                .lte("next_crawl_at", now_iso)
                .not_.is_("next_crawl_at", "null")
                .execute()
                .data
            )
            all_skipped = True
            for r in due_rows:
                sid = r["source_id"]
                crawl_url = (r.get("url") or "").strip() or f"https://{sid}"
                if sid in _crawling_sources:
                    print(f"[scheduler] Skipping {sid} — already crawling")
                    continue
                all_skipped = False
                print(f"[scheduler] Recrawling {sid} from {crawl_url}")
                asyncio.create_task(crawl_and_store(crawl_url, max_pages=MAX_PAGES_DEFAULT, is_recrawl=True))

            # All due sources are still running — wait before re-checking to avoid spin
            if due_rows and all_skipped:
                await asyncio.sleep(60)

        except Exception as e:
            print(f"[scheduler] Error: {e}")
            await asyncio.sleep(60)  # back off briefly on unexpected errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(auto_recrawl_scheduler())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Crawl4AI RAG API", lifespan=lifespan)

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
    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=30000)),
            timeout=45,
        )
    except asyncio.TimeoutError:
        print(f"[crawl] Timeout fetching {url} — skipping")
        return "", []
    if not result.success:
        print(f"[crawl] Failed: {url} — {result.error_message}")
        return "", []
    content = result.markdown.raw_markdown if result.markdown else ""
    links = extract_internal_links(result, url)
    return content, links


async def crawl_and_store(url: str, max_pages: int = MAX_PAGES_DEFAULT, is_recrawl: bool = False) -> str:
    """Crawl a URL and its subpages, store everything in Supabase.

    When is_recrawl=True, existing articles are preserved and only new URLs are added.
    """
    source_id = urlparse(url).netloc
    _crawling_sources.add(source_id)
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        supabase = get_supabase_client()
        source_id = urlparse(url).netloc

        # For recrawls, pre-load every URL already in the DB for this source.
        # We add them to `visited` so the crawl loop skips them entirely —
        # no fetching, no embedding, no wasted API calls.
        # The root URL is excluded so we still crawl the homepage to discover new links.
        if is_recrawl:
            try:
                existing_rows = (
                    supabase.table("crawled_pages")
                    .select("url")
                    .eq("source_id", source_id)
                    .execute()
                    .data
                )
                already_crawled = {row["url"] for row in existing_rows} - {url}
                print(f"[crawl] Recrawl: skipping {len(already_crawled)} already-indexed URLs")
            except Exception as e:
                print(f"[crawl] Could not fetch existing URLs: {e}")
                already_crawled = set()
        else:
            already_crawled = set()

        visited: set[str] = already_crawled.copy()
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

        print(f"[crawl] Starting {'re' if is_recrawl else ''}crawl of {url} (max {max_pages} new pages)")

        new_pages_crawled = 0
        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
            while new_pages_crawled < max_pages:
                item = next_url()
                if item is None:
                    break
                current_url, _ = item
                if current_url in visited:
                    continue
                visited.add(current_url)
                new_pages_crawled += 1

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

        pages_crawled = new_pages_crawled

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
        update_source_info(supabase, source_id, site_summary, total_words, url=url)
        add_documents_to_supabase(
            supabase, all_urls, all_chunk_numbers, all_chunks,
            all_metadatas, url_to_full_doc,
            skip_existing=is_recrawl,
        )
        set_next_crawl(supabase, source_id)

        print(f"[crawl] Done. {pages_crawled} pages, {len(all_chunks)} chunks stored.")
        next_crawl_dt = datetime.now(timezone.utc) + timedelta(hours=RECRAWL_INTERVAL_HOURS)
        next_crawl_str = next_crawl_dt.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"Successfully crawled **{source_id}**.\n\n"
            f"- **{pages_crawled} page{'s' if pages_crawled != 1 else ''}** crawled"
            f"{' (limit reached)' if pages_crawled >= max_pages else ''}\n"
            f"- **{len(all_chunks)} chunks** stored\n"
            f"- **{total_words:,} words** indexed\n"
            f"- **Auto-recrawl** scheduled every {RECRAWL_INTERVAL_HOURS}h (next: {next_crawl_str})\n\n"
            f"You can now ask questions about this content."
        )

    except Exception as e:
        print(f"[crawl] Error: {e}")
        return f"Error crawling {url}: {str(e)}"
    finally:
        _crawling_sources.discard(source_id)


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
        (r"^what (happened|occurred|took place) (to|with|at|in|after|during|when)\b", ""),
        (r"^how (did|does|do|was|were)\b", ""),
        (r"^why (did|does|do|was|were|is|are)\b", ""),
        (r"^(can you |please )?(explain|describe|tell me about|summarize)\b", ""),
        (r"^(tell me|give me|show me) (what |about |the )?(details? of |story of |info (on|about) )?", ""),
    ]
    for pattern, replacement in rewrites:
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE).strip()
    # Remove trailing question mark
    q = q.rstrip("?").strip()
    return q if q else query


def get_rag_context(query: str) -> tuple[str, bool]:
    """
    Search the knowledge base and return (context_text, found_results).
    Searches all sources globally, then applies source-diversity balancing so that
    no single source dominates the context when multiple sites cover the same topic.
    """
    # For broad summarisation queries lower the threshold so more articles pass through
    broad_summary = any(w in query.lower() for w in {"summarize", "summarise", "summary", "last 24", "today", "recent", "latest", "everything"})
    SIMILARITY_THRESHOLD = 0.2 if broad_summary else 0.3
    SIMILARITY_THRESHOLD_SOFT = 0.10  # relaxed threshold used before falling back to keyword search
    FETCH_COUNT = 40 if broad_summary else 25   # fetch more candidates for summary queries
    FINAL_COUNT = 20   # max chunks passed to the model
    CHUNKS_PER_SOURCE = 4  # guaranteed slots per source in the final context

    try:
        supabase = get_supabase_client()

        # 1. Primary vector search with rewritten query
        rewritten = _rewrite_query_for_embedding(query)
        results = search_documents(supabase, rewritten, match_count=FETCH_COUNT)

        # Also search with the original query and merge
        if rewritten.lower() != query.lower():
            orig_results = search_documents(supabase, query, match_count=FETCH_COUNT)
            seen_ids = {r.get("id") for r in results}
            for r in orig_results:
                if r.get("id") not in seen_ids:
                    results.append(r)

        # 2. Keyword fallback
        stopwords = {"what", "does", "this", "that", "with", "from", "have", "been",
                     "besides", "exist", "there", "compare", "other", "about", "which",
                     "would", "could", "should", "their", "these", "those", "than"}
        keywords = [w for w in re.findall(r"[a-zA-Z]{3,}", query) if w.lower() not in stopwords]
        if keywords:
            kw_results = search_documents(supabase, " ".join(keywords[:6]), match_count=FETCH_COUNT)
            seen_ids = {r.get("id") for r in results}
            for r in kw_results:
                if r.get("id") not in seen_ids:
                    results.append(r)

        # 3. Broad-query fallback for "top news / latest" style questions
        broad_keywords = {"top", "latest", "recent", "news", "headlines", "stories", "today"}
        if set(query.lower().split()) & broad_keywords:
            extra = search_documents(supabase, "latest news articles headlines", match_count=8)
            seen_ids = {r.get("id") for r in results}
            for r in extra:
                if r.get("id") not in seen_ids:
                    results.append(r)

        # 4. Drop index/profile chunks — they contain only headlines or metadata,
        #    not article body text, so they produce misleading "no content" answers.
        _INDEX_TYPES = {"featured_stories", "site_profile", "article_index"}
        before_filter = len(results)
        results = [
            r for r in results
            if (r.get("metadata") or {}).get("type", "page") not in _INDEX_TYPES
        ]
        print(f"[rag] raw results: {before_filter}, after index-type filter: {len(results)}")

        # Log top raw scores so we can diagnose threshold issues
        if results:
            top_scores = sorted(
                [r.get("similarity") or 0 for r in results], reverse=True
            )[:5]
            print(f"[rag] top similarity scores: {[round(s, 3) for s in top_scores]}")
        else:
            print("[rag] search_documents returned 0 results (embedding error or empty DB?)")

        # Filter by similarity threshold — try strict first, relax if nothing passes
        strict = [r for r in results if (r.get("similarity") or 0) >= SIMILARITY_THRESHOLD]
        if strict:
            results = strict
        else:
            soft = [r for r in results if (r.get("similarity") or 0) >= SIMILARITY_THRESHOLD_SOFT]
            if soft:
                print(f"[rag] strict threshold ({SIMILARITY_THRESHOLD}) empty — soft threshold ({SIMILARITY_THRESHOLD_SOFT}) found {len(soft)} results")
            else:
                print(f"[rag] both thresholds empty — all scores below {SIMILARITY_THRESHOLD_SOFT}, falling back to full-text")
            results = soft

        # Recency boost: articles published (or crawled if no publish date) in the last
        # 24 hours get a +0.15 score bump so they rank above older content.
        now_utc = datetime.now(timezone.utc)
        for r in results:
            meta = r.get("metadata") or {}
            published_date_str = meta.get("published_date")
            crawled_at_str = meta.get("crawled_at")
            ref_str = published_date_str or crawled_at_str
            if ref_str:
                try:
                    ref_dt = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
                    age_hours = (now_utc - ref_dt).total_seconds() / 3600
                    if age_hours <= 24:
                        r["_boosted_score"] = (r.get("similarity") or 0) + 0.15
                    else:
                        r["_boosted_score"] = r.get("similarity") or 0
                except Exception:
                    r["_boosted_score"] = r.get("similarity") or 0
            else:
                r["_boosted_score"] = r.get("similarity") or 0

        results.sort(key=lambda r: r["_boosted_score"], reverse=True)

        # 5. Full-text fallback — runs when vector search finds nothing OR only found
        #    soft-threshold results (score < SIMILARITY_THRESHOLD). In the soft case we
        #    merge full-text hits in rather than replacing, so the keyword-matched article
        #    is included alongside the weak vector result.
        used_soft = bool(results) and not strict
        if (not results or used_soft) and keywords:
            try:
                # Exclude generic verbs/adverbs that appear in almost every article —
                # they produce noisy matches and dilute the result set.
                _GENERIC_WORDS = {
                    "happened", "occurred", "said", "told", "added", "noted", "stated",
                    "according", "after", "before", "during", "while", "also", "just",
                    "still", "already", "then", "when", "because", "however", "although",
                }
                topic_keywords = [k for k in keywords if k.lower() not in _GENERIC_WORDS]
                # Fall back to all keywords if filtering left nothing
                search_terms_pool = topic_keywords if topic_keywords else keywords
                # Prefer longer (more distinctive) terms, take up to 4
                search_terms = sorted(search_terms_pool, key=len, reverse=True)[:4]
                ft_results = []
                seen_ft_ids: set = set()
                for term in search_terms:
                    rows = (
                        supabase.table("crawled_pages")
                        .select("id, url, chunk_number, content, metadata, source_id")
                        .ilike("content", f"%{term}%")
                        .limit(10)
                        .execute()
                        .data
                    )
                    for row in rows:
                        rid = row.get("id")
                        meta_type = (row.get("metadata") or {}).get("type", "page")
                        if rid not in seen_ft_ids and meta_type not in _INDEX_TYPES:
                            seen_ft_ids.add(rid)
                            row["similarity"] = 0.0
                            ft_results.append(row)
                if ft_results:
                    # Rank by how many distinct search terms appear in the content —
                    # chunks that match more keywords are more likely to be the right article.
                    all_terms_lower = [t.lower() for t in search_terms_pool]
                    for row in ft_results:
                        content_lower = (row.get("content") or "").lower()
                        row["_ft_score"] = sum(1 for t in all_terms_lower if t in content_lower)
                    ft_results.sort(key=lambda r: r["_ft_score"], reverse=True)
                    if used_soft:
                        # Merge: keep soft vector results, add full-text hits not already present
                        existing_ids = {r.get("id") for r in results}
                        added = [r for r in ft_results if r.get("id") not in existing_ids]
                        results.extend(added)
                        print(f"[rag] soft-threshold vector + full-text fallback: {len(results)} total results ({len(added)} from full-text)")
                    else:
                        print(f"[rag] vector search empty — full-text fallback found {len(ft_results)} results")
                        results = ft_results
            except Exception as ft_err:
                print(f"[rag] full-text fallback error: {ft_err}")

        if not results:
            return "", False

        # 5. Source-diversity cap:
        #    Walk results best-first and include a chunk only if its source hasn't
        #    already filled its cap. This prevents one dominant source from taking
        #    all slots, but never forces in irrelevant chunks just to represent a source.
        MAX_CHUNKS_PER_SOURCE = 5
        source_counts: dict[str, int] = {}
        balanced: list = []
        for r in results:
            sid = r.get("source_id", "unknown")
            if source_counts.get(sid, 0) < MAX_CHUNKS_PER_SOURCE:
                balanced.append(r)
                source_counts[sid] = source_counts.get(sid, 0) + 1
            if len(balanced) >= FINAL_COUNT:
                break

        print(f"[rag] sending {len(balanced)} chunks to model: {[r.get('url','?') for r in balanced[:5]]}")

        parts = []
        for r in balanced:
            url = r.get("url", "")
            content = r.get("content", "")
            if not content:
                continue
            meta = r.get("metadata") or {}

            # Use published_date for recency when available; fall back to crawled_at
            published_date_str = meta.get("published_date")
            crawled_at_str = meta.get("crawled_at")

            recency_label = "ARCHIVE"
            if published_date_str:
                try:
                    pub_dt = datetime.fromisoformat(published_date_str.replace("Z", "+00:00"))
                    age_hours = (now_utc - pub_dt).total_seconds() / 3600
                    if age_hours <= 24:
                        recency_label = "RECENT"
                except Exception:
                    pass
            elif crawled_at_str:
                try:
                    crawled_dt = datetime.fromisoformat(crawled_at_str)
                    age_hours = (now_utc - crawled_dt).total_seconds() / 3600
                    if age_hours <= 24:
                        recency_label = "RECENT"
                except Exception:
                    pass

            # Build label line — include all available metadata so the model can answer
            # questions like "when was this published?" or "who wrote this?"
            label_parts = [recency_label, f"Source: {url}"]
            if published_date_str:
                label_parts.append(f"Published: {published_date_str}")
            if meta.get("title"):
                label_parts.append(f"Title: {meta['title']}")
            if meta.get("author"):
                label_parts.append(f"Author: {meta['author']}")

            parts.append(f"[{' | '.join(label_parts)}]\n{content}")

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
            "- The knowledge base may contain information that is more current than your training data. "
            "Whenever there is any conflict between what the context says and what you believe to be true "
            "from training — including titles, roles, statuses, dates, or any other facts — "
            "the context is always authoritative. Never silently correct or update what the context says.\n"
            "- When referring to people or events, use only the descriptions in the context — "
            "do not apply your own knowledge of their current role or status.\n"
            "- Each context chunk is labeled [RECENT] or [ARCHIVE] based on the article's "
            "publish date (or crawl date when no publish date is available). When the user asks "
            "about 'latest', 'recent', 'today', or 'new' articles/news, prioritise [RECENT] "
            "chunks. If there are RECENT chunks, lead with those. If there are also ARCHIVE "
            "chunks, you may include them but clearly note they are older. Never refuse to answer "
            "just because some chunks are ARCHIVE — summarise whatever is available and be clear "
            "about what is recent vs older. "
            "Each chunk may also include 'Published: <date>', 'Title: <title>', and "
            "'Author: <name>' fields — use these when answering questions about when something "
            "was published, who wrote it, or what the article is called.\n"
            "- When the context contains coverage of the same topic from multiple sources, "
            "synthesize all perspectives into a single combined answer. Note where sources "
            "agree, and explicitly highlight any differences in framing, emphasis, or detail "
            "between them.\n"
            "- ALWAYS end your response with a small sources section in this exact format — "
            "a plain text line '**Sources:**' (not a heading) followed by a markdown list where "
            "each item is the article title as a clickable link, e.g. `- [Article Title](url)`. "
            "Only include URLs that actually appear in the context below. "
            "Do not include this section if no context was found.\n\n"
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
            source["is_crawling"] = sid in _crawling_sources

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


@app.post("/sources/{source_id}/recrawl")
async def recrawl_source(source_id: str):
    """Trigger an immediate recrawl of a source."""
    if source_id in _crawling_sources:
        return {"success": False, "error": f"{source_id} is already being crawled"}
    try:
        supabase = get_supabase_client()
        row = supabase.table("sources").select("url").eq("source_id", source_id).execute().data
        if not row:
            return {"success": False, "error": "Source not found"}
        crawl_url = (row[0].get("url") or "").strip() or f"https://{source_id}"
        asyncio.create_task(crawl_and_store(crawl_url, max_pages=MAX_PAGES_DEFAULT, is_recrawl=True))
        return {"success": True, "message": f"Recrawl of {source_id} started from {crawl_url}"}
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
async def chat_completions(request: ChatRequest, x_openai_key: Optional[str] = Header(default=None)):
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )

    api_key = x_openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="OpenAI API key is required. Provide it via the X-OpenAI-Key header.")

    client = openai.OpenAI(api_key=api_key)
    import openai as _openai_module
    _openai_module.api_key = api_key  # used by utils.py embedding calls

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
            stream = client.chat.completions.create(
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

    response = client.chat.completions.create(
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
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8052")))
    print(f"Starting Crawl4AI RAG API on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
