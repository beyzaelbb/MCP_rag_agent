import streamlit as st
import requests
import json
import time
from datetime import datetime, timezone

API_URL = "http://localhost:8052"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.5  # seconds between retries


def api_get(path: str, **kwargs) -> requests.Response | None:
    """GET with automatic retries. Returns None if all attempts fail."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return requests.get(f"{API_URL}{path}", **kwargs)
        except requests.exceptions.ConnectionError:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    return None


def api_post(path: str, **kwargs) -> requests.Response | None:
    """POST with automatic retries. Returns None if all attempts fail."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return requests.post(f"{API_URL}{path}", **kwargs)
        except requests.exceptions.ConnectionError:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    return None


def api_delete(path: str, **kwargs) -> requests.Response | None:
    """DELETE with automatic retries. Returns None if all attempts fail."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return requests.delete(f"{API_URL}{path}", **kwargs)
        except requests.exceptions.ConnectionError:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    return None


def backend_offline_msg():
    st.error(
        "Cannot reach the backend after 3 attempts. "
        "Make sure it is running with: `uv run python src/api_server.py`"
    )

st.set_page_config(
    page_title="Crawl4AI RAG",
    page_icon="🕷️",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_title" not in st.session_state:
    st.session_state.active_title = "New Chat"
if "sidebar_tab" not in st.session_state:
    st.session_state.sidebar_tab = "chats"  # "chats" or "sources"


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_conversation(conv_id: str, messages: list, title: str | None = None):
    payload = {"messages": messages}
    if title:
        payload["title"] = title
    try:
        requests.put(f"{API_URL}/conversations/{conv_id}", json=payload, timeout=10)
    except Exception:
        pass


def auto_title(first_user_message: str) -> str:
    return first_user_message[:50].strip() + ("…" if len(first_user_message) > 50 else "")


def load_conversation(conv_id: str):
    resp = api_get(f"/conversations/{conv_id}", timeout=10)
    if resp is None:
        backend_offline_msg()
        return
    data = resp.json().get("conversation", {})
    st.session_state.active_conversation_id = conv_id
    st.session_state.messages = data.get("messages", [])
    st.session_state.active_title = data.get("title", "New Chat")


def new_chat():
    resp = api_post("/conversations", timeout=10)
    if resp is None:
        backend_offline_msg()
        return
    conv = resp.json().get("conversation")
    if conv:
        st.session_state.active_conversation_id = conv["id"]
        st.session_state.messages = []
        st.session_state.active_title = "New Chat"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🕷️ Crawl4AI RAG")

    tab_chats, tab_sources = st.tabs(["💬 Chats", "🌐 Sources"])

    # ── Chats tab ──
    with tab_chats:
        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            try:
                new_chat()
                st.rerun()
            except Exception as e:
                st.error(f"Could not create chat: {e}")

        st.divider()

        convs_resp = api_get("/conversations", timeout=10)
        conversations = convs_resp.json().get("conversations", []) if convs_resp else []

        if not conversations:
            st.caption("No chats yet. Click '+ New Chat' to start.")
        else:
            for conv in conversations:
                cid = conv["id"]
                title = conv.get("title", "New Chat")
                is_active = cid == st.session_state.active_conversation_id

                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    label = f"**{title}**" if is_active else title
                    if st.button(label, key=f"conv_{cid}", use_container_width=True):
                        try:
                            load_conversation(cid)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading chat: {e}")
                with col_del:
                    if st.button("🗑", key=f"delconv_{cid}"):
                        resp = api_delete(f"/conversations/{cid}", timeout=10)
                        if resp is None:
                            backend_offline_msg()
                        else:
                            if st.session_state.active_conversation_id == cid:
                                st.session_state.active_conversation_id = None
                                st.session_state.messages = []
                                st.session_state.active_title = "New Chat"
                            st.rerun()

    # ── Sources tab ──
    with tab_sources:
        st.subheader("Add a Source")
        url_input = st.text_input("URL to crawl", placeholder="https://example.com/docs")
        max_pages = st.slider("Max pages to crawl", min_value=1, max_value=100, value=20,
                              help="Crawls the page and follows internal links up to this limit.")
        if st.button("Crawl & Index", use_container_width=True):
            if url_input.strip():
                with st.spinner(f"Crawling {url_input} (up to {max_pages} pages) …"):
                    resp = api_post("/v1/chat/completions", json={
                        "model": "crawl4ai-rag",
                        "messages": [{"role": "user", "content": f"crawl {url_input} max_pages={max_pages}"}],
                        "stream": False,
                    }, timeout=300)
                    if resp is None:
                        backend_offline_msg()
                    else:
                        msg = resp.json()["choices"][0]["message"]["content"]
                        st.success(msg)
                        st.rerun()
            else:
                st.warning("Please enter a URL.")

        st.divider()
        st.subheader("Indexed Sources")

        sources_resp = api_get("/sources", timeout=10)
        sources = sources_resp.json().get("sources", []) if sources_resp else []

        if not sources:
            st.caption("No sources indexed yet.")
        else:
            for source in sources:
                sid = source["source_id"]
                word_count = source.get("total_word_count", 0)
                page_count = source.get("page_count", 0)
                summary = source.get("summary", "")
                next_crawl_at = source.get("next_crawl_at")
                is_crawling = source.get("is_crawling", False)

                # Format next recrawl time for display
                if is_crawling:
                    crawl_badge = "🔄 Crawling now…"
                elif next_crawl_at:
                    try:
                        dt = datetime.fromisoformat(next_crawl_at.replace("Z", "+00:00"))
                        now_utc = datetime.now(timezone.utc)
                        diff = dt - now_utc
                        hours_left = int(diff.total_seconds() // 3600)
                        mins_left = int((diff.total_seconds() % 3600) // 60)
                        if diff.total_seconds() < 0:
                            crawl_badge = "⏰ Recrawl overdue"
                        elif hours_left > 0:
                            crawl_badge = f"⏰ Recrawl in {hours_left}h {mins_left}m"
                        else:
                            crawl_badge = f"⏰ Recrawl in {mins_left}m"
                    except Exception:
                        crawl_badge = "⏰ Auto-recrawl on"
                else:
                    crawl_badge = ""

                expander_label = f"🌐 {sid}  ·  {page_count} page{'s' if page_count != 1 else ''}  ·  {word_count:,} words"
                if crawl_badge:
                    expander_label += f"  ·  {crawl_badge}"

                with st.expander(expander_label):
                    if summary:
                        st.caption(summary)

                    col_del, col_recrawl = st.columns([1, 1])
                    with col_del:
                        if st.button(f"🗑 Delete entire source", key=f"del_{sid}", type="secondary", use_container_width=True):
                            with st.spinner(f"Deleting {sid} …"):
                                del_resp = api_delete(f"/sources/{sid}", timeout=15)
                                if del_resp is None:
                                    backend_offline_msg()
                                elif del_resp.json().get("success"):
                                    st.success(f"Deleted {sid}")
                                    st.rerun()
                                else:
                                    st.error(del_resp.json().get("error", "Unknown error"))
                    with col_recrawl:
                        recrawl_disabled = is_crawling
                        recrawl_label = "🔄 Crawling…" if is_crawling else "🔄 Recrawl Now"
                        if st.button(recrawl_label, key=f"recrawl_{sid}", use_container_width=True, disabled=recrawl_disabled):
                            resp = api_post(f"/sources/{sid}/recrawl", timeout=10)
                            if resp is None:
                                backend_offline_msg()
                            elif resp.json().get("success"):
                                st.success(f"Recrawl of {sid} started!")
                                st.rerun()
                            else:
                                st.error(resp.json().get("error", "Could not start recrawl"))

                    st.markdown("**Crawled pages:**")
                    pages_resp = api_get(f"/sources/{sid}/pages", timeout=10)
                    pages = pages_resp.json().get("pages", []) if pages_resp else []
                    pages = [p for p in pages if p.get("type") not in ("featured_stories", "site_profile", "article_index")]

                    if not pages:
                        st.caption("No pages found.")
                    else:
                        for page in pages:
                            p_url = page["url"]
                            p_title = page["title"] or p_url
                            p_preview = page.get("preview", "")
                            p_chunks = page.get("chunk_count", 0)

                            pcol1, pcol2 = st.columns([5, 1])
                            with pcol1:
                                st.markdown(f"**{p_title}**")
                                st.caption(f"{p_chunks} chunk{'s' if p_chunks != 1 else ''} · [{p_url}]({p_url})")
                                if p_preview:
                                    st.markdown(
                                        f"<small style='color:gray'>{p_preview[:150]}…</small>",
                                        unsafe_allow_html=True,
                                    )
                            with pcol2:
                                if st.button("🗑", key=f"delpage_{p_url}", help=f"Delete {p_title}"):
                                    dr = api_delete("/pages", params={"url": p_url}, timeout=10)
                                    if dr is None:
                                        backend_offline_msg()
                                    elif dr.json().get("success"):
                                        st.success("Deleted page")
                                        st.rerun()
                                    else:
                                        st.error(dr.json().get("error", "Error"))
                            st.divider()

        # Auto-refresh only while at least one source is being crawled
        if any(s.get("is_crawling") for s in sources):
            time.sleep(3)
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────

if st.session_state.active_conversation_id is None:
    # Landing screen — no active chat
    st.markdown("## Welcome to Crawl4AI RAG 👋")
    st.markdown(
        "Use the **＋ New Chat** button in the sidebar to start a conversation, "
        "or select a previous chat from the list."
    )
    st.markdown(
        "To add knowledge, go to the **🌐 Sources** tab and paste a URL to crawl."
    )
else:
    st.subheader(st.session_state.active_title)

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user input
    if prompt := st.chat_input("Ask anything about your indexed sources…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Auto-title on first message
        if len(st.session_state.messages) == 1:
            title = auto_title(prompt)
            st.session_state.active_title = title
            save_conversation(st.session_state.active_conversation_id, [], title=title)

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                with requests.post(  # streaming requires raw requests (no wrapper)
                    f"{API_URL}/v1/chat/completions",
                    json={
                        "model": "crawl4ai-rag",
                        "messages": st.session_state.messages,
                        "stream": True,
                    },
                    stream=True,
                    timeout=60,
                ) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk["choices"][0]["delta"]
                                token = delta.get("content", "")
                                full_response += token
                                placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                pass

                placeholder.markdown(full_response)

            except requests.exceptions.ConnectionError:
                # Retry once before giving up on the stream
                time.sleep(RETRY_DELAY)
                try:
                    with requests.post(
                        f"{API_URL}/v1/chat/completions",
                        json={"model": "crawl4ai-rag", "messages": st.session_state.messages, "stream": False},
                        timeout=60,
                    ) as retry_resp:
                        full_response = retry_resp.json()["choices"][0]["message"]["content"]
                        placeholder.markdown(full_response)
                except Exception:
                    full_response = "Backend unavailable after retry. Make sure `uv run python src/api_server.py` is running."
                    placeholder.error(full_response)
            except Exception as e:
                full_response = f"Error: {e}"
                placeholder.error(full_response)

        # Save assistant reply and persist to Supabase
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_conversation(
            st.session_state.active_conversation_id,
            st.session_state.messages,
        )
