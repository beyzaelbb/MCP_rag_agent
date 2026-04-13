# Crawl4AI RAG Workspace

This repository implements a document crawling and retrieval system built around [Crawl4AI](https://crawl4ai.com), [Supabase](https://supabase.com/), [OpenAI](https://platform.openai.com/), and optional [Neo4j](https://neo4j.com/).

It currently exposes the same core knowledge base through three different interfaces:

- An MCP server for AI agents and coding assistants
- An OpenAI-compatible chat API with built-in crawl-and-index behavior
- A Streamlit frontend for managing sources and chatting with the indexed knowledge base

It also includes an optional knowledge graph pipeline for parsing Python repositories into Neo4j and checking AI-generated scripts for hallucinated imports, classes, methods, or usage patterns.

## What The Project Does

At a high level, the project:

1. Crawls web pages, sitemaps, text files, and recursively discovered internal links
2. Chunks the extracted content and stores it in Supabase with vector embeddings
3. Lets users and agents retrieve relevant content through RAG
4. Optionally extracts large code examples into a separate searchable index
5. Optionally parses Python repositories into Neo4j for repository-aware code validation

## Main Components

### 1. MCP server

File: [src/crawl4ai_mcp.py](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/src/crawl4ai_mcp.py)

This is the agent-facing server. It runs as an MCP service over SSE by default and exposes tools for crawling, retrieval, code-example search, and knowledge graph operations.

Implemented MCP tools:

- `crawl_single_page`
- `smart_crawl_url`
- `get_available_sources`
- `perform_rag_query`
- `search_code_examples`
- `check_ai_script_hallucinations`
- `query_knowledge_graph`
- `parse_github_repository`

### 2. OpenAI-compatible API

File: [src/api_server.py](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/src/api_server.py)

This service provides a chat-completions style API around the same Supabase-backed knowledge base. It supports:

- `GET /v1/models`
- `POST /v1/chat/completions`
- conversation CRUD endpoints
- source and page management endpoints

It also detects crawl intent inside user messages. If a message contains commands like `crawl https://... max_pages=20`, it crawls and indexes that source before returning a response.

### 3. Streamlit frontend

File: [frontend/app.py](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/frontend/app.py)

The frontend is a lightweight operator UI for:

- creating and revisiting chat sessions
- triggering crawl jobs
- viewing indexed sources
- browsing crawled pages
- deleting pages or entire sources

### 4. Shared utilities

File: [src/utils.py](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/src/utils.py)

This module contains the core data-layer logic:

- Supabase client creation
- embedding creation
- batch insert logic
- contextual embedding enrichment
- code block extraction
- source summarization
- vector search for documents and code examples

### 5. Knowledge graph utilities

Files live under [knowledge_graphs](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/knowledge_graphs).

These scripts support:

- parsing Python repositories into Neo4j using AST-based extraction
- analyzing AI-generated Python scripts
- validating those scripts against the repository graph
- querying the stored graph interactively

## Implemented Features

### Crawling and indexing

- Single-page crawling
- Smart URL handling for normal pages, sitemap URLs, and `.txt` files
- Recursive internal-link crawling for websites
- Parallel batch crawling
- URL prioritization so article-like pages are crawled earlier
- URL skipping rules for low-value pages such as pagination, tags, login, cart, and policy routes
- Markdown-aware chunking with preferences for code block and paragraph boundaries
- Source-level summaries stored per domain
- Site profile and featured-story synthetic chunks added during deep crawl mode
- First crawl replaces existing records; recrawls only add new pages — old articles are never deleted

### Auto-recrawl scheduler

- Every crawled source automatically schedules a recrawl 24 hours after indexing
- The scheduler sleeps until the exact due time rather than polling on a fixed interval
- When a recrawl fires, already-indexed URLs are skipped entirely at the crawl loop level — no redundant fetching or re-embedding
- Only new pages discovered since the last crawl are inserted
- Auto-recrawls default to `max_pages=20`; manual crawls via the chat interface accept a user-specified limit
- A `POST /sources/{source_id}/recrawl` endpoint allows triggering an immediate recrawl from the UI

### Retrieval and RAG

- Vector similarity search over crawled content across all indexed sources simultaneously
- Source-diversity cap: results ranked globally by similarity, with each source limited to 5 chunks maximum so no single site dominates when multiple sources cover the same topic
- Featured-story and site-profile index chunks excluded from RAG so the model only receives actual article body content
- Full-text fallback: when vector search returns no results (e.g. for specific named entities or non-ASCII names), a keyword-based `ilike` content search is run against Supabase automatically
- Optional hybrid retrieval combining vector search with keyword matching
- Optional reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Query rewriting for better semantic retrieval
- Listing-intent detection for questions like “what articles do you have?”
- Source-aware filtering
- Multi-source synthesis: when multiple sources cover the same topic the model is instructed to combine perspectives and highlight differences in framing or emphasis
- Every answer ends with a `**Sources:**` section of clickable article links derived only from context actually used in the answer
- OpenAI-backed answer generation constrained to indexed knowledge

### Embedding cache

- In-memory cache for all OpenAI embedding API calls, bounded at 2000 entries with FIFO eviction
- Thread-safe; no TTL required since embeddings are deterministic
- Covers all call paths: user queries, rewritten queries, keyword fallback queries, and document chunk embeddings
- Cache hits are logged so utilization is visible in server output

### Code-example retrieval

- Optional extraction of large code blocks from crawled markdown
- LLM-generated summaries for each extracted code example
- Separate `code_examples` vector index in Supabase
- Dedicated MCP search tool for code-oriented retrieval

### API and frontend operations

- OpenAI-compatible `/v1/chat/completions`
- Streaming and non-streaming responses
- Conversation persistence in Supabase
- Source listing, page listing, source deletion, and page deletion
- `POST /sources/{source_id}/recrawl` for immediate manual recrawl
- Streamlit UI for chat and source management
- Source expander shows live recrawl countdown (`⏰ Recrawl in Xh Ym`) or active crawl indicator (`🔄 Crawling now…`)
- Frontend auto-refreshes every 3 seconds while any source is actively being crawled, and stops automatically when idle

### Knowledge graph and hallucination detection

- Optional Neo4j integration
- GitHub repository parsing into repository, file, class, method, function, and import relationships
- AST-based analysis of Python scripts
- Validation of imports, class usage, methods, and function calls against the parsed graph
- Interactive graph queries for repositories, classes, methods, and custom Cypher

## Architecture

### Storage layer

Supabase stores three main datasets:

- `sources`: source/domain summaries, aggregate counts, original crawl URL, and `next_crawl_at` timestamp for the auto-recrawl scheduler
- `crawled_pages`: chunked documentation or web content with embeddings
- `code_examples`: extracted code snippets with embeddings and summaries
- `conversations`: frontend/API chat history

Schema file: [crawled_pages.sql](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/crawled_pages.sql)

### Model usage

OpenAI is used for:

- embeddings via `text-embedding-3-small`
- source summaries
- contextual chunk descriptions
- code example summaries
- answer generation in the API server

The code also contains commented placeholders for switching some operations back to a local Ollama setup, but the active implementation is OpenAI-based.

## Retrieval Modes

These environment flags control the advanced behavior:

- `USE_CONTEXTUAL_EMBEDDINGS=true`
  Enriches each chunk with LLM-generated context before embedding.

- `USE_HYBRID_SEARCH=true`
  Combines semantic search with exact keyword matches.

- `USE_AGENTIC_RAG=true`
  Extracts code examples into a separate searchable index.

- `USE_RERANKING=true`
  Reranks retrieved results with a cross-encoder model.

- `USE_KNOWLEDGE_GRAPH=true`
  Enables Neo4j-backed repository parsing and hallucination-checking tools.

## Prerequisites

- Python `3.12+`
- [uv](https://docs.astral.sh/uv/) or Docker
- A Supabase project
- An OpenAI API key
- Neo4j only if knowledge graph features are enabled

## Installation

### Option 1: Local Python setup

```bash
git clone <your-repo-url>
cd mcp-crawl4ai-rag
uv venv
source .venv/bin/activate
uv pip install -e .
crawl4ai-setup
```

### Option 2: Docker setup

```bash
git clone <your-repo-url>
cd mcp-crawl4ai-rag
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
```

The Docker image currently runs the MCP server entrypoint from [Dockerfile](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/Dockerfile).

## Environment Variables

Create a `.env` file in the project root.

```env
# MCP server
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# API server
API_PORT=8052

# OpenAI
OPENAI_API_KEY=your_openai_api_key
MODEL_CHOICE=gpt-4o-mini

# Supabase
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Retrieval features
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Neo4j, only required when USE_KNOWLEDGE_GRAPH=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

## Supabase Setup

Before running the project, execute the SQL in [crawled_pages.sql](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/crawled_pages.sql) inside your Supabase SQL editor.

This creates:

- vector-enabled content tables
- similarity search functions for pages and code examples
- a conversation table for the frontend/API
- indexes and policies required by the current implementation

If you have an existing database from a previous version, run this migration to add the scheduler columns:

```sql
alter table sources add column if not exists url text;
alter table sources add column if not exists next_crawl_at timestamp with time zone;
```

## Running The Project

### Run the MCP server

```bash
uv run python src/crawl4ai_mcp.py
```

By default this uses SSE and serves on `http://localhost:8051/sse`.

Connection config example: [mcpo_config.json](/Users/beyzanurelbeyoglu/Desktop/mcp-crawl4ai-rag/mcpo_config.json)

### Run the API server

```bash
uv run python src/api_server.py
```

Default API base URL:

```text
http://localhost:8052
```

### Run the Streamlit frontend

```bash
uv run streamlit run frontend/app.py
```

The frontend expects the API server at `http://localhost:8052`.

## MCP Tool Reference

### `crawl_single_page`

Crawls one page, chunks it, stores it in Supabase, updates source metadata, and optionally extracts code examples.

### `smart_crawl_url`

Chooses the crawl strategy automatically:

- `.txt` URL: fetch as text content
- sitemap URL: parse and crawl listed pages
- regular page: recursively crawl internal links

### `get_available_sources`

Returns indexed sources from Supabase so an agent can decide what to query.

### `perform_rag_query`

Searches the crawled content index, optionally limited to a source.

### `search_code_examples`

Searches the code example index. This is most useful when `USE_AGENTIC_RAG=true`.

### `parse_github_repository`

Parses a GitHub repository into Neo4j for graph-based repository understanding.

### `check_ai_script_hallucinations`

Analyzes a Python file and validates its code usage against the Neo4j graph.

### `query_knowledge_graph`

Provides structured graph exploration commands and supports custom Cypher queries.

## API Behavior

The API server is not only a wrapper around retrieval. It also adds some practical behavior:

- If a user message contains a URL and crawl intent, it triggers indexing directly
- If a user asks for a list of available articles or pages, it bypasses semantic RAG and enumerates indexed URLs from the database
- For standard questions, it builds a constrained system prompt with retrieved context and calls OpenAI chat completions

This makes the API usable as both a chat endpoint and a lightweight operator endpoint for source ingestion.

## Knowledge Graph Workflow

When enabled, the intended flow is:

1. Start Neo4j
2. Run the MCP server with `USE_KNOWLEDGE_GRAPH=true`
3. Parse a Python GitHub repository with `parse_github_repository`
4. Run `check_ai_script_hallucinations` on an AI-generated `.py` file
5. Inspect the graph with `query_knowledge_graph` if needed

The repository parser is AST-based and designed to insert code structure into Neo4j directly rather than relying on LLM extraction.

## Current Limitations

- The Docker setup is centered on the MCP server only
- The frontend and API server are separate processes and are not containerized here
- Knowledge graph features require extra infrastructure and are better suited to local `uv` execution than the current Docker path
- The implementation is currently tailored to OpenAI embeddings and chat generation
- The frontend hardcodes the API server URL to `http://localhost:8052`
- There is no formal test suite in the current repository snapshot

## Repository Layout

```text
.
├── frontend/
│   └── app.py
├── knowledge_graphs/
│   ├── ai_hallucination_detector.py
│   ├── ai_script_analyzer.py
│   ├── hallucination_reporter.py
│   ├── knowledge_graph_validator.py
│   ├── parse_repo_into_neo4j.py
│   ├── query_knowledge_graph.py
│   └── test_script.py
├── src/
│   ├── api_server.py
│   ├── crawl4ai_mcp.py
│   ├── todo.txt
│   └── utils.py
├── crawled_pages.sql
├── Dockerfile
├── mcpo_config.json
├── pyproject.toml
└── README.md
```

## Summary

This project is no longer just an MCP server. In its current form, it is a small retrieval platform with:

- a crawl-and-index pipeline
- a Supabase-backed RAG layer
- an OpenAI-compatible chat API
- a Streamlit operations UI
- optional Neo4j-based repository validation tooling

That makes it suitable both as an AI-agent backend and as a supervised demo system for crawling, indexing, retrieval, and AI code validation workflows.
