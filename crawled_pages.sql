-- Enable the pgvector extension
create extension if not exists vector;

-- ── Sources ───────────────────────────────────────────────────────────────────

create table if not exists sources (
    source_id text primary key,
    summary text,
    total_word_count integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table sources enable row level security;

create policy "Allow public read access to sources"
    on sources for select to public using (true);

-- ── Crawled pages ─────────────────────────────────────────────────────────────

create table if not exists crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number),
    foreign key (source_id) references sources(source_id)
);

create index if not exists idx_crawled_pages_embedding on crawled_pages using ivfflat (embedding vector_cosine_ops);
create index if not exists idx_crawled_pages_metadata on crawled_pages using gin (metadata);
create index if not exists idx_crawled_pages_source_id on crawled_pages (source_id);

alter table crawled_pages enable row level security;

create policy "Allow public read access to crawled_pages"
    on crawled_pages for select to public using (true);

create or replace function match_crawled_pages (
    query_embedding vector(1536),
    match_count int default 10,
    filter jsonb DEFAULT '{}'::jsonb,
    source_filter text DEFAULT NULL
) returns table (
    id bigint,
    url varchar,
    chunk_number integer,
    content text,
    metadata jsonb,
    source_id text,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        id, url, chunk_number, content, metadata, source_id,
        1 - (crawled_pages.embedding <=> query_embedding) as similarity
    from crawled_pages
    where metadata @> filter
        AND (source_filter IS NULL OR source_id = source_filter)
    order by crawled_pages.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- ── Code examples ─────────────────────────────────────────────────────────────

create table if not exists code_examples (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    summary text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number),
    foreign key (source_id) references sources(source_id)
);

create index if not exists idx_code_examples_embedding on code_examples using ivfflat (embedding vector_cosine_ops);
create index if not exists idx_code_examples_metadata on code_examples using gin (metadata);
create index if not exists idx_code_examples_source_id on code_examples (source_id);

alter table code_examples enable row level security;

create policy "Allow public read access to code_examples"
    on code_examples for select to public using (true);

create or replace function match_code_examples (
    query_embedding vector(1536),
    match_count int default 10,
    filter jsonb DEFAULT '{}'::jsonb,
    source_filter text DEFAULT NULL
) returns table (
    id bigint,
    url varchar,
    chunk_number integer,
    content text,
    summary text,
    metadata jsonb,
    source_id text,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        id, url, chunk_number, content, summary, metadata, source_id,
        1 - (code_examples.embedding <=> query_embedding) as similarity
    from code_examples
    where metadata @> filter
        AND (source_filter IS NULL OR source_id = source_filter)
    order by code_examples.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- ── Conversations ─────────────────────────────────────────────────────────────

create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    title text not null default 'New Chat',
    messages jsonb not null default '[]'::jsonb,
    created_at timestamp with time zone default timezone('utc', now()) not null,
    updated_at timestamp with time zone default timezone('utc', now()) not null
);

create index if not exists idx_conversations_updated_at on conversations (updated_at desc);

alter table conversations enable row level security;

create policy "Allow all access to conversations"
    on conversations for all to public
    using (true)
    with check (true);
