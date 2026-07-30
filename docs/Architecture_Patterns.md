# System Architecture & Architectural Patterns

## AUTOSAR Document Intelligence Assistant — Objective 2

---

## Q5: System Architecture Diagram (HLD)

The system architecture diagram is provided as the HLD image:

![AUTOSAR Document Intelligence Assistant - HLD](../ChatGPT%20Image%20Jun%209,%202026,%2012_19_34%20AM.png)

### Architecture Overview

The HLD comprises **5 core layers** plus **cross-cutting concerns**:

| Layer | Type | Components | Technology |
|-------|------|-----------|-----------|
| **1. Input Sources** | Data Entry | AUTOSAR PDFs, User NL Queries | File uploads, REST API |
| **2. Ingestion & Indexing Pipeline** | ML (One-time) | Document Parser → Chunker → Embedder → Enricher → Indexer | PyMuPDF, nomic-embed-text, ChromaDB |
| **3. Retrieval & Reasoning Engine** | ML (Runtime) | Query Understanding → Semantic Search → Re-ranking → Context Assembly → Prompt Construction → LLM Inference | ChromaDB, llama3.2 via Ollama |
| **4. Response & Output** | Non-ML | Answer + Citations + Source Chunks | FastAPI JSON response |
| **5. Feedback & Learning Loop** | ML Support | User Feedback → Analytics → Continuous Improvement | JSON store, metrics |
| **API Layer** | Non-ML | POST /ingest, POST /query, GET /health, POST /feedback | FastAPI, uvicorn |
| **Cross-Cutting** | Non-ML | Security, Scalability, Reliability, Observability, Compliance | structlog, circuit breaker, heartbeat |

### ML vs Non-ML Components

```
ML Components:
├── Embedding Model (nomic-embed-text) — learned text representations
├── Vector Similarity Search (HNSW index) — approximate nearest neighbor
├── LLM Re-ranker (llama3.2) — relevance scoring
├── LLM Generator (llama3.2) — answer generation
└── Feedback Loop — continuous improvement signals

Non-ML Components:
├── PDF Parser — rule-based text extraction
├── Chunker — algorithmic text splitting
├── API Gateway — REST endpoint routing
├── Health Monitor — heartbeat tactic
├── Metadata Store — JSON persistence
├── Correlation ID Middleware — request tracing
└── Circuit Breaker — fault tolerance
```

---

## Q6 & Q7: Architectural Patterns (Selection, Justification & Implementation)

---

### Pattern 1: Retrieval-Augmented Generation (RAG)

#### What is RAG?

RAG is an architectural pattern that **decouples knowledge from reasoning** in LLM-based systems:

1. **Knowledge** is stored externally in a searchable database (vector store)
2. **Retrieval** finds relevant knowledge for a given query
3. **Augmentation** injects retrieved knowledge into the LLM's prompt
4. **Generation** produces an answer grounded in the retrieved knowledge

```
┌──────────┐     ┌────────────────┐     ┌─────────────┐     ┌──────────┐
│  User    │     │   Retriever    │     │  Augmenter   │     │ Generator│
│  Query   │────▶│  (Vector DB    │────▶│  (Context    │────▶│  (LLM)   │
│          │     │   Search)      │     │   Builder)   │     │          │
└──────────┘     └───────┬────────┘     └─────────────┘     └────┬─────┘
                         │                                        │
                  ┌──────▼────────┐                        ┌─────▼──────┐
                  │  ChromaDB     │                        │  Grounded  │
                  │  (Embeddings) │                        │  Answer +  │
                  │               │                        │  Citations │
                  └───────────────┘                        └────────────┘
```

#### Why RAG for This Application?

| Reason | Explanation |
|--------|-------------|
| **No fine-tuning needed** | AUTOSAR specs change between versions (R22-11, R23-11). RAG updates knowledge by re-ingesting PDFs — no model retraining required |
| **Eliminates hallucination** | By grounding answers in retrieved documents, the LLM cannot fabricate information not in the specs |
| **Explainability** | Every answer can cite its source (document, page, section) — critical for safety-critical automotive domain |
| **Cost-effective** | Works with any off-the-shelf LLM — no expensive fine-tuning GPU time or proprietary API costs |
| **Updatable knowledge** | New AUTOSAR specifications are added via `/ingest` without any model changes |

#### Implementation

**Key files implementing the RAG pattern:**

| Component | File | Purpose |
|-----------|------|---------|
| Retriever | [search.py](../app/services/retrieval/search.py) | Embed query → cosine similarity search in ChromaDB |
| Re-ranker | [reranker.py](../app/services/retrieval/reranker.py) | LLM scores each chunk's relevance (0-10) |
| Augmenter | [context.py](../app/services/retrieval/context.py) | Assemble ranked chunks with source markers |
| Prompt Builder | [prompt.py](../app/services/inference/prompt.py) | Inject context into versioned prompt template |
| Generator | [llm.py](../app/services/inference/llm.py) | Generate answer via Ollama (llama3.2) |
| Citation Extractor | [citations.py](../app/services/inference/citations.py) | Parse [Source N] from answer, compute confidence |
| Orchestrator | [router.py](../app/services/retrieval/router.py) | Wire all stages together in POST /query |

**Core RAG flow (from `router.py`):**

```python
# POST /query — Full RAG Pipeline
async def query_documents(request: QueryRequest):
    # 1. RETRIEVE: Semantic search
    search_results = await semantic_search(query=request.question, top_k=10)
    
    # 2. RE-RANK: LLM-based relevance scoring
    ranked_results = await rerank(query=request.question, search_results=search_results, top_n=5)
    
    # 3. AUGMENT: Build context with source markers
    context = assemble_context(ranked_results)
    
    # 4. BUILD PROMPT: System prompt + context + query
    system_prompt = get_system_prompt()
    query_prompt = build_query_prompt(question=request.question, context=context)
    
    # 5. GENERATE: LLM produces grounded answer
    answer = await generate_completion(prompt=query_prompt, system_prompt=system_prompt)
    
    # 6. EXTRACT: Citations and confidence
    citations = extract_citations(answer, ranked_results)
    confidence = compute_confidence_score(ranked_results, citations)
    
    return QueryResponse(answer=answer, citations=citations, confidence=confidence)
```

---

### Pattern 2: Microservices Architecture

#### What is Microservices Architecture?

The system is decomposed into **independent, loosely-coupled services** that communicate via REST APIs. Each service:
- Has a single responsibility
- Can be developed, tested, and deployed independently
- Exposes its own health endpoint
- Has independent error handling and retry logic

```
┌──────────────────────────────────────────────────────┐
│                    API GATEWAY                        │
│  FastAPI (main.py) — Routing, CORS, Correlation IDs  │
└──────┬─────────────────┬────────────────┬────────────┘
       │                 │                │
       ▼                 ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  INGESTION   │  │  RETRIEVAL   │  │    LLM       │
│  SERVICE     │  │  SERVICE     │  │  INFERENCE   │
│              │  │              │  │  SERVICE     │
│ POST /ingest │  │ POST /query  │  │              │
│ GET /ingest/ │  │ POST /query/ │  │ (Internal)   │
│   documents  │  │   search     │  │              │
│              │  │              │  │              │
│ • parser.py  │  │ • search.py  │  │ • llm.py     │
│ • chunker.py │  │ • reranker.py│  │ • prompt.py  │
│ • embedder.py│  │ • context.py │  │ • citations  │
│ • router.py  │  │ • router.py  │  │   .py        │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                │
       ▼                 ▼                ▼
┌──────────────────────────────────────────────────────┐
│                 SHARED DATA LAYER                     │
│  ChromaDB (Vector Store) │ Metadata (JSON) │ Feedback│
└──────────────────────────────────────────────────────┘
```

#### Why Microservices for This Application?

| Reason | Explanation |
|--------|-------------|
| **Independent scaling** | Ingestion is CPU-bound (PDF parsing, embedding); Retrieval is I/O-bound (ChromaDB query); LLM Inference is GPU-bound. Each has different resource needs. |
| **Fault isolation** | If the LLM service fails (Ollama crash, model unloaded), document ingestion continues to work. The circuit breaker prevents cascading failures. |
| **Independent development** | Team members can work on different services simultaneously (one on ingestion, another on retrieval). |
| **Technology flexibility** | Each service can use the optimal tools for its domain. |
| **Heartbeat tactic** | Each service exposes `/health` for monitoring — a direct implementation of the heartbeat architectural tactic from M3. |

#### Implementation

**Service boundaries defined in `main.py`:**

```python
# main.py — API Gateway wires all microservice routers
app.include_router(health_router)      # GET /health, GET /health/metrics
app.include_router(ingestion_router)   # POST /ingest/upload, POST /ingest/local
app.include_router(retrieval_router)   # POST /query, POST /query/search  
app.include_router(feedback_router)    # POST /feedback, GET /feedback/analytics
```

**Heartbeat tactic implementation (from `health.py`):**

```python
async def heartbeat_loop():
    """Background task: check all service health every 30 seconds."""
    while True:
        await asyncio.gather(
            check_ollama_api(),        # Is Ollama reachable?
            check_embedding_model(),   # Is nomic-embed-text loaded?
            check_llm_model(),         # Is llama3.2 loaded?
            check_vector_store(),      # Is ChromaDB accessible?
        )
        # Log any unhealthy services
        for name, health in services.items():
            if health.status != ServiceStatus.HEALTHY:
                logger.warning("service_unhealthy", service=name, 
                             failures=health.consecutive_failures)
        await asyncio.sleep(30)  # Check every 30 seconds
```

**Circuit breaker pattern (from `llm.py`):**

```python
class CircuitBreaker:
    """Fail fast after consecutive LLM failures."""
    
    def can_proceed(self) -> bool:
        if self.state == "OPEN":
            # After reset_seconds, try one test request
            if time.time() - self.last_failure > self.reset_seconds:
                self.state = "HALF_OPEN"
                return True
            return False  # Fail fast — don't wait for timeout
        return True
    
    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= 5:
            self.state = "OPEN"  # Stop trying for 60 seconds
```

**Cross-service correlation IDs (from `main.py`):**

```python
@app.middleware("http")
async def correlation_id_middleware(request, call_next):
    """Inject correlation ID for cross-service request tracing."""
    corr_id = request.headers.get("X-Correlation-ID", generate_correlation_id())
    correlation_id_var.set(corr_id)  # Available to all loggers
    
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    return response
```

---

## Mapping to M3/M4 Module Topics

| Module Topic | Implementation in This Project |
|-------------|-------------------------------|
| **M3: RAG Model for LLMs** | Core architecture — retrieval → augmentation → generation pipeline |
| **M3: Microservices Pattern** | Three isolated services (Ingestion, Retrieval, LLM Inference) with independent routers |
| **M3: Feature Store Pattern** | ChromaDB as the vector store — pre-computed embeddings cached for fast retrieval |
| **M3: Heartbeat Tactic** | Background task checking service health every 30s with consecutive failure tracking |
| **M4: API Design** | REST endpoints: POST /ingest, POST /query, GET /health, POST /feedback |
| **M4: Logging & Debugging** | structlog with JSON output and per-request correlation IDs |
| **M4: Version Control** | Prompt templates versioned in `prompts/` directory; embedding model version tracked per document |
| **M4: Code Practices** | Type hints, docstrings, Pydantic validation, pytest test suite (46 tests) |
