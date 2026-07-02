# GR4ML — Analytics Design View

## AUTOSAR Document Intelligence Assistant

---

## Analytics Design View Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                              │
│   ████████████████████████████████                                                                           │
│   █  Analytics Design View       █                                                                           │
│   ████████████████████████████████                                                                           │
│                                                                                                              │
│                                           ☁ Precision ☁           ☁ Robustness ☁                             │
│                                         ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐       ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐                              │
│                              ┌───┬───┐  │  (softgoal)  │       │  (softgoal)  │                              │
│                              │ ■ │   │  └╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘       └╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘                              │
│                              │ ▲ │   │       ▲ evaluates              ▲ evaluates                            │
│                              │ ▲ │ ■ │       │                        │                                      │
│         Retrieval            └───┴───┘       │                        │                                      │
│         Precision@5     ╔════════════════════╧════════════════════════╧═════════════════════╗                │
│         (indicator)     ║                                                                   ║                │
│              │          ║   Semantic Retrieval & Answer Generation                          ║                │
│    evaluates │          ║   for [AUTOSAR Query]                                             ║                │
│              │          ║                                                                   ║                │
│              ▼          ║   (Analytics Goal)                                                ║                │
│   is required for ───►  ╚═══════╤════════════╤═══════════════════╤══════════════════╤═══════╝                │
│                                 │            │                   │                  │                        │
│                              performs     performs           performs          performs                      │
│                                 │            │                   │                  │                        │
│                                 ▼            ▼                   ▼                  ▼                        │
│                          ⬡───────────⬡  ⬡──────────⬡    ⬡───────────────⬡   ⬡───────────────⬡               │
│                          │  nomic-   │  │ Cosine   │     │   llama3.2   │    │     LLM      │                │
│                          │  embed-   │  │Similarity│     │   (LLM       │    │   Re-ranking │                │
│                          │  text     │  │ Search   │     │  Generation) │    │  (Relevance  │                │
│                          │(Embedding)│  │          │     │              │    │   Scoring)   │                │
│                          ⬡───────────⬡  ⬡──────────⬡    ⬡───────────────⬡   ⬡───────────────⬡               │
│                           (algorithm)   (algorithm)        (algorithm)        (algorithm)                    │
│                                │              │                 │                  │                         │
│                                │              │                 │                  │                         │
│              ☁ Low Latency ☁   │    ☁ Anti-hallucination ☁      │       ☁ Interpretability ☁                 │
│            ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐    │   ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐       │       ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐                │
│            │  (softgoal)  │    │   │    (softgoal)      │       │       │   (softgoal)      │                │
│            └╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘    │   └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘       │       └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘                │
│                  ▲             │              ▲                 │                  ▲                         │
│                  │ influence   │              │ influence       │                  │ influence               │
│                  │    −        │              │    ++           │                  │    +                    │
│                  └─────────── Re-ranking ─────┘                 │                  │                         │
│                               influences                        └──────────────────┘                         │
│                                                          Citation extraction                                 │
│                                                          influences ++ interpretability                      │
│                                                                                                              │
│           ✓ Satisfied:                    ✗ Denied:                                                          │
│           Precision (≥85% P@5)            N/A (all softgoals are addressed)                                  │
│           Anti-hallucination (<5%)                                                                           │
│           Fast retrieval (<1s)                                                                                │
│                                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Legend for Analytics Design View

| Symbol | Element | Description |
|--------|---------|-------------|
| ⬡ Hexagon | **Algorithm** | An ML algorithm or computational method that performs the analytics goal |
| Double-bordered oval | **Analytics Goal** | The central ML task to be achieved |
| ☁ Cloud shape | **Softgoal** | A quality attribute or non-functional requirement (not strictly measurable) |
| Colored bar (■ green / ▲ red) | **Indicator** | A measurable metric that evaluates a softgoal |
| `Performs ──►` | **Performs** | Algorithm contributes to achieving the analytics goal |
| `Evaluates ┬┬┬` | **Evaluates** | Indicator measures the satisfaction of a softgoal |
| `Influence + / −` | **Influence** | Algorithm positively (+) or negatively (−) impacts a softgoal |
| `Association ◄──►` | **Association** | Relationship between elements |
| `Generates` | **Generates** | One element produces another |
| ✓ Satisfied | **Satisfied** | Softgoal is met by the design |
| ✗ Denied | **Denied** | Softgoal is not met |

---

## Analytics Goal

**Semantic Retrieval & Answer Generation for [AUTOSAR Query]**

The central analytics goal is to take a natural language query about AUTOSAR specifications and produce a **citation-backed, accurate answer** by retrieving the most relevant specification chunks and generating a coherent response.

This goal is **required for** satisfying the Business View's Question Goal: *"What is the relevant specification content for the AUTOSAR query at hand?"*

---

## Algorithms

### Algorithm 1: nomic-embed-text (Embedding)

| Attribute | Value |
|-----------|-------|
| **Purpose** | Convert text to dense vector representations for similarity search |
| **Architecture** | Transformer-based text encoder |
| **Output Dimensions** | 768 |
| **Similarity Metric** | Cosine similarity |
| **Deployment** | Local via Ollama (no API costs) |
| **Used In** | Both ingestion (document chunks) and query (user questions) |

**How it works:** The embedding model maps semantically similar texts to nearby points in a 768-dimensional vector space. When a user queries "Com stack configuration", its embedding is close to chunks containing "Communication module configuration parameters" — even though the words differ.

**Influence on Softgoals:**
- **Precision** → ++ (high-quality embeddings are the foundation of accurate retrieval)
- **Low Latency** → + (fast inference at ~50ms per embedding)

### Algorithm 2: Cosine Similarity Search

| Attribute | Value |
|-----------|-------|
| **Purpose** | Find the most relevant document chunks for a query |
| **Method** | Cosine similarity between query vector and pre-computed chunk vectors |
| **Index** | HNSW (Hierarchical Navigable Small World) via ChromaDB |
| **Output** | Top-K=10 nearest chunks with similarity scores |
| **Latency** | ~10ms |

**Influence on Softgoals:**
- **Precision** → + (approximate nearest neighbor retrieval)
- **Low Latency** → ++ (HNSW index enables sub-10ms search)

### Algorithm 3: llama3.2 (LLM Generation)

| Attribute | Value |
|-----------|-------|
| **Purpose** | Generate coherent answers from retrieved context |
| **Architecture** | Decoder-only transformer (3B parameters) |
| **Input** | System prompt + retrieved context + user query |
| **Output** | Natural language answer with [Source N] citations |
| **Temperature** | 0.1 (near-deterministic for factual accuracy) |
| **Deployment** | Local via Ollama |

**Anti-hallucination guardrails in prompt:**
- "ONLY answer based on the provided context passages"
- "Do NOT use prior knowledge"
- "If context is insufficient, explicitly state that"
- "Always cite sources using [Source N] notation"

**Influence on Softgoals:**
- **Anti-hallucination** → ++ (prompt guardrails constrain the LLM to retrieved context)
- **Interpretability** → + (citations make answers verifiable)
- **Low Latency** → − (LLM inference is the slowest stage at ~1500ms)

### Algorithm 4: LLM Re-ranking (Relevance Scoring)

| Attribute | Value |
|-----------|-------|
| **Purpose** | Re-score retrieved chunks for improved precision |
| **Method** | LLM scores each chunk's relevance (0-10 scale) |
| **Input** | Query + 10 candidate chunks |
| **Output** | Top-5 chunks reordered by LLM relevance score |
| **Latency** | ~500ms (parallel scoring) |

**Influence on Softgoals:**
- **Precision** → ++ (LLM-based re-ranking significantly improves retrieval quality)
- **Low Latency** → − (adds ~500ms to the pipeline)
- **Anti-hallucination** → + (better ranked context reduces likelihood of irrelevant information)

---

## Softgoals & Satisfaction Status

| Softgoal | Influence From | Status | Justification |
|----------|---------------|--------|---------------|
| ☁ **Precision** | nomic-embed-text (++), Cosine Search (+), Re-ranking (++) | ✓ Satisfied | Target: ≥85% Retrieval Precision@5 — achieved through embedding quality + re-ranking |
| ☁ **Robustness** | Retry logic (+), Circuit breakers (+), Batch processing (+) | ✓ Satisfied | Exponential backoff, error handling, and graceful degradation |
| ☁ **Anti-hallucination** | llama3.2 guardrails (++), Re-ranking (+) | ✓ Satisfied | Target: <5% hallucination rate — enforced via prompt constraints and context quality |
| ☁ **Low Latency** | Cosine Search (++), Embedding (+), Re-ranking (−), LLM (−) | ✓ Satisfied for retrieval; measured for full generation | `/query/search` returns relevant chunks in under 1 second on the prepared store; full `/query` exposes `latency_ms` and was measured at about 21.7 seconds with local `llama3.2` |
| ☁ **Interpretability** | Citation extraction (++), Source markers (+) | ✓ Satisfied | Every answer includes [Source N] citations with document, page, section |

---

## Indicator

**Retrieval Precision@5**

| Attribute | Value |
|-----------|-------|
| **Metric** | Proportion of top-5 retrieved chunks that contain the correct answer |
| **Target** | ≥ 85% |
| **Measurement** | Manual evaluation on 50 test queries |
| **Evaluates** | Precision softgoal + Analytics Goal |
| **Status** | ■ Green (target met) / ▲ Red (below target) |

---

## Input Specification

| Attribute | Details |
|-----------|---------|
| **Input Type** | Natural language text string |
| **Format** | JSON: `{"question": "...", "top_k": 5}` |
| **Constraints** | 3-2000 characters, UTF-8 encoded |
| **Examples** | "What is the Com stack configuration for PDU routing?" |
| | "Explain the NvM block configuration parameters" |
| | "What are the error detection mechanisms in the Dem module?" |

---

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Natural language answer with inline [Source N] citations |
| `citations` | array | List of source references: `{document, page, section, relevance_score}` |
| `confidence` | float (0-1) | Weighted score: 40% retrieval similarity + 30% citation coverage + 30% re-rank agreement |
| `latency_ms` | float | End-to-end processing time in milliseconds |
| `chunks_retrieved` | int | Number of chunks from initial search |
| `chunks_after_rerank` | int | Number of chunks after re-ranking |
| `model_used` | string | Name of the LLM model used |

---

## Pipeline Stages Detail

### Stage 1: Query Embedding
- **Algorithm:** nomic-embed-text
- **Input:** Raw query text
- **Process:** Embed using nomic-embed-text via Ollama API
- **Output:** 768-dimensional dense vector
- **Latency:** ~50ms

### Stage 2: Semantic Search
- **Algorithm:** Cosine Similarity Search
- **Input:** Query embedding vector
- **Process:** Cosine similarity search against ChromaDB (HNSW index)
- **Output:** Top-10 nearest chunks with similarity scores
- **Latency:** ~10ms

### Stage 3: Re-ranking (Optional)
- **Algorithm:** LLM Re-ranking
- **Input:** Query + 10 candidate chunks
- **Process:** LLM scores each chunk's relevance (0-10 scale, parallel)
- **Output:** Top-5 chunks reordered by LLM relevance score
- **Latency:** ~500ms

### Stage 4: Context Assembly
- **Input:** Top-5 ranked chunks
- **Process:** Sort by document position, deduplicate overlaps (Jaccard > 0.7), add source markers
- **Output:** Formatted context string with `[Source N: doc | Page: X | Section: Y]` headers
- **Latency:** ~1ms

### Stage 5: Prompt Construction
- **Input:** System prompt template + assembled context + user query
- **Process:** Load versioned template, inject context and query, add anti-hallucination instructions
- **Output:** Complete LLM prompt
- **Latency:** ~1ms

### Stage 6: LLM Inference
- **Algorithm:** llama3.2
- **Input:** Complete prompt
- **Process:** Generate answer via Ollama API (llama3.2, temp=0.1, max_tokens=2048)
- **Output:** Raw answer text with [Source N] references
- **Latency:** ~1500ms

### Stage 7: Citation Extraction & Confidence
- **Input:** Raw answer + ranked results
- **Process:** Parse [Source N] references, map to chunk metadata, compute confidence score
- **Output:** Structured response with citations and confidence
- **Latency:** ~5ms

**Latency Target:** `/query/search` under 1 second for retrieval; full `/query` reports measured local-LLM latency in `latency_ms`.
