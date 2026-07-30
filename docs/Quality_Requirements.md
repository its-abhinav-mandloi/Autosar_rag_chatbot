# Top 3 Quality Requirements

## AUTOSAR Document Intelligence Assistant

---

## Quality Requirement Selection Rationale

The three quality requirements below were selected based on:
1. **Domain criticality** — AUTOSAR is used in safety-critical automotive systems
2. **User impact** — what matters most to embedded systems engineers
3. **Architectural influence** — requirements that drive significant design decisions

---

## Quality Requirement 1: Accuracy (Retrieval Precision)

> **The system shall achieve retrieval precision@5 ≥ 85% for AUTOSAR domain queries.**

### Definition
Retrieval Precision@5 measures whether the top 5 retrieved document chunks actually contain the information needed to answer the query. For a query about "CAN Driver initialization API", at least 4 out of 5 retrieved chunks should be from the CAN Driver specification's initialization section.

### Justification

| Factor | Explanation |
|--------|-------------|
| **Primary value proposition** | The system's core purpose is providing **correct, relevant answers**. If retrieval quality is low, the LLM generates answers from irrelevant context, producing wrong technical information. |
| **Safety-critical domain** | AUTOSAR specifications govern software in vehicles — brake controllers, engine management, airbag systems. Incorrect technical information could lead to defective implementations with real safety consequences. |
| **Downstream impact** | Retrieval precision is the foundation of the RAG pipeline. Even the best LLM cannot produce correct answers from irrelevant context. |
| **Measurability** | Precision@5 is a well-defined metric that can be evaluated manually on a test set of 50 queries. |

### Architectural Impact

This quality requirement drives several design decisions:

1. **Semantic chunking by section headings** — Keeps related content together, improving retrieval relevance
2. **LLM-based re-ranking** — Adds a second relevance scoring stage beyond embedding similarity
3. **Metadata enrichment** — Attaching section headings and requirement IDs enables filtering
4. **Embedding model selection** — nomic-embed-text chosen for strong retrieval benchmark performance

### Measurement Plan

```
Precision@5 = (# of relevant chunks in top 5) / 5

Test Protocol:
1. Create 50 test queries with known ground-truth documents/sections
2. For each query, retrieve top-5 chunks
3. Human evaluator marks each chunk as relevant/irrelevant
4. Compute average Precision@5 across all queries
5. Target: ≥ 85% (i.e., ≥ 4.25 relevant chunks per query on average)
```

---

## Quality Requirement 2: Explainability (Citation Transparency)

> **Every answer shall include verifiable citations to source documents, sections, and page numbers. The hallucination rate shall be < 5%.**

### Definition
Explainability means the user can **trace every claim in the answer back to a specific source**. Citations include: document name, page number, and section heading. Hallucination rate measures the percentage of claims in answers that are not supported by the retrieved context.

### Justification

| Factor | Explanation |
|--------|-------------|
| **Verification requirement** | Engineers must verify answers against the original specification before using them in design decisions. An answer without citations is useless — engineers won't trust it. |
| **LLM hallucination risk** | Even the best LLMs can fabricate plausible-sounding but incorrect information. In the AUTOSAR domain, this could mean citing non-existent requirement IDs or wrong API parameters. |
| **Regulatory context** | Automotive software development often follows ISO 26262 (functional safety). Traceability of design decisions to specification requirements is a compliance requirement. |
| **User trust** | Engineers will only adopt the tool if they can verify its outputs. Citations build trust incrementally. |

### Architectural Impact

1. **Source markers in context** — Each chunk injected into the prompt is tagged with `[Source N: doc | Page: X | Section: Y]`
2. **Anti-hallucination prompt engineering** — System prompt explicitly instructs: "ONLY answer based on provided context" and "NEVER fabricate information"
3. **Citation extraction module** — Post-processes LLM output to parse [Source N] references and map them to original chunks
4. **Confidence scoring** — Combines retrieval similarity, citation coverage, and re-rank agreement into a 0-1 confidence score
5. **Implicit citation fallback** — If LLM doesn't cite sources explicitly, top-3 sources are included as "implicit" citations

### Measurement Plan

```
Hallucination Rate = (# of unsupported claims) / (# of total claims) × 100

Test Protocol:
1. Generate answers for 50 test queries
2. Human evaluator identifies all factual claims in each answer
3. For each claim, verify against the retrieved context:
   - Supported: claim can be found in or directly inferred from context
   - Unsupported: claim has no basis in the retrieved context
4. Compute hallucination rate
5. Target: < 5% (i.e., < 1 in 20 claims is unsupported)
```

---

## Quality Requirement 3: Availability (System Reliability)

> **The system shall maintain 99% uptime with graceful degradation when individual services fail.**

### Definition
Availability measures the percentage of time the system is operational and responsive. Graceful degradation means that if one component fails (e.g., the LLM service), other components (e.g., ingestion, health monitoring) continue to function.

### Justification

| Factor | Explanation |
|--------|-------------|
| **Daily workflow dependency** | Engineers use this tool during active development sprints. System downtime directly blocks their work and forces fallback to slow manual search. |
| **Multiple failure points** | The system depends on Ollama (embedding model + LLM), ChromaDB, and the FastAPI server — any can fail independently. |
| **Local deployment** | Unlike cloud services with built-in redundancy, this runs on a developer's machine. Hardware reboots, Ollama model unloading, and disk space issues are real risks. |
| **Service isolation** | A bug in the LLM inference should not crash the document ingestion pipeline. |

### Architectural Impact

1. **Microservices decomposition** — Ingestion, Retrieval, and Inference are isolated services with independent routers
2. **Heartbeat tactic** — Background task pings all services every 30 seconds and tracks consecutive failures
3. **Circuit breaker pattern** — After 5 consecutive LLM failures, the circuit breaker opens and fails fast for 60 seconds (prevents cascading timeouts)
4. **Health endpoints** — `GET /health` returns status of every dependency with timestamps
5. **Retry with exponential backoff** — Transient failures (network timeout, model loading) are retried 3 times with 2s/4s/8s delays
6. **Graceful degradation** — If LLM is down, `/health` reports "degraded" (not crashed), and ingestion continues to work

### Measurement Plan

```
Uptime = (total_time - downtime) / total_time × 100

Test Protocol:
1. Deploy system and run heartbeat monitoring for 7 days
2. Record all periods where /health returns "unavailable" for any critical service
3. Simulate failure scenarios:
   a. Stop Ollama → verify degraded status reported within 60s
   b. Kill ChromaDB process → verify health reports vector_store unavailable
   c. Restart Ollama → verify automatic recovery detected
4. Compute uptime percentage
5. Target: ≥ 99% (i.e., < 1.68 hours downtime per week)
```

---

## Quality Requirements Summary

| # | Quality Attribute | Requirement | Metric | Target | Primary Pattern |
|---|------------------|------------|--------|--------|-----------------|
| 1 | **Accuracy** | Retrieval precision for AUTOSAR queries | Precision@5 | ≥ 85% | RAG + Re-ranking |
| 2 | **Explainability** | Citation transparency, anti-hallucination | Hallucination Rate | < 5% | Prompt Engineering + Citation Extraction |
| 3 | **Availability** | Uptime with graceful degradation | System Uptime | ≥ 99% | Heartbeat Tactic + Circuit Breaker |

### Why These Three?

These three quality attributes form a **trust triangle** for the system:

```
        Accuracy
       /        \
      /   TRUST   \
     /              \
Explainability ─── Availability
```

- **Accuracy** ensures the system gives **correct answers**
- **Explainability** ensures users can **verify** those answers
- **Availability** ensures the system is **always there** when needed

Without any one of these, the system fails to deliver value:
- High accuracy without explainability → users can't trust it
- High accuracy without availability → users can't use it
- High availability without accuracy → system is reliably wrong
