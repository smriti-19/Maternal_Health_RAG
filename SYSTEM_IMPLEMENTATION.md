# Maternal Health RAG + Multi-Stage Guardrails (Reference Implementation)

This repo contains a reference implementation of a maternal health chatbot pipeline with:
1) Retrieval-Augmented Generation (RAG)
2) Pre-LLM safety routing for emergencies
3) Post-LLM label-based gating for urgency


---

## High-Level Flow

For each user question:

1. **Pre-LLM Guardrails (Emergency-only)**
   - If the question looks like a crisis/emergency (NOW-*), return a fixed crisis template immediately.
   - Otherwise continue to RAG.

2. **RAG Retrieval**
   - Translate query to English (optional)
   - Expand the query (synonyms / related terms)
   - Retrieve from:
     - Dense vector store (FAISS + multilingual E5 embeddings)
     - BM25 retriever
   - Merge results using Reciprocal Rank Fusion (RRF)
   - Optional cross-encoder reranking

3. **LLM Answer with Label Instruction**
   - LLM is prompted to output a label on the **first line**:
     - NOW-MH, NOW-DV, NOW-MED, SAME-DAY, PASS
   - The rest of the response is the actual answer.

4. **Post-LLM Label Routing**
   - If label is NOW-*: return crisis template (no citations)
   - If label is SAME-DAY: return SAME-DAY template + additional info + citations
   - If PASS: return normal RAG answer with citations

---

## Files and Inputs

### Environment variables (required)
- `OPENAI_API_KEY`: API key for the ChatOpenAI model
- `FAISS_DB_DIR`: path to the FAISS index directory
- `BM25_PATH`: path to the serialized BM25 index (pickle)

### Questions file
- `questions.csv`
- First column is treated as the user query text.

### Policy file
- `guardrails/policy.yaml`
- Contains stage-detection regex patterns and guardrail configuration.

---

## Key Components (by code section)

### 1) Helper utilities

**normalize_text / dedup_docs**
- Normalizes doc text and removes duplicates based on normalized `page_content`.
- Used to reduce repeated context chunks.

**reciprocal_rank_fusion (RRF)**
- Combines rankings from dense retrieval and BM25 retrieval.
- Goal: improve recall and robustness by merging both signals.

**translate_query_to_english**
- Uses the LLM to translate the query into English before retrieval.

**expand_query**
- Uses the LLM to generate synonym/related-term expansions.
- Each expanded query is used for retrieval.

**parse_llm_label**
- Parses the LLM response first line for a valid label.
- If none is found, defaults to PASS.

---

### 2) GuardrailsRouter (Pre-LLM emergency check)

Purpose:
- Catch high-risk queries BEFORE retrieval/generation.

How it works:
- Loads `guardrails/policy.yaml`
- Detects a coarse stage (`maternal_pregnant`, `maternal_postpartum`, `newborn_0_2mo`) using regex patterns.
- Runs an **embedding similarity** check:
  - Encodes the user text (translated to English if needed)
  - Computes cosine similarity to a set of emergency exemplar phrases
  - If similarity > `EMERGENCY_THRESHOLD`, routes to emergency handling

Outputs (routing dict):
- `guardrail_action`: CRISIS or PASS_THROUGH
- `guardrail_crisis_type`: NOW-MH / NOW-DV / NOW-MED
- `guardrail_stage`: inferred stage bucket
- `emergency_sim`: similarity score (debugging)

Important:
- This stage is currently simplified: it focuses on CRISIS-level emergencies.
- SAME-DAY is primarily handled by post-LLM label gating (see below).

---

### 3) EnhancedMedicalRAG (Retrieval + Generation + Post-LLM routing)

#### Retrieval: `hybrid_retrieve`
- Expands query using LLM
- For each expanded query:
  - Dense retrieval: FAISS similarity search
  - Sparse retrieval: BM25 get relevant docs
- Merges results with RRF
- Adds a couple of extra “hint” retrievals:
  - If query mentions trimester/week/month/postpartum: adds "pregnancy stage info"
  - If query has symptoms keywords (pain/bleeding/etc): adds "maternal health concerns"
- Deduplicates documents
- Returns top_k docs

#### Reranking: `rerank_documents`
- Optional cross-encoder reranking to improve precision.
- Returns top `rerank_top_k`.

#### Generation: `rag_prompt`
- Prompts LLM to:
  - output a label first line (NOW-MH / NOW-DV / NOW-MED / SAME-DAY / PASS)
  - answer strictly using retrieved context
  - keep language consistent with user input
  - avoid certain disallowed content (drug brands, fetal sex selection, off-topic)

#### Post-LLM routing logic
After LLM response:
- If label in NOW-*:
  - return fixed crisis template (no citations)
- If label is SAME-DAY:
  - return SAME-DAY template + “Additional Information” + citations
- Else PASS:
  - return normal answer + citations

---

## What this system does NOT do (current limitations)

- Does not use numeric gestational age (e.g., "28 weeks") as a structured field.
  - Stage detection is regex-based and coarse.
- FAISS/BM25 artifacts are not included in the repo.
  - They must be built or provided separately.
- Not production hardened:
  - no caching
  - limited observability / metrics
  - no strict schema validation for inputs/outputs


