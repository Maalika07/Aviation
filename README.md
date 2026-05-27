# Project

---

## common.py
Summary
- Core utilities, data models, dataset loader and synthetic feature derivation for the airport dataset.
- Provides stable I/O for KnowledgeDocument objects used across the pipeline.

Key exports
- Constants
  - PROJECT_DIR, DATASET_PATH, ARTIFACTS_DIR, OPERATION_ID_COLUMN
- Data classes
  - KnowledgeDocument(doc_id, title, content, source_type, metadata)
    - combined_text()  title + content
    - to_dict() / from_dict()  JSON (de)serialization
  - RetrievedDocument(doc: KnowledgeDocument, score: float, retriever: str, rank: int)
- Dataset helpers
  - load_airport_dataset()  pd.DataFrame
    - Validates CSV existence, builds derived columns, ensures unique operation_id.
- Internal helpers (used by load_airport_dataset)
  - _derive_columns(df: DataFrame)  DataFrame
    - Adds scheduled_departure, airline extraction, origin/destination (hash-mapped), terminal/gate_id, runway_id, weather-derived fields (temperature, visibility, wind), runway_congestion, gate_occupancy_pct, baggage/staff availability, maintenance_flag, actual_turnaround_minutes, departure_delay_minutes, disruption_risk/target.
    - Uses deterministic RNG (seeded) and small noise to make features realistic.
  - _build_operation_ids(df)  Series of OP-000001 style IDs
- Persistence / text utils
  - save_documents(path: Path, docs: list[KnowledgeDocument])  JSON file
  - load_documents(path: Path)  list[KnowledgeDocument]
  - normalize_text(text: str)  normalized string
  - tokenize(text: str)  list[str]

Techniques & rationale
- Dataclasses for compact, typed document models  easy JSON serialization and consistent metadata shape.
- Deterministic pseudo-random generation (numpy default_rng with fixed seed) to ensure reproducible synthetic perturbations.
- Feature engineering inside _derive_columns centralizes dataset enrichment so downstream components can rely on consistent fields.
- Tokenization and normalization utilities standardize text processing across retrievers.
- Saving knowledge base artifacts to ARTIFACTS_DIR enables caching and faster iteration.

How it is used
- All other modules import KnowledgeDocument / RetrievedDocument and dataset loader. The format and metadata keys are expected by the KB builder, retrievers, graph, RL, and agent workflow.

---

## phase_1_knowledge_base.py
Summary
- Builds the airport knowledge base (KB) consisting of:
  - Static SOP/policy documents
  - Aggregated briefs (per weather / gate / runway / terminal)
  - Historical operation records (one document per dataset row)
- Persists KB to ARTIFACTS_DIR/airport_knowledge_base.json and validates cache.

Key exports
- KNOWLEDGE_PATH (artifact path)
- build_static_documents()  list[KnowledgeDocument]
  - Returns a list of static SOP docs (title/content/metadata).
- build_historical_documents()  list[KnowledgeDocument]
  - Iterates dataset rows, produces rich structured textual "Operational linkage record" with metadata including operation_id, flight_id, runway_id, gate, resource metrics, etc.
  - doc_id uses prefix "history::OP-..." and includes ISO scheduled_departure in title.
- build_aggregated_brief_documents()  list[KnowledgeDocument]
  - Computes groupby statistics and creates concise briefs for weather, gate, runway, terminal (doc_id like "brief::weather::rain").
- build_knowledge_base(refresh: bool=False)  list[KnowledgeDocument]
  - Loads cached KB if valid; otherwise rebuilds static + aggregated + historical docs and saves JSON.
- _knowledge_base_is_valid(docs) internal validation helper.

Techniques & rationale
- Template-driven textual documents: embedding structured, consistent sections makes retrieval easier and allows retrievers / LLMs to find signals (e.g., explicit operation ids).
- Aggregation (Pandas groupby) to precompute briefs used for analytical queries  reduces on-the-fly computation and provides short evidence documents for RAG.
- Caching to JSON for fast startup: KB rebuild is expensive (per-row docs), so caching prevents repeated rebuilds.
- Metadata-rich documents: including operation_id, flight_id, resource fields  enables exact-match boosting and graph linking.

How its used
- AirportAgenticWorkflow calls build_knowledge_base(refresh=...) at startup to obtain the corpus for retrievers.
- Historical docs provide evidence for RAG and join to RL/graph outputs by operation_id.

---

## phase_2_retrievers.py
Summary
- Two complementary retrieval backends:
  - VectorRetriever  semantic vector retrieval (hashing or optional dense SBERT)
  - KeywordRetriever  classic token / TF-IDF style retrieval
- Utilities: query expansion (domain synonyms), environment checks, exact-match boosting.

Key exports & members
- expand_query(query: str)  str
  - Extends query with domain synonyms when relevant tokens are present.
- VectorRetriever(docs: Iterable[KnowledgeDocument])
  - Builds either:
    - Dense sentence-transformer embeddings (if ENABLE_LOCAL_DENSE_RETRIEVER or running in Colab with permission) OR
    - Sparse HashingVectorizer (sklearn) index (default).
  - search(query, top_k)  list[RetrievedDocument]
    - If dense, dot product between doc embeddings and query embedding; else sparse matrix multiplication.
    - Adds _exact_match_bonus for flight/gate/runway-like tokens.
  - HashingVectorizer params: ngram_range=(1,2), n_features=4096, alternate_sign=False, l2 normalization.
- KeywordRetriever(docs: Iterable[KnowledgeDocument])
  - Precomputes token lists and counters, computes IDF via log formula.
  - search(query, top_k) uses TF weighting (1+log(tf)) * idf and exact-match bonus.

Key function: _exact_match_bonus(doc, query_tokens)  float
- Adds score boosts for flight-like tokens (alphanumeric), tokens with separators (/ or -), and domain keywords to prefer precise matches.

Techniques & rationale
- Hybrid design (dense + sparse + keyword) provides robustness:
  - HashingVectorizer is memory-efficient and avoids storing a full vocabulary (good for constrained environments).
  - Optional dense SBERT (SentenceTransformer) is used when available for higher semantic quality.
  - Keyword retriever offers precise term frequency signals and IDF weighting useful for structured tokens (flight IDs).
- Query expansion with domain synonyms increases recall for domain-specific phrasing.
- Exact-match boosting ensures entities like flight IDs, gates, runway identifiers are surfaced strongly even if vector similarity is imperfect.

How its used
- AirportAgenticWorkflow initializes both retrievers and uses them for hybrid retrieval; results are fused via RRF.

Environment variables
- ENABLE_LOCAL_DENSE_RETRIEVER, COLAB_USE_DENSE_RETRIEVER, DENSE_RETRIEVER_MODEL, ALLOW_EMBEDDING_DOWNLOAD influence dense backend selection.

---
## phase_3_rrf.py
Summary
- Implements Reciprocal Rank Fusion (RRF) to combine multiple ranked lists (retriever outputs) into one fused ranking.

Function
- reciprocal_rank_fusion(ranked_lists: list[list[RetrievedDocument]], k: int = 60, top_k: int = 8) -> list[RetrievedDocument]
  - Sums 1 / (k + rank) per document across result lists.
  - Tracks representative document object and retriever trace (joined retriever names).
  - Returns top_k fused results with fused scores and retriever provenance.

Techniques & rationale
- RRF is simple, robust, and unsupervised; it balances rank positions rather than raw scores which may be on different scales across retrievers.
- Deduplication by doc_id and retriever provenance are preserved for explainability.

How its used
- AirportAgenticWorkflow calls RRF with vector_hits, keyword_hits, and graph_hits to produce fused_evidence used for LLM prompting.

---

## phase_4_graph_intelligence.py
Summary
- In-memory lightweight "causal" graph over operations using dictionary-based adjacency indexed by attributes (flight_id, gate_id, runway_id, weather_condition, terminal).
- Responsible for resolving natural-language mentions of specific operations and for graph-walk retrieval.

Key class: AirportGraphIntelligence
- __init__(): loads dataset, builds _op_to_row and indices for flight/gate/runway/weather/terminal.
- _build_graph(): populates mapping dicts from dataframe rows.
- _best_in(ops, key="departure_delay_minutes") -> operation id with maximal key (used as tie-breaker).
- _parse_thresholds(query) -> dict thresholds (e.g., departure_delay_minutes, runway_congestion).
- _filter_by_thresholds(thresholds) -> filtered DataFrame.
- resolve_operation(query) -> Optional[str]
  - Resolution priority:
    1. Flight ID regex (e.g., "UK439")
    2. Gate ID (T2-G26)
    3. Exact runway ID (must actually exist in dataset)
    4. Numeric thresholds (delay > X)
    5. Superlatives (which runway/terminal/gate worst)
    6. Weather keyword
    7. Global highest-delay fallback
- resolve_by_context(query) -> dict with op_id and small context (delay, congestion)
- search(query, top_k) -> list[RetrievedDocument]
  - Seeds from resolved op; graph-walk scoring: add scores for same gate (0.6), runway (0.4), weather (0.3); rank and return documents (doc_id prefixed "graph::").

Techniques & rationale
- Lightweight graph uses attribute-based adjacency (no external DB) for very fast startup and simple graph traversals that capture operational proximity (same gate/runway/weather).
- Deterministic, explainable heuristics for resolution and scoring 1 good for operations where explicit identifiers or structured attributes are present.
- Seed + edge scoring yields related evidence for contextual RAG.

How its used
- Used both to resolve user queries to an operation ID (feeding the RL module) and to surface graph-linked evidence to the fusion step.

---

## phase_5_rl_optimizer.py
Summary
- A compact tabular Q-learning agent (AirportResourceRL) that recommends operational actions to reduce delay per operation.

Key exports
- ACTIONS  list of available discrete actions (e.g., "reroute_ground_traffic", "prioritize_fueling").
- AirportResourceRL:
  - __init__(): loads dataset, initializes q_table dict keyed by discretized state tuples.
  - _discretize(delay, congestion, weather_risk, maintenance, fuel, staff_pct) -> 6	1tuple state
    - Buckets reduce continuous space to manageable discrete table indices.
  - _simulate_reduction(action_idx, features) -> float
    - Domain-specific simulation function estimating expected delay reduction per action based on current features (weather, congestion, fuel, staff, pax, gate occupancy).
  - train(episodes=2000): tabular Q-learning (alpha=0.10, gamma=0.90, eps-greedy eps=0.30)
    - Samples dataset rows, applies epsilon-greedy, computes reward = reduction / max(delay,1), TD update to q_table.
  - recommend(operation_id) -> dict with action, policy_mode ("rl_trained" or "heuristic"), reason, projected_delay, expected_delay_reduction

Techniques & rationale
- Tabular Q-learning with discretization:
  - Keeps implementation simple and transparent; effective when feature space can be bucketed.
  - Discretization trades granularity for tractability (memory and sample efficiency).
- Simulation-based immediate reward (expected delay reduction) lets training happen without a live environment.
- Heuristic fallback ensures recommendations even for unseen states or before training.
- Action-specific parametric formulas encapsulate domain knowledge into reward dynamics.

How its used
- AirportAgenticWorkflow trains this at startup and uses recommend(operation_id) to supply an RL action and explainable projected improvements included in LLM context.

---

## phase_6_agent_workflow.py
Summary
- Orchestrator that integrates KB, retrievers, RRF fusion, graph intelligence, RL optimizer, and LLM (Groq client).
- Implements query classification, context assembly, hybrid retrieval, and LLM prompt templating.

Key class: AirportAgenticWorkflow
- __init__(refresh_knowledge=False)
  - Loads dataset and KB, initializes VectorRetriever and KeywordRetriever, builds graph, trains RL (2000 episodes), and creates Groq client using GROQ_API_KEY.
- _classify_query(query) -> str
  - Classifies into 'threshold', 'analytical', 'specific', 'scenario', or 'general' using regex and keywords.
- _build_threshold_context(query) -> str
  - For threshold queries, filters dataset and produces a compact summary table string (count, top 5, weather breakdown, maintenance count).
- _build_analytical_context(query) -> str
  - Builds aggregate stats for runway / terminal / weather / gates depending on query.
- _hybrid_retrieve(query) -> fused hits
  - Calls vector, keyword, and graph retrievers and fuses via reciprocal_rank_fusion.
- _generate_answer(...) -> str
  - Builds RAG evidence snippet, RL block, constructs structured user message based on query_type, calls Groq API chat completion with _SYSTEM_PROMPT.
- run(query) -> dict
  - End-to-end: classifies, retrieves, resolves operation, gets RL rec, builds extra context, generates final LLM answer; returns selection, RL rec, fused hits, final_answer.

Techniques & rationale
- Hybrid pipeline: combines semantic vectors, keyword signals, and graph context to maximize recall & precision.
- RRF fusion makes multisource fusion robust to scale mismatch.
- Query classification steers context assembly and ensures the LLM sees the right structured evidence for the question type.
- Template-based LLM prompt: constrains LLM to be direct, numeric, and operational  helps obtain concise operational answers.
- Explicit provenance: fused_hits and retriever names surface to user for explainability.

Configuration & env vars
- GROQ_API_KEY required for Groq client; _GROQ_MODEL default can be overridden via GROQ_MODEL.

How its used
- Main entrypoint for user queries (project.py / run.py call this). Returns structures ready for printing or programmatic use.

---

## Phase_7_voice_ai
Voice AI using Whisper

## project.py
Summary
- CLI entrypoint (argparse) demonstrating the agentic workflow (similar to run.py).
- Builds AirportAgenticWorkflow (optionally refresh KB), runs a single query (defaults to the highest-delay flight), prints summary: selected flight/op, RL recommendation, top fused evidence, and final answer.

Why present separately
- Provides a convenient script with slightly different printing / messaging. Useful as an example or alternative CLI wrapper.

Key commands
- default_query() returns a natural default question about the highest-delay flight.
- main() parses --query and --refresh-kb then runs workflow and prints results.

---

## run.py
Summary
- Another CLI driver (very similar to project.py) that:
  - Ensures project folder on sys.path, builds workflow, prints a structured multi-step startup log, runs query and prints detailed results.
- Slightly different messages but same end-to-end orchestration.

Why both exist
- Offers different CLI flavors / messages for interactive runs or unit runs. Either script can be used to exercise the pipeline.

---

## validate_project.py
Summary
- A suite of sanity checks and end-to-end validations to assert pipeline integrity.
- Useful for CI or local dev confidence that dataset, KB, retrievers, RRF, graph, RL, and workflow produce consistent outputs.

Key checks implemented
- Dataset schema and unique operation identifiers.
- Knowledge base uniqueness and that historical doc count equals dataset rows.
- Hybrid retriever sanity: VectorRetriever and KeywordRetriever return hits.
- RRF fusion deduplication and non-empty output.
- Graph intelligence: resolve and search for duplicated flight IDs.
- RL optimizer: train and recommend for resolved operation and verify recommendation structure.
- End-to-end: instantiate AirportAgenticWorkflow and run several scenario queries asserting final_answer, rl_recommendation, fused_hits exist.

Techniques & rationale
- Defensive asserts provide faster feedback during development and ensure invariants (e.g., one historical doc per dataset row).
- Reuses actual components so the validation exercises real code paths (KB, retrievers, graph, RL, workflow).

Usage
- run_validation() is called in __main__ when run directly. Good candidate for automated tests.

---

