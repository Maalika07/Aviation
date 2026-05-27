from __future__ import annotations

from typing import Iterable

try:
    from Project.common import OPERATION_ID_COLUMN, load_airport_dataset
    from Project.phase_1_knowledge_base import build_knowledge_base
    from Project.phase_2_retrievers import KeywordRetriever, VectorRetriever
    from Project.phase_3_rrf import reciprocal_rank_fusion
    from Project.phase_4_graph_intelligence import AirportGraphIntelligence
    from Project.phase_5_rl_optimizer import AirportResourceRL
    from Project.phase_6_agent_workflow import AirportAgenticWorkflow
except ImportError:
    from common import OPERATION_ID_COLUMN, load_airport_dataset
    from phase_1_knowledge_base import build_knowledge_base
    from phase_2_retrievers import KeywordRetriever, VectorRetriever
    from phase_3_rrf import reciprocal_rank_fusion
    from phase_4_graph_intelligence import AirportGraphIntelligence
    from phase_5_rl_optimizer import AirportResourceRL
    from phase_6_agent_workflow import AirportAgenticWorkflow

def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

def _print_check(title: str) -> None:
    print(f"[check] {title}")

def run_validation() -> None:
    df = load_airport_dataset()
    _print_check("dataset schema and unique operation identifiers")
    _assert(OPERATION_ID_COLUMN in df.columns,
            "operation_id column must exist after loading the dataset")
    _assert(df[OPERATION_ID_COLUMN].is_unique,
            "operation_id values must be unique")
    _assert(df["scheduled_departure"].notna().all(),
            "scheduled_departure must not contain null values")

    docs = build_knowledge_base(refresh=True)
    _print_check("knowledge base uniqueness and integrity")
    doc_ids = [doc.doc_id for doc in docs]
    _assert(len(doc_ids) == len(set(doc_ids)),
            "knowledge base doc IDs must be unique")
    historical_docs = [doc for doc in docs if doc.source_type == "historical_record"]
    _assert(len(historical_docs) == len(df),
            "historical document count must match dataset row count")
    _assert(
        all(OPERATION_ID_COLUMN in doc.metadata for doc in historical_docs[:50]),
        "historical docs must carry operation_id metadata",
    )

    vector  = VectorRetriever(docs)
    keyword = KeywordRetriever(docs)
    _print_check("hybrid retrievers")
    vector_hits  = vector.search(
        "Why is flight SG9626 delayed and what should operations do next?", top_k=5)
    keyword_hits = keyword.search(
        "gate congestion mitigation and fuel shortage response", top_k=5)
    _assert(len(vector_hits)  > 0, "vector retriever must return hits")
    _assert(len(keyword_hits) > 0, "keyword retriever must return hits")

    _print_check("RRF fusion")
    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], top_k=5)
    _assert(len(fused) > 0, "RRF must return fused results")
    _assert(
        len({item.doc.doc_id for item in fused}) == len(fused),
        "RRF output must be deduplicated by doc ID",
    )

    graph = AirportGraphIntelligence()
    duplicate_flight_id = df[df["flight_id"].duplicated(keep=False)]["flight_id"].iloc[0]
    _print_check("graph intelligence duplicate-flight disambiguation")
    resolved_operation = graph.resolve_operation(
        f"Why is flight {duplicate_flight_id} delayed?")
    _assert(resolved_operation is not None,
            "graph resolver must return an operation for a duplicated flight ID")
    graph_hits = graph.search(
        f"Why is flight {duplicate_flight_id} delayed?", top_k=3)
    _assert(len(graph_hits) > 0,
            "graph search must return hits for a duplicated flight query")
    _assert(OPERATION_ID_COLUMN in graph_hits[0].doc.metadata,
            "graph documents must expose operation_id metadata")

    _print_check("RL optimizer")
    rl = AirportResourceRL()
    rl.train(episodes=2000)
    rl_result = rl.recommend(resolved_operation)
    _assert(rl_result[OPERATION_ID_COLUMN] == resolved_operation,
            "RL recommendation must preserve the resolved operation_id")
    _assert(rl_result["projected_delay_minutes"] is not None,
            "RL recommendation must provide projected delay minutes")
    _assert(rl_result["expected_delay_reduction_minutes"] >= 0,
            "RL expected delay reduction must be non-negative")

    _print_check("end-to-end workflow scenarios")
    workflow = AirportAgenticWorkflow(refresh_knowledge=False)
    queries: Iterable[str] = [
        "Why is flight SG9626 operationally high risk, what dependencies are causing the disruption, and what should the airport do next?",
        f"Why is flight {duplicate_flight_id} delayed and what should operations do next?",
        "Show congestion risk around gate T2-G26 and recommend a mitigation action.",
        "How does Heavy Rain affect airport disruption patterns?",
    ]
    for query in queries:
        result = workflow.run(query)
        _assert(result.get("final_answer"),
                f"workflow must produce a final answer for query: {query}")
        _assert(result.get("rl_recommendation"),
                f"workflow must produce an RL recommendation for query: {query}")
        _assert(result.get("fused_hits"),
                f"workflow must produce fused hits for query: {query}")

    _print_check("Phase 7 — Voice AI pipeline (text-mode validation, no mic required)")
    _validate_phase_7(workflow)

    print("\n✅  All project validation checks passed (Phases 1-7).")

def _validate_phase_7(workflow: AirportAgenticWorkflow) -> None:
    try:
        from Phase_7_voice_ai import (
            AirportVoiceAI,
            QueryPreprocessor,
            ResponseFormatter,
        )
    except ImportError as exc:
        print(f"  [skip] Phase 7 not importable ({exc}) — skipping voice checks.")
        print("         Install:  pip install openai-whisper gtts sounddevice playsound")
        return

    _print_check("Phase 7 — QueryPreprocessor transcript cleaning")
    preprocessor = QueryPreprocessor()

    cleaned = preprocessor.process("Why is flight UK four three nine delayed? Thank you for watching")
    _assert("thank you for watching" not in cleaned.lower(),
            "preprocessor must remove Whisper hallucinations")

    cleaned2 = preprocessor.process("what is the status of flight six E three one five")
    _assert("6" in cleaned2,
            "preprocessor must convert number words to digits in flight IDs")

    _assert(not preprocessor.is_valid("um"),
            "preprocessor must reject trivially short transcripts")
    _assert(preprocessor.is_valid("Why is flight UK439 delayed?"),
            "preprocessor must accept a valid airport query")

    _print_check("Phase 7 — ResponseFormatter console and voice output")
    formatter   = ResponseFormatter()
    dummy_result = {
        "selected_flight_id":    "UK439",
        "selected_operation_id": "OP-004326",
        "rl_recommendation": {
            "action":                            "reroute_ground_traffic",
            "policy_mode":                       "rl_trained",
            "expected_delay_reduction_minutes": 11.9,
            "projected_delay_minutes":          22.1,
            "reason":                           "Runway congestion exceeds safe threshold.",
        },
        "fused_hits": [],
        "final_answer": "Flight UK439 is at high risk due to fog and runway congestion.",
    }
    console_out = formatter.format_console(dummy_result, "Test query")
    voice_out   = formatter.format_voice(dummy_result)

    _assert(isinstance(console_out, str) and len(console_out) > 50,
            "ResponseFormatter.format_console must return a non-empty string")
    _assert("UK439" in console_out,
            "console output must contain the flight ID")
    _assert("reroute_ground_traffic" in console_out,
            "console output must contain the recommended action")
    _assert(isinstance(voice_out, str) and len(voice_out) > 20,
            "ResponseFormatter.format_voice must return a non-empty string")
    _assert("UK439" in voice_out,
            "voice output must reference the flight")

    _print_check("Phase 7 — AirportVoiceAI text-mode end-to-end run")

    voice_ai = AirportVoiceAI(
        whisper_model = "base",
        enable_tts    = False,
        refresh_kb    = False,
    )
    voice_ai.workflow = workflow

    voice_queries = [
        "Which flights are most at risk during peak hour fog conditions?",
        "What action should be taken when runway congestion exceeds eighty?",
        "Show the worst gate for departure delays and recommend a fix.",
    ]

    for vq in voice_queries:
        result = voice_ai.run_text(vq)
        _assert(result.get("final_answer"),
                f"Voice AI must produce a final_answer for: '{vq}'")
        _assert(result.get("rl_recommendation"),
                f"Voice AI must produce an rl_recommendation for: '{vq}'")
        _assert(result.get("fused_hits"),
                f"Voice AI must return fused_hits (RAG evidence) for: '{vq}'")

        answer = result["final_answer"]
        _assert(len(answer) > 40,
                f"Voice AI final_answer is too short for query: '{vq}'")

    print("  ✅  Phase 7 validation passed (QueryPreprocessor, ResponseFormatter, "
          "AirportVoiceAI text-mode).")

if __name__ == "__main__":
    run_validation()