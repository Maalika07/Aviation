from __future__ import annotations

import argparse

try:
    from Project.common import load_airport_dataset
    from Project.phase_6_agent_workflow import AirportAgenticWorkflow
except ImportError:
    from common import load_airport_dataset
    from phase_6_agent_workflow import AirportAgenticWorkflow


def default_query() -> str:
    df = load_airport_dataset()
    row = df.sort_values("departure_delay_minutes", ascending=False).iloc[0]
    return (
        f"Why is flight {row['flight_id']} operationally high risk, "
        "what dependencies are causing the disruption, and what should the airport do next?"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic AI airport operations demo with Hybrid RAG, RRF, graph intelligence, and RL."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Operational question for the airport agent.",
    )
    parser.add_argument(
        "--refresh-kb",
        action="store_true",
        help="Rebuild the airport knowledge base before running.",
    )
    args = parser.parse_args()

    workflow = AirportAgenticWorkflow(refresh_knowledge=args.refresh_kb)
    query = args.query or default_query()
    result = workflow.run(query)

    print("\n=== Agentic Smart Airport Operations ===")
    print(f"Query: {query}")
    print(f"Selected flight: {result.get('selected_flight_id')}")
    print(f"Selected operation: {result.get('selected_operation_id')}")
    print(f"RL recommendation: {result.get('rl_recommendation')}")
    print("\nTop fused evidence:")
    for item in result.get("fused_hits", [])[:5]:
        print(f"  {item.rank}. {item.doc.title} | {item.retriever} | {item.score:.4f}")
    print("\nFinal answer:\n")
    print(result.get("final_answer", "No answer produced."))


if __name__ == "__main__":
    main()
