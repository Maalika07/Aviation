from __future__ import annotations

import os
import re
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    from Project.common import OPERATION_ID_COLUMN, load_airport_dataset
    from Project.phase_1_knowledge_base import build_knowledge_base
    from Project.phase_2_retrievers import KeywordRetriever, VectorRetriever
    from Project.phase_3_rrf import reciprocal_rank_fusion
    from Project.phase_4_graph_intelligence import AirportGraphIntelligence
    from Project.phase_5_rl_optimizer import AirportResourceRL
except ImportError:
    from common import OPERATION_ID_COLUMN, load_airport_dataset
    from phase_1_knowledge_base import build_knowledge_base
    from phase_2_retrievers import KeywordRetriever, VectorRetriever
    from phase_3_rrf import reciprocal_rank_fusion
    from phase_4_graph_intelligence import AirportGraphIntelligence
    from phase_5_rl_optimizer import AirportResourceRL

_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_SYSTEM_PROMPT = """\
You are an expert airport operations AI assistant. Your job is to directly answer
the operator's question using the retrieved evidence and RL recommendation provided.

Rules:
- ALWAYS answer the actual question asked first.
- If the question asks for a count, list, comparison, or pattern -- answer that.
- If the question asks for a specific operation or flight -- describe it.
- Use numbers from the evidence. Do not invent figures.
- Keep the answer to 4-6 sentences. Be direct and operational in tone.
- Do NOT open with "The primary disruption factors" every time.
  Match your opening to the question type."""

class AirportAgenticWorkflow:
    def __init__(self, refresh_knowledge: bool = False) -> None:
        print("    Loading dataset ...")
        self.df = load_airport_dataset()

        print("    Building knowledge base ...")
        self.docs = build_knowledge_base(refresh=refresh_knowledge)

        print("    Initialising retrievers ...")
        self.vector_retriever = VectorRetriever(self.docs)
        self.keyword_retriever = KeywordRetriever(self.docs)

        print("    Building graph intelligence ...")
        self.graph = AirportGraphIntelligence()

        print("    Training RL optimizer (2000 episodes) ...")
        self.rl = AirportResourceRL()
        self.rl.train(episodes=2000)

        print("    Connecting to Groq API ...")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _classify_query(self, query: str) -> str:
        q = query.lower()
        if re.search(r'(delay|congestion)\s*(above|over|>|greater)', q):
            return "threshold"
        if re.search(r'\b([A-Z]{2,3}\d{3,5}|T\d-G\d+)\b', query):
            return "specific"
        if any(w in q for w in ("highest", "worst", "most", "average", "compare",
                                "which", "what runway", "what terminal", "ranking")):
            return "analytical"
        if any(w in q for w in ("fog", "storm", "rain", "weather", "visibility",
                                "sop", "what if", "scenario", "during", "when")):
            return "scenario"
        return "general"

    def _build_threshold_context(self, query: str) -> str:
        q = query.lower()
        filtered = self.df.copy()

        delay_m = re.search(r'delay\s*(?:above|over|>|greater\s*than)\s*(\d+)', q)
        cong_m = re.search(r'congestion\s*(?:above|over|>|greater\s*than)\s*(\d+)', q)

        if delay_m:
            filtered = filtered[filtered["departure_delay_minutes"] > float(delay_m.group(1))]
        if cong_m:
            filtered = filtered[filtered["runway_congestion"] > float(cong_m.group(1))]

        if filtered.empty:
            return "No operations matched the specified thresholds."

        total = len(filtered)
        top5 = filtered.sort_values("departure_delay_minutes", ascending=False).head(5)

        lines = [
            f"Total operations matching criteria: {total}",
            "",
            "Top 5 by departure delay:",
        ]
        for _, r in top5.iterrows():
            lines.append(
                f" {r[OPERATION_ID_COLUMN]} | Flight {r['flight_id']} | "
                f"Delay {r['departure_delay_minutes']:.0f} min | "
                f"Congestion {r['runway_congestion']:.0f}/100 | "
                f"Weather: {r['weather_condition']} | "
                f"Maintenance: {'YES' if r['maintenance_flag'] else 'NO'} | "
                f"Fuel trucks: {r['fuel_trucks_available']}"
            )

        weather_dist = filtered["weather_condition"].value_counts().to_dict()
        lines.append(f"\nWeather breakdown: {weather_dist}")
        maint_count = int(filtered["maintenance_flag"].sum())
        lines.append(f"Operations with active maintenance flag: {maint_count} / {total}")
        return "\n".join(lines)

    def _build_analytical_context(self, query: str) -> str:
        q = query.lower()
        lines = []

        if "runway" in q:
            stats = (
                self.df.groupby("runway_id")["departure_delay_minutes"]
                .agg(["mean", "count"])
                .sort_values("mean", ascending=False)
                .reset_index()
            )
            lines.append("Runway average departure delays (ranked):")
            for _, r in stats.iterrows():
                lines.append(f" Runway {r['runway_id']}: avg {r['mean']:.1f} min  ({int(r['count'])} operations)")

        if "terminal" in q:
            stats = (
                self.df.groupby("terminal")[["departure_delay_minutes","gate_occupancy_pct","staff_available_pct"]]
                .mean()
                .sort_values("departure_delay_minutes", ascending=False)
                .reset_index()
            )
            lines.append("Terminal performance (ranked by avg delay):")
            for _, r in stats.iterrows():
                lines.append(
                    f" {r['terminal']}: avg delay {r['departure_delay_minutes']:.1f} min | "
                    f"gate occ {r['gate_occupancy_pct']:.1f}% | staff {r['staff_available_pct']:.1f}%"
                )

        if "weather" in q or "fog" in q or "storm" in q or "rain" in q:
            stats = (
                self.df.groupby("weather_condition")[["departure_delay_minutes","runway_congestion"]]
                .mean()
                .sort_values("departure_delay_minutes", ascending=False)
                .reset_index()
            )
            lines.append("Weather condition impact (ranked by avg delay):")
            for _, r in stats.iterrows():
                lines.append(
                    f" {r['weather_condition']}: avg delay {r['departure_delay_minutes']:.1f} min | "
                    f"avg congestion {r['runway_congestion']:.1f}/100"
                )

        if "gate" in q:
            stats = (
                self.df.groupby("gate_id")["departure_delay_minutes"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            lines.append("Top 10 gates by avg delay:")
            for _, r in stats.iterrows():
                lines.append(f" Gate {r['gate_id']}: avg {r['departure_delay_minutes']:.1f} min")

        return "\n".join(lines) if lines else ""

    def _hybrid_retrieve(self, query: str, top_k: int = 6):
        vector_hits = self.vector_retriever.search(query, top_k=top_k)
        keyword_hits = self.keyword_retriever.search(query, top_k=top_k)
        graph_hits = self.graph.search(query, top_k=4)
        return reciprocal_rank_fusion([vector_hits, keyword_hits, graph_hits], top_k=8)

    def _generate_answer(
        self,
        query: str,
        query_type: str,
        fused_hits: list,
        rl_rec: dict,
        operation_id: Optional[str],
        flight_id: Optional[str],
        extra_context: str = "",
    ) -> str:
        rag_evidence = "\n\n".join(
            f"[Evidence {i+1}] {hit.doc.title}\n{hit.doc.content[:400]}"
            for i, hit in enumerate(fused_hits[:4])
        )

        rl_block = (
            f"RL Recommendation (for operation {operation_id}):\n"
            f" Action : {rl_rec.get('action')}\n"
            f" Expected reduction: {rl_rec.get('expected_delay_reduction_minutes')} min\n"
            f" Projected delay : {rl_rec.get('projected_delay_minutes')} min remaining\n"
            f" Policy mode : {rl_rec.get('policy_mode')}\n"
            f" Reason : {rl_rec.get('reason')}"
        ) if rl_rec else "No RL recommendation available."

        if query_type == "threshold":
            user_msg = (
                f"QUESTION: {query}\n\n"
                f"=== MATCHING OPERATIONS DATA ===\n{extra_context}\n\n"
                f"=== RL RECOMMENDATION FOR WORST CASE ===\n{rl_block}\n\n"
                f"=== SUPPORTING RAG EVIDENCE ===\n{rag_evidence}\n\n"
                f"Answer the question: how many operations match, what do they have in common, "
                f"and what is the recommended action?"
            )
        elif query_type == "analytical":
            user_msg = (
                f"QUESTION: {query}\n\n"
                f"=== AGGREGATED STATISTICS ===\n{extra_context}\n\n"
                f"=== SUPPORTING RAG EVIDENCE ===\n{rag_evidence}\n\n"
                f"Answer the question directly using the statistics above."
            )
        else:
            row_data = ""
            if operation_id and flight_id:
                rows = self.df[self.df[OPERATION_ID_COLUMN] == operation_id]
                if not rows.empty:
                    r = rows.iloc[0]
                    row_data = (
                        f"Operation: {operation_id} | Flight: {flight_id}\n"
                        f"Delay: {r['departure_delay_minutes']:.0f} min | "
                        f"Congestion: {r['runway_congestion']:.0f}/100 | "
                        f"Weather: {r['weather_condition']} | "
                        f"Maintenance: {'YES' if r['maintenance_flag'] else 'NO'} | "
                        f"Fuel trucks: {r['fuel_trucks_available']} | "
                        f"Staff: {r['staff_available_pct']:.0f}% | "
                        f"Passengers: {r['passenger_count']}"
                    )
            user_msg = (
                f"QUESTION: {query}\n\n"
                f"=== OPERATION DETAILS ===\n{row_data}\n\n"
                f"=== RL RECOMMENDATION ===\n{rl_block}\n\n"
                f"=== RAG EVIDENCE ===\n{rag_evidence}\n\n"
                f"Answer the question directly."
            )

        response = self.client.chat.completions.create(
            model=_GROQ_MODEL,
            max_tokens=600,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content

    def run(self, query: str) -> dict:
        query_type = self._classify_query(query)

        fused_hits = self._hybrid_retrieve(query)

        operation_id: Optional[str] = self.graph.resolve_operation(query)
        flight_id: Optional[str] = None
        if operation_id:
            rows = self.df[self.df[OPERATION_ID_COLUMN] == operation_id]
            if not rows.empty:
                flight_id = str(rows.iloc[0]["flight_id"])

        rl_recommendation: dict = {}
        if operation_id:
            rl_recommendation = self.rl.recommend(operation_id)

        extra_context = ""
        if query_type == "threshold":
            extra_context = self._build_threshold_context(query)
        elif query_type == "analytical":
            extra_context = self._build_analytical_context(query)

        final_answer = self._generate_answer(
            query, query_type, fused_hits,
            rl_recommendation, operation_id, flight_id,
            extra_context,
        )

        return {
            "selected_flight_id": flight_id,
            "selected_operation_id": operation_id,
            "rl_recommendation": rl_recommendation,
            "fused_hits": fused_hits,
            "final_answer": final_answer,
        }
    
    