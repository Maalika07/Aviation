from __future__ import annotations

import re
from typing import Optional

try:
    from Project.common import KnowledgeDocument, RetrievedDocument, OPERATION_ID_COLUMN, load_airport_dataset
except ImportError:
    from common import KnowledgeDocument, RetrievedDocument, OPERATION_ID_COLUMN, load_airport_dataset


_WEATHER_RISK = {"Storm": 1.0, "Fog": 0.8, "Rain": 0.6, "Windy": 0.3, "Clear": 0.0}


class AirportGraphIntelligence:
    

    def __init__(self) -> None:
        self.df = load_airport_dataset()
        self._op_to_row:      dict[str, dict]       = {}
        self._flight_to_ops:  dict[str, list[str]]  = {}
        self._gate_to_ops:    dict[str, list[str]]  = {}
        self._runway_to_ops:  dict[str, list[str]]  = {}
        self._weather_to_ops: dict[str, list[str]]  = {}
        self._terminal_to_ops: dict[str, list[str]] = {}
        self._build_graph()



    def _build_graph(self) -> None:
        for row in self.df.itertuples(index=False):
            oid = str(getattr(row, OPERATION_ID_COLUMN))
            self._op_to_row[oid] = {
                "flight_id":               str(row.flight_id),
                "gate_id":                 str(row.gate_id),
                "runway_id":               str(row.runway_id),
                "terminal":                str(row.terminal),
                "weather_condition":       str(row.weather_condition),
                "departure_delay_minutes": float(row.departure_delay_minutes),
                "runway_congestion":       float(row.runway_congestion),
                "gate_occupancy_pct":      float(row.gate_occupancy_pct),
                "fuel_trucks_available":   int(row.fuel_trucks_available),
                "staff_available_pct":     float(row.staff_available_pct),
                "maintenance_flag":        int(row.maintenance_flag),
                "passenger_count":         int(row.passenger_count),
                "disruption_risk":         float(row.disruption_risk),
                OPERATION_ID_COLUMN:       oid,
            }
            self._flight_to_ops.setdefault(str(row.flight_id),        []).append(oid)
            self._gate_to_ops.setdefault(str(row.gate_id),            []).append(oid)
            self._runway_to_ops.setdefault(str(row.runway_id),        []).append(oid)
            self._weather_to_ops.setdefault(str(row.weather_condition),[]).append(oid)
            self._terminal_to_ops.setdefault(str(row.terminal),       []).append(oid)

 

    def _best_in(self, ops: list[str], key: str = "departure_delay_minutes") -> str:
        return max(ops, key=lambda oid: self._op_to_row[oid][key])

    def _parse_thresholds(self, query: str) -> dict[str, float]:
        """Extract numeric thresholds like 'delay above 30', 'congestion > 80'."""
        thresholds: dict[str, float] = {}
        patterns = [
            (r"delay\s*(?:above|over|>|greater\s*than)\s*(\d+)",    "departure_delay_minutes"),
            (r"congestion\s*(?:above|over|>|greater\s*than)\s*(\d+)", "runway_congestion"),
            (r"delay\s*(?:below|under|<|less\s*than)\s*(\d+)",      "delay_max"),
        ]
        for pattern, col in patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                thresholds[col] = float(m.group(1))
        return thresholds

    def _filter_by_thresholds(self, thresholds: dict[str, float]):
        filtered = self.df.copy()
        if "departure_delay_minutes" in thresholds:
            filtered = filtered[filtered["departure_delay_minutes"] > thresholds["departure_delay_minutes"]]
        if "runway_congestion" in thresholds:
            filtered = filtered[filtered["runway_congestion"] > thresholds["runway_congestion"]]
        if "delay_max" in thresholds:
            filtered = filtered[filtered["departure_delay_minutes"] < thresholds["delay_max"]]
        return filtered

   

    def resolve_operation(self, query: str) -> Optional[str]:
      
        q_low = query.lower()

        # 1. Explicit flight ID
        m = re.search(r'\b([A-Z]{2,3}\d{3,5})\b', query)
        if m:
            ops = self._flight_to_ops.get(m.group(1))
            if ops:
                return self._best_in(ops)

        # 2. Explicit gate ID
        m = re.search(r'\b(T\d-G\d+)\b', query, re.IGNORECASE)
        if m:
            ops = self._gate_to_ops.get(m.group(1).upper())
            if ops:
                return self._best_in(ops)

        # 3. Exact runway ID — only match IDs that actually exist in the dataset
        #    This prevents "30 minutes" or "80 congestion" from being misread as runways.
        for runway_id in self._runway_to_ops:
            pattern = r'(?<![A-Z\d])' + re.escape(runway_id) + r'(?![A-Z\d])'
            if re.search(pattern, query, re.IGNORECASE):
                return self._best_in(self._runway_to_ops[runway_id])

        # 4. Numerical threshold filtering
        thresholds = self._parse_thresholds(query)
        if thresholds:
            filtered = self._filter_by_thresholds(thresholds)
            if not filtered.empty:
                row = filtered.sort_values("departure_delay_minutes", ascending=False).iloc[0]
                return str(row[OPERATION_ID_COLUMN])

        # 5a. Analytical: "which/what runway … highest/worst/most delayed"
        if "runway" in q_low and any(w in q_low for w in ("highest", "worst", "most", "top", "maximum")):
            runway_avg = self.df.groupby("runway_id")["departure_delay_minutes"].mean()
            worst_runway = runway_avg.idxmax()
            return self._best_in(self._runway_to_ops[worst_runway])

        # 5b. Analytical: "which terminal … highest/worst"
        if "terminal" in q_low and any(w in q_low for w in ("highest", "worst", "most", "top", "maximum")):
            terminal_avg = self.df.groupby("terminal")["departure_delay_minutes"].mean()
            worst_terminal = terminal_avg.idxmax()
            return self._best_in(self._terminal_to_ops[worst_terminal])

        # 5c. Analytical: "which gate … highest/worst"
        if "gate" in q_low and any(w in q_low for w in ("highest", "worst", "most", "top", "maximum")):
            gate_avg = self.df.groupby("gate_id")["departure_delay_minutes"].mean()
            worst_gate = gate_avg.idxmax()
            return self._best_in(self._gate_to_ops[worst_gate])

        # 6. Weather keyword
        for weather in self._weather_to_ops:
            if weather.lower() in q_low:
                return self._best_in(self._weather_to_ops[weather])

        # 7. Global fallback
        if self._op_to_row:
            return self._best_in(list(self._op_to_row.keys()))
        return None

 

    def resolve_by_context(self, query: str) -> Optional[dict]:
        oid = self.resolve_operation(query)
        if oid is None:
            return None
        row = self._op_to_row[oid]
        return {
            "op_id": oid,
            "context": {
                "delay":      row["departure_delay_minutes"],
                "congestion": row["runway_congestion"],
            },
        }


    def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
       
        res = self.resolve_by_context(query)
        if res is None:
            return []

        seed_oid = res["op_id"]
        seed_row = self._op_to_row[seed_oid]

        # Graph-walk: accumulate candidates with edge-type scores
        candidate_scores: dict[str, float] = {seed_oid: 1.0}

        for oid in self._gate_to_ops.get(seed_row["gate_id"], [])[:6]:
            candidate_scores[oid] = candidate_scores.get(oid, 0.0) + 0.6

        for oid in self._runway_to_ops.get(seed_row["runway_id"], [])[:6]:
            candidate_scores[oid] = candidate_scores.get(oid, 0.0) + 0.4

        for oid in self._weather_to_ops.get(seed_row["weather_condition"], [])[:6]:
            candidate_scores[oid] = candidate_scores.get(oid, 0.0) + 0.3

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RetrievedDocument] = []
        for rank, (oid, score) in enumerate(ranked, start=1):
            r = self._op_to_row[oid]
            content = (
                f"Graph-linked operation {oid}: Flight {r['flight_id']} "
                f"→ Gate {r['gate_id']} | Runway {r['runway_id']} | "
                f"Weather: {r['weather_condition']} | "
                f"Delay: {r['departure_delay_minutes']:.0f} min | "
                f"Congestion: {r['runway_congestion']:.0f}/100 | "
                f"Maintenance flag: {r['maintenance_flag']} | "
                f"Disruption risk: {r['disruption_risk']:.3f}"
            )
            doc = KnowledgeDocument(
                doc_id=f"graph::{oid}",
                title=f"Graph node — {r['flight_id']} ({oid})",
                content=content,
                source_type="graph_node",
                metadata={
                    OPERATION_ID_COLUMN:           oid,
                    "flight_id":                   r["flight_id"],
                    "departure_delay_minutes":     r["departure_delay_minutes"],
                    "runway_congestion":           r["runway_congestion"],
                    "weather_condition":           r["weather_condition"],
                },
            )
            results.append(RetrievedDocument(doc=doc, score=score, retriever="graph", rank=rank))
        return results

    def close(self) -> None:
        pass