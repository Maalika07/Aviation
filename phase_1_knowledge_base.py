from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    from Project.common import ARTIFACTS_DIR, KnowledgeDocument, OPERATION_ID_COLUMN, load_airport_dataset, load_documents, save_documents
except ImportError:
    from common import ARTIFACTS_DIR, KnowledgeDocument, OPERATION_ID_COLUMN, load_airport_dataset, load_documents, save_documents


KNOWLEDGE_PATH = ARTIFACTS_DIR / "airport_knowledge_base.json"

STATIC_SOPS = [
    (
        "sop_weather_thunderstorm",
        "Thunderstorm Disruption SOP",
        "When thunderstorms are active, operations should slow pushback sequencing, prioritize flights already boarded, reduce runway conflict pressure, and communicate expected turnaround extensions to gate control and apron teams.",
    ),
    (
        "sop_weather_fog",
        "Low Visibility and Fog SOP",
        "During fog or poor visibility, operators should expect reduced runway throughput, longer taxi buffers, slower stand handling, and a higher need for proactive gate resequencing to prevent propagation delays.",
    ),
    (
        "sop_gate_congestion",
        "Gate Congestion Mitigation SOP",
        "If gate occupancy approaches terminal capacity, operations should trigger alternate gate search, protect arriving flights with short turnaround windows, and reroute flexible departures before a queue forms.",
    ),
    (
        "sop_fuel_shortage",
        "Fuel Truck Shortage SOP",
        "Low fuel truck availability should trigger priority fueling for high-passenger and near-departure flights, consolidation of refueling tasks by terminal, and escalation to standby contractors if buffers collapse.",
    ),
    (
        "sop_baggage_staff_constraints",
        "Baggage and Staff Constraint SOP",
        "When baggage vehicles or ramp staff are limited, operations should batch baggage loading by gate cluster, reassign flexible crew pools, and protect flights with heavy passenger loads first.",
    ),
    (
        "sop_runway_congestion",
        "Runway Congestion Control SOP",
        "High runway congestion requires deconflicting pushback timing, coordinating departures by readiness, and delaying low-priority pushbacks when staging traffic would worsen bottlenecks.",
    ),
    (
        "sop_maintenance_escalation",
        "Maintenance Escalation SOP",
        "If maintenance support is required, the airport control team should parallelize inspection, fueling, and baggage preparation when safe, while immediately flagging risk of missed slot or gate overstay.",
    ),
    (
        "sop_passenger_surge",
        "Passenger Surge Management SOP",
        "Large passenger loads increase boarding, baggage, and fueling coordination pressure. Flights with high passenger counts should receive earlier stand preparation and stronger departure-readiness checks.",
    ),
]


def build_static_documents() -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            source_type="sop",
            metadata={"category": "policy"},
        )
        for doc_id, title, content in STATIC_SOPS
    ]


def build_historical_documents() -> list[KnowledgeDocument]:
    df = load_airport_dataset()
    docs: list[KnowledgeDocument] = []
    for row in df.itertuples(index=False):
        scheduled_departure = row.scheduled_departure.isoformat()
        
        # Explicitly build structural text dependencies inside the content body
        content = (
            f"=== OPERATIONAL LINKAGE RECORD ===\n"
            f"Operation ID: {row.operation_id} | Flight ID: {row.flight_id} | Handler Airline: {row.airline}\n"
            f"Route Mapping: From {row.origin} to {row.destination}\n\n"
            f"--- INFRASTRUCTURE ROUTING RELATIONSHIPS ---\n"
            f"- Flight {row.flight_id} is dynamically routed through Terminal {row.terminal} to Gate Node {row.gate_id}.\n"
            f"- The operation utilizes Runway Asset {row.runway_id} for departure execution.\n\n"
            f"--- CASUAL ENVIRONMENTAL & RESOURCE DEPENDENCIES ---\n"
            f"- Environmental Impact: Weather condition is {row.weather_condition} (Visibility: {row.visibility_km} km, Wind: {row.wind_speed_kts} kts).\n"
            f"- Infrastructure Bottlenecks: Runway congestion index stands at {row.runway_congestion}/100, while physical Gate Occupancy is at {row.gate_occupancy_pct}%.\n"
            f"- Asset Supply Constraints: Fuel truck availability is at {row.fuel_trucks_available} units, baggage vehicles stand at {row.baggage_vehicles_available} units, and the baseline Staff Readiness metric is {row.staff_available_pct}%.\n"
            f"- Safety/Technical Link: Maintenance intervention flag is evaluated as status code {row.maintenance_flag}.\n\n"
            f"--- DOWNSTREAM DELAY PROPAGATION OUTCOMES ---\n"
            f"- Turnaround Efficiency window was scheduled for {row.scheduled_turnaround_minutes} minutes, but required {row.actual_turnaround_minutes} minutes of true execution time.\n"
            f"- Resulting Congestion Delay: This interconnected relationship network generated an actual departure delay of {row.departure_delay_minutes} minutes for Flight {row.flight_id}.\n"
            f"- Predictive Risk Matrix: Initial structural disruption probability score was calculated at {row.disruption_risk}."
        )
        
        docs.append(
            KnowledgeDocument(
                doc_id=f"history::{row.operation_id}",
                title=f"Operational record for {row.flight_id} at {row.scheduled_departure:%Y-%m-%d %H:%M}",
                content=content,
                source_type="historical_record",
                metadata={
                    OPERATION_ID_COLUMN: row.operation_id,
                    "flight_id": row.flight_id,
                    "airline": row.airline,
                    "origin": row.origin,
                    "destination": row.destination,
                    "terminal": row.terminal,
                    "gate_id": row.gate_id,
                    "runway_id": row.runway_id,
                    "weather_condition": row.weather_condition,
                    "runway_congestion": int(row.runway_congestion),
                    "gate_occupancy_pct": int(row.gate_occupancy_pct),
                    "fuel_trucks_available": int(row.fuel_trucks_available),
                    "baggage_vehicles_available": int(row.baggage_vehicles_available),
                    "staff_available_pct": int(row.staff_available_pct),
                    "maintenance_flag": int(row.maintenance_flag),
                    "passenger_count": int(row.passenger_count),
                    "departure_delay_minutes": int(row.departure_delay_minutes),
                    "disruption_risk": row.disruption_risk,
                    "scheduled_departure": scheduled_departure,
                },
            )
        )
    return docs


def build_aggregated_brief_documents() -> list[KnowledgeDocument]:
    df = load_airport_dataset()
    docs: list[KnowledgeDocument] = []

    # Weather impact relationship profiling
    weather_stats = (
        df.groupby("weather_condition")[["departure_delay_minutes", "runway_congestion"]]
        .mean()
        .sort_values("departure_delay_minutes", ascending=False)
    )
    for weather, stats in weather_stats.iterrows():
        docs.append(
            KnowledgeDocument(
                doc_id=f"brief::weather::{weather.lower().replace(' ', '_')}",
                title=f"Weather impact brief for {weather}",
                content=(
                    f"=== SYSTEMIC WEATHER RELATIONSHIP PROFILE ===\n"
                    f"Active Environmental Variable: {weather}\n\n"
                    f"DEPENDENCY OVERLAPS:\n"
                    f"- Delay Correlation: {weather} causes a baseline average departure delay of {stats['departure_delay_minutes']:.1f} minutes.\n"
                    f"- Infrastructure Stress: Directly links to a systematic runway congestion rating of {stats['runway_congestion']:.1f}/100.\n"
                    f"Cross-Reference Note: Multi-agent planners must flag this threshold when analyzing upstream weather bottlenecks."
                ),
                source_type="aggregated_brief",
                metadata={"weather_condition": weather},
            )
        )

    # Gate allocation capacity profiling
    gate_stats = (
        df.groupby("gate_id")[["departure_delay_minutes", "gate_occupancy_pct"]]
        .mean()
        .sort_values("departure_delay_minutes", ascending=False)
    )
    for gate_id, stats in gate_stats.iterrows():
        docs.append(
            KnowledgeDocument(
                doc_id=f"brief::gate::{gate_id}",
                title=f"Gate operations brief for {gate_id}",
                content=(
                    f"=== INFRASTRUCTURE NODE OPERATION BRIEF ===\n"
                    f"Monitored Asset Location: Gate {gate_id}\n\n"
                    f"DEPENDENCY OVERLAPS:\n"
                    f"- Spatial Footprint: Operates with a historical mean gate occupancy index of {stats['gate_occupancy_pct']:.1f}%.\n"
                    f"- Backlog Outcome: Linked structurally to a baseline node departure delay of {stats['departure_delay_minutes']:.1f} minutes."
                ),
                source_type="aggregated_brief",
                metadata={"gate_id": gate_id},
            )
        )

    # Runway throughput profiling
    runway_stats = (
        df.groupby("runway_id")[["departure_delay_minutes", "runway_congestion"]]
        .mean()
        .sort_values("departure_delay_minutes", ascending=False)
    )
    for runway_id, stats in runway_stats.iterrows():
        docs.append(
            KnowledgeDocument(
                doc_id=f"brief::runway::{runway_id.replace('/', '_')}",
                title=f"Runway operations brief for {runway_id}",
                content=(
                    f"=== RUNWAY CAPACITY RELATIONSHIP BRIEF ===\n"
                    f"Monitored Traffic Asset: Runway {runway_id}\n\n"
                    f"DEPENDENCY OVERLAPS:\n"
                    f"- Load Profile: Handles an average strategic runway congestion profile of {stats['runway_congestion']:.1f}/100.\n"
                    f"- Delay Attribution: Operations traversing this node sustain a mean delay of {stats['departure_delay_minutes']:.1f} minutes."
                ),
                source_type="aggregated_brief",
                metadata={"runway_id": runway_id},
            )
        )

    # Sector resource profiling
    grouped = df.groupby("terminal")[["departure_delay_minutes", "gate_occupancy_pct", "staff_available_pct"]].mean()
    for terminal, stats in grouped.iterrows():
        docs.append(
            KnowledgeDocument(
                doc_id=f"brief::terminal::{terminal}",
                title=f"Terminal performance brief for {terminal}",
                content=(
                    f"=== SECTOR PERFORMANCE & RESOURCE FOOTPRINT ===\n"
                    f"Airport Sector: {terminal}\n\n"
                    f"DEPENDENCY OVERLAPS:\n"
                    f"- Structural Backlog: Generates an average localized sector delay of {stats['departure_delay_minutes']:.1f} minutes.\n"
                    f"- Gate Congestion footprint: {stats['gate_occupancy_pct']:.1f}% physical utilization capacity.\n"
                    f"- Labor Constraints: Ground operations function with a baseline staff availability index of {stats['staff_available_pct']:.1f}%."
                ),
                source_type="aggregated_brief",
                metadata={"terminal": terminal},
            )
        )
    return docs


def _knowledge_base_is_valid(docs: list[KnowledgeDocument]) -> bool:
    if not docs:
        return False
    doc_ids = [doc.doc_id for doc in docs]
    if len(doc_ids) != len(set(doc_ids)):
        return False
    dataset = load_airport_dataset()
    expected_historical = len(dataset)
    expected_total = len(build_static_documents()) + len(build_aggregated_brief_documents()) + expected_historical
    if len(docs) != expected_total:
        return False
    historical_docs = [doc for doc in docs if doc.source_type == "historical_record"]
    if len(historical_docs) != expected_historical:
        return False
    for doc in historical_docs[:10]:
        if OPERATION_ID_COLUMN not in doc.metadata:
            return False
    return True


def build_knowledge_base(refresh: bool = False) -> list[KnowledgeDocument]:
    if KNOWLEDGE_PATH.exists() and not refresh:
        cached_docs = load_documents(KNOWLEDGE_PATH)
        if _knowledge_base_is_valid(cached_docs):
            return cached_docs

    docs = build_static_documents() + build_aggregated_brief_documents() + build_historical_documents()
    save_documents(KNOWLEDGE_PATH, docs)
    return docs


def main() -> None:
    docs = build_knowledge_base(refresh=True)
    print(f"Knowledge base built with {len(docs)} documents.")
    print(f"Saved to: {Path(KNOWLEDGE_PATH)}")


if __name__ == "__main__":
    main()