from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from Project.common import OPERATION_ID_COLUMN, load_airport_dataset
except ImportError:
    from common import OPERATION_ID_COLUMN, load_airport_dataset

ACTIONS: list[str] = [
    "standard_ops",
    "reroute_ground_traffic",
    "extend_turnaround_buffer",
    "prioritize_fueling",
    "reassign_baggage_crew",
    "request_maintenance_fast_track",
]

_WEATHER_RISK: dict[str, float] = {
    "Storm": 1.0, "Fog": 0.8, "Rain": 0.6, "Windy": 0.3, "Clear": 0.0
}

_ACTION_REASONS: dict[str, str] = {
    "standard_ops":
        "All operational metrics are within nominal bounds; standard procedures apply.",
    "reroute_ground_traffic":
        "Runway congestion exceeds safe threshold; rerouting ground traffic will reduce bottleneck pressure.",
    "extend_turnaround_buffer":
        "Adverse weather requires an additional time buffer to protect the departure window.",
    "prioritize_fueling":
        "Low fuel truck availability is on the critical path; prioritisation shortens turnaround time.",
    "reassign_baggage_crew":
        "Staff shortage is constraining baggage loading; crew reallocation will release the gate earlier.",
    "request_maintenance_fast_track":
        "Maintenance flag is active; fast-tracking inspection prevents gate overstay.",
}

class AirportResourceRL:
    def __init__(self) -> None:
        self.df = load_airport_dataset()
        self.n_actions = len(ACTIONS)
        self.q_table: dict[tuple, np.ndarray] = {}
        self._trained = False

    def _weather_risk(self, weather: str) -> float:
        return _WEATHER_RISK.get(weather, 0.3)

    def _discretize(
        self,
        delay: float,
        congestion: float,
        weather_risk: float,
        maintenance: int,
        fuel: float,
        staff_pct: float,
    ) -> tuple:
        d = min(int(delay // 10), 5)
        c = min(int(congestion // 20), 4)
        w = min(int(weather_risk * 3), 2)
        mf = int(bool(maintenance))
        f = 0 if fuel >= 6 else (1 if fuel >= 3 else 2)
        s = 0 if staff_pct >= 85 else (1 if staff_pct >= 70 else 2)
        return (d, c, w, mf, f, s)

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def _simulate_reduction(self, action_idx: int, features: dict) -> float:
        delay = features["departure_delay_minutes"]
        congestion = features["runway_congestion"]
        weather_r = features["weather_risk"]
        maintenance = features["maintenance_flag"]
        fuel = features["fuel_trucks_available"]
        staff_pct = features["staff_available_pct"]
        pax = features["passenger_count"]
        gate_occ = features["gate_occupancy_pct"]

        action = ACTIONS[action_idx]

        if action == "standard_ops":
            penalty = 1.0 - 0.4 * (congestion / 100) - 0.3 * weather_r
            return round(min(3.0, delay * 0.06 * max(0.1, penalty)), 2)

        if action == "reroute_ground_traffic":
            congestion_factor = max(0.0, (congestion - 40) / 60)
            gate_factor = max(0.0, (gate_occ - 50) / 50)
            base = delay * 0.38 * congestion_factor * (1 + 0.3 * gate_factor)
            return round(min(16.0, base), 2)

        if action == "extend_turnaround_buffer":
            pax_factor = min(1.5, pax / 150)
            weather_factor = 0.5 + weather_r
            return round(min(9.0, delay * 0.20 * weather_factor * pax_factor), 2)

        if action == "prioritize_fueling":
            fuel_scarcity = max(0.0, (8 - fuel) / 8)
            return round(min(11.0, delay * 0.28 * (0.4 + fuel_scarcity)), 2)

        if action == "reassign_baggage_crew":
            staff_gap = max(0.0, (100 - staff_pct) / 100)
            pax_load = min(1.4, pax / 160)
            return round(min(8.0, delay * 0.20 * (0.4 + staff_gap) * pax_load), 2)

        if action == "request_maintenance_fast_track":
            maint_multiplier = 1.6 if maintenance == 1 else 0.25
            return round(min(13.0, delay * 0.32 * maint_multiplier), 2)

        return 0.0

    def _row_to_features(self, row) -> dict:
        weather = str(row[2]) if hasattr(row, '__getitem__') else str(row)
        return {
            "departure_delay_minutes": float(row[0]),
            "runway_congestion": float(row[1]),
            "weather_condition": weather,
            "weather_risk": self._weather_risk(weather),
            "maintenance_flag": int(row[3]),
            "fuel_trucks_available": float(row[4]),
            "staff_available_pct": float(row[5]),
            "passenger_count": int(row[6]),
            "gate_occupancy_pct": float(row[7]),
        }

    def train(self, episodes: int = 2000) -> None:
        rng = np.random.default_rng(42)
        alpha = 0.10
        gamma = 0.90
        eps = 0.30

        cols = [
            "departure_delay_minutes", "runway_congestion", "weather_condition",
            "maintenance_flag", "fuel_trucks_available", "staff_available_pct",
            "passenger_count", "gate_occupancy_pct",
        ]
        data = self.df[cols].values
        n = len(data)

        for _ in range(episodes):
            idx = int(rng.integers(0, n))
            features = self._row_to_features(data[idx])

            state = self._discretize(
                features["departure_delay_minutes"],
                features["runway_congestion"],
                features["weather_risk"],
                features["maintenance_flag"],
                features["fuel_trucks_available"],
                features["staff_available_pct"],
            )
            q = self._get_q(state)

            action_idx = int(rng.integers(0, self.n_actions)) if rng.random() < eps \
                         else int(np.argmax(q))

            reduction = self._simulate_reduction(action_idx, features)
            reward = reduction / max(features["departure_delay_minutes"], 1.0)

            next_state = self._discretize(
                max(0.0, features["departure_delay_minutes"] - reduction),
                features["runway_congestion"] * 0.9,
                features["weather_risk"],
                features["maintenance_flag"],
                features["fuel_trucks_available"],
                features["staff_available_pct"],
            )
            td_target = reward + gamma * float(np.max(self._get_q(next_state)))
            q[action_idx] += alpha * (td_target - q[action_idx])

        self._trained = True

    def recommend(self, operation_id: str) -> dict:
        rows = self.df[self.df[OPERATION_ID_COLUMN] == operation_id]
        if rows.empty:
            return {
                OPERATION_ID_COLUMN: operation_id,
                "action": "standard_ops",
                "policy_mode": "default",
                "reason": "Operation not found; defaulting to standard ops.",
                "projected_delay_minutes": 0,
                "expected_delay_reduction_minutes": 0,
            }

        row = rows.iloc[0]
        features = {
            "departure_delay_minutes": float(row["departure_delay_minutes"]),
            "runway_congestion": float(row["runway_congestion"]),
            "weather_condition": str(row["weather_condition"]),
            "weather_risk": self._weather_risk(str(row["weather_condition"])),
            "maintenance_flag": int(row["maintenance_flag"]),
            "fuel_trucks_available": float(row["fuel_trucks_available"]),
            "staff_available_pct": float(row["staff_available_pct"]),
            "passenger_count": int(row["passenger_count"]),
            "gate_occupancy_pct": float(row["gate_occupancy_pct"]),
        }

        state = self._discretize(
            features["departure_delay_minutes"],
            features["runway_congestion"],
            features["weather_risk"],
            features["maintenance_flag"],
            features["fuel_trucks_available"],
            features["staff_available_pct"],
        )

        if self._trained and state in self.q_table:
            action_idx = int(np.argmax(self.q_table[state]))
            policy_mode = "rl_trained"
        else:
            delay = features["departure_delay_minutes"]
            if features["maintenance_flag"] == 1 and delay > 15:
                action_idx = ACTIONS.index("request_maintenance_fast_track")
            elif features["runway_congestion"] > 70 and delay > 20:
                action_idx = ACTIONS.index("reroute_ground_traffic")
            elif features["weather_risk"] > 0.7:
                action_idx = ACTIONS.index("extend_turnaround_buffer")
            elif features["fuel_trucks_available"] <= 3:
                action_idx = ACTIONS.index("prioritize_fueling")
            elif features["staff_available_pct"] < 70:
                action_idx = ACTIONS.index("reassign_baggage_crew")
            else:
                action_idx = ACTIONS.index("standard_ops")
            policy_mode = "heuristic"

        action = ACTIONS[action_idx]
        reduction = self._simulate_reduction(action_idx, features)
        projected = round(max(0.0, features["departure_delay_minutes"] - reduction), 1)

        return {
            OPERATION_ID_COLUMN: operation_id,
            "action": action,
            "policy_mode": policy_mode,
            "reason": _ACTION_REASONS.get(action, "Recommended based on operational state."),
            "projected_delay_minutes": projected,
            "expected_delay_reduction_minutes": reduction,
        }