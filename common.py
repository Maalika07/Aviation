
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR   = Path(__file__).resolve().parent
DATASET_PATH  = PROJECT_DIR / "realistic_airport_dataset_improved.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
OPERATION_ID_COLUMN = "operation_id"


@dataclass
class KnowledgeDocument:
    doc_id:      str
    title:       str
    content:     str
    source_type: str
    metadata:    dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        return f"{self.title}\n{self.content}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id":      self.doc_id,
            "title":       self.title,
            "content":     self.content,
            "source_type": self.source_type,
            "metadata":    self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeDocument":
        return cls(
            doc_id=payload["doc_id"],
            title=payload["title"],
            content=payload["content"],
            source_type=payload["source_type"],
            metadata=payload.get("metadata", {}),
        )


@dataclass
class RetrievedDocument:
    doc:       KnowledgeDocument
    score:     float
    retriever: str
    rank:      int




_RNG = np.random.default_rng(42)
_AIRPORTS = ["BOM", "DEL", "BLR", "HYD", "MAA", "CCU", "AMD", "PNQ", "COK",
             "GOI", "JAI", "LKO", "BHO", "IXC", "DXB", "SIN", "DOH", "LHR"]
_ZONE_TERMINAL = {"A": "T1", "B": "T2", "C": "T3", "D": "T4", "E": "T5"}
_RUNWAYS = ["09L", "09R", "27L", "27R", "14", "32"]

_WEATHER_TEMP    = {"Clear": 29, "Windy": 26, "Rain": 23, "Fog": 19, "Storm": 21}
_WEATHER_VIS     = {"Clear": 10.0, "Windy": 8.0, "Rain": 3.5, "Fog": 1.2, "Storm": 2.0}
_WEATHER_WIND    = {"Clear": 6,   "Windy": 22,  "Rain": 14,  "Fog": 4,   "Storm": 38}


_CONG_NUM        = {"Low": 22, "Medium": 54, "High": 83}


_STAFF_PCT       = {"Low": 71, "Medium": 84, "High": 95}


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    n  = len(df)
    rng = np.random.default_rng(42)   

    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=False)
    df["scheduled_arrival"] = df["timestamp"]
    # scheduled_departure = arrival + scheduled turnaround
    df["scheduled_departure"] = df["timestamp"] + pd.to_timedelta(
        df["scheduled_turnaround_minutes"], unit="min"
    )

   
    df["airline"] = df["flight_id"].str.extract(r"^([A-Z0-9]{2,3})(?=\d)")[0].fillna("XX")

    pool = _AIRPORTS
    idx_orig = df["flight_id"].apply(lambda fid: hash(fid + "org") % len(pool))
    idx_dest = df["flight_id"].apply(lambda fid: hash(fid + "dst") % len(pool))
    df["origin"]      = idx_orig.map(lambda i: pool[i])
    df["destination"] = idx_dest.map(lambda i: pool[i])
 
    same = df["origin"] == df["destination"]
    df.loc[same, "destination"] = df.loc[same, "origin"].map(
        lambda o: pool[(pool.index(o) + 1) % len(pool)]
    )
    zone = df["gate"].str.extract(r"^([A-Z])")[0].fillna("C")
    df["terminal"] = zone.map(_ZONE_TERMINAL).fillna("T3")
    gate_num       = df["gate"].str.extract(r"(\d+)")[0].fillna("1")
    df["gate_id"]  = df["terminal"] + "-G" + gate_num  


    df["runway_id"] = df["flight_id"].apply(
        lambda fid: _RUNWAYS[hash(fid) % len(_RUNWAYS)]
    )

    df["temperature_c"]  = (
        df["weather_condition"].map(_WEATHER_TEMP).fillna(27).astype(float)
        + rng.normal(0, 2, n)
    ).round(1)

    df["visibility_km"]  = df["weather_condition"].map(_WEATHER_VIS).fillna(8.0)

    df["wind_speed_kts"] = (
        df["weather_condition"].map(_WEATHER_WIND).fillna(10)
        + rng.integers(-3, 8, n)
    ).clip(0, 60)

   
    df["runway_congestion"] = (
        df["runway_congestion"].map(_CONG_NUM).fillna(40).astype(int)
        + rng.integers(-8, 8, n)
    ).clip(0, 100).astype(int)

  
    df["gate_occupancy_pct"] = (
        45
        + df["peak_hour"] * 22
        + df["runway_congestion"] * 0.18
        + rng.integers(-8, 8, n)
    ).clip(15, 99).astype(int)


    df["baggage_vehicles_available"] = (
        8 - (df["baggage_delay_minutes"] / 12).astype(int)
        + rng.integers(0, 2, n)
    ).clip(1, 9).astype(int)

  
    df["staff_available_pct"] = (
        df["staff_availability"].map(_STAFF_PCT).fillna(82).astype(int)
        + rng.integers(-4, 4, n)
    ).clip(55, 100).astype(int)

    df["maintenance_flag"] = (
        ((df["staff_availability"] == "Low") & (df["baggage_delay_minutes"] > 18))
        | (df["weather_condition"] == "Storm")
        | (df["fuel_trucks_available"] <= 2)
    ).astype(int)

    
    df["actual_turnaround_minutes"] = (
        df["scheduled_turnaround_minutes"] + df["actual_turnaround_delay_minutes"]
    )

    
    df["departure_delay_minutes"] = df["actual_turnaround_delay_minutes"].astype(int)

    
    df["disruption_risk"]   = df["delay_probability"].round(3)
    df["disruption_target"] = (df["actual_turnaround_delay_minutes"] > 14).astype(int)

    return df




def _build_operation_ids(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [f"OP-{i:06d}" for i in range(1, len(df) + 1)],
        index=df.index,
        dtype="string",
    )


def load_airport_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {DATASET_PATH}"
            f"\nMake sure 'realistic_airport_dataset_improved.csv' is in:\n  {PROJECT_DIR}"
        )

    df = pd.read_csv(DATASET_PATH)

    
    df = _derive_columns(df)

    
    if OPERATION_ID_COLUMN not in df.columns:
        df.insert(0, OPERATION_ID_COLUMN, _build_operation_ids(df))
    df[OPERATION_ID_COLUMN] = df[OPERATION_ID_COLUMN].astype("string")

    if not df[OPERATION_ID_COLUMN].is_unique:
        df[OPERATION_ID_COLUMN] = _build_operation_ids(df).astype("string")

    return df




def save_documents(path: Path, docs: list[KnowledgeDocument]) -> None:
    path.write_text(
        json.dumps([doc.to_dict() for doc in docs], indent=2),
        encoding="utf-8",
    )


def load_documents(path: Path) -> list[KnowledgeDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeDocument.from_dict(item) for item in payload]



def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9/._ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9/._-]+", normalize_text(text))
