"""Self-contained demo: in-memory SQLite data plus a matching ontology. No credentials needed."""
from __future__ import annotations

import datetime as dt
import random
import sqlite3

from ..security.identity import UserContext
from ..semantic import Attribute, Concept, Edge, Example, Grant, Metric, Ontology, Synonym

DEV_USERS = {
    "dev-analyst": UserContext("u-analyst", "analyst", {"regions": ["West", "South"]}, tenant="demo"),
    "dev-admin": UserContext("u-admin", "admin", {"regions": ["West", "South", "North"]}, tenant="demo"),
}

# Stand-ins for ontology.verified_query rows: offered to the planner as few-shot examples.
EXAMPLES = [
    Example(
        "peak power per row in tier III",
        {
            "metrics": ["max_power_kw"],
            "dimensions": ["rack.row"],
            "filters": [{"ref": "facility.tier", "op": "=", "values": ["III"]}],
            "limit": 100,
        },
    ),
    Example(
        "average power by day for the last 7 days",
        {
            "metrics": ["avg_power_kw"],
            "time_range": {"ref": "power_reading.ts", "last": 7, "unit": "day"},
            "time_grain": {"ref": "power_reading.ts", "grain": "day"},
            "limit": 100,
        },
    ),
]


def build_sqlite(seed: int = 7, now: dt.datetime | None = None) -> sqlite3.Connection:
    """Hourly readings for the 24 hours before `now` (default: the current UTC hour), so relative
    time questions ("last 6 hours") always have data. Values depend only on the seed."""
    now = (now or dt.datetime.now(dt.timezone.utc)).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    start = now - dt.timedelta(hours=23)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(
        """
        CREATE TABLE facilities (facility_id INTEGER PRIMARY KEY, facility_name TEXT, region TEXT, tier TEXT);
        CREATE TABLE racks (rack_id INTEGER PRIMARY KEY, row_label TEXT, facility_id INTEGER);
        CREATE TABLE power_readings (rack_id INTEGER, power_kw REAL, reading_ts TEXT);
        """
    )
    rng = random.Random(seed)
    facilities = [(1, "Mumbai-1", "West", "III", 6.0), (2, "Chennai-1", "South", "III", 5.0), (3, "Delhi-1", "North", "IV", 8.0)]
    conn.executemany("INSERT INTO facilities VALUES (?,?,?,?)", [f[:4] for f in facilities])
    rack_id = 0
    for fid, _, _, _, base in facilities:
        for i in range(4):
            rack_id += 1
            conn.execute("INSERT INTO racks VALUES (?,?,?)", (rack_id, f"R{i // 2 + 1}", fid))
            for hour in range(24):
                kw = round(base + rng.uniform(-1.0, 1.5) + i * 0.3, 2)
                ts = (start + dt.timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%S")
                conn.execute("INSERT INTO power_readings VALUES (?,?,?)", (rack_id, kw, ts))
    conn.commit()
    conn.execute("PRAGMA query_only = ON")
    return conn


def build_ontology() -> Ontology:
    concepts = [
        Concept(
            "power_reading", "sqlite", "power_readings", "Rack power draw samples",
            {
                "power_kw": Attribute("power_kw", "power_kw", "double", "measure", description="Power in kW", unit="kW"),
                "ts": Attribute("ts", "reading_ts", "timestamp", "time", description="Reading time"),
            },
        ),
        Concept(
            "rack", "sqlite", "racks", "Physical rack",
            {
                "id": Attribute("id", "rack_id", "bigint", "key"),
                "row": Attribute("row", "row_label", "string", "dimension", description="Row label in the hall"),
            },
        ),
        Concept(
            "facility", "sqlite", "facilities", "Data centre facility",
            {
                "name": Attribute("name", "facility_name", "string", "dimension", description="Facility name"),
                "region": Attribute("region", "region", "string", "dimension", description="Geographic region"),
                "tier": Attribute("tier", "tier", "string", "dimension", description="Uptime tier"),
            },
        ),
    ]
    edges = [
        Edge("power_reading", "rack", "rack_id", "rack_id", "many_to_one"),
        Edge("rack", "facility", "facility_id", "facility_id", "many_to_one"),
    ]
    metrics = [
        Metric("avg_power_kw", "power_reading", "avg", "power_kw", "Average rack power draw", "kW"),
        Metric("max_power_kw", "power_reading", "max", "power_kw", "Peak rack power draw", "kW"),
        Metric("rack_count", "rack", "count", "id", "Number of racks"),
    ]
    synonyms = [
        Synonym("average power", "metric", "avg_power_kw"),
        Synonym("power draw", "metric", "avg_power_kw"),
        Synonym("peak power", "metric", "max_power_kw"),
        Synonym("how many racks", "metric", "rack_count"),
        Synonym("number of racks", "metric", "rack_count"),
        Synonym("rack count", "metric", "rack_count"),
        Synonym("maximum power", "metric", "max_power_kw"),
        Synonym("facility", "attribute", "facility.name"),
        Synonym("site", "attribute", "facility.name"),
        Synonym("geography", "attribute", "facility.region"),
        Synonym("area", "attribute", "facility.region"),
    ]
    scope = dict(scope_ref="facility.region", claim="regions")
    grants = [
        Grant("analyst", "power_reading", "internal", **scope),
        Grant("analyst", "rack", "internal", **scope),
        Grant("analyst", "facility", "internal", **scope),
        Grant("admin", "power_reading", "restricted"),
        Grant("admin", "rack", "restricted"),
        Grant("admin", "facility", "restricted"),
    ]
    return Ontology(concepts, edges, metrics, synonyms, grants, release_id=1, examples=EXAMPLES)
