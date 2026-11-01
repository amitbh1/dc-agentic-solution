from __future__ import annotations

from dataclasses import dataclass

SENS = {"public": 0, "internal": 1, "restricted": 2}


@dataclass(frozen=True)
class Attribute:
    name: str
    column: str
    data_type: str
    role: str  # key | dimension | measure | time
    sensitivity: str = "internal"
    description: str = ""
    unit: str | None = None


@dataclass(frozen=True)
class Concept:
    name: str
    source: str  # sqlglot dialect: databricks | snowflake | sqlite
    table: str
    description: str
    attrs: dict


@dataclass(frozen=True)
class Edge:
    left: str
    right: str
    left_key: str  # physical column
    right_key: str  # physical column
    cardinality: str  # many_to_one | one_to_many | one_to_one (left -> right)
    priority: int = 100  # lower wins when several shortest join paths exist


@dataclass(frozen=True)
class Metric:
    id: str
    concept: str
    agg: str  # sum | avg | min | max | count
    attr: str
    description: str = ""
    unit: str | None = None
    distinct: bool = False


@dataclass(frozen=True)
class Grant:
    role: str
    concept: str
    max_sensitivity: str = "internal"
    scope_ref: str | None = None  # concept.attribute restricted by a user claim
    claim: str | None = None


@dataclass(frozen=True)
class Synonym:
    term: str
    kind: str  # concept | attribute | metric
    target: str
    vector: tuple | None = None  # embedding of term, from ontology.synonym.embedding


@dataclass(frozen=True)
class Example:
    """A human-verified question and plan (ontology.verified_query), used as a few-shot example."""

    question: str
    plan: dict
    vector: tuple | None = None
