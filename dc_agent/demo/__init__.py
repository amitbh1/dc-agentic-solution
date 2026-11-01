"""Self-contained demo: in-memory SQLite, a matching ontology and dev users."""
from .data import DEV_USERS, EXAMPLES, build_ontology, build_sqlite

__all__ = ["DEV_USERS", "EXAMPLES", "build_ontology", "build_sqlite"]
