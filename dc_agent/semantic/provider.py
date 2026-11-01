from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..observability.metrics import ONTOLOGY_PROBLEMS, ONTOLOGY_RELOADS
from .ontology import Ontology
from .validation import validate_ontology

log = logging.getLogger("dc_agent.ontology")


class OntologyProvider:
    """Serves the active ontology and reloads it on a TTL. Keeps the last good copy on failure.

    Each new release is validated; problems are logged and exported as a gauge so a broken
    release (for example a concept missing its row scope) raises an alert.
    """

    def __init__(self, loader: Callable[[], Ontology], ttl_s: int = 60):
        self._loader = loader
        self._ttl = ttl_s
        self._onto: Ontology | None = None
        self._at = 0.0
        self._lock = threading.Lock()

    def get(self) -> Ontology:
        with self._lock:
            if self._onto is None or time.monotonic() - self._at > self._ttl:
                try:
                    onto = self._loader()
                    ONTOLOGY_RELOADS.labels("ok").inc()
                    if self._onto is None or onto.release_id != self._onto.release_id:
                        self._report(onto)
                    self._onto = onto
                except Exception:
                    ONTOLOGY_RELOADS.labels("failed").inc()
                    if self._onto is None:
                        raise
                    log.warning("ontology reload failed; serving previous copy", exc_info=True)
                self._at = time.monotonic()
            return self._onto

    @staticmethod
    def _report(onto: Ontology) -> None:
        problems = validate_ontology(onto)
        ONTOLOGY_PROBLEMS.set(len(problems))
        for p in problems:
            log.warning("ontology problem", extra={"fields": {"release": onto.release_id, "problem": p}})
