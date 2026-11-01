"""Validate an ontology release before publishing it (CI gate for ontology changes).

    python -m dc_agent.jobs.validate          # the active release in DC_PG_DSN
    python -m dc_agent.jobs.validate --demo   # the built-in demo ontology

Exit code 1 if there are problems, for example a concept without a row scope for a scoped role.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)

    from ..semantic import validate_ontology

    if args.demo:
        from ..demo import build_ontology

        onto = build_ontology()
    else:
        from ..core.config import Settings
        from ..semantic import load_from_postgres

        onto = load_from_postgres(Settings.from_env().pg_dsn)
    problems = validate_ontology(onto)
    for p in problems:
        print(p)
    print(f"release {onto.release_id}: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
