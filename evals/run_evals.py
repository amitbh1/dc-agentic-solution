"""Regression evals: score the agent on a golden set by RESULT match, not SQL text.

    python -m evals.run_evals                                    # in-process, demo mode
    python -m evals.run_evals --base-url https://agent.internal \
        --tokens-file tokens.json                                # against a deployed API

tokens.json maps the `user` labels in golden.jsonl to bearer tokens. A case may list `history`:
questions asked first in the same session (for follow-ups). Exit code 1 if the pass rate is below
--min-pass. For production, keep a golden set built from verified queries on staging
data, and run it on every prompt, model, ontology or code change.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def load_cases(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_inprocess(cases: list) -> list:
    from dc_agent.bootstrap import build_agent
    from dc_agent.core.config import Settings
    from dc_agent.agent import outcome_of

    bundle = build_agent(Settings(mode="demo"))
    results = []
    for i, case in enumerate(cases):
        user = bundle.dev_users[case["user"]]
        config = {"configurable": {"thread_id": f"{user.user_id}:eval-{i}"}}
        for prior in case.get("history", []):
            bundle.agent.invoke({"question": prior, "user": user.to_dict()}, config=config)
        state = bundle.agent.invoke({"question": case["question"], "user": user.to_dict()}, config=config)
        results.append({"outcome": outcome_of(state), "columns": state.get("columns", []), "rows": state.get("rows", [])})
    return results


def run_http(cases: list, base_url: str, tokens: dict) -> list:
    results = []
    for i, case in enumerate(cases):
        token = tokens.get(case["user"], case["user"])
        session = f"eval-{i}-{uuid.uuid4().hex[:8]}"
        body = {}
        for question in [*case.get("history", []), case["question"]]:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/ask",
                data=json.dumps({"question": question, "session_id": session}).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.load(resp)
            except urllib.error.HTTPError as exc:
                body = {"outcome": f"http_{exc.code}", "columns": [], "rows": []}
        results.append(body)
    return results


def check(case: dict, result: dict) -> list:
    errors = []
    if result.get("outcome") != case["expect_outcome"]:
        errors.append(f"outcome {result.get('outcome')!r} != {case['expect_outcome']!r}")
        return errors
    cols, rows = result.get("columns", []), result.get("rows", [])
    if "expect_columns" in case and cols != case["expect_columns"]:
        errors.append(f"columns {cols} != {case['expect_columns']}")
    for col, expected in case.get("expect_values", {}).items():
        if col not in cols:
            errors.append(f"missing column {col}")
            continue
        got = {r[cols.index(col)] for r in rows}
        if got != set(expected):
            errors.append(f"{col}: {sorted(got)} != {sorted(expected)}")
    if len(rows) < case.get("min_rows", 0):
        errors.append(f"only {len(rows)} rows")
    if "max_rows" in case and len(rows) > case["max_rows"]:
        errors.append(f"{len(rows)} rows, expected at most {case['max_rows']}")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).with_name("golden.jsonl")))
    parser.add_argument("--base-url")
    parser.add_argument("--tokens-file")
    parser.add_argument("--min-pass", type=float, default=1.0)
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))
    if args.base_url:
        tokens = json.loads(Path(args.tokens_file).read_text()) if args.tokens_file else {}
        results = run_http(cases, args.base_url, tokens)
    else:
        results = run_inprocess(cases)

    passed = 0
    for case, result in zip(cases, results):
        errors = check(case, result)
        passed += not errors
        print(f"{'PASS' if not errors else 'FAIL'} {case['id']}" + "".join(f"\n     {e}" for e in errors))
    rate = passed / len(cases)
    print(f"\n{passed}/{len(cases)} passed ({rate:.0%})")
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
