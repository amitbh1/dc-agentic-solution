"""LangGraph wiring: a fixed graph with one bounded repair loop and one optional human approval.

    guard_input -> ground -> planner -> compile -> cost_check -> execute -> check_result [-> enrich] -> respond
                              ^-----------'  (repairable errors, at most max_repairs times)

cost_check may pause the run (interrupt) until the user approves an expensive query. enrich is
added only when tools are configured. Every node can short-circuit to respond; a fatal error
anywhere ends the turn with a safe message.
"""
from __future__ import annotations

import functools
import logging
import time

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph

from ..observability import audit
from ..observability.metrics import NODE_ERRORS, NODE_SECONDS
from .deps import Deps
from .messages import MSG, fatal
from .nodes import AgentNodes
from .state import AgentState

log = logging.getLogger("dc_agent.graph")


def instrument(name: str, fn):
    """Latency metric per node, and a safety net: an unexpected exception ends the turn
    gracefully instead of failing the whole request."""

    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        try:
            return fn(state)
        except GraphBubbleUp:  # LangGraph control flow (interrupts), not an error
            raise
        except Exception as exc:
            log.exception("node failed", extra={"fields": {"node": name}})
            NODE_ERRORS.labels(name).inc()
            audit("node_failed", node=name, error=repr(exc)[:300])
            return {**fatal("internal", "internal", f"{name} failed"), "answer": MSG["internal"]}
        finally:
            NODE_SECONDS.labels(name).observe(time.perf_counter() - t0)

    return wrapper


def build_graph(deps: Deps, checkpointer=None):
    n = AgentNodes(deps)
    max_repairs = deps.settings.max_repairs
    with_tools = bool(deps.tools)

    def can_repair(state: AgentState) -> bool:
        return bool(state.get("error")) and not state.get("error_fatal") and state.get("attempts", 0) <= max_repairs

    def after_guard(state: AgentState) -> str:
        return "respond" if state.get("error") else "ground"

    def after_ground(state: AgentState) -> str:
        return "respond" if state.get("clarification") or state.get("error") else "planner"

    def after_planner(state: AgentState) -> str:
        if state.get("clarification") or state.get("error_fatal"):
            return "respond"
        if state.get("plan") and not state.get("error"):
            return "compile"
        return "planner" if can_repair(state) else "respond"

    def after_compile(state: AgentState) -> str:
        if state.get("sql"):
            return "cost_check"
        return "planner" if can_repair(state) else "respond"

    def after_cost(state: AgentState) -> str:
        return "respond" if state.get("error") else "execute"

    def after_execute(state: AgentState) -> str:
        return "respond" if state.get("error") else "check_result"

    nodes = [
        ("guard_input", n.guard_input), ("ground", n.ground), ("planner", n.planner), ("compile", n.compile),
        ("cost_check", n.cost_check), ("execute", n.execute), ("check_result", n.check_result),
        ("respond", n.respond),
    ]
    if with_tools:
        nodes.append(("enrich", n.enrich))
    g = StateGraph(AgentState)
    for name, fn in nodes:
        g.add_node(name, instrument(name, fn))

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, {"ground": "ground", "respond": "respond"})
    g.add_conditional_edges("ground", after_ground, {"planner": "planner", "respond": "respond"})
    g.add_conditional_edges(
        "planner", after_planner, {"compile": "compile", "planner": "planner", "respond": "respond"}
    )
    g.add_conditional_edges(
        "compile", after_compile, {"cost_check": "cost_check", "planner": "planner", "respond": "respond"}
    )
    g.add_conditional_edges("cost_check", after_cost, {"execute": "execute", "respond": "respond"})
    g.add_conditional_edges("execute", after_execute, {"check_result": "check_result", "respond": "respond"})
    if with_tools:
        g.add_edge("check_result", "enrich")
        g.add_edge("enrich", "respond")
    else:
        g.add_edge("check_result", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)
