import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("DC_MODE", "demo")
    monkeypatch.setenv("DC_RATE_BURST", "3")
    from dc_agent.api import app

    with TestClient(app) as c:
        yield c


def auth(token):
    return {"Authorization": f"Bearer {token}"}


Q = {"question": "What is the average power by region?"}


def test_requires_auth(client):
    assert client.post("/ask", json=Q).status_code == 401
    assert client.post("/ask", json=Q, headers=auth("wrong")).status_code == 401


def test_health_ready_metrics(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_ask_scopes_rows_and_hides_sql_from_analyst(client):
    r = client.post("/ask", json=Q, headers=auth("dev-analyst"))
    body = r.json()
    assert r.status_code == 200 and body["outcome"] == "ok"
    assert {row[0] for row in body["rows"]} == {"West", "South"}
    assert "sql" not in body
    assert r.headers["X-Request-ID"]


def test_admin_gets_sql(client):
    body = client.post("/ask", json=Q, headers=auth("dev-admin")).json()
    assert "sql" in body and {row[0] for row in body["rows"]} == {"West", "South", "North"}


def test_rate_limit_returns_429(client):
    codes = [client.post("/ask", json=Q, headers=auth("dev-analyst")).status_code for _ in range(5)]
    assert 429 in codes


def test_oversized_and_invalid_input_rejected(client):
    assert client.post("/ask", json={"question": "x" * 2000}, headers=auth("dev-admin")).status_code == 422
    assert client.post("/ask", json={**Q, "session_id": "bad id!"}, headers=auth("dev-admin")).status_code == 422


def test_busy_session_returns_409_without_spending_rate_limit(client):
    b = client.app.state.bundle
    user = b.dev_users["dev-analyst"]
    assert b.locks.try_acquire(f"{user.user_id}:s1")
    body = {**Q, "session_id": "s1"}
    codes = [client.post("/ask", json=body, headers=auth("dev-analyst")).status_code for _ in range(5)]
    assert codes == [409] * 5
    b.locks.release(f"{user.user_id}:s1")
    assert client.post("/ask", json=body, headers=auth("dev-analyst")).status_code == 200


def test_stream_emits_nodes_then_result(client):
    r = client.post("/ask/stream", json=Q, headers=auth("dev-admin"))
    events = [line.split(": ", 1)[1] for line in r.text.splitlines() if line.startswith("event: ")]
    assert events[0] == "node" and events[-1] == "result"


@pytest.fixture()
def approval_client(monkeypatch):
    monkeypatch.setenv("DC_MODE", "demo")
    monkeypatch.setenv("DC_APPROVAL_SCAN_BYTES", "1000")
    from dc_agent.api import create_app

    with TestClient(create_app()) as c:
        yield c


def test_approval_over_http(approval_client):
    c = approval_client
    body = {**Q, "session_id": "ap1"}
    paused = c.post("/ask", json=body, headers=auth("dev-admin")).json()
    assert paused["outcome"] == "approval_required" and paused["approval"]["estimated_bytes"] > 1000
    assert "sql" not in paused and paused["rows"] == []
    done = c.post("/ask/approve", json={"session_id": "ap1", "approve": True}, headers=auth("dev-admin")).json()
    assert done["outcome"] == "ok" and len(done["rows"]) == 3
    again = c.post("/ask/approve", json={"session_id": "ap1", "approve": True}, headers=auth("dev-admin"))
    assert again.status_code == 409


def test_other_user_cannot_approve(approval_client):
    c = approval_client
    c.post("/ask", json={**Q, "session_id": "ap2"}, headers=auth("dev-admin"))
    r = c.post("/ask/approve", json={"session_id": "ap2", "approve": True}, headers=auth("dev-analyst"))
    assert r.status_code == 409  # the analyst's own thread has nothing pending


def test_stream_reports_approval_required(approval_client):
    r = approval_client.post("/ask/stream", json={**Q, "session_id": "ap3"}, headers=auth("dev-admin"))
    result = [line for line in r.text.splitlines() if line.startswith("data: ")][-1]
    assert '"outcome": "approval_required"' in result


def test_feedback_records_rating(client):
    assert client.post("/feedback", json={"rating": "up"}, headers=auth("dev-admin")).status_code == 409
    client.post("/ask", json={**Q, "session_id": "fb"}, headers=auth("dev-admin"))
    r = client.post("/feedback", json={"session_id": "fb", "rating": "up", "comment": "spot on"}, headers=auth("dev-admin"))
    assert r.status_code == 200
    down = client.post("/feedback", json={"session_id": "fb", "rating": "down"}, headers=auth("dev-admin"))
    assert down.status_code == 200
    up_rec, down_rec = client.app.state.bundle.feedback.all()[-2:]
    assert up_rec.question == Q["question"] and up_rec.plan["metrics"] == ["avg_power_kw"] and up_rec.comment == "spot on"
    assert down_rec.question is None and down_rec.q_hash == up_rec.q_hash
    bad = client.post("/feedback", json={"rating": "meh"}, headers=auth("dev-admin"))
    assert bad.status_code == 422


def test_chunked_body_over_limit_is_rejected(client):
    def chunks():
        for _ in range(20):
            yield b"x" * 1024

    r = client.post("/ask", content=chunks(), headers={**auth("dev-admin"), "Content-Type": "application/json"})
    assert r.status_code == 413


def test_response_has_notes_and_outcome_fields(client):
    body = client.post("/ask", json=Q, headers=auth("dev-admin")).json()
    assert body["notes"] == [] and body["outcome"] == "ok" and body["release"] == 1
