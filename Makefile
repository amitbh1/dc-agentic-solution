.PHONY: install test integration eval lint run demo docker validate drift embed promote retention
install:
	pip install -r requirements-dev.txt
test:
	pytest -q
# needs docker compose up -d (or any Postgres with pgvector and a Redis)
integration:
	DC_TEST_PG_DSN=postgresql://postgres:dev-only-change-me@localhost:5432/dcagent \
	DC_TEST_REDIS_URL=redis://localhost:6379/0 pytest -q tests/integration
eval:
	python -m evals.run_evals
lint:
	ruff check .
demo:
	python demo.py
run:
	uvicorn dc_agent.api:app --reload
docker:
	docker build -t dc-agent .
# scheduled jobs (prod): see docs/RUNBOOK.md for cadence
validate:
	python -m dc_agent.jobs.validate
drift:
	python -m dc_agent.jobs.drift
embed:
	python -m dc_agent.jobs.embed
promote:
	python -m dc_agent.jobs.promote
retention:
	python -m dc_agent.jobs.retention
