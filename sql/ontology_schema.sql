-- Ontology store for the data centre agent (Postgres 15+, pgvector, pg_trgm).
-- Loaded by dc_agent/semantic/postgres.py into dc_agent.semantic.Ontology. The agent.feedback table
-- is written by the API (POST /feedback) and read by dc_agent.jobs.promote.
-- Embedding columns are vector(768) to match DC_EMBED_DIM (Vertex text-embedding-005).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE SCHEMA IF NOT EXISTS ontology;

-- A release is a published snapshot id. Use it in plan-cache keys and audit logs.
CREATE TABLE ontology.release (
    id         bigserial PRIMARY KEY,
    note       text NOT NULL,
    is_active  boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX one_active_release ON ontology.release (is_active) WHERE is_active;

CREATE TABLE ontology.concept (
    id             bigserial PRIMARY KEY,
    name           text NOT NULL UNIQUE CHECK (name ~ '^[a-z][a-z0-9_]*$'),
    description    text NOT NULL,
    source         text NOT NULL CHECK (source IN ('databricks', 'snowflake')),
    physical_table text NOT NULL,
    status         text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'stale', 'deprecated')),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE ontology.attribute (
    id              bigserial PRIMARY KEY,
    concept_id      bigint NOT NULL REFERENCES ontology.concept (id) ON DELETE CASCADE,
    name            text NOT NULL CHECK (name ~ '^[a-z][a-z0-9_]*$'),
    physical_column text NOT NULL,
    data_type       text NOT NULL,
    role            text NOT NULL CHECK (role IN ('key', 'dimension', 'measure', 'time')),
    description     text,
    unit            text,
    sensitivity     text NOT NULL DEFAULT 'internal' CHECK (sensitivity IN ('public', 'internal', 'restricted')),
    UNIQUE (concept_id, name),
    UNIQUE (id, concept_id)
);

-- Keys are physical column names, as the compiler expects.
-- priority: lower wins when several join paths exist.
CREATE TABLE ontology.relationship (
    id               bigserial PRIMARY KEY,
    left_concept_id  bigint NOT NULL REFERENCES ontology.concept (id) ON DELETE CASCADE,
    right_concept_id bigint NOT NULL REFERENCES ontology.concept (id) ON DELETE CASCADE,
    left_key         text NOT NULL,
    right_key        text NOT NULL,
    cardinality      text NOT NULL CHECK (cardinality IN ('many_to_one', 'one_to_many', 'one_to_one')),
    priority         int NOT NULL DEFAULT 100,
    description      text,
    CHECK (left_concept_id <> right_concept_id),
    UNIQUE (left_concept_id, right_concept_id, left_key, right_key)
);

CREATE TABLE ontology.metric (
    id          text PRIMARY KEY CHECK (id ~ '^[a-z][a-z0-9_]*$'),
    concept_id  bigint NOT NULL REFERENCES ontology.concept (id) ON DELETE CASCADE,
    attribute_id bigint NOT NULL,
    agg         text NOT NULL CHECK (agg IN ('sum', 'avg', 'min', 'max', 'count')),
    is_distinct boolean NOT NULL DEFAULT false,
    description text NOT NULL,
    unit        text,
    FOREIGN KEY (attribute_id, concept_id) REFERENCES ontology.attribute (id, concept_id)
);

-- Business terms the agent grounds against. Embeddings must match your model's dimension.
CREATE TABLE ontology.synonym (
    id           bigserial PRIMARY KEY,
    term         text NOT NULL,
    concept_id   bigint REFERENCES ontology.concept (id) ON DELETE CASCADE,
    attribute_id bigint REFERENCES ontology.attribute (id) ON DELETE CASCADE,
    metric_id    text REFERENCES ontology.metric (id) ON DELETE CASCADE,
    embedding    vector(768),
    CHECK (num_nonnulls(concept_id, attribute_id, metric_id) = 1)
);
CREATE INDEX synonym_term_trgm ON ontology.synonym USING gin (term gin_trgm_ops);
CREATE INDEX synonym_embedding ON ontology.synonym USING hnsw (embedding vector_cosine_ops);

-- Structured row scoping: the compiler adds "attribute IN (values of the user's claim)".
CREATE TABLE ontology.role_grant (
    role                   text NOT NULL,
    concept_id             bigint NOT NULL REFERENCES ontology.concept (id) ON DELETE CASCADE,
    max_sensitivity        text NOT NULL DEFAULT 'internal' CHECK (max_sensitivity IN ('public', 'internal', 'restricted')),
    row_scope_attribute_id bigint REFERENCES ontology.attribute (id),
    row_scope_claim        text,
    PRIMARY KEY (role, concept_id),
    CHECK ((row_scope_attribute_id IS NULL) = (row_scope_claim IS NULL))
);

-- Written by a scheduled job that diffs Unity Catalog and Snowflake information_schema.
CREATE TABLE ontology.drift_event (
    id           bigserial PRIMARY KEY,
    concept_id   bigint NOT NULL REFERENCES ontology.concept (id) ON DELETE CASCADE,
    attribute_id bigint REFERENCES ontology.attribute (id) ON DELETE CASCADE,
    kind         text NOT NULL CHECK (kind IN ('table_missing', 'column_missing', 'type_changed', 'new_column')),
    detail       jsonb NOT NULL DEFAULT '{}',
    detected_at  timestamptz NOT NULL DEFAULT now(),
    resolved_at  timestamptz
);

CREATE FUNCTION ontology.mark_stale() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE ontology.concept
       SET status = 'stale', updated_at = now()
     WHERE id = NEW.concept_id AND status = 'active'
       AND NEW.kind IN ('table_missing', 'column_missing', 'type_changed');
    RETURN NEW;
END $$;

CREATE TRIGGER drift_marks_stale AFTER INSERT ON ontology.drift_event
    FOR EACH ROW EXECUTE FUNCTION ontology.mark_stale();

-- Human-verified question -> plan -> SQL examples: few-shot source and evaluation set.
CREATE TABLE ontology.verified_query (
    id                bigserial PRIMARY KEY,
    question          text NOT NULL,
    question_embedding vector(768),
    plan              jsonb NOT NULL,
    sql               text NOT NULL,
    dialect           text NOT NULL,
    release_id        bigint NOT NULL REFERENCES ontology.release (id),
    verified_by       text,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX verified_query_embedding ON ontology.verified_query USING hnsw (question_embedding vector_cosine_ops);

-- Adjacency list in both directions; fan_out marks one-to-many traversal.
-- Stale or deprecated concepts drop out, so they cannot be joined.
CREATE VIEW ontology.join_graph AS
SELECT l.name AS from_concept, r.name AS to_concept,
       e.left_key AS from_key, e.right_key AS to_key,
       (e.cardinality = 'one_to_many') AS fan_out, e.priority
  FROM ontology.relationship e
  JOIN ontology.concept l ON l.id = e.left_concept_id AND l.status = 'active'
  JOIN ontology.concept r ON r.id = e.right_concept_id AND r.status = 'active'
UNION ALL
SELECT r.name, l.name, e.right_key, e.left_key,
       (e.cardinality = 'many_to_one'), e.priority
  FROM ontology.relationship e
  JOIN ontology.concept l ON l.id = e.left_concept_id AND l.status = 'active'
  JOIN ontology.concept r ON r.id = e.right_concept_id AND r.status = 'active';

-- Seed data: example Databricks tables. Replace with your own concepts before use.
INSERT INTO ontology.release (note, is_active) VALUES ('initial seed', true);

INSERT INTO ontology.concept (name, description, source, physical_table) VALUES
    ('power_reading', 'Rack power draw samples', 'databricks', 'main.dc.power_readings'),
    ('rack',          'Physical rack',           'databricks', 'main.dc.racks'),
    ('facility',      'Data centre facility',    'databricks', 'main.dc.facilities');

INSERT INTO ontology.attribute (concept_id, name, physical_column, data_type, role, unit)
SELECT c.id, v.name, v.col, v.dtype, v.role, v.unit
  FROM (VALUES
        ('power_reading', 'power_kw', 'power_kw',      'double',    'measure',   'kW'),
        ('power_reading', 'ts',       'reading_ts',    'timestamp', 'time',      NULL),
        ('rack',          'id',       'rack_id',       'string',    'key',       NULL),
        ('rack',          'row',      'row_label',     'string',    'dimension', NULL),
        ('facility',      'name',     'facility_name', 'string',    'dimension', NULL),
        ('facility',      'region',   'region',        'string',    'dimension', NULL),
        ('facility',      'tier',     'tier',          'string',    'dimension', NULL)
       ) AS v(concept, name, col, dtype, role, unit)
  JOIN ontology.concept c ON c.name = v.concept;

INSERT INTO ontology.relationship (left_concept_id, right_concept_id, left_key, right_key, cardinality)
SELECT l.id, r.id, v.lk, v.rk, v.card
  FROM (VALUES
        ('power_reading', 'rack',     'rack_id',     'rack_id',     'many_to_one'),
        ('rack',          'facility', 'facility_id', 'facility_id', 'many_to_one')
       ) AS v(lc, rc, lk, rk, card)
  JOIN ontology.concept l ON l.name = v.lc
  JOIN ontology.concept r ON r.name = v.rc;

INSERT INTO ontology.metric (id, concept_id, attribute_id, agg, description, unit)
SELECT 'avg_power_kw', a.concept_id, a.id, 'avg', 'Average rack power draw', 'kW'
  FROM ontology.attribute a
  JOIN ontology.concept c ON c.id = a.concept_id
 WHERE c.name = 'power_reading' AND a.name = 'power_kw';

INSERT INTO ontology.synonym (term, metric_id) VALUES
    ('average power', 'avg_power_kw'),
    ('power draw', 'avg_power_kw');

INSERT INTO ontology.synonym (term, attribute_id)
SELECT 'geography', a.id
  FROM ontology.attribute a
  JOIN ontology.concept c ON c.id = a.concept_id
 WHERE c.name = 'facility' AND a.name = 'region';

-- Grounding query (run exact/trigram first, then embeddings):
--   SELECT term, concept_id, attribute_id, metric_id
--     FROM ontology.synonym
--    ORDER BY embedding <=> $1
--    LIMIT 5;

-- User feedback on answers (POST /feedback). The question text is stored only for 'up' ratings.
CREATE SCHEMA IF NOT EXISTS agent;
CREATE TABLE agent.feedback (
    id         bigserial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    user_id    text NOT NULL,
    role       text NOT NULL,
    tenant     text,
    thread_id  text NOT NULL,
    request_id text,
    rating     text NOT NULL CHECK (rating IN ('up', 'down')),
    comment    text CHECK (length(comment) <= 500),
    question   text,
    q_hash     text NOT NULL,
    plan       jsonb,
    sql        text,
    dialect    text,
    release_id bigint,
    outcome    text NOT NULL
);
CREATE INDEX feedback_promotion ON agent.feedback (release_id, q_hash) WHERE plan IS NOT NULL;
CREATE INDEX feedback_created ON agent.feedback (created_at);
