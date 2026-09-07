"""API route tests against a fixture-backed in-memory store."""
import pytest
from fastapi.testclient import TestClient

from eval_card_registry.main import app
from eval_card_registry.store.hf_store import get_store
from eval_card_registry.store import schemas as s
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.services.log_writer import ResolveLogWriter


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    """Replace the module-level store singleton with a fresh in-memory store."""
    from eval_card_registry.store import hf_store

    store = hf_store.RegistryStore()
    store._tables = {name: s.empty(name) for name in [
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)

    # Set up app.state so route dependencies work without lifespan
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")  # disabled (no bucket)
    return store


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_stats(self, client):
        r = client.get("/api/v1/stats")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "benchmarks" in data


# The lean HTTP ResolveResponse is the type-agnostic CORE + ancestry +
# resolution_detail. Type-specific entity fields are NOT on the resolve response;
# they live on the entity GET endpoints (models / benchmarks / families /
# composites). These are the fields a consumer must fetch there, not from resolve.
_DROPPED_RESOLVE_FIELDS = {
    "parent_canonical_id", "resolved_leaf_id", "root_model_id",
    "lineage_origin_org_id", "model_group_id", "model_family_id",
    "lineage_origin_model_id", "lineage_origin_model_org_id",
    "inference_platform", "resolution_granularity", "parents",
    "open_weights", "release_date", "params_billions",
    "family_key", "composite_keys", "category",
}
_CORE_RESOLVE_FIELDS = {
    "raw_value", "entity_type", "canonical_id", "strategy", "confidence",
    "created_new", "resolution_source", "review_status", "ancestry",
    "resolution_detail",
}


class TestResolve:
    def test_resolve_unknown_creates_draft(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "UnknownBenchmark",
            "entity_type": "benchmark",
            "source_config": "test_cfg",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["canonical_id"] is not None
        assert data["created_new"] is True
        assert data["review_status"] == "draft"

    def test_resolve_batch(self, client):
        r = client.post("/api/v1/resolve/batch", json=[
            {"raw_value": "BenchA", "entity_type": "benchmark"},
            {"raw_value": "BenchB", "entity_type": "benchmark"},
        ])
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_resolve_response_is_lean_core(self, client):
        """The HTTP response carries ONLY the type-agnostic core + ancestry
        + resolution_detail — no type-specific entity fields."""
        r = client.post("/api/v1/resolve", json={
            "raw_value": "SomeBench", "entity_type": "benchmark",
        })
        assert r.status_code == 200
        data = r.json()
        assert set(data) == _CORE_RESOLVE_FIELDS
        assert _DROPPED_RESOLVE_FIELDS.isdisjoint(data)
        assert data["raw_value"] == "SomeBench"
        assert data["entity_type"] == "benchmark"
        assert isinstance(data["ancestry"], list)
        assert isinstance(data["resolution_detail"], dict)


class TestEntityCRUD:
    def test_create_and_get_benchmark(self, client):
        r = client.post("/api/v1/benchmarks", json={
            "id": "my-bench",
            "display_name": "My Benchmark",
            "review_status": "reviewed",
        })
        assert r.status_code == 201

        r2 = client.get("/api/v1/benchmarks/my-bench")
        assert r2.status_code == 200
        assert r2.json()["display_name"] == "My Benchmark"

    def test_patch_benchmark(self, client):
        client.post("/api/v1/benchmarks", json={"id": "patch-bench", "display_name": "Old Name"})
        r = client.patch("/api/v1/benchmarks/patch-bench", json={"display_name": "New Name"})
        assert r.status_code == 200
        assert r.json()["display_name"] == "New Name"

    def test_get_nonexistent_returns_404(self, client):
        r = client.get("/api/v1/benchmarks/does-not-exist")
        assert r.status_code == 404

    def test_list_models(self, client):
        client.post("/api/v1/models", json={"id": "org/model-1", "display_name": "Model 1"})
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_list_search_with_null_columns_serializes(self, client):
        """Regression: list endpoints with a search term used to 500 when
        matching rows had nullable columns (pd.NA / NaN) — those fields
        must serialize to JSON null, not raise."""
        client.post("/api/v1/benchmarks", json={"id": "math", "display_name": "MATH"})
        r = client.get("/api/v1/benchmarks?search=math")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["parent_benchmark_id"] is None
        assert rows[0]["description"] is None

    def test_get_entity_with_null_columns_serializes(self, client):
        client.post("/api/v1/benchmarks", json={"id": "nullish", "display_name": "N"})
        r = client.get("/api/v1/benchmarks/nullish")
        assert r.status_code == 200
        body = r.json()
        assert body["parent_benchmark_id"] is None
        assert body["dataset_repo"] is None

    def test_post_model_with_parents_round_trips(self, client):
        """Regression: `parents` is a list-of-dicts on the API side but
        a JSON-encoded string in the parquet column. The route's
        `_JSON_FIELDS` set must include `parents` so POST→GET round-trips
        cleanly and the GET response decodes the column to a list."""
        body = {
            "id": "lab/test-model",
            "display_name": "Test Model",
            "org_id": "lab-org",
            "parents": [
                {"id": "lab/parent-a", "relationship": "variant", "axis": "size"},
                {"id": "lab/parent-b", "relationship": "finetune"},
            ],
        }
        r = client.post("/api/v1/models", json=body)
        assert r.status_code == 201, r.text

        def _strip_axis_none(parents):
            # Pydantic serializes Optional[axis] as `axis: None` for edges
            # where the input omitted axis. Drop None for comparison.
            return [{k: v for k, v in p.items() if v is not None} for p in parents]

        post_parents = r.json()["parents"]
        assert isinstance(post_parents, list)
        assert _strip_axis_none(post_parents) == body["parents"]

        # GET round-trip: same shape (axis=None survives the JSON round-trip
        # because POST stored it that way via Pydantic model_dump).
        g = client.get("/api/v1/models/lab/test-model")
        assert g.status_code == 200
        get_parents = g.json()["parents"]
        assert isinstance(get_parents, list)
        assert _strip_axis_none(get_parents) == body["parents"]


class TestNonFiniteBounds:
    def test_infinite_metric_bound_is_serialised_as_infinity_string(self, client):
        client.post("/api/v1/metrics", json={
            "id": "perplexity-like", "display_name": "Perplexity-like",
            "score_type": "continuous", "lower_is_better": True,
            # The wire form: JSON has no infinity literal, so the string is the
            # input form too (pydantic parses it to float("inf")).
            "min_score": 1.0, "max_score": "Infinity",
        })
        r = client.get("/api/v1/metrics/perplexity-like")
        assert r.status_code == 200
        body = r.json()
        assert body["min_score"] == 1.0
        assert body["max_score"] == "Infinity"
        assert "Infinity" in r.text and "inf" not in r.text.replace("Infinity", "")

    def test_validation_error_echoing_an_infinite_input_still_renders(self, client):
        """`1e400` parses to inf, and the 422 echoes the offending input
        back; the handler must use the registry response class or the echo
        itself raises and the client gets a 500."""
        r = client.post("/api/v1/metrics", content=b'{"id": [1e400], "display_name": "x"}',
                        headers={"content-type": "application/json"})
        assert r.status_code == 422, r.text
        body = r.json()
        assert body["detail"][0]["input"] == ["Infinity"]
        r = client.post("/api/v1/metrics", content=b"[1e400]",
                        headers={"content-type": "application/json"})
        assert r.status_code == 422, r.text
