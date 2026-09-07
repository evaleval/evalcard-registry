"""Contract tests for the lean resolve HTTP API + full hierarchy scope.

Covers, against a fully-populated in-memory hierarchy store:
  - `ancestry` (model group/family chain; benchmark→family→composite;
    family→composite; composite/root → []),
  - typed `resolution_detail.<type>` (model granularity; benchmark
    level/matched_subset incl. the slice + subset-fold cases),
  - GET /families/{id} and GET /composites/{id},
  - composite/family resolving via resolve(entity_type=...).

The fixtures/-backed gate invariants (tests/test_gate_invariants.py) own
the resolution-outcome guarantees; this module owns the RESPONSE SHAPE.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eval_card_registry.main import app
from eval_card_registry.store import hf_store, schemas as s
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.services.log_writer import ResolveLogWriter


def _row(table: str, **vals) -> dict:
    """Build a full row for `table` with schema defaults + overrides.
    JSON-encodes list/dict overrides for the *_ids / *_keys / configs
    columns so the stored shape matches the parquet contract."""
    base = {col: None for col in s._SCHEMAS[table]}
    base.update({"review_status": "reviewed",
                 "created_at": "2026-01-01T00:00:00+00:00",
                 "updated_at": "2026-01-01T00:00:00+00:00"})
    for k, v in vals.items():
        base[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return base


def _alias(raw, etype, cid):
    return {"id": f"{etype}:{raw}", "raw_value": raw, "entity_type": etype,
            "canonical_id": cid, "source_config": None, "source_field": None,
            "status": "confirmed", "strategy": "exact", "confidence": 1.0,
            "notes": None, "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00"}


@pytest.fixture
def hier_store(monkeypatch):
    """A store with a complete model + benchmark hierarchy.

    Models:  acme/widget-7b-instruct  (group acme/widget-7b, family acme/widget)
    Benchmarks: bench-pro (family fam-x), bench-sub (parent_benchmark_id bench-pro / slice)
    Families: fam-x  (benchmark_ids [bench-pro], composite_keys [comp-suite])
    Composites: comp-suite (family_id fam-x)
    """
    store = hf_store.RegistryStore()
    tables = {name: s.empty(name) for name in [
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "canonical_orgs", "canonical_families",
        "canonical_composites", "aliases", "resolution_log", "eval_results",
        "sync_runs",
    ]}

    tables["canonical_models"] = pd.DataFrame([
        _row("canonical_models", id="acme/widget-7b-instruct",
             display_name="Widget 7B Instruct", org_id="acme",
             model_group_id="acme/widget-7b", model_family_id="acme/widget",
             resolution_granularity="variant", parents="[]"),
        # HF-backed (sourced from the HF oracle): hf_repo_id == its own id.
        _row("canonical_models", id="meta-llama/Llama-3.1-8B",
             display_name="Llama 3.1 8B", org_id="meta",
             resolution_source="hf", parents="[]"),
        # HF-backed via hub-stats metadata (resolution_source != hf, but
        # metadata.hf_id == its own id): hf_repo_id == its own id.
        _row("canonical_models", id="mistralai/Mistral-7B-v0.1",
             display_name="Mistral 7B v0.1", org_id="mistral",
             resolution_source="models_dev",
             metadata={"hf_id": "mistralai/Mistral-7B-v0.1"}, parents="[]"),
        # Shadow slug: metadata.hf_id points at a DIFFERENT real repo, so its
        # OWN id is not a real HF repo -> hf_repo_id is None.
        _row("canonical_models", id="kimi/kimi-x",
             display_name="Kimi X", org_id="moonshotai",
             resolution_source="inferred",
             metadata={"hf_id": "moonshotai/Kimi-X"}, parents="[]"),
    ])
    tables["canonical_benchmarks"] = pd.DataFrame([
        _row("canonical_benchmarks", id="bench-pro", display_name="Bench Pro"),
        _row("canonical_benchmarks", id="bench-sub", display_name="Bench Sub",
             parent_benchmark_id="bench-pro"),
        _row("canonical_benchmarks", id="mmlu", display_name="MMLU"),
    ])
    tables["eval_harnesses"] = pd.DataFrame([
        _row("eval_harnesses", id="helm", display_name="HELM"),
    ])
    tables["canonical_families"] = pd.DataFrame([
        _row("canonical_families", id="fam-x", display_name="Fam X",
             category="reasoning", benchmark_ids=["bench-pro", "bench-sub"],
             composite_keys=["comp-suite"]),
    ])
    tables["canonical_composites"] = pd.DataFrame([
        _row("canonical_composites", id="comp-suite", display_name="Comp Suite",
             source_configs=["cfg-a", "cfg-b"], family_id="fam-x"),
    ])
    tables["aliases"] = pd.DataFrame([
        _alias("acme/widget-7b-instruct", "model", "acme/widget-7b-instruct"),
        _alias("meta-llama/Llama-3.1-8B", "model", "meta-llama/Llama-3.1-8B"),
        _alias("mistralai/Mistral-7B-v0.1", "model", "mistralai/Mistral-7B-v0.1"),
        _alias("kimi/kimi-x", "model", "kimi/kimi-x"),
        _alias("bench-pro", "benchmark", "bench-pro"),
        _alias("bench-sub", "benchmark", "bench-sub"),
        _alias("mmlu", "benchmark", "mmlu"),
        _alias("Anatomy", "benchmark", "mmlu"),
        _alias("helm", "harness", "helm"),
        _alias("fam-x", "family", "fam-x"),
        _alias("comp-suite", "composite", "comp-suite"),
    ])

    store._tables = tables
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


@pytest.fixture
def client(hier_store):
    return TestClient(app, raise_server_exceptions=True)


# --- ancestry --------------------------------------------------------------

class TestAncestry:
    def test_model_ancestry_group_then_family(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "acme/widget-7b-instruct", "entity_type": "model"})
        assert r.status_code == 200
        data = r.json()
        assert data["canonical_id"] == "acme/widget-7b-instruct"
        assert data["ancestry"] == [
            {"canonical_id": "acme/widget-7b", "level": "group"},
            {"canonical_id": "acme/widget", "level": "family"},
        ]

    def test_benchmark_ancestry_family_then_composite(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "bench-pro", "entity_type": "benchmark"})
        assert r.status_code == 200
        data = r.json()
        # bench-pro -> family fam-x -> composite comp-suite (the full chain).
        assert data["ancestry"] == [
            {"canonical_id": "fam-x", "level": "family"},
            {"canonical_id": "comp-suite", "level": "composite"},
        ]

    def test_family_ancestry_composite(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "fam-x", "entity_type": "family"})
        assert r.status_code == 200
        assert r.json()["ancestry"] == [
            {"canonical_id": "comp-suite", "level": "composite"}]

    def test_composite_is_root_empty_ancestry(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "comp-suite", "entity_type": "composite"})
        assert r.status_code == 200
        assert r.json()["ancestry"] == []


# --- resolution_detail -----------------------------------------------------

class TestResolutionDetail:
    def test_model_detail_granularity(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "acme/widget-7b-instruct", "entity_type": "model"})
        # Off-HF model: granularity carried, hf_repo_id None.
        assert r.json()["resolution_detail"] == {
            "granularity": "variant", "hf_repo_id": None}

    def test_model_detail_hf_repo_id_oracle_backed(self, client):
        # resolution_source == "hf" -> the match is a real HF repo id.
        r = client.post("/api/v1/resolve", json={
            "raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model"})
        assert r.json()["resolution_detail"]["hf_repo_id"] == "meta-llama/Llama-3.1-8B"

    def test_model_detail_hf_repo_id_metadata_confirmed(self, client):
        # resolution_source != hf, but metadata.hf_id == own id -> HF-backed.
        r = client.post("/api/v1/resolve", json={
            "raw_value": "mistralai/Mistral-7B-v0.1", "entity_type": "model"})
        assert r.json()["resolution_detail"]["hf_repo_id"] == "mistralai/Mistral-7B-v0.1"

    def test_model_detail_hf_repo_id_shadow_slug_is_none(self, client):
        # metadata.hf_id points at a DIFFERENT repo -> own id is NOT a real HF
        # repo, so hf_repo_id is None (the shadow slug must not claim it).
        r = client.post("/api/v1/resolve", json={
            "raw_value": "kimi/kimi-x", "entity_type": "model"})
        assert r.json()["resolution_detail"]["hf_repo_id"] is None

    def test_model_detail_hf_repo_id_normalized_match(self, client):
        # A casing/separator-variant input still resolves and surfaces the
        # HF-true repo id (the value of matching a HF id loosely).
        r = client.post("/api/v1/resolve", json={
            "raw_value": "meta-llama/llama-3.1-8b", "entity_type": "model"})
        d = r.json()
        assert d["canonical_id"] == "meta-llama/Llama-3.1-8B"
        assert d["resolution_detail"]["hf_repo_id"] == "meta-llama/Llama-3.1-8B"

    def test_benchmark_detail_plain(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "bench-pro", "entity_type": "benchmark"})
        assert r.json()["resolution_detail"] == {
            "level": "benchmark", "matched_subset": None}

    def test_benchmark_detail_slice(self, client):
        """A decomposed slice canonical (parent_benchmark_id set) reports
        level=slice."""
        r = client.post("/api/v1/resolve", json={
            "raw_value": "bench-sub", "entity_type": "benchmark"})
        d = r.json()["resolution_detail"]
        assert d["level"] == "slice"

    def test_benchmark_detail_matched_subset_fold(self, client):
        """A subset string folded onto a parent canonical (Anatomy -> mmlu)
        surfaces the subset via matched_subset without a slice entity."""
        r = client.post("/api/v1/resolve", json={
            "raw_value": "Anatomy", "entity_type": "benchmark"})
        d = r.json()["resolution_detail"]
        assert d["matched_subset"] == "Anatomy"
        assert r.json()["canonical_id"] == "mmlu"

    def test_composite_detail_empty(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "comp-suite", "entity_type": "composite"})
        assert r.json()["resolution_detail"] == {}

    def test_harness_detail_empty_without_version_strip(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "helm", "entity_type": "harness"})
        assert r.json()["resolution_detail"] == {}

    def test_harness_detail_records_a_stripped_version(self, client):
        """The one harness case that carries keys: the resolver reached the
        match by dropping a trailing version token. Additive — every other
        harness response keeps `{}`."""
        r = client.post("/api/v1/resolve", json={
            "raw_value": "helm 1.2.3", "entity_type": "harness"})
        body = r.json()
        assert body["canonical_id"] == "helm"
        assert body["strategy"] == "normalized"
        assert body["resolution_detail"] == {
            "harness_version_stripped": "1.2.3",
            "bare_name": "helm",
            "bare_tier": "exact",
        }


# --- new endpoints ---------------------------------------------------------

class TestFamilyCompositeEndpoints:
    def test_get_family(self, client):
        r = client.get("/api/v1/families/fam-x")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "fam-x"
        assert body["category"] == "reasoning"
        assert body["benchmark_ids"] == ["bench-pro", "bench-sub"]
        assert body["composite_keys"] == ["comp-suite"]

    def test_get_family_404(self, client):
        assert client.get("/api/v1/families/nope").status_code == 404

    def test_get_composite(self, client):
        r = client.get("/api/v1/composites/comp-suite")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "comp-suite"
        assert body["source_configs"] == ["cfg-a", "cfg-b"]
        assert body["family_id"] == "fam-x"

    def test_get_composite_404(self, client):
        assert client.get("/api/v1/composites/nope").status_code == 404

    def test_list_families_and_composites(self, client):
        assert client.get("/api/v1/families").status_code == 200
        assert client.get("/api/v1/composites").status_code == 200


# --- resolve via the composite / family entity_types -----------------------

class TestResolveNewEntityTypes:
    def test_resolve_family_entity_type(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "fam-x", "entity_type": "family"})
        assert r.status_code == 200
        assert r.json()["canonical_id"] == "fam-x"
        assert r.json()["entity_type"] == "family"

    def test_resolve_composite_entity_type(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "comp-suite", "entity_type": "composite"})
        assert r.status_code == 200
        assert r.json()["canonical_id"] == "comp-suite"
        assert r.json()["entity_type"] == "composite"
