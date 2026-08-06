# eval-card-registry

Entity resolution registry for AI evaluation data. Maps raw model, benchmark, metric, and harness names from the EEE datastore to stable canonical IDs, and stores resolved evaluation results in a flat mapping table (`eval_results`).

---

## Quickstart

Resolve a raw string against the hosted registry:

```bash
curl -X POST https://evaleval-entity-registry.hf.space/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value": "MATH Level 5", "entity_type": "benchmark"}'
```

```json
{
  "raw_value": "MATH Level 5",
  "entity_type": "benchmark",
  "canonical_id": "math-level-5",
  "strategy": "exact",
  "confidence": 1.0,
  "created_new": false,
  "resolution_source": null,
  "review_status": "reviewed",
  "ancestry": [],
  "resolution_detail": {"level": "benchmark", "matched_subset": "MATH Level 5"}
}
```

(`math-level-5` is a standalone benchmark, so its `ancestry` is empty;
`matched_subset` echoes the raw label because it differs from the canonical
id. A benchmark that belongs to a family or is a decomposed subset carries a
non-empty `ancestry` and/or `level: "slice"` — see the [API section](#api).)

The resolve response is a **type-agnostic core** — identical shape for every
entity type — plus two hierarchy fields:

- `ancestry`: an ordered `[{canonical_id, level}]` chain from the matched
  entity's immediate parent up to the root (`[]` when self is a root). A model
  resolves to e.g. `[{group}, {family}]`; a benchmark to e.g.
  `[{family}, {composite}]`.
- `resolution_detail`: a typed sub-object keyed by `entity_type` — `model`:
  `{granularity, hf_repo_id}`; `benchmark`: `{level, matched_subset}`; others
  `{}`.

`entity_type` + the `ancestry` levels tell you which entity endpoint(s) to
follow for the full entity structure (group/family/lineage/params for models;
`family_key`/`composite_keys`/`category` for benchmarks; members for families
and composites) — note a benchmark resolve may land on a family or composite,
so this is a hint, not a guaranteed flat `GET /{entity_type}s/{id}`. The
type-specific entity fields live ONLY on those GET endpoints, never on
resolve. (The in-process `eval_entity_resolver.ResolutionResult` stays the
rich union for the producer path.)

`entity_type` is one of `benchmark`, `model`, `metric`, `harness`, `org`,
`composite`, `family`. See the [API section](#api) for batch resolve, entity
browsing (including `GET /families/{id}` and `GET /composites/{id}`), and the
full endpoint list.

---

## Local development

```bash
git clone <repo>
cd eval-card-registry
uv sync
cp .env.example .env          # defaults work for local dev
```

**1. Seed the registry with known entities:**

```bash
uv run eval-card-registry seed --local
```

This loads orgs, models, benchmarks, metrics, and harnesses from `seed/` into `fixtures/*.parquet`. You should see counts printed for each entity type. (After any change that renames canonical ids, `rm fixtures/*.parquet` before reseeding — or pass `--prune-stale` — since the seed upserts by id and does not prune renamed-away rows.)

**2. Check what's in the registry:**

```bash
uv run eval-card-registry stats --local
```

Expected output:

```
  models      total=7148  draft=...
  benchmarks  total=2592  draft=...
  metrics     total=27  draft=0
  harnesses   total=11  draft=0

  aliases        total=29255  uncertain=0
  eval_results   total=0
  resolution_log total=0
  sync_runs      total=0
```

(Counts are illustrative — they grow as seed data and refreshes land. Models and
orgs are seeded from `seed/models/` + `seed/orgs*.yaml`; a fresh checkout seeds a
populated `canonical_models` table, not an empty one.)

**3. Sync an EEE config — resolve entities and populate the mapping table:**

```bash
uv run eval-card-registry sync --config hfopenllm_v2 --local
```

This downloads the EEE dataset config from HuggingFace (first run will take a few minutes), resolves every raw string to a canonical entity, and writes results to `fixtures/eval_results.parquet` — the mapping table (one row per model × benchmark × metric result).

**4. Verify results:**

```bash
uv run eval-card-registry stats --local
```

You should now see `eval_results`, `aliases`, and entity counts populated. Each row in `eval_results` looks like:

```json
{
  "evaluation_id": "hfopenllm_v2/...",
  "result_index": 0,
  "source_config": "hfopenllm_v2",
  "model_id": "meta-llama/Llama-3.1-8B",
  "harness_id": "lm-evaluation-harness",
  "benchmark_id": "ifeval",
  "parent_benchmark_id": null,
  "metric_id": "accuracy",
  "benchmark_card_id": null,
  "score": 0.42,
  "score_details": "{\"score\": 0.42}"
}
```

---

## How it works

Raw strings from EEE (e.g. `"MATH Level 5"`, `"lm-evaluation-harness"`) are resolved to canonical IDs (`math`, `lm-evaluation-harness`) through a strategy chain: exact alias match → HF id check (models only — a raw value that IS a valid HF repo id, per the local hub-stats index or a rate-limited live Hub API check, resolves to itself in HF-true casing and outranks any alias fold) → normalized match (collapses case + all separators — spaces, hyphens, underscores, and slashes) → normalized-tier HF id check → fuzzy stem match (strips known suffixes like `-fc`/`-prompt`, normalizes org prefixes) → auto-create draft. Every resolution is logged with its strategy and confidence score.

Resolve requests accept `mode: "resolve" | "exact"` (default `resolve`). Exact mode stops after the alias and HF-id-check steps: no fuzzy inference, no draft creation, no alias writes — a miss is an honest `no_match`.

**Models** are grounded in a three-tier source-of-truth: HuggingFace (the
`fixed_hf_model_id` oracle + the hub-stats index) → models.dev catalog →
name-based inference for the off-HF tail. Canonical ids follow one ladder: an
external id is adopted verbatim — the real HF repo id (HF-true casing) first,
else the OpenRouter model id for models.dev-only models served there — and only
a model on neither gets an invented `{org}/{slug}` id. `org_id` stays the
curated developer for all three rungs, decoupled from the id prefix; it
resolves through a two-tier org model — the HF org
spelling is preserved for community uploaders, while curated developer remaps
fold alternate namespaces into one parent (`meta-llama`/`facebook` → `meta`,
`qwen`/`THUDM`/`zai-org` → `alibaba`/`zai`, …). Models also carry
group/family/lineage membership and an optional `inference_platform`.

The bulk of the model universe is generated: `seed/models/sources/*.generated.yaml`
is committed and is the source of truth for the HF + models.dev tail, regenerated
by the scripts in `scripts/` (the models.dev source refreshes daily via the
`refresh-models` workflow, which is core-aware — it only adds coverage, never
clobbers a curated id — and gated by an invariant suite before it commits).
`seed/models/core.yaml` is the small curated-override layer on top: closed-API
models and hand judgement calls. The loader merges sources → core with
field-level merge (aliases / tags / `parents` union; other scalars prefer
non-empty, last write wins).

Canonical entities start as `draft` and can be promoted to `reviewed`. Aliases that fall below the confidence threshold are flagged `uncertain` for human review.

---

## ID conventions

The rules a canonical id and its aliases follow. (Resolution *mechanics* are in
"How it works" above; the contribution *workflow* is in `CONTRIBUTING.md`.)

**Models** — the canonical id follows one ladder, and an external id is
adopted **verbatim**: the **real HuggingFace repo id, in HF-true casing**
(`Qwen/Qwen2.5-7B-Instruct`, not `qwen/qwen2.5-7b-instruct`) → else the
**OpenRouter model id** for a models.dev-only model served there
(`anthropic/claude-3-haiku`, `z-ai/glm-4-32b` — upstream org prefix kept) →
else an invented `{org}/{slug}`. The ladder is rung-monotone: a fresh id from
a strictly higher rung always wins over a committed lower-rung id (the loser
survives as an alias); OpenRouter-over-invented promotions run only in a
deliberate migration pass. `org_id` stays the curated developer org on every
rung, decoupled from the id prefix. Sources are grounded HF-first —
**HuggingFace → models.dev → name-based inference** — and when they disagree,
HF casing wins.

- **Closed / off-HF models** therefore carry their OpenRouter id when served
  there, and an invented lowercase `{org}/{name}` (e.g.
  `anthropic/claude-opus-4.6`) only when on neither HF nor OpenRouter.
- **Dated snapshots are legitimate identity** (`gpt-5.4-2026-03-05`); date
  suffixes are deliberately **not** stripped. A new dated slug gets its **own
  canonical** with a `{relationship: variant, axis: version}` parent edge to the
  base — do NOT alias it onto the base, which silently folds the snapshot into
  the family pointer and loses the snapshot's `release_date`.
- **Effort tiers (`-high`/`-medium`/`-low`/`-minimal`), mode variants (e.g.
  `-thinking`), and quantized repos (e.g. `-FP8`, GGUF uploads) all follow one
  rule:** when a canonical with that suffix exists in the registry, it is a
  **first-class canonical** and the resolver matches that leaf (e.g.
  `google/gemini-3-1-pro-preview-high` resolves exactly; benchmark scores
  differ measurably across quantization levels, so the leaf identity matters).
  The fuzzy suffix-strip is only a **fallback** when no such canonical exists
  (e.g. `openai/gpt-5-high` → `openai/gpt-5`). Curation policy, separate from
  that resolver behavior: don't hand-mint effort-tier ids in `core.yaml` —
  effort-tier canonicals arrive from the generated sources when upstream
  catalogs list them. Per-run inference detail lives in the eval record's
  `generation_args` / `model_route_id`, not in the id.

**Benchmarks, metrics, harnesses** — lowercase, hyphenated slugs
(`ai2-reasoning-challenge-arc`, `pass-at-1`, `lm-evaluation-harness`) — a
different scheme from models.

**Aliases are gap-filling only.** Each alias must be a form no generator canonical
already claims; the loader unions them (case-insensitive dedup). Don't add
mechanical variants — the `normalized` matcher already collapses case + all
separators + dots-between-digits, so add an alias only for forms it can't reach (a
*removed* separator, a token reshape, a `-v2`, a date suffix, a marketing name).
Model aliases go in `seed/models/enrichments/aliases.yaml` (`{id, aliases}`
entries); never hand-edit `seed/models/sources/*.generated.yaml` — those files
are machine-generated and rewritten by the daily refresh.

**`core.yaml` is the minimal override floor** — closed-API models and hand
judgement calls layered on top of the generated sources. Two distinct knobs
drop ids from the merge:
- `skip_source_ids` drops an id from the generated **sources/enrichments
  only** — core stays authoritative. This is the right tool when a generator
  ships wrong data for a model `core` curates correctly.
- `skip_ids` drops the id from **every layer, including core** — the entity
  vanishes from the registry entirely.

---

## Project layout

```
eval-card-registry/
├── packages/eval-entity-resolver/        # Standalone resolver package (uv workspace member)
├── src/eval_card_registry/               # FastAPI service + CLI
│   ├── api/                              # Route handlers
│   ├── services/                         # resolution_service, ingestion pipeline
│   └── store/                            # In-memory store backed by HF Dataset parquet
├── seed/                                 # Known canonical entities (YAML)
│   ├── orgs.yaml                         # Curated orgs (labs + developer remaps)
│   ├── orgs.generated.yaml               # Community orgs (HF-cased, generated)
│   ├── inference_platforms.yaml          # Hosting / gateway platforms
│   ├── benchmarks.yaml / metrics.yaml / harnesses.yaml
│   ├── composites.yaml / families.yaml    # Backend hierarchy taxonomy hints
│   ├── slice_overrides.yaml               # Benchmark-vs-slice overrides — shipped to HF and read by the PRODUCER (not the registry resolver)
│   └── models/                           # Model seed (canonical_id = real HF repo id)
│       ├── core.yaml                     # Minimal curated overrides: closed-API + hand judgement calls (+ skip_ids / skip_source_ids)
│       ├── enrichments/                  # Curated overlays (aliases.yaml — model alias bridges; parents.yaml — curated lineage edges)
│       └── sources/                      # Generated source-of-truth layer (committed; regenerated by scripts/ — NEVER hand-edited):
│           ├── hf_oracle.generated.yaml      #   HF source-of-truth (Tier-1)
│           ├── hub_stats.generated.yaml      #   HF hub-stats index
│           ├── models_dev.generated.yaml     #   models.dev catalog (full re-cased)
│           ├── models_dev_catalog.generated.yaml  #   models.dev-only additions (catalog split)
│           └── tier3_inferred.generated.yaml #   name-based inference (Tier-3)
├── scripts/                              # Source generators + daily refresh scripts (refresh_from_modelsdev.py, refresh_from_hub_stats.py, generate_*_seed.py, publish_registry_data.py)
├── fixtures/                             # Local parquet files for offline dev/tests
└── tests/
```

The service and the resolver package are separate. The resolver can be imported directly by other pipelines (e.g. AutoBenchmarkCard) without pulling in the full service.

---

## CLI reference

All commands require `uv run` prefix (or install the package first with `uv pip install -e .`).

```bash
# Seed known entities from seed/ YAML files
uv run eval-card-registry seed --local

# Print entity counts, draft counts, uncertain aliases
uv run eval-card-registry stats --local

# Sync one EEE config — resolves entities, writes to eval_results table
uv run eval-card-registry sync --config hfopenllm_v2 --local

# Sync all configs
uv run eval-card-registry sync --all --local

# Re-resolve everything (after updating seed data or fuzzy matching logic)
uv run eval-card-registry sync --config hfopenllm_v2 --rerun --local
```

Drop `--local` and configure `.env` with HF credentials to read/write from HF Hub instead of `fixtures/`.

---

## API

Start the server:

```bash
LOCAL_MODE=true uv run uvicorn eval_card_registry.main:app --reload
```

Base path: `http://localhost:8000/api/v1`

**Resolve a raw string:**

```bash
curl -X POST http://localhost:8000/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value": "rewardbench-chat", "entity_type": "benchmark"}'
```

```json
{
  "raw_value": "rewardbench-chat",
  "entity_type": "benchmark",
  "canonical_id": "rewardbench-chat",
  "strategy": "exact",
  "confidence": 1.0,
  "created_new": false,
  "resolution_source": null,
  "review_status": "reviewed",
  "ancestry": [{"canonical_id": "reward-bench", "level": "family"}],
  "resolution_detail": {"level": "slice", "matched_subset": null}
}
```

(`rewardbench-chat` is a decomposed subset of `rewardbench`, so it reports
`level: "slice"`, and it belongs to the `reward-bench` family, so its
`ancestry` chain points there.)

The HTTP resolve response is a **type-agnostic core** (identical shape for every
entity type) + `ancestry` + a typed `resolution_detail`:

- `raw_value`, `entity_type` — echo of the request.
- `canonical_id`, `strategy`, `confidence`, `created_new`, `resolution_source`,
  `review_status` — the match facts.
- `ancestry` — ordered `[{canonical_id, level}]` from the matched entity's
  immediate parent up to the root; `[]` when self is a root. A model resolves
  to e.g. `[{group}, {family}]`; a benchmark to e.g. `[{family}, {composite}]`.
- `resolution_detail` — typed sub-object keyed by `entity_type`:
  - `model`: `{ "granularity": variant|group|family|null, "hf_repo_id": str|null }`.
    `hf_repo_id` is the matched canonical's HF repo id, but ONLY when the
    registry can attest it at runtime (the HF oracle — `resolution_source="hf"` —
    or a hub-stats-confirmed match). `null` is conservative: it covers off-HF /
    closed models AND genuine HF repos whose provenance isn't runtime-verifiable
    (models.dev-only / name-inferred entries), so non-null is reliable but `null`
    is not proof a model is off-HF.
  - `benchmark`: `{ "level": composite|family|benchmark|slice, "matched_subset": str|null }`.
    `level="slice"` means the matched benchmark is itself a decomposed subset of a
    parent (it carries a `parent_benchmark_id`), e.g. `rewardbench-chat` under
    `rewardbench`. `matched_subset` is INDEPENDENT of `level`: it echoes the raw
    surface form whenever it differs from the canonical id by more than case
    (e.g. `"MATH Level 5"` → `math-level-5`), so a plain `level="benchmark"`
    match can still carry one.
  - `composite` / `family` / `metric` / `harness` / `org`: `{}` (reserved).

**Response shape & consistency.** The ten top-level fields above are ALWAYS
present on every `/resolve` and `/resolve/batch` response, for every entity type
and even on a no-match — fields with nothing to report come back explicitly
`null` (`canonical_id`, `resolution_source`, `review_status`) rather than being
omitted, and `ancestry` is always a list. The one field whose *inner* keys vary
is `resolution_detail`: it is a polymorphic payload selected by `entity_type`
(model and benchmark carry different keys; metric/harness/org/composite/family
are `{}`), and it is `{}` on a **no-match** (no match → no detail). So treat
`resolution_detail` as "present-but-possibly-empty" and key into it by
`entity_type` + a successful match. Example no-match:

```json
{
  "raw_value": "totally-not-a-real-model",
  "entity_type": "model",
  "canonical_id": null,
  "strategy": "no_match",
  "confidence": 0.0,
  "created_new": false,
  "resolution_source": null,
  "review_status": null,
  "ancestry": [],
  "resolution_detail": {}
}
```

For a model resolve the chain carries group + family membership:

```bash
curl -X POST http://localhost:8000/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value": "meta-llama/Llama-3.1-8B-Instruct", "entity_type": "model"}'
```

```json
{
  "raw_value": "meta-llama/Llama-3.1-8B-Instruct",
  "entity_type": "model",
  "canonical_id": "meta-llama/Llama-3.1-8B-Instruct",
  "strategy": "exact",
  "confidence": 1.0,
  "created_new": false,
  "resolution_source": "hf",
  "review_status": "reviewed",
  "ancestry": [{"canonical_id": "meta-llama/llama-3.1", "level": "family"}],
  "resolution_detail": {"granularity": null, "hf_repo_id": "meta-llama/Llama-3.1-8B-Instruct"}
}
```

A model's `canonical_id` is the **real HF repo id** (e.g.
`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B`) — you can build
`huggingface.co/{canonical_id}`. Developer grouping is carried separately by
`org_id` (the canonical parent, e.g. `meta`, `alibaba`), which is decoupled from
the id prefix, so `Qwen/…` and `meta-llama/…` resolve to org `alibaba` / `meta`.
Models not on HF (proprietary, or API-only) keep a `{org}/{name}` id.

The type-specific ENTITY structure (for models: group/family/lineage/`parents`/
`open_weights`/`release_date`/`params_billions`; for benchmarks: `family_key`/
`composite_keys`/`category`; for families/composites: their members) lives ONLY
on the entity GET endpoints — never on resolve. `entity_type` + the `ancestry`
levels tell you which endpoint(s) to follow; note a benchmark resolve may land
on a family or composite, so this is a hint, not a guaranteed flat
`GET /{entity_type}s/{id}`. (The in-process
`eval_entity_resolver.ResolutionResult` stays the rich union for the producer
path — see "Using the resolver standalone" below.)

**Batch resolve:**

```bash
POST /api/v1/resolve/batch
Body: [{ "raw_value": "...", "entity_type": "..." }, ...]
```

**Entity CRUD** (models, benchmarks, metrics, harnesses):

```
GET    /api/v1/benchmarks?search=math&review_status=draft
GET    /api/v1/benchmarks/{id}
POST   /api/v1/benchmarks
PATCH  /api/v1/benchmarks/{id}
```

Model IDs containing `/` (e.g. `meta-llama/Llama-3.1-8B`) work in path params directly.

**Families and composites** (read-only — these are first-class hierarchy entities a benchmark resolve's `ancestry` points at):

```
GET    /api/v1/families            # canonical_families
GET    /api/v1/families/{id}        # benchmark_ids, category, composite_keys
GET    /api/v1/composites           # canonical_composites
GET    /api/v1/composites/{id}      # source_configs, family_id
```

**Aliases:**

```
GET    /api/v1/aliases?status=uncertain&entity_type=benchmark
PATCH  /api/v1/aliases/{id}   # confirm, reject, or correct an alias
```

**Health and stats:**

```
GET  /api/v1/health
GET  /api/v1/stats
```

Interactive docs at `http://localhost:8000/docs`.

---

## Using the resolver standalone

The `eval-entity-resolver` package can be used independently — no service required. It returns the **rich** in-process `ResolutionResult` (root-collapse for quantized chains, `parents`, `open_weights`, `release_date`, `params_billions`, the model lineage fields, benchmark `family_key`/`category`, plus `ancestry` and `resolution_detail`). The HTTP `POST /resolve` is a LEAN projection of this — core + `ancestry` + `resolution_detail` only — so the producer (which consumes the dataclass in-process) keeps the full union while external HTTP callers get the stable lean shape:

```python
from eval_entity_resolver import Resolver, ResolverConfig

# Load both aliases AND canonical entities from the production HF Dataset:
resolver = Resolver.from_hf("evaleval/entity-registry-data",
                            config=ResolverConfig(threshold=0.85))

# Or from a local parquet directory (e.g. after `eval-card-registry seed --local`):
resolver = Resolver.from_parquet("./fixtures/")

result = resolver.resolve(
    raw_value="meta-llama/Llama-3.1-8B-Instruct",
    entity_type="model",           # model | benchmark | metric | harness | org | composite | family
    source_config=None,             # optional; scopes to per-config aliases
)
# result is a `ResolutionResult` dataclass — the RICH in-process union (the HTTP
# POST /resolve is a lean projection of it):
#   raw_value, entity_type, source_config — echo of inputs
#   canonical_id          — the matched entity = the real HF repo id for models;
#                           None on no_match
#   strategy, confidence  — match info
#   review_status         — "draft" | "reviewed"
#   ancestry              — [{canonical_id, level}] from immediate parent to root
#   resolution_detail     — typed sub-object keyed by entity_type
#   # Models only:
#   org_id                — canonical parent org (decoupled from the id prefix)
#   model_group_id        — identity-group root (folds version/quantized/mode);
#                           always set (self for singletons)
#   model_family_id       — family-release root
#   lineage_origin_model_id / lineage_origin_model_org_id
#                         — deepest non-variant (finetune/quant) ancestor + its org;
#                           None at origin
#   parents               — full typed-edge list
#   open_weights / release_date / params_billions / inference_platform
#   # Benchmarks only:
#   family_key            — curated family id (falls back to self id for singletons)
#   composite_keys        — list of composites containing this benchmark; [] when none
#   category              — curated category for the family (general / agentic /
#                           reasoning / knowledge / multimodal / tool-use / math /
#                           security / factuality / reward-modelling / safety / code /
#                           instruction-following / other); None when not curated
```

A model's `canonical_id` is the matched entity itself (the real HF repo id). Its
place in the identity graph is carried separately: `model_group_id` is the group
root (a quantized/versioned variant and its base share one group) and
`model_family_id` is the family-release root.

```python
>>> r = resolver.resolve("meta-llama/Llama-3.1-8B-Instruct", "model")
>>> r.canonical_id, r.org_id, r.model_group_id, r.model_family_id
('meta-llama/Llama-3.1-8B-Instruct', 'meta', 'meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/llama-3.1')
```

If you really want the bare matcher (no metadata enrichment), you can construct `Resolver` without a `CanonicalStore`:

```python
from eval_entity_resolver import AliasStore, Resolver
resolver = Resolver(AliasStore.from_parquet("./fixtures/"))  # no canonical_store
# Now `result` only has `canonical_id`, `strategy`, `confidence`. All other
# fields are None. Useful when you don't have the canonical_models parquet
# (e.g. an alias-only HF dataset) or just want to avoid the lookup cost.
```

Install from this workspace:

```bash
uv add eval-entity-resolver --workspace
```

---

## Tests

```bash
uv run pytest
```

Tests use the in-memory fixture store — no HF credentials or network needed.

---

## Resolution behaviour

| Alias status | Meaning |
|---|---|
| `auto` | Resolved above confidence threshold — no review needed |
| `uncertain` | Below threshold or no match — auto-created draft, flagged for review |
| `confirmed` | Manually verified |
| `rejected` | Wrong match, excluded from future resolution |

Resolution order for a given `(entity_type, raw_value)`:

1. Config-scoped alias (`source_config` matches)
2. Global alias (`source_config` is null)
3. Resolver chain (exact → normalized → fuzzy → auto-create draft)

Resolving the same raw string twice returns the same canonical ID. Re-running with `--rerun` re-evaluates existing aliases — prior resolution log entries are preserved.

---

## ID conventions

| Entity | Format | Example |
|---|---|---|
| Model (on HF) | real HF repo id, HF-true casing | `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B` |
| Model (not on HF) | `{org_id}/{model-slug}` | `anthropic/claude-opus-4.5`, `openai/gpt-4o` |
| Benchmark / Metric / Harness | lowercase slug | `math`, `lm-evaluation-harness` |
| `eval_results` row ID | `sha256(evaluation_id:result_index)[:16]` | `a3f2b1c9d4e5f678` |

Entity IDs use human-readable slugs (not hashes) because they appear in seed files, API responses, and are referenced during manual curation. Internal row IDs (like `eval_results.id`) use deterministic hashes for uniform length and collision resistance.

---

## HF Hub deployment

> [!IMPORTANT]
> **Merging a seed change does not make it live.** The hosted resolver
> (`https://evaleval-entity-registry.hf.space`) answers from the deployed
> dataset repo, not from this git repo, so a new canonical, alias, benchmark,
> metric or harness in `seed/` reaches consumers only after someone re-runs
> `seed` against the Hub and the Space reloads. Until then a downstream adapter
> resolving the new raw value still gets `no_match` — and, because
> `POST /api/v1/resolve` auto-creates a `draft` on a miss, an unlucky consumer
> can mint the very duplicate the seed change was meant to prevent. Treat the
> deploy as part of merging a seed PR.

For production, configure `.env`:

```
LOCAL_MODE=false
HF_TOKEN=hf_...
HF_DATASET_REPO=org/eval-card-registry
```

Then run the same commands without `--local`:

```bash
uv run eval-card-registry seed
uv run eval-card-registry sync --config hfopenllm_v2
```

Data is stored as one parquet config per table in the HF Dataset repo.

---

## HF Space deployment (query-only API)

The service can be deployed to a HuggingFace Space as a **query-only** disambiguation API — read-only resolve + entity/alias GETs, no writes.

**Architecture:**
- **Space** (`evaleval/entity-registry`) — Docker SDK, runs FastAPI on port 7860
- **Dataset repo** (`evaleval/entity-registry-data`) — entity parquet tables, read at startup
- **Storage Bucket** (`evaleval/entity-registry-storage`) — async resolve logs, written periodically

**Read-only mode behaviour:**
- `POST /resolve` runs the full resolver chain but does NOT auto-create draft entities or write aliases on no_match — `canonical_id` is `null`
- `POST`/`PATCH` entity + alias endpoints return `405 Method Not Allowed`
- Only 5 tables (models, benchmarks, metrics, harnesses, aliases) are loaded — `eval_results`, `resolution_log`, `sync_runs` are skipped
- Every resolve request is logged asynchronously to the Storage Bucket (buffered in memory, flushed every 5 min as partitioned parquet)

**Deploy:**

```bash
# Prerequisites: create the Space, Dataset repo, and Storage Bucket on HF;
# seed + sync the Dataset repo with entity data locally first.

bash deploy/push-to-space.sh
```

Configure the Space in HF Space Settings:

| Variable | Type | Value |
|---|---|---|
| `HF_TOKEN` | Secret | Token with read access to dataset + write access to log bucket |
| `HF_DATASET_REPO` | Variable | `evaleval/entity-registry-data` |
| `HF_LOG_BUCKET` | Variable | `evaleval/entity-registry-storage` |

`READ_ONLY=true` and `LOCAL_MODE=false` are set in the Dockerfile ENV.

**Local test of read-only mode:**

```bash
READ_ONLY=true LOCAL_MODE=true uv run uvicorn eval_card_registry.main:app --reload
```

See `deploy/END_TO_END.md` for a step-by-step verification guide (local smoke
test, Docker test, Space deploy + checks).

---

## TODO
- Combine logic with EEE codebase's model registry and evalcard backend metric registry
- Verify metric extraction logic — although likely partially addressed with future schema versions and fixes.
- Clean up how we implement registry updates + check against regression
- Populate `benchmark_card_id` once an auto-benchmarkcard has been generated and linked for each benchmark.
- Implement walking and backfilling for lineage
- Clarify entity type 
