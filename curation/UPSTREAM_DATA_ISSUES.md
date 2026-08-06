# Upstream data issues — to flag + track

Known-wrong records in the upstream sources that the registry corrects LOCALLY
(see `seed/models/enrichments/upstream_corrections.yaml` + `core.yaml`
`skip_source_ids`) pending an upstream fix. Once an upstream record is corrected,
remove the corresponding local override and this entry.

## evaleval/EEE_datastore

- **`openai/GPT-J-6B` and `openai/GPT-NeoX-20B`** — GPT-J and GPT-NeoX are
  **EleutherAI** models. The EEE datastore records evaluations under the
  `openai/` namespace (alongside the correct `EleutherAI/...` records); HF marks
  the `openai/` ids `unresolved_not_found_or_inaccessible`. The developer
  attribution is wrong upstream.
  - Local handling: map the wrong raw → the real EleutherAI repo
    (`upstream_corrections.yaml`); drop the tier3 mints of the wrong ids
    (`core.yaml` `skip_source_ids`).
  - **Action: flag to the EEE maintainers to fix the source records.**

## models.dev

- **`abacusai/Dracarys-72B-Instruct`** — models.dev gives this real Qwen2-72B
  repo the display/alias **"Llama 3.1 70B Dracarys 2"**, which denotes a
  different (Llama-3.1-70B) model. The HF record for this id is correct
  (`Dracarys-72B-Instruct`).
  - Local handling: override the display back to `Dracarys-72B-Instruct`
    (`upstream_corrections.yaml`). The stray models.dev alias is low-impact (not
    an EEE evaluation id); it clears when models.dev corrects the catalog entry
    or when the generator gains display/id size-consistency hygiene.
  - **Action: flag to models.dev to correct the catalog label.**

- **Vendor-prefix noise (`alibaba/cogview4`, `openai/wizardlm-2-8x22b`)** —
  models.dev keys some models under a vendor that did not make them: CogView4 is
  ZhipuAI/Z.AI (real repo `THUDM/CogView4-6B`), and WizardLM-2 is Microsoft, not
  OpenAI. The wrong prefix had been baked into the curated `canonical_id`.
  - Local handling: re-key the canonicals to the correct developer
    (`zai/cogview4`, `microsoft/WizardLM-2-8x22B`) in `core.yaml`, keeping the
    wrong forms as resolvable aliases.
  - A wider, low-severity pattern: models.dev also prefixes many community
    finetunes with a base vendor (e.g. `meta/<TheDrummer-model>`,
    `nvidia/<google-or-mistral-model>`). These resolve CORRECTLY in the registry
    (the alias maps the wrong upstream string to the right canonical), so they
    are kept as aliases, not dropped — dropping would lose resolution of a real
    upstream string. No local data is wrong; the noise is purely upstream.
  - **Action: flag to models.dev to correct the vendor prefixes.**

## Coverage gaps found while resolving the LEXam leaderboard

Resolving the 36 models on the [LEXam](https://lexam-benchmark.github.io/)
leaderboard for the Every Eval Ever `lexam` adapter surfaced four checkpoints
that have **no HF-anchored canonical**, so the leaderboard label resolves only
to an API-catalog draft (`resolution_source: models_dev` / `inferred`) or to a
different model. The adapter falls back to the canonical HF repo id and records
`model_id_resolution: hf_canonical`, which is the honest answer but keeps the
same physical model split across two id namespaces
(`deepseek-ai/...` vs `deepseek/...`).

- **`deepseek-ai/DeepSeek-V3.2-Exp`** — the HF repo exists (created
  2025-09-29) and its siblings `deepseek-ai/DeepSeek-V3.2` and
  `deepseek-ai/DeepSeek-V3.2-Speciale` are present and `reviewed`, but the
  `-Exp` checkpoint is absent. `DeepSeek-V3.2-Exp` therefore resolves to the
  models_dev draft `deepseek/deepseek-v3-2-exp`.
- **`Qwen3-Next` resolves to a draft that cannot name a variant.** Both released
  variants ARE registered as reviewed hf canonicals
  (`Qwen/Qwen3-Next-80B-A3B-Instruct`, `-Thinking`), but the bare label
  `Qwen3-Next` resolves to the models_dev draft `alibaba/qwen3-next`, and there
  are seven `qwen3-next*` ids in total (2 reviewed hf, 5 drafts across
  `alibaba/` and `aliyun/`). A leaderboard that prints only "Qwen3-Next" is
  therefore unresolvable to actual weights — not for lack of a canonical, but
  because the label is ambiguous between two of them.
- **`utter-project/EuroLLM-9B-Instruct`** — absent; only the base
  `utter-project/EuroLLM-9B` is present, so an instruct-model label can only
  resolve to the base model.
- **`meta-llama/Llama-3.1-405B-Instruct`** — absent; only the Together-hosted
  `Meta-Llama-3.1-405B-Instruct-Turbo` drafts are present.

**Locally handled now:** only `utter-project/EuroLLM-9B-Instruct` is added as a
curated `hf` canonical — it has no minted slug twin and its parent is already a
canonical, so it introduces no duplicate identity.

**The other two are aliased onto the canonicals that already exist**
(`DeepSeek-V3.2-Exp` → `deepseek/deepseek-v3.2-exp`, `Llama-3.1-405B-it` →
`meta/llama-3-1-405b-instruct`) rather than minted as HF-anchored twins. I tried
minting them and `tests/test_gate_invariants.py` rejected it, correctly:

- `test_no_real_hf_id_duplicated_by_slug` — adding `deepseek-ai/DeepSeek-V3.2-Exp`
  beside `deepseek/deepseek-v3.2-exp` *is* the duplication this repo forbids.
- `test_no_dangling_parent_edges` — the HF base repos
  (`DeepSeek-V3.2-Exp-Base`, `Llama-3.1-405B`) are not canonicals here, so the
  `parents` edges I copied off the model card broke the lineage walks.
- Folding the slug twins to satisfy the first test then tripped
  `test_phase0_oracle_lineage_no_loss` and
  `test_no_parent_edge_names_a_renamed_away_id`, because dated siblings name
  those slugs as parents.

The lesson is that HF-vs-slug reconciliation is a curation pass with lineage
consequences, not something a leaderboard adapter's companion PR should carry.

**Also reverted for the same reason:** folding the three pure mode variants
(`deepseek/deepseek-v3-2-reasoning`, `-exp-prompt-thinking`,
`fireworks/deepseek-v3p2-thinking`) onto their checkpoints. It broke three
lineage invariants — the dated `-0925` siblings name those drafts as parents, so
skipping them dropped `model_group_id` / `model_family_id` and lost typed oracle
edges. The mode-as-identity problem is real and stays documented below; the fix
needs to move the parent edges at the same time.

**Action: reconcile the HF/slug pairs and the mode drafts as a dedicated pass,**
moving parent edges with them. This PR now touches no lineage.

**Why the generator missed them (diagnosed, not guessed):** the hub_stats
generator records the raw strings it has examined in
`seed/models/sources/hub_stats.state.json` `rows_checked_at_etag` (12,659
entries). None of the three appears there, while their siblings do
(`deepseek-ai/DeepSeek-V3.2` ✓, `meta-llama/Llama-3.3-70B-Instruct` ✓). So they
were never *filtered out* — they are absent from the generator's upstream input
at the pinned `parquet_etag`, which no re-run of the current snapshot will fix.

- **Action: refresh the hub-stats input** (or add these to whatever seeds it)
  so the generator carries them. The `core.yaml` entries above are a floor, not
  a fix: once the generator reproduces them, they can be thinned back.

### Ambiguous instruct alias

- **`gemma2-9b-it` → `google/gemma-2-9b`** (seed alias) points an `-it` form at
  the **base** model, while the correctly-formed `gemma-2-9b-it` →
  `google/gemma-2-9b-it` exists alongside it. Any resolver that normalizes
  punctuation sees both and can land on the base model for an instruct-tuned
  evaluation.
  - **Cannot be fixed from the enrichment layer — attempted and reverted.**
    `models_dev.generated.yaml` declares `gemma2-9b-it` on the *base* entry
    `google/gemma-2-9b` itself. Re-declaring it on `google/gemma-2-9b-it` trips
    the seed's own alias-collision guard ("the same alias is declared by more
    than one canonical ... the owner would be seed-order-dependent"), and the
    enrichment layer unions aliases rather than moving them: there is
    `skip_ids`/`skip_source_ids` for *entities*, but no equivalent for one wrong
    alias.
  - Impact is narrow: only the exact spelling `gemma2-9b-it` is affected. The
    conventional `gemma-2-9b-it` and display forms like `Gemma-2-9B-it` resolve
    correctly to the instruct model.
  - **Action: fix upstream in models.dev, or add an alias-level skip** so a
    known-wrong single alias can be re-pointed without dropping its entity.

### Mode-as-model drafts

`deepseek/deepseek-v3-2-reasoning`, `deepseek/deepseek-v3.2-thinking`,
`deepseek/deepseek-v3-2-exp-prompt-thinking` and
`fireworks/deepseek-v3p2-thinking` are thinking/non-thinking **modes** of a
single checkpoint rather than separate models (DeepSeek's API changelog,
2025-12-01: `deepseek-chat` and `deepseek-reasoner` are the non-thinking and
thinking modes of DeepSeek-V3.2). `deepseek-v3.2-thinking` is already an alias
of `deepseek-ai/DeepSeek-V3.2`, which is the right treatment; the remaining
draft canonicals bake a generation setting into model identity.

  - **Attempted in this PR and reverted** (see the coverage-gap section above):
    redirecting them broke three lineage invariants, because the dated `-0925`
    siblings name these drafts as parents. Any fold has to move those parent
    edges in the same change.
  - Previously described as for the three ids whose *only* distinguishing axis
    is the mode: `deepseek/deepseek-v3-2-reasoning`,
    `deepseek/deepseek-v3-2-exp-prompt-thinking` and
    `fireworks/deepseek-v3p2-thinking` move into `skip_source_ids` with their raw
    forms bridged onto the checkpoint, so those strings now resolve to weights
    rather than to a setting. No EEE adapter emits any of them.
  - Note on removal: on an existing table the three draft *rows* survive,
    alias-less. `skip_source_ids` stops them being re-absorbed from
    `tier3_inferred` and a fresh build never creates them, but `--prune-stale`
    removes only **reviewed** entities, so incremental runs keep the drafts.
    Resolution is correct either way; the orphan rows need a draft-pruning pass.
  - **Rationale, since the registry curates mode ids elsewhere**
    (`anthropic/claude-haiku-4.5-thinking`, `claude-opus-4.7-non-reasoning`):
    those are closed models where the mode-suffixed endpoint is the only thing
    to point at. DeepSeek publishes the weights, so a weight-anchored canonical
    exists and the mode belongs in an EEE record's `generation_config`. If
    maintainers prefer mode-as-canonical uniformly, the three lines revert.
  - **Left alone:** the dated variants (`deepseek/deepseek-v3-2-0925`,
    `-reasoning-0925`). A date is a legitimate identity axis, so folding them
    would lose information rather than de-duplicate it.

### Build-only canonical shadowing a seeded dated snapshot

- **`anthropic/Claude-3.7-Sonnet`** is declared by **no checked-in seed file** —
  it exists only as a `resolution_source: inferred`, `review_status: draft` entry
  materialized into the built tables, where it owns the display forms
  `Claude 3.7 Sonnet` / `claude-37-sonnet`. (The deployed Space already answers
  the conventional `anthropic/claude-sonnet-3.7` for those forms, so this is a
  build-state divergence rather than a live wrong answer — which is also
  independent confirmation that dropping the twin is the right direction.) The seed meanwhile carries `anthropic/claude-3-7-sonnet-20250219`
  (plus a `-thinking` twin), which is the exact model string LEXam's own
  `litellm_eval.py` calls, and which follows the documented closed-model form
  `{org_id}/{slug}` with a dated snapshot.
  - Effect: a consumer resolving the display name gets an odd-cased id with no
    seed provenance, while the dated canonical that names the actual weights
    sits beside it — two ids for one model, the split this file exists to track.
  - **Fixed in this PR.** No Every Eval Ever adapter emits the capitalized id —
    the one apparent hit was a regex artifact (`sciarena` maps the display
    string `Claude-3-7-Sonnet` to a developer). Since models.dev already
    declares the display forms on the conventional lowercase
    `anthropic/claude-sonnet-3.7`, adding the `tier3_inferred` materialization to
    `skip_source_ids` needs no alias block: `Claude-3.7-Sonnet` /
    `Claude 3.7 Sonnet` / `claude-37-sonnet` now resolve to
    `anthropic/claude-sonnet-3.7`, verified on a local rebuild, and the LEXam
    adapter emits that id.
  - **Residual risk:** a record already published under the capitalized id no
    longer matches the canonical. The raw display forms still resolve, so
    re-ingest lands on the right model; only previously written `model_info.id`
    values are stale. BenchPress's pinned resolver snapshot (unmerged) also
    needs a refresh.
  - **Still open, deliberately:** which of the remaining Claude 3.7 ids is
    canonical (`claude-sonnet-3.7` vs the dated `claude-3-7-sonnet-20250219` vs
    `claude-3.7-sonnet-thinking`). This PR removes only the unconventional twin;
    consolidating the rest is a naming call.

## Metric ids that name no quantity

Every Eval Ever's `metric identity` validator warns when `metric_id` is a word
that names no particular quantity, because such an id silently merges unrelated
numbers from every source that picked the same word. Six ids in
`seed/metrics.yaml` are exactly that shape, four of them `reviewed`:

  | id | review_status | bounds |
  |---|---|---|
  | `average` | draft | [0.0, 1.0] |
  | `elo` | reviewed | [None, None] |
  | `mean-score` | reviewed | [None, None] |
  | `overall` | draft | [0.0, 1.0] |
  | `rank` | reviewed | [1.0, None] |
  | `score` | reviewed | [None, None] |

An Elo or a rank is only comparable inside one leaderboard's pool, and
`score` / `overall` / `average` / `mean-score` say nothing about what was
measured — the unbounded `min_score: null` on several of them is the same
information gap seen from the other side. Their presence means a contributor who
resolves "Score" against the registry is handed an id the datastore validator
will then flag.

- **Action: decide per id** — retire it in favour of qualified slugs (the
  registry already prefers `mteb-score`, `codegolf.score`), or keep it with a
  note that it is intentionally source-local. Either way the alias forms that
  currently point at it need a target, so this is a curation pass rather than a
  deletion.
