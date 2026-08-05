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
- **`Qwen/Qwen3-Next-80B-A3B-Instruct` / `-Thinking`** — absent; `Qwen3-Next`
  resolves to the models_dev draft `alibaba/qwen3-next`, which does not
  distinguish the two released variants.
- **`utter-project/EuroLLM-9B-Instruct`** — absent; only the base
  `utter-project/EuroLLM-9B` is present, so an instruct-model label can only
  resolve to the base model.
- **`meta-llama/Llama-3.1-405B-Instruct`** — absent; only the Together-hosted
  `Meta-Llama-3.1-405B-Instruct-Turbo` drafts are present.

**Locally handled now:** three of the four are added as curated `hf` canonicals
in `seed/models/core.yaml` with the leaderboard labels as aliases
(`deepseek-ai/DeepSeek-V3.2-Exp`, `meta-llama/Llama-3.1-405B-Instruct`,
`utter-project/EuroLLM-9B-Instruct`). The superseded models.dev draft
`deepseek/deepseek-v3-2-exp` is listed in `core.yaml` `skip_source_ids`, since
keeping both would make `deepseek-v3-2-exp` resolve ambiguously — the same
failure mode as the `gemma2-9b-it` alias below.

`Qwen3-Next` is deliberately **not** added: the leaderboard label does not say
which released variant was evaluated (`Qwen/Qwen3-Next-80B-A3B-Instruct` vs
`-Thinking`), so the variant-agnostic `alibaba/qwen3-next` stays until the
benchmark authors confirm. Adding a canonical would be a guess about which
weights produced the score.

- **Action: check why the HF generator missed these repos** (all four exist on
  the Hub). The curated entries above are a floor, not a fix — once the
  generator picks them up, the `core.yaml` entries can be thinned back to
  whatever the generator does not reproduce.

### Ambiguous instruct alias

- **`gemma2-9b-it` → `google/gemma-2-9b`** (seed alias) points an `-it` form at
  the **base** model, while the correctly-formed `gemma-2-9b-it` →
  `google/gemma-2-9b-it` exists alongside it. Any resolver that normalizes
  punctuation sees both and can land on the base model for an instruct-tuned
  evaluation.
  - **Action: re-point `gemma2-9b-it` at `google/gemma-2-9b-it`.**

### Mode-as-model drafts

`deepseek/deepseek-v3-2-reasoning`, `deepseek/deepseek-v3.2-thinking`,
`deepseek/deepseek-v3-2-exp-prompt-thinking` and
`fireworks/deepseek-v3p2-thinking` are thinking/non-thinking **modes** of a
single checkpoint rather than separate models (DeepSeek's API changelog,
2025-12-01: `deepseek-chat` and `deepseek-reasoner` are the non-thinking and
thinking modes of DeepSeek-V3.2). `deepseek-v3.2-thinking` is already an alias
of `deepseek-ai/DeepSeek-V3.2`, which is the right treatment; the remaining
draft canonicals bake a generation setting into model identity.

- **Action: fold the mode drafts into their checkpoint canonicals as aliases.**

### Build-only canonical shadowing a seeded dated snapshot

- **`anthropic/Claude-3.7-Sonnet`** is what the resolver returns for the raw
  forms `Claude 3.7 Sonnet` / `claude-37-sonnet` (confirmed aliases), but that id
  is declared by **no checked-in seed file** — it exists only as a
  `resolution_source: inferred`, `review_status: draft` entry in the built
  tables. The seed meanwhile carries `anthropic/claude-3-7-sonnet-20250219`
  (plus a `-thinking` twin), which is the exact model string LEXam's own
  `litellm_eval.py` calls, and which follows the documented closed-model form
  `{org_id}/{slug}` with a dated snapshot.
  - Effect: a consumer resolving the display name gets an odd-cased id with no
    seed provenance, while the dated canonical that names the actual weights
    sits beside it — two ids for one model, the split this file exists to track.
  - The LEXam adapter keeps the resolver's current answer rather than
    re-pointing a widely used id on its own, and reports the mismatch in its
    `registry_snapshot.json` under `models_absent_from_seed`.
  - **Action: decide which id is canonical for Claude 3.7 Sonnet.** If the
    dated one wins, fold the inferred entry in via `skip_source_ids` plus an
    alias bridge (the treatment already used for `deepseek/deepseek-v3-2-exp`);
    if the display-name root wins, declare it in `core.yaml` so it has seed
    provenance.
