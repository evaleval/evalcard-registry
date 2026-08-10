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

## EleutherAI/lm-evaluation-harness

- **`ter` is registered as higher-is-better** — `lm_eval/api/metrics.py` declares
  `@register_metric(metric="ter", higher_is_better=True)` while the aggregation it
  registers says "Lower is better" in its own docstring and computes
  `sacrebleu.corpus_ter`, an edit rate. `higher_is_better` is not a sort key or a
  display hint: `is_higher_better()` (`api/registry.py`) feeds
  `Task.higher_is_better()`, which lm-eval writes into its results JSON, and the
  same table correctly says `False` for `perplexity`, `word_perplexity`,
  `byte_perplexity`, `bits_per_byte` and `brier_score`. So `ter` is the lone
  inverted entry, and a run that does not override it per task
  (`api/task.py` reads `higher_is_better` out of a task's `metric_list`) reports
  the wrong direction.
  - Local handling: the `ter` metric here states `lower_is_better: true`, the
    metric's actual direction, and records the contradiction in its `metadata`.
  - **Action: flag to the lm-evaluation-harness maintainers.** Remove this entry
    and the metadata note once `higher_is_better=False` lands.

- **`chrf` is documented as chrF++ but computes chrF** — the same file's `chrf`
  aggregation opens "chrF++ is a tool for automatic evaluation…" and hedges
  "Higher is better  # TODO I think", but calls `sacrebleu.corpus_chrf(preds, refs)`
  with the defaults, and sacrebleu is chrF at `word_order=0` and chrF++ only at
  `word_order=2`. The computed number is chrF; the direction it registers is
  correct.
  - Local handling: this registry keeps `chrf` and `chrf-plus-plus` as separate
    canonicals, and `chrf`'s metadata says which one lm-eval's stat is, so the
    docstring cannot pull a consumer onto the wrong entry.
  - **Action: flag to the lm-evaluation-harness maintainers** — a docstring fix,
    no behaviour change.
