### Objective and scope

Follow the main output scope. Keep `parser.py` limited to parsing and splitting;
run augmentation, audit, previews and checks separately using unsaved execution
code. Use local files and installed libraries only; do not create helper modules.

Loading this subprompt requires actual additional training rows. Follow the main
prompt's splitting, information-preservation and preprocessing rules; no model
training is required.

Support independent observations with a scalar classification or regression target.
Do not apply row-wise perturbations to ordered sequences or linked measurements
without a task-specific method preserving those dependencies. Fail clearly when
no justified augmentation is available; do not switch tasks or invent target values.

### Select transformations

Use only documented continuous input measurements whose plausible variation leaves
all known target candidates and annotations unchanged. Missing target annotations
do not establish label invariance; use task evidence. Targets, IDs, timestamps,
group keys, categorical values and provenance are never perturbation inputs.

| Method | Operation and conditions |
| --- | --- |
| Additive measurement noise | Add zero-mean noise justified by measurement uncertainty; respect units, valid ranges and joint constraints. |
| Multiplicative measurement variation | Apply a positive factor when proportional measurement variation is meaningful and target-preserving; preserve signs and physical constraints. |

Do not apply methods automatically. Derive strengths, affected fields, sampling
rules and variant counts from domain evidence and original train, with no universal
numeric defaults or test-data tuning. A small change is not automatically safe,
especially for regression, threshold labels or fault detection. Training spread
alone does not establish measurement uncertainty or label invariance.

Preserve dependencies between measurements and recompute derived input fields from
their documented formulas. Never perturb one-hot columns independently, round
continuous perturbations into invented discrete values, or clip invalid rows into
plausibility. Exclude methods requiring changed targets. Duplication, resampling,
SMOTE, cross-class mixing and generative synthesis are outside this policy.

### Generate and audit

Freeze original splits and retain every baseline row. Generate distinct non-identity
variants in cleaned input space before applying the already fitted preprocessing;
never refit on synthetic rows. Keep missing source values missing and preserve
unchanged targets. Keep source/group provenance. If an exported observation ID
must be unique, assign a deterministic derivative ID and record its parent in the
audit; never change group IDs or add source/ID columns merely for auditing.
Changed measurements describe a synthetic observation. Use the main prompt's same export schema.

Apply a documented count policy to comparable sources and explain ineligible cases.
Report original/derived counts and class or target distributions; explain any
change caused by eligibility or count rules. Do not silently rebalance targets.
Use a recorded seed and stable `hashlib`-derived per-source/per-variant
seeds, independent of traversal order. Reject invalid proposals using a documented
bounded procedure; report failure rather than retrying indefinitely or adding copies.

Generate every variant directly from an original training row. Write
`augmentation.csv` with:

```text
row_index,original_row_index,technique,seed,parameters,original_sha256,augmented_sha256
```

Indices are zero-based data-row positions in the final `train.csv` (excluding its
header), assigned after final ordering. They identify the derivative and its
retained original without adding audit columns to the model table. The audit must
resolve against that exact exported file.
Store `parameters` as a Python list of `[name, value]` pairs, using nested lists
when needed. Serialize with canonical `repr()` and read with `ast.literal_eval`;
never use dictionaries or JSON in table cells.
Record affected fields, original cleaned values, generated values and settings.
Hash a documented canonical serialization of exported model features and targets,
excluding IDs. Require a real feature change after preprocessing and serialization;
otherwise reject the proposal. Never refit preprocessing on derivatives. Keep the
preprocessing settings needed for reproduction in `report.md`.

### Validate and report

In addition to main checks, verify parent lineage, unchanged targets, valid units/
ranges/dependencies, count-policy agreement and unchanged evaluation membership.
Require actual changes in exported model features, not just source columns or IDs;
reject variants identical to any original training row or accepted variant when
compared on exported model features, excluding identifiers. Independently reproduce rows, hashes and preprocessing
from recorded parents/settings/seeds. Never use test rows as parents or parameter
estimates. A positive number of usable variants is required for success.

Report selected/rejected methods and evidence, affected fields, settings, count
policy, seed, library versions, original/derived counts and validation. Include a
compact before/after sample and distribution comparison in `report.md`, and print
training counts. Report success only after real variants and audits are exported.
