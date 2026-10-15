### Objective and scope

Follow the main output allowlist. Use unsaved execution code; do not create an
augmentation module, helper scripts or additional audit/configuration files.

Implement offline tabular augmentation outside `parser.py`. Loading this subprompt
requires actual additional training rows. Follow the main prompt's splitting,
information-preservation, preprocessing and validation rules; no model training.

Support independent observations with a scalar classification or regression target.
Do not apply row-wise perturbations to ordered sequences or linked measurements
without a task-specific method preserving those dependencies. Fail clearly when
no justified augmentation is available; do not switch tasks or invent target values.

### Select transformations

Use only documented continuous input measurements whose plausible variation leaves
all known target candidates and annotations unchanged. Targets, IDs, timestamps,
group keys, categorical values and provenance are never perturbation inputs.

| Method | Operation and conditions |
| --- | --- |
| Additive measurement noise | Add zero-mean noise justified by measurement uncertainty or natural training variation; respect units, valid ranges and joint constraints. |
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
unchanged targets. Keep source/group provenance, assign stable unique derivative
keys and retain parent values in the audit. Changed measurements describe the
synthetic row, not an observed recording. Use the main prompt's same export schema.

Apply a documented count policy to comparable sources and explain ineligible cases.
Use a configurable recorded seed and stable `hashlib`-derived per-source/per-variant
seeds, independent of traversal order. Reject invalid proposals using a documented
bounded procedure; report failure rather than retrying indefinitely or adding copies.

Write `augmentation.csv` with
`sample_id,original_sample_id,technique,seed,parameters,original_sha256,augmented_sha256`.
Reference stable exported row keys (map differently named natural keys explicitly).
Store changed fields, parent values, generated values and exact settings in canonical
JSON `parameters`. Hash a documented canonical serialization of the modeled source
measurements and targets; exclude synthetic IDs so re-identification is not a change.

### Validate and report

In addition to main checks, verify parent lineage, unchanged targets, valid units/
ranges/dependencies, count-policy agreement and unchanged evaluation membership.
Require actual changes in exported model features, not just source columns or IDs;
check for duplicate variants. Independently reproduce rows, hashes and preprocessing
from recorded parents/settings/seeds. Never use test rows as parents or parameter
estimates. A positive number of usable variants is required for success.

Report selected/rejected methods and evidence, affected fields, settings, count
policy, seed, library versions, original/derived counts and validation. Include a
compact before/after sample and distribution comparison in `report.md`, and print
training counts. Report success only after real variants and audits are exported.
