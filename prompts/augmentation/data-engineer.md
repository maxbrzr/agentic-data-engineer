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

Transform only documented input measurements whose plausible variation leaves
all known target candidates and annotations unchanged. Missing target annotations
do not establish label invariance; use task evidence. Targets, IDs, timestamps,
group keys and provenance are never distance or noise inputs. Do not numerically
interpolate categorical values; use category-aware methods below. Synthetic time
coordinates are allowed only under the temporal interpolation rule.

| Method | Operation and conditions |
| --- | --- |
| Additive measurement noise | Add zero-mean noise justified by measurement uncertainty; respect units, valid ranges and joint constraints. |
| Multiplicative measurement variation | Apply a positive factor when proportional measurement variation is meaningful and target-preserving; preserve signs and physical constraints. |
| SMOTE (numeric classification oversampling) | Select an original training sample from a class to oversample and randomly choose one of its eligible same-class k-nearest neighbors in feature space. Generate `x_new = x_a + lambda * (x_b - x_a)` with a random scalar weight shared across features; reject identity outcomes. This uses linear feature-space interpolation, not temporal interpolation or missing-value imputation. Exclude IDs and targets from distances; justify scaling, metric, k and class sampling counts from original train. Require enough compatible neighbors and plausible joint values. |
| Borderline-SMOTE (borderline-1) | Detect minority-class training samples near a class boundary using neighborhoods containing all classes, then generate samples toward compatible same-class neighbors. Use borderline-1 under this policy; variants using other-class generating neighbors are excluded by the target-compatibility rule. Check noise and outliers before selecting boundary samples. |
| ADASYN | Allocate more synthetic samples to minority-class training samples with a higher proportion of other-class neighbors, then interpolate toward eligible same-class neighbors. This adaptive allocation differs from selecting only borderline samples. Check noise, outliers and overlap; record achieved counts, which can differ from requested counts. |
| SMOTENC / SMOTEN | Use SMOTENC for mixed continuous/categorical inputs and SMOTEN for all-categorical inputs. Declare categorical fields explicitly; preserve valid categories and joint constraints. Never interpolate category codes or one-hot vectors as continuous measurements. |
| Random oversampling | Sample original training rows with replacement under an explicit, justified count policy. Exact copies are allowed for this method only; report them as duplicates, not new information. |
| Smoothed bootstrap | For continuous numeric measurements, sample training parents and add noise with a documented dispersion/shrinkage policy. Standard noise distributions may be unbounded; reject physically invalid proposals under this policy. Respect correlated features and physical constraints; do not perturb categorical codes. |
| Temporal linear interpolation | Add synthetic training observations at interior times between compatible measurements in the same entity, recording and continuous segment. Use actual elapsed-time weights for irregular sampling; require documented smoothness, bounded gaps and stable labels. No extrapolation, crossing recording gaps or label transitions. |

The conditions above are project eligibility rules, not universal guarantees of
label preservation or properties of every library implementation.

Do not apply methods automatically. Derive strengths, affected fields, sampling
rules and variant counts from domain evidence and original train, with no universal
numeric defaults or test-data tuning. A small change is not automatically safe,
especially for regression, threshold labels or fault detection. Training spread
alone does not establish measurement uncertainty or label invariance.

Preserve dependencies between measurements and recompute derived input fields from
their documented formulas. Never perturb one-hot columns independently, round
continuous perturbations into invented discrete values, or clip invalid rows into
plausibility. Exclude methods requiring changed targets. Cross-class mixup with
soft labels, regression-target interpolation and generative synthesis remain outside
this policy. Do not remove baseline rows through undersampling or cleaning samplers.

For multi-parent methods, all parents must have matching known values for every
retained target candidate and compatible annotations; missing labels do not prove
compatibility. Same-class membership alone does not prove a plausible interpolation.
For regression, use target-preserving noise/scaling or duplication when justified;
do not assume interpolated inputs preserve a continuous outcome.

Temporal interpolation is explicitly authorized only for additional synthetic
training rows after the baseline split. It does not replace the main prompt's
alignment procedure or change original/evaluation measurements. Keep all endpoints
inside original train and preserve group/recording identity. Never use a measurement
that would be unavailable at prediction time; two-sided interpolation is unsuitable
for a causal forecasting input if its right endpoint lies in that input's future.
Interpolate compatible continuous channels coherently; copy only demonstrably
constant categorical attributes and targets. Retain original missing values; do not
present filling missing cells as additional observations. Document synthetic times,
endpoints, weights, maximum gap and causal availability. Do not apply ordinary
row-wise SMOTE to dependent time-series samples.

### Generate and audit

Freeze original splits and retain every baseline row. Generate additional
variants (non-identity except for explicitly selected random oversampling) in cleaned input space before applying the already fitted preprocessing;
never refit on synthetic rows. Keep missing source values missing and preserve
unchanged targets. Keep source/group provenance. If an exported observation ID
must be unique, assign a deterministic derivative ID and track its parents in
memory; never change group IDs or add source/ID columns merely for auditing.
Changed measurements describe a synthetic observation. Use the main prompt's same export schema.
Never use timestamps or bin indices as arbitrary synthetic IDs. Recompute derived
bin indices from synthetic timestamps using the same origin and width as originals;
multiple observations may legitimately share a bin. If a separate observation ID
is necessary for uniqueness, use it consistently for original and synthetic rows
in the shared export schema. Preserve chronological ordering within each recording
when exporting time-series data, then resolve audit references to final row positions.

Apply a documented count policy to comparable sources and explain ineligible cases.
Report original/derived counts and class or target distributions; explain any
change caused by eligibility or count rules. Do not silently rebalance targets.
Use a recorded seed and stable `hashlib`-derived per-source/per-variant
seeds, independent of traversal order. Reject invalid proposals using a documented
bounded procedure; report failure rather than retrying indefinitely or falling back to unplanned copies.

Generate every variant directly from one or more original training rows, never
from another synthetic row. Fit neighbor search, scaling and sampler statistics on
original train only. Retain the already fitted export preprocessing; any temporary
sampler representation must map back to the export schema. If cross-validation is
performed later, regenerate augmentation separately inside each training fold.

Append accepted augmented observations directly to `train.csv` alongside all
original training rows, using the same export schema. Keep `test.csv` as the
unchanged evaluation split; never augment it. Write `augmentation.csv` as lineage
and transformation documentation, not as a replacement for augmented `train.csv`.

Write one audit row per added training observation with this schema:

```text
row_index,original_row_index,technique,seed,parameters,original_sha256,augmented_sha256
```

Indices are zero-based data-row positions in final `train.csv`, excluding the
header and assigned after final ordering. `original_row_index` identifies a retained
original parent; for multiple parents it identifies the anchor. In `parameters`,
record ordered `parent_row_indices`, corresponding `parent_sha256` values, weights
or categorical choices, affected fields and settings. Include every generating
parent and verify that all parent references resolve to retained original training
rows. Serialize parameters as a Python list of `[name, value]` pairs with nested
lists where needed, serialized as valid Python list literals and read with
`ast.literal_eval`, never `eval()`.

Hash a documented canonical serialization of exported model features and targets,
excluding IDs. Hashes must resolve against the final exported file. Record the
training-pool selection, ordering, metric, count policy and deterministic generation
procedure in `report.md`; include compact before/after examples there.
Require a real feature change after export serialization for synthetic methods;
otherwise reject the proposal unless it is explicit random oversampling. Never
refit preprocessing on derivatives. Document preprocessing settings in `report.md`.

### Validate and report

In addition to main checks, verify parent lineage, unchanged targets, valid units/
ranges/dependencies, count-policy agreement and unchanged evaluation membership.
For synthetic methods, require actual changes in exported model features, not just
source columns or IDs; reject variants identical to originals or accepted variants
on model features excluding identifiers. For random oversampling, instead verify
exact feature/target equality with the parent and the intended multiplicities. Independently reproduce rows and preprocessing
from the documented procedure and exported audit parents/settings/seeds. Never use test rows as parents or parameter
estimates. A positive number of usable variants is required for success.

Report selected/rejected methods and evidence, affected fields, settings, count
policy, seed, library versions, original/derived counts and validation. Include a
compact before/after sample and distribution comparison in `report.md`, and print
training counts. Report success only after augmented rows are exported in `train.csv`, the
`augmentation.csv` is exported, the files pass validation and the report records
the augmentation results.
