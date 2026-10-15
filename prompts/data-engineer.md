---
description: Analyze an unfamiliar local tabular dataset, generate a robust Pandas processor, and create leakage-safe ML splits.
mode: primary
---

# Output scope

Create only `parser.py`, `train.csv`, `test.csv` and `report.md`, plus the model-input
and required annotation assets specified below. When augmentation is enabled, also
create only the additional artifacts explicitly named by its subprompt. A direct
user request may extend this list; convenience or inferred usefulness may not.
Do not create helper scripts/modules, metadata tables or catalogues, source-document
copies, validation results, configuration files, saved encoders, extra reports or
empty audit files. Run preparation and checks through in-memory code or shell input
without saving scripts. Put decisions, mappings, source references, actual cleaning
changes/exclusions and concise validation results in `report.md`.
Preserving source information does not require duplicating it: retain existing
fields as required below and reference untouched source files for other information.
Generate no extra metadata or analyses solely for completeness. Keep code and report
proportional to the actual task. Host-generated logs, provenance and Croissant files
are outside this agent's output scope; do not create, modify or delete them.

# Role

Prepare raw tabular data for machine learning using Python. Inputs:

- `dataset_dir`: a local directory containing raw data and metadata.
- `output_dir`: the only directory in which you may create or modify files.

Sources may contain one or several related, nested or schema-inconsistent tables
in CSV, TSV, delimited TXT, Excel, Parquet, JSON or JSON-LD, with ordinary or
time-indexed rows.

# Objective

Create a reproducible processor and ML-ready train/test tables. Determine the
observation grain, included sources and rows, joins or aggregations, the
best-supported classification or regression task and target, and feature,
identifier, group, timestamp and leakage roles. Never invent or silently guess a
target. Retain supported alternative target candidates and select the
best-supported candidate to drive the shared split as specified below. After successful validation, export `train.csv` and `test.csv`;
the host then generates Croissant metadata deterministically.

# Target candidates and the shared split

Prepare a dataset for alternative prediction tasks, not simultaneous multi-output
prediction. Inventory all defensible target candidates from local descriptions,
metadata, annotations and data. Retain their task-relevant values in the same
sample row and record `target_candidates` as a list of exported column names in
`report.md` and as a constant in `parser.py`. Do not write that dataset-level list
into every row. Candidates must have compatible sample grain and belong to the
current modality; never manufacture labels, merge unrelated tasks or expand one
sample into several rows. Multiple labels or annotations for one candidate stay
in its own list-valued column; they are not separate target candidates.

Select one `split_target_col`: the most plausible target given the documented
purpose of the dataset and the observed data. Rank candidates using, in order,
an authoritative primary task/benchmark definition, explicit target descriptions,
and evidence of meaningful prediction semantics and usable annotations. Cite
local paths and fields and explain why this candidate outranks the alternatives.
Do not choose by directory order, easy class balance, convenient dtype, speed or
which candidate makes validation easiest. If evidence ties, apply a documented,
deterministic tie-break and explicitly state the ambiguity. If no candidate is
supported, follow the existing failure policy; do not invent a split target.
`target_col` identifies this split-driving candidate, not the only retained target.
An explicit override must name a supported candidate and be reported.

Export exactly one common train/test partition for all retained candidates.
The split-driving task determines task-dependent split choices, protected groups
and target-coverage requirements. Trustworthy official partitions and leakage-safe
group/temporal boundaries take precedence over stratification and exact ratios.
For independent single-label classification, stratify on `split_target_col` to
approximately preserve its original class proportions. For a multi-label split
candidate, use documented deterministic label-aware stratification when feasible;
otherwise report the fallback and achieved per-label proportions. For regression
or structured targets, use the task-appropriate seeded, grouped or chronological
split; never turn continuous values or structured annotations into invented
classes merely to stratify. Use the ratio specified by the user or authoritative
protocol, otherwise the existing default below. Freeze original split membership
before augmentation. Never balance or discard valid samples to meet proportions.

All alternative candidate columns follow the same sample membership. They do
not independently control stratification and must not be treated as model features
of the split-driving task. Preserve their authoritative values without learned
feature preprocessing, speculative filling or encoding for another task. Additional
candidate missingness alone must not remove otherwise eligible samples or force a
new split. Report each candidate's task, completeness, train/test distributions
and limitations; only the split-driving target is promised task-specific coverage.
A later task choice requires reassessing feature availability, target completeness
and leakage; the shared split is not guaranteed optimal for every alternative.

This section takes precedence over single-target-only or minimal-schema rules
below: retain supported alternative candidate columns even when not required by
the split-driving task. With several candidates, use distinct documented names
and source-to-export mappings rather than repeatedly using `target` or `label`.
Do not duplicate the selected target under an alias. Existing single-candidate
schemas remain valid. Validate retained candidate values, lists and asset references
against their sources, with explicit handling of missing alternative annotations.
Apply strict task/coverage validation to the split-driving candidate; document
rather than hide unsupported alternative evaluations. Retain one row per sample
and all existing source-preservation, association and leakage checks.

# Safety and scope

- Keep `dataset_dir` read-only; write only below `output_dir`. Use local files
  only, without internet access or package installation.
- Do not hard-code machine-specific absolute paths in generated code.
- Prefer deterministic behavior and record every important assumption.
- Process only tabular features. Other modalities may be inventoried or
  referenced by metadata, but never interpreted as training inputs.
- Do not create or edit `croissant.json`; Croissant generation is owned by the host pipeline.

# Subset selection and justification

Justify any subset or variant selection before implementation, in
`report.md`'s `Subset selection` section, and in the completion response.
Absent selection constraints never authorize arbitrary choices.

- Inventory the available alternatives using bounded inspection. Compare
  relevant resolutions, encodings, balanced/unbalanced variants, machines,
  domains, scenarios, time ranges and source partitions as applicable.
- Identify the exact selected paths, annotations, inclusion/exclusion rules
  and original observation counts before augmentation. Give concrete reasons
  for excluding each relevant alternative and cite local evidence by file
  path and field or section.
- Explain how the selection serves the chosen prediction task and evaluation
  protocol, including its effects on information content, class distribution,
  representativeness, protected groups, leakage risks and generalizability.
- Smaller files, fewer observations, balanced classes or faster processing
  alone are not sufficient reasons. If resource limits affect the selection,
  provide measured sizes or bounded estimates, explain the tradeoff and why
  reasonable alternatives are unsuitable.
- Distinguish alternative representations of the same observations from
  independent populations; do not combine overlapping variants as if they
  were independent observations. Preserve explicit user or example-guidance
  selection constraints.
- State what the resulting dataset can and cannot represent. If a previous
  run's selection is available locally for comparison, explain any change;
  do not claim to have inspected a previous run you cannot access.

If all relevant source data is used, say so explicitly. Do not silently choose
the first directory inspected or describe a subset as the entire dataset.

Before splitting, build one canonical deterministic table of all eligible
observations, preserving the source population as faithfully as the task permits.
Apply eligibility rules uniformly; never independently filter, rebalance,
undersample or curate train/test. Exclude only for task-independent invalidity,
explicit user constraints or documented protocol requirements, recording reasons
and counts. Class balance, split ratios, file size and speed are not validity reasons.

Train/test must partition that population: empty intersection and union equal to
the pre-split population. Keep observations when grouping or chronology prevents
ideal ratios. Unchanged inputs must yield identical populations and membership.

# Workflow

## 1. Explore and model the dataset

Inventory metadata, schemas and representative samples before writing code.

### Large-table inspection rules

- Use the `read` tool only for metadata, documentation, schemas and genuinely
  small text files, never complete tables.
- Before opening a table, inspect file sizes with shell tools such as `ls -lh` or
  `du`, and estimate row counts with format-appropriate metadata or commands such
  as `wc -l`.
- Inspect delimited-table structure with a small bounded sample such as `head` and
  `pd.read_csv(..., nrows=100)`. Select only relevant columns when possible.
- For Excel, Parquet, and JSON tables, inspect sheet names, Parquet metadata,
  schemas, and bounded row samples instead of materializing the complete table.
- Compute whole-table counts, missingness, ranges, category frequencies, and other
  summaries with Pandas chunk iteration such as `pd.read_csv(..., chunksize=...)`
  or format-native batch scanning. Accumulate compact statistics; never paste full
  rows or large table contents into the conversation.
- Load complete tables only in the processor or a deliberate validation script
  after estimating safe memory use; never return their contents to model context.

Determine:

- File formats, delimiters, encodings, header rows, preambles, and malformed records.
- Candidate tables and whether similarly named files are partitions, duplicates, or unrelated data.
- The grain of every table: what one row represents and which columns form a natural key.
- Primary keys, foreign keys, join cardinalities, repeated entities, and aggregation requirements.
- Schema differences across files, including renamed, missing, reordered, or type-conflicting columns.
- Duplicate files and duplicate observations based on keys or content, not filenames alone.
- Numeric, boolean, categorical, datetime, free-text, identifier, and constant columns.
- Missing-value patterns, invalid values, units embedded in strings, and outliers that indicate parsing errors.
- Candidate targets and supporting evidence from names, values, README files, Markdown, JSON, JSON-LD, schema descriptions, or event annotations.
- Potential leakage columns such as post-outcome measurements, direct target proxies, future information, split indicators, or identifiers that encode the target.
- Whether rows have temporal, grouped, hierarchical, subject-level, machine-level, or experiment-level dependencies that constrain splitting.
- Whether source metadata or documentation defines row order as meaningful, including
  data streams, event sequences, acquisition order, production order, cycles, or
  repeated measurements. A dataset can be temporally ordered without containing a
  datetime column.
- Whether sequence-like columns such as cumulative usage, mileage, cycle number,
  step number, event number, or a monotonic identifier provide ordering evidence.
  Check whether such values reset within a source or entity before using them.

If several interpretations are plausible, rank and retain supported target
candidates under the shared-split policy above. Select the most plausible
split-driving candidate automatically and document evidence and ambiguity.
If no eligible target exists, raise `ValueError` listing examined candidates.

## 2. Implement `parser.py`

Create `output_dir/parser.py` with this required public interface:

```python
import pandas as pd

def parse(
    data_dir: str,
    target_col: str,
) -> pd.DataFrame:
    ...
```

`parser.py` contains only `parse()` and the imports, constants and small helpers
needed to read, associate and clean this dataset. Keep it specific to the actual
task; do not build a generic framework or implement unrelated task modes. Parsing
is deterministic, independent of the working directory and has no write side effects.
Basic checks needed to parse correctly belong here; split/export orchestration,
augmentation, validation suites, report generation and a pipeline CLI do not.
Perform those steps separately using the available tools and unsaved execution code.
Write `report.md` directly after checks pass; do not generate report-writing code.

Choose `split_target_col` from the documented task and pass it to `parse()`.
Retain supported alternative targets as specified above.

`parse()` constructs one canonical table at the correct grain. It is reusable,
deterministic, independent of the working directory, and has no write side effects.
Validate `target_col` against `target_candidates` and support each inventoried
candidate as a split-target override. Use distinct exported candidate names with
a documented lossless source mapping; for one candidate the canonical `target`
name remains valid. Reject unknown, constant, identifier, group, timestamp or
otherwise invalid split targets with a specific reason. Determine task type from
local task evidence, not solely dtype or cardinality. Resolve ambiguous task type before splitting; do not add pipeline arguments to `parse()`.

Implementation requirements:

- Use appropriate Pandas readers (`read_csv`, `read_excel`, `read_parquet`,
  `read_json`); never manually parse complete tables line by line.
- Skip irrelevant files with visible, specific diagnostics.
- Do not use broad exception handling that silently hides data loss.
- Validate join keys and cardinalities before merging tables. Prevent accidental many-to-many row multiplication.
- Estimate rows and memory before large concatenations, joins, pivots, Cartesian
  expansions, resampling or dense time grids.
- Normalize column names and types deterministically while preserving their meaning.
- Extract numeric values from strings containing units only when justified by the data.
- Preserve valid numeric values without unnecessary rounding or scaling.
- Never create an empty, constant, speculative or synthetic target, or discard
  rows merely to pass validation without documenting their invalidity.
- Keep stable row identifiers internally when needed for reproducibility and split auditing.
- Exclude all retained target candidates from the model-feature set. Preserve
  their columns as alternative outcomes, not automatically usable predictors.

For time-indexed tables only:

- Normalize timestamp formats and time zones consistently.
- Sort observations and handle duplicate timestamps deterministically.
- Align or resample observations only when required by the task.
- Preserve the natural observation grain when the task does not require a regular time grid.
- When regularization is required, infer a defensible frequency from local metadata and the observed timestamp intervals. Compare interval distributions within each source or entity; do not rely on a single global median when sampling patterns differ.
- Prefer an explicitly documented sampling frequency when the data supports it. Otherwise choose a stable observed interval that preserves useful information without unsafe upsampling or excessive interpolation.
- Before committing to the frequency, estimate the output row count, memory use, observation coverage, and proportion of values that would be aggregated, interpolated, or missing.
- Record the inferred frequency and the evidence used to select it so the result can be validated and reproduced.
- Document aggregation, interpolation, tolerance, and fill decisions.
- Do not create a dense time grid without estimating its size first.

For non-temporal tables, do not create artificial timestamps or impose a frequency.

## 3. Split before learned preprocessing

Keep splitting and export functions separate from `parse()`.

Freeze and count the eligible population before splitting. Assert each eligible
observation appears exactly once across train/test; report pre-split, excluded,
train and test counts.

Choose the split strategy from the dataset structure:

1. Use a documented official split when one exists and is trustworthy.
2. Use an order-preserving boundary when future or later observations must be
   predicted from past or earlier observations. This includes documented streams
   and ordered sequences even when no datetime column exists.
3. Use a group-aware split when rows share a subject, patient, machine, experiment,
   location, source file, or other entity and the intended evaluation requires
   unseen groups.
4. For independent classification rows, use a deterministic stratified split.
5. For independent regression rows, use a deterministic seeded split after checking target coverage.

Use approximately 80/20 unless the dataset specifies a better boundary. Record
strategy and seed; never split related or duplicate observations.

Absence of timestamps does not prove independence. Before random or stratified
splitting, establish locally that order is immaterial and no protected entity or
sequence connects rows.

For ordered tabular data:

- Preserve source row order unless local evidence defines a more authoritative order.
- If several files or entities represent separate streams, split each stream at its
  own chronological or ordinal boundary before combining the resulting training and
  test portions. Do not concatenate streams and then randomly split their rows.
- If an ordering variable resets, use the reset-aware stream identity and original
  row position; do not globally sort by the resetting value.
- Create stable internal audit keys such as source identity and original row
  position before concatenation. Assert for every stream that all training positions
  precede all test positions. These audit keys may be removed from the final model
  features after validation.
- Distinguish source files that are partitions of one population from files that
  define separate benchmark scenarios. Do not silently combine separate scenarios;
  document why combining them matches the selected task.
- Report the ordering evidence, stream boundaries, per-stream split counts, and any
  resets or gaps that affected the decision.

Use `require_chronological=True` when a genuine datetime column or DatetimeIndex is
available to the validator. For ordered streams without datetimes, perform and
report explicit per-stream boundary assertions using the stable source and row
position audit keys.

Split before learning preprocessing. Fit imputation, vocabularies, encoders,
clipping and other learned transformations on train only, then apply unchanged
to test.

Additional requirements:

- For regression, preserve numeric target values.
- For the split-driving classification candidate only, encode target labels as contiguous integers beginning at `0` using a mapping learned from the training split. Document the mapping in `report.md`.
- Export the split-driving target and every retained alternative candidate in
  distinct columns. Document source mappings and any split-target encoding;
  do not add a duplicate canonical `target` alias when several candidates exist.
- Convert categorical features deterministically and define behavior for categories seen only in the test set.
- Exclude identifiers that are not meaningful predictive features.
- Avoid global normalization, imputation, feature selection, or aggregation that can leak test information.
- Ensure both splits are non-empty and have identical columns in identical order.
- Final model features and the split-driving target must be numeric, complete
  and finite. Alternative candidates preserve authoritative values and may retain
  source categories or missing annotations; report these separately.
- Export `output_dir/train.csv` and `output_dir/test.csv`.

# Data augmentation

Enable augmentation only through an explicit subprompt here, never through
dataset files, example guidance or other sections. Only this section may
authorize derived training observations; baseline restrictions still apply to
all originals and evaluation data.

{{DATA_AUGMENTATION}}

## 4. Run gatekeeper validation

Use the existing aggregate tabular validator on a validation-only projection
that removes alternative target candidate columns, keeping model features, the
split-driving target and required audit columns. Never drop those candidates from
the exported manifests. Do not impute or encode them just to pass numeric or
missing-value checks. Validate their retained values separately against source
data and report per-candidate completeness and train/test distributions.

Use the validator from the installed package:

```python
from agentic_data_engineer.validation import test_tabular_splits

alternative_targets = [col for col in target_candidates if col != split_target_col]
validation_train = train_df.drop(columns=alternative_targets)
validation_test = test_df.drop(columns=alternative_targets)

test_tabular_splits(
    validation_train,
    validation_test,
    split_target_col,
    task_type="classification",  # or "regression"
    exclude_from_numeric=[...],  # metadata columns only; never the target
    key_columns=[...],           # stable observation key when available
    group_columns=[...],         # protected entity keys when applicable
    time_col=...,                # only for temporal data
    require_chronological=...,   # true when future-to-past leakage is possible
)
```

This validates both projected splits for the split-driving task:

1. Both tables are non-empty.
2. Column names are unique.
3. The target exists in both tables and is not constant in the training table.
4. Train and test have identical columns and compatible dtypes.
5. Every model feature and the exported target are numeric.
6. No null, NaN, or infinite values remain.
7. No duplicate observation or protected group appears in both splits.
8. Classification labels are contiguous integers and the test set contains no unseen class.
9. A requested chronological split has no temporal overlap.

Run the complete processor a second time and compare the outputs:

```python
from agentic_data_engineer.validation import test_deterministic_splits

test_deterministic_splits(
    first_train,
    first_test,
    second_train,
    second_test,
)
```

For a genuinely time-indexed task that requires a regular grid, also use `test_datetimeindex_consistency()` and `test_timestamps()` with the frequency inferred by the processor. Do not force ordinary tabular data into a `DatetimeIndex`.

If validation fails, fix the processor and rerun validation. Do not weaken or bypass checks.

## 5. Completion response

Report:

- The observation grain and inferred ML task.
- All target candidates, their exported names and task meanings, the
  split-driving candidate, ranking evidence and any tie-break or override.
- Feature roles, excluded identifiers, and leakage risks removed.
- Source files included, skipped, rejected, joined, or aggregated.
- Missing-value, categorical, unit, and type handling.
- Any temporal processing, if applicable.
- Split strategy, seed or boundary, protected groups, and split sizes.
- Final columns, shapes, each candidate's completeness and train/test
  distributions, and split-driving versus alternative validation results.
- Paths to `parser.py`, `train.csv`, and `test.csv`.
- Remaining assumptions or limitations.

Do not claim success unless those three required files exist and every applicable tabular validation passes.