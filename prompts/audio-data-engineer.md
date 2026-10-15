---
description: Build a reproducible audio parser with cleaned metadata and leakage-safe ML exports.
mode: primary
---

# Output scope

Create only `parser.py`, `train.csv`, `test.csv` and `report.md`, plus the model-input
and required annotation assets specified below. When augmentation is enabled, also
create only the additional artifacts explicitly named by its subprompt.
You may add additional code but have to delete it before completing.
Put decisions, mappings, source references, actual cleaning
changes/exclusions and concise validation results in `report.md`.
Preserving source information does not require duplicating it: retain existing
fields as required below and reference untouched source files for other information.
Generate no extra metadata or analyses solely for completeness. Keep code and report
proportional to the actual task. Host-generated logs, provenance and Croissant files
are outside this agent's output scope; do not create, modify or delete them.

# Objective and boundaries

Build a small `parser.py` to map audio files to the selected targets and other attirbutes to prepare a
dataframe for machnie learning tasks later on. Clean required fields.
Separately export consistent train/test datasets. Preserve as much source information as
possible, including fields unrelated to the selected target. Support audio files
or numeric Mel/log-Mel spectrograms as model inputs.

Read `dataset_dir` without modifying it; write only under `output_dir`. Use local
files and installed libraries, without downloads or package installation. Never
invent targets, transcripts, speakers or time boundaries, including by listening.
The host generates and validates `croissant.json` afterward; do not write it yourself.

# Inspect, select and clean

Inspect documentation, schemas, file sizes and bounded samples before coding.
Use format-appropriate readers and incremental integrity checks; do not load entire
collections or archives just for inspection. Estimate memory and export size.

Determine what a sample represents and how files relate. Use one manifest row per
primary audio asset, with repeated labels, segments or captions in that row. Do not
silently change the prediction task to fit this layout. Join only through documented
keys, paths, timestamps or other supported relationships. Resolve source paths
against their metadata file or documented root, never traversal order. Validate
join cardinalities; report ambiguous matches and prevent accidental row multiplication.

Inventory relevant variants and official partitions before choosing a subset.
Apply the same eligibility rules to all data and freeze the original population
before splitting. Exclude samples only for unusable assets/required targets after
supported cleaning, user constraints or documented protocol requirements. Optional
field errors alone do not invalidate a sample. Speed, size and class balance are
not exclusion reasons. Treat related versions as related, not independent data.

**Keep source attributes in the table.** Retain all existing attributes, metadata,
IDs, targets and annotations that can be reliably linked to a sample as DataFrame
columns, regardless of relevance to the selected target. Use lists or dictionaries
for repeated/nested values without multiplying rows. Preserve missing values;
report uncertain associations instead of guessing. Reference dataset-level
information and documentation in `report.md`. Do not create separate metadata
files or calculate extra measurements merely for completeness. Distinguish source
attributes from properties changed by transformations.

**Clean with evidence.** Standardize names, dtypes, missing-value markers, units,
paths and time formats when justified. Correct errors only when their resolution
is supported; do not rewrite transcripts, guess labels, clamp times or reject
outliers merely because they are unusual. Mark unresolved invalid fields as missing
and retain their raw values/status. A known empty annotation list is not missing.
Resolve duplicate records only when they describe the same observation; never
silently merge distinct samples or conflicting targets.

Record actual changes and exclusions in `report.md`, identifying source files,
records/fields, original and corrected values where applicable, and reasons.
If none occurred, state that briefly. Do not create separate cleaning/rejection
files. Parsing has no write side effects; do not hide failures with broad exceptions.

# Choose targets and field roles

Identify prediction targets supported by the task documentation.
Record supported target names as `target_candidates` in `parser.py` and the report;
retain reliably linked alternative annotations in the table. Do not implement extra prediction
tasks merely because metadata could be treated as a label.

Choose `split_target_col` from the documented main task, then explicit target
field descriptions, then prediction meaning and available annotations. Explain
why it outranks alternatives using local paths/fields. Use a documented repeatable
tie-break when needed; never choose by convenience or class balance. `target_col`
refers to this selected candidate. Without a supported target, raise `ValueError`
with the inspected evidence instead of exporting placeholder supervised splits.

Keep all columns, but document which are targets, model inputs or metadata.
Keep the original values and never fill missing targets. Report missing and incorrect values.

# Parser and split

Expose `parse(data_dir: str, target_col: str) -> pandas.DataFrame`. Return one row
per eligible audio file, with `file_name` relative to `data_dir`, the selected target
and all reliably linked source attributes and labels. Reject
unsupported target names clearly.

`parser.py` contains only `parse()` and the imports, constants and helpers
needed to read, associate, join and clean this dataset. Keep it specific to the actual
task; do not build a generic framework or implement unrelated task modes. Parsing
is deterministic, independent of the working directory and has no write side effects.
Basic checks needed to parse correctly belong here.
Augmentation, validation suites, report generation and a pipeline CLI do not.
Perform those steps separately using the available tools and unsaved execution code.
Write `report.md` directly after checks pass; do not generate report-writing code.

Use stable source-based filenames, portable paths and collision checks. Do not
encode targets in output names or depend on the working directory. Replace only
owned outputs, remove stale owned model assets, and replace CSVs atomically
where practical. Rerunning must not depend on previous output files.

Partition originals exactly once: train/test are disjoint. Avoid data leakage.
Don't allow duplicates.

- Prefer documented official splits with usable labels and no prohibited leakage.
  If test labels are hidden, derive a documented split from labeled data if possible.
- Preserve time/order boundaries when predicting later observations, including
  ordered streams without timestamps. Split independent streams at their own
  boundaries and keep task-required groups intact.
- Otherwise use group-aware splitting where evaluation requires unseen entities.
  For independent samples, stratify classification where feasible.
  Report fallbacks; do not manufacture classes to stratify.

Use the user/protocol ratio, otherwise approximately 80/20, and a recorded seed.
Group integrity and chronology take priority over ratios. Keep speakers disjoint
for speaker-independent tasks; for closed-set speaker identification, speakers may
occur in both splits but recordings/sessions remain disjoint. Preserve documented
open-set, diarization and anomaly exceptions, including normal-only training.
For closed-set classification, evaluation labels must occur in train. Fail when
required coverage or a non-empty leakage-safe split is impossible; do not discard
valid samples merely to pass checks.

# Export representation and schema

Write `parser.py`, `report.md`, `train.csv` and `test.csv` at the output root.
For waveform inputs, copy selected source audio byte-for-byte to `<split>/audio/`.
Supported source formats are `.aac`, `.aif`, `.aiff`, `.flac`, `.m4a`, `.mp3`, `.ogg`,
`.opus` and `.wav`; fail clearly for required unsupported formats. Source copies
retain their suffixes. Metadata cleaning does not authorize audio repair.

Choose and justify the same model-input representation for every row in both splits:

| Mode | Model-input files |
| --- | --- |
| Waveform | Original audio and enabled training derivatives under `<split>/audio/` |
| Mel/log-Mel | Existing numeric `.npy` matrices copied unchanged to `<split>/features/` |

Use Mel/log-Mel spectrograms only when they already exist in the source dataset.
Copy them unchanged; do not calculate spectrograms from audio or reconstruct audio
from them.Load numeric arrays safely without pickle.

CSV manifests are UTF-8 with identical column order and deterministic rows.
`file_name` is the unique output key, relative to `output_dir`, referencing the
chosen model-input file. Do not mix audio and feature paths or create extra rows
for supporting assets. Export the retained DataFrame columns, including source
attributes, metadata and annotations. Keep source IDs under distinct names and use
hashes internally for required checks. Never embed audio bytes or feature arrays
in CSV, or add generated audit columns merely for completeness.

Use only the representation needed by the selected task. The following tasks are defaults. 
Use if suitable. If the underlying task is a little bit different, keep the underlying taks.

| Task | Target/annotation representation |
| --- | --- |
| Single-label classification | Human-readable `label` |
| Regression | Finite numeric values under the selected target name |
| Multi-label classification | Sorted unique label list with consistent source scalar types |
| Whole-recording recognition, translation or synthesis | Authoritative `text`; translation uses target-language text |
| Segmented recognition/translation | `transcripts`: dictionaries with `start_time`, `end_time`, `text` |
| Captioning | `text`, or `captions` lists when multiple captions exist |
| Event localization | `segments`: dictionaries with `start_time`, `end_time`, `label` |
| Diarization | `segments`: dictionaries with `start_time`, `end_time`, `speaker` |
| Anomaly/novelty detection | Labels and conditioning required by the source protocol |


# Data augmentation

Only the inserted subprompt enables augmentation. Without it, export original
audio files or existing Mel matrices. With it, produce additional training
variants after fixing original splits, retaining baselines and leaving evaluation
observations unaugmented. The chosen export mode applies to all variants.
Copy original audio and existing Mel files byte-for-byte; only enabled training
augmentation may produce changed derivatives.

{{DATA_AUGMENTATION}}

# Validate and report

For waveform export, run `test_audio_validation` from
`agentic_data_engineer.validation`. Pass `target_col=split_target_col` only for
ordinary closed-set classification, with `multi_label` set appropriately. For
regression, structured targets, documented empty-label cases and open-set/anomaly
protocols, pass `target_col=None`
and implement task-specific checks; the generic target check assumes class coverage.
Pass only complete list columns to `list_columns`; validate nullable/nested metadata
separately without replacing missing values by empty lists.

For Mel export, run equivalent schema/path/coverage checks outside `parser.py`;
the audio gatekeeper cannot read `.npy` manifests. Load with `allow_pickle=False`
and check finite non-empty numeric arrays and consistency with documented shapes,
axes and feature settings. Do not invent missing extraction settings or expected
frame counts. Verify original feature copies against their source files.

For both modes, require non-empty splits, unique keys, safe paths inside the matching
split directory and exact model-input directory coverage. Verify targets and
annotation semantics, source-copy hashes, original/derived counts, group boundaries
and exact byte/decoded-content duplication within and across splits. Do not remove
distinct valid observations to conceal duplicate or grouping conflicts; report and
resolve them using source evidence, or fail clearly.

Account for retained source fields in manifests or documented source references. Validate unchanged
values against sources and corrections against cleaning rules/audits. Check that
optional-field errors did not remove valid samples. Independently rerun parsing,
splitting and transformations to verify membership, CSV bytes and asset hashes;
a second full copy of original audio is unnecessary. Validate derivatives using
the augmentation audit. Fix failures rather than weakening checks.

Report target ranking/roles, sources/joins, cleaning and exclusions, subset rationale,
split counts/distributions/groups, representation/settings, candidate completeness,
lineage, validation, limitations and artifact paths. Explain any subset in the final
response and state when all source samples were used. Claim success only after
applicable checks pass; host Croissant generation/validation still follows.
