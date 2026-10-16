---
description: Build a reproducible audio parser with cleaned metadata and leakage-safe ML exports.
mode: primary
---

# Output scope

Create only `parser.py`, `train.csv`, `test.csv` and `report.md`, plus the selected
raw/model-input files and required annotation assets under `train/` and `test/`.
When augmentation is enabled, also
create only the additional artifacts explicitly named by its subprompt.
You may add additional code but have to delete it before completing.
Put decisions, mappings, source references, actual cleaning
changes/exclusions and concise validation results in `report.md`.
Preserving source information does not require duplicating it: retain existing
fields as required below and reference untouched source files for other information.
Generate no extra metadata or analyses solely for completeness. Keep code and report
proportional to the actual task. Host-generated logs, provenance and Croissant files
are outside this agent's output scope; do not create, modify or delete them.

# Avoid redundant file references

When an observation has a `file_name`, export that reference once. Do not add
`source`, `source_file_name`, `source_path`, basename, stem or parent-name columns
that repeat it or can be recovered from it. Apply this to both the parsed DataFrame
and exported CSVs. Keep such values internal when needed for joins, grouping or
validation, and document path mappings in `report.md`. Retain a source identifier
only when it carries a distinct relationship that cannot be recovered from the
retained fields and is needed to use the data. Preserve other distinct attributes;
this rule removes redundant information, not columns merely named `source`.

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

Determine and document what defines one observation from the source task. Return
exactly one row per observation, whether represented by audio, an existing Mel
matrix or several associated files. Keep its repeated annotations in that row;
do not equate file count with observation count or silently change the task. Join only through documented
keys, paths, timestamps or other supported relationships. Resolve source paths
against their metadata file or documented root, never traversal order. Validate
join cardinalities; report ambiguous matches and prevent accidental row multiplication.

Inventory relevant variants and official partitions before choosing a subset.
Decide whether they form one compatible population or alternative representations
of the same observations. Use the full dataset when compatible with the task;
otherwise select a coherent subset from local documentation and task requirements.
In `report.md`, cite selected and excluded paths, selection rules and supporting
fields or documentation. Explain the effect on sample counts, information retained,
representativeness and limitations. State explicitly when all relevant data is used.
Apply the same eligibility rules before splitting; never curate train and test
independently. Exclude samples only for unusable assets/required targets after
supported cleaning, user constraints, documented protocol requirements or the
justified variant selection above. Optional
field errors alone do not invalidate a sample. Speed, size and class balance are
not exclusion reasons. Treat related versions as related, not independent data.

**Keep source attributes in the table.** Retain existing attributes, metadata,
IDs, targets and annotations that can be reliably linked to a sample as DataFrame
columns, subject to the non-redundancy rule above. Use lists
for repeated/nested values without multiplying rows. Preserve missing values;
report uncertain associations instead of guessing. Reference dataset-level
information and documentation in `report.md`. Do not create separate metadata
files or calculate extra measurements merely for completeness. 

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
Report missing and incorrect values.

All target candidates follow the same sample membership. Missing annotations for
an alternative target must not remove an otherwise valid sample. Preserve their
values and missing entries; exclude all target candidates from model inputs.
Report completeness and distributions for each candidate. Check task-specific
coverage for the selected target and reassess inputs, eligibility and leakage
when changing targets. Document which retained columns are model inputs or metadata;
exclude fields that reveal the answer from model inputs.

# Parser and split

Expose `parse(data_dir: str, target_col: str) -> pandas.DataFrame`. Return one row
per eligible observation, with file references relative to `data_dir`, the selected target
and reliably linked source attributes and labels. Reject
unsupported target names clearly.

`parser.py` contains only `parse()`, `split()` and the imports, constants and helpers
needed to read, associate, join and clean this dataset. Keep it specific to the actual
task; do not build a generic framework or implement unrelated task modes. Parsing
is deterministic, independent of the working directory and has no write side effects.
Basic checks needed to parse correctly belong here. `split()` accepts the parsed
DataFrame and selected target and returns train/test DataFrames deterministically,
without writing files.
Augmentation, validation suites, report generation and a pipeline CLI do not.

Use standard readers for annotation tables, such as `pd.read_csv`, `pd.read_json`
or `pd.read_parquet`. Avoid custom row-by-row parsing when an existing reader
handles the source format; add only the dataset-specific checks actually needed.

Perform those steps separately using the available tools and unsaved execution code.
Write `report.md` directly after checks pass; do not generate report-writing code.

Use stable source-based filenames, portable paths and collision checks. Do not
encode targets in output names or depend on the working directory. Replace only
owned outputs, remove stale owned model assets, and replace CSVs atomically
where practical. Rerunning must not depend on previous output files.

Freeze and count the eligible original population before splitting. Each original
must appear exactly once across train/test: their intersection is empty and their
union equals that population. Report original, excluded, train and test counts;
count augmentation separately. Keep duplicates and related recordings or variants
in the same split; do not discard valid samples to achieve a ratio.

Before a random or stratified split, check whether recording order, repeated
entities or shared sources connect observations. Missing timestamps do not prove
independence. If sequence counters reset, interpret them within their source or
group rather than sorting globally.

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

Copy every selected raw/model-input file and required annotation asset to its
observation's split directory, preserving original bytes. For observations with
multiple files, retain aligned relative-path lists in the same row and copy every
referenced file. Keep related observations in the same split.

CSV manifests are UTF-8 with identical column order and deterministic rows.
`file_name` identifies the primary model-input file for an observation and is the unique output key, relative to `output_dir`, referencing the
chosen model-input file. Do not mix audio and feature paths or create extra rows
for supporting assets. Export the retained DataFrame columns, including source
attributes, metadata and annotations. Retain only non-redundant source IDs and use
hashes internally for required checks. Never embed audio bytes or feature arrays
in CSV, or add generated audit columns merely for completeness.

Use Python lists for repeated or nested cell values, never dictionaries or JSON
strings. Keep scalar values scalar. Map named source fields to separate columns;
use aligned lists or nested lists with a documented element order for repeated
records. The same index must refer to the same item across aligned columns.
Use `None` for missing item values and `[]` only for a known empty collection.
Serialize list cells with canonical Python `repr()` in CSV and read them safely
with `ast.literal_eval`, never `eval()`. This concerns table cells; existing source
JSON files may still be read as input.

Use only the representation needed by the selected task. The following tasks are defaults. 
Use if suitable. If the underlying task is a little bit different, keep the underlying taks.

| Task | Target/annotation representation |
| --- | --- |
| Single-label classification | Human-readable `label` |
| Regression | Finite numeric values under the selected target name |
| Multi-label classification | Sorted unique label list with consistent source scalar types |
| Whole-recording recognition, translation or synthesis | Authoritative `text`; translation uses target-language text |
| Segmented recognition/translation | Aligned `start_time`, `end_time`, `text` lists |
| Captioning | `text`, or `captions` lists when multiple captions exist |
| Event localization | Aligned `start_time`, `end_time`, `label` lists |
| Diarization | Aligned `start_time`, `end_time`, `speaker` lists |
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
