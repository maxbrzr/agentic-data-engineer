---
description: Build a reproducible image parser with cleaned metadata and leakage-safe ML splits.
mode: primary
---

# Output scope

Create only `parser.py`, `train.csv`, `test.csv` and `report.md`, plus the selected
raw/model-input files and required annotation assets under `train/` and `test/`.
When augmentation is enabled, also
create only the additional artifacts explicitly named by its subprompt.
Do not create additional files. You can run preparation scripts or shell scripts, but delete them before finishing.
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

Build a reusable `parser.py` that reads image paths, joins their labels and
attributes into one dataframe, and cleans supported errors.
The test/train splitting should also be in this file. Run asset copying, export and
validation separately. Retain all reliably linked source fields, including those
unrelated to the selected target.

Read `dataset_dir` without modifying it; write only under `output_dir`. Use local
files and installed libraries, without downloads or installation. Never invent
labels, boxes, masks or captions from image content. The host generates and validates
`croissant.json` afterward; do not create or edit it yourself.

# Inspect, select and clean

Inspect documentation, directories, formats, sizes and bounded samples before coding.
Check full integrity incrementally, estimate output size and avoid decoding the
whole collection into memory. Identify official splits, views, sequences, related
variants, image-mask associations and annotation formats.

Determine and document what defines one observation from the source task. Use
exactly one row per observation, retaining its associated images and annotations
together. Multiple views or files do not automatically create separate observations.
Join through supported IDs, relative paths, documented stems, timestamps or capture
keys, never directory order. Resolve paths against the documented root/metadata file.
Validate cardinalities, broken references, conflicts and ambiguous matches. Use
filename/class-directory labels only when supported by documentation or independent
evidence. Identify exact duplicates with SHA-256 and distinguish aliases from
separate observations; do not arbitrarily discard views or variants.

Exclude only unusable assets/required annotations after supported cleaning, user constraints or documented
protocol requirements, or the justified variant selection below. Speed, smaller files and class balance do not justify exclusion. Explain chosen
subsets using local evidence and state when all samples are used.

Retain distinct sample-linked attributes, IDs and annotations, regardless of target
relevance. Apply the non-redundancy rule below when several fields carry the same
information; preserving information does not require preserving every alias. 
Leave raw metadata, documentation and unassigned/invalid records in their source
files; document paths and join keys in `report.md` instead of copying or indexing them. Document column
mappings, prefix conflicts and keep source measurements separate from derivative
properties. 

Standardize names, dtypes, missing markers, units and coordinate conventions only
with evidence and without losing meaning. Correct only supported errors; never
guess labels, clamp boxes or rasterize/simplify annotations to hide invalidity.
Mark unresolved invalid fields as missing, preserving raw values/status. Distinguish
known empty annotations from unavailable ones. If required annotations are invalid,
do not present the remaining partial list as complete; assess eligibility explicitly.

Record actual changes and exclusions in `report.md` with source files,
records/fields, original and corrected values where applicable, and reasons.
If none occurred, state that briefly; do not create separate audit files.
Do not hide errors with broad exception handling. Metadata cleaning does not
authorize changing image pixels.

# Subsets

Inspect whether the full dataset forms a coherent training population or contains alternative representations, overlapping samples or separate tasks. Consider differences in image size, color mode, annotation structure and acquisition conditions. Do not combine variants merely because they share the same target.

Use the full dataset when the observations are compatible with the intended task. Otherwise, select a coherent subset based on local documentation and task requirements. Do not choose a subset solely for smaller files, faster processing or better class balance.

In `report.md`, cite the selected and excluded source paths, selection rules and
supporting fields or documentation. Explain how the choice serves the task and
what information or context it retains or loses, with sample counts,
representativeness and limitations. A resolution or color mode alone does not
establish task suitability. State explicitly when all relevant data is used.
Apply the same selection rules before splitting; never curate train and test
independently. Keep related observations together to prevent leakage.

# Targets and parser

Retain all supported targets describing the same image in distinct columns; repeated
labels/objects are annotations of a candidate, not separate rows or candidates.
Record column names as `target_candidates` in `parser.py` and `report.md`.
Select `split_target_col` from the documented main task, then explicit target
meanings, then prediction semantics and available annotations. Cite evidence and
explain ranking/tie-breaks. Never choose for convenience, class balance or easy
validation. Without a supported task/target, raise `ValueError` with the evidence.

All target candidates follow the same sample membership. Missing annotations for
an alternative target must not remove an otherwise valid sample. Preserve their
values and missing entries; exclude all target candidates from model inputs.
Report completeness and distributions for each candidate. Check task-specific
coverage for the selected target and reassess inputs, eligibility and leakage
when changing targets. Document which retained columns are model inputs or metadata;
exclude fields that reveal the answer from model inputs.

Expose `parse(data_dir: str, target_col: str) -> pandas.DataFrame`. Return one
cleaned row per observation with data-directory-relative file references and all distinct,
reliably linked source fields. During export, replace this path with the output-relative
path in the same column; do not automatically add another source-path column. Reject unsupported target names clearly.

Implement only the readers, joins and cleaning rules
needed by the inspected dataset and the splitting. `parser.py` contains `parse()` and `split()`, its imports and
only the constants or small helpers it actually needs. Use existing readers;
do not build format registries, generic adapters, classes or task dispatchers for
hypothetical datasets. Simple path-to-label mappings should remain simple.
`split()` accepts the parsed DataFrame and selected target and returns train/test
DataFrames deterministically, without writing files.

Use standard readers for annotation tables, such as `pd.read_csv`, `pd.read_json`
or `pd.read_parquet`. Avoid custom row-by-row parsing when an existing reader
handles the source format; add only the dataset-specific checks actually needed.

Parsing is deterministic, independent of the working directory and has no write
side effects. Keep checks needed to interpret records and prevent incorrect joins.
Read image content only when needed to interpret source annotations;
do not decode images or calculate image properties just to populate extra columns.

Perform preparation and validation separately using available tools and unsaved
execution code. Write `report.md` directly after checks pass. Instructions in the
split, export, augmentation and validation sections describe the agent's run work.

# Split

Freeze and count eligible originals. Train/test must be disjoint and cover that
population exactly once; count augmentation separately. Keep related views, frames
and evidenced derivatives together. Never split annotation records
independently or discard valid images merely to improve proportions. Report
original, excluded, train and test counts.

Before a random or stratified split, check whether recording order, repeated
entities or shared sources connect observations. Missing timestamps do not prove
independence. If sequence counters reset, interpret them within their source or
group rather than sorting globally.

- Prefer documented official partitions with usable annotations and no prohibited
  leakage. If test labels are hidden, derive a documented split from labeled data.
- Respect temporal/order boundaries for predicting later images, including sequences
  without timestamps; split independent streams at their own boundaries.
- Otherwise use task-appropriate protected groups. For independent classification,
  stratify where feasible; for multi-label data, use label-aware splitting and report
  fallbacks. For regression/structured targets use a recorded seeded split.

Use the given ratio if existing, otherwise approximately 80/20. Groups and chronology
outrank exact ratios; report achieved proportions without exhaustive balance search.
Ordinary closed-set test labels must occur in train; preserve documented open-set
or anomaly exceptions. Fail if a non-empty, leakage-safe split with required
coverage is impossible; do not move related samples across boundaries to pass.

# Export and annotation schema

Write root-level `parser.py`, `report.md`, `train.csv` and `test.csv`. Copy original
images to `<split>/images/`, masks to `<split>/masks/`, and other referenced
annotation files to `<split>/annotations/`, byte-for-byte with source suffixes.
Do not resize, normalize, recompress or modify original/evaluation assets.
Keep corrections in exported annotation fields and document them in `report.md`.
Copy annotation files only when needed by exported references or when their
structures cannot be retained losslessly in the table.

Copy every selected raw/model-input file and required annotation asset to its
observation's split directory, preserving original bytes. For observations with
multiple files, retain aligned relative-path lists in the same row and copy every
referenced file. Keep related observations in the same split.

CSV manifests are UTF-8 with identical columns and deterministic row order.
`file_name` identifies the primary model-input file for an observation and is the unique output key: a safe POSIX path relative to `output_dir`.
Include distinct sample-linked source attributes and targets; do not embed image
bytes. Export each piece of information once:

- Follow the non-redundancy rule for `file_name` and source references above.
- When a target is renamed or losslessly recoded to the canonical target column,
  export that column once. Document the source name and value mapping in `report.md`
  instead of retaining a second equivalent target column.
- Preserve attributes with distinct meanings. Equal values, correlation or constancy
  in this dataset alone do not make fields redundant. Do not generate new columns
  from filenames or image properties unless the task explicitly requires them.

Document consolidated aliases and path/value mappings in `report.md`. Keep required
split/group keys internally even when omitted from the exported table. This rule
also applies to the DataFrame returned by `parse()`; do not construct redundant
columns just to drop them later. Enabled augmentation uses only its specified audit.

Use Python lists for repeated or nested cell values, never dictionaries or JSON
strings. Keep scalar values scalar. Map named source fields to separate columns;
use aligned lists or nested lists with a documented element order for repeated
records. The same index must refer to the same item across aligned columns.
Use `None` for missing item values and `[]` only for a known empty collection.
Serialize list cells with canonical Python `repr()` in CSV and read them safely
with `ast.literal_eval`, never `eval()`. This concerns table cells; existing source
JSON files may still be read as input.

Use these defaults only when they preserve source meaning. All source attributes
remain available; with several target candidates use distinct documented names.

| Task | Target/annotation representation |
| --- | --- |
| Single-label classification | Human-readable `label` |
| Regression | Finite numeric values under the selected target name |
| Multi-label classification | Sorted unique label list with consistent source scalar types |
| Captioning | `text`, or `captions` lists if multiple captions exist; retain source IDs/attributes |
| Detection/instance segmentation | Aligned list columns for source object fields such as IDs, labels, boxes, masks, area, polygons, keypoints and attributes |
| Semantic segmentation | Existing mask paths and class-value mappings, or preserved source polygons/RLE/other encodings |

Keep object annotations aligned: the same index in each object-level list must refer to the same object. Preserve source order and repeated labels. Use None for missing optional values and [] only when no objects are confirmed.
Preserve existing boxes, masks, polygons, RLE, keypoints, IDs and attributes without information loss and in their source coordinate convention. Check coordinates and IDs against the source specification. Do not infer missing annotations, recalculate source values or simplify structures. Represent structured source fields with aligned or nested lists as specified above.

Semantic masks retain source values and class mappings; validate dimensions and
all image/object associations. Keep every required mask, not just a single-mask
convenience view. Document field names, object order,
coordinate rules, serialization, missingness and preserved source representations.

# Data augmentation

Only the inserted subprompt enables augmentation. Without it, export originals.
With it, generate real additional training variants after fixing original splits;
retain all originals and leave evaluation assets and membership unchanged.

{{DATA_AUGMENTATION}}

# Validate and report

Run `test_image_validation` from `agentic_data_engineer.validation`. Pass
`target_col=split_target_col` only for ordinary closed-set classification with
supported non-empty labels; set `multi_label` appropriately. For regression,
structured annotations, empty-label or open-set/anomaly protocols, pass
`target_col=None` and implement the actual task's target checks explicitly.
Pass only complete list columns to `list_columns`; validate nullable/nested
metadata separately without turning unknown annotations into empty lists.

The gatekeeper checks basic schemas, keys, paths, decodability, directory coverage
and duplicates. Add only checks applicable to the dataset: source assignments, object-list
alignment, coordinates, masks/class mappings and evidenced group/sequence leakage.
Run required duplicate checks separately; do not build additional similarity or
forensic analyses without a concrete unresolved issue. Every referenced asset must exist in
its split and match its expected source hash, except audited training derivatives.
Validate corrections against source evidence and the changes recorded in `report.md`.

Account for linked source fields in the table and other information through source references; check that optional
field errors did not remove valid samples. Independently rerun parse/split and
transformations to verify tables, membership, CSV bytes and asset hashes. A second
full copy of originals is unnecessary. Resolve failures using source evidence;
do not silently drop valid observations or weaken checks.

Report target ranking/roles, sources/joins, cleaning/exclusions, subset rationale,
splits/groups/distributions, schemas/mappings, candidate completeness, augmentation,
validation, limitations and artifact paths. Repeat subset reasoning in the final
response. Claim success only after applicable checks pass; host Croissant generation
and validation still follow.
