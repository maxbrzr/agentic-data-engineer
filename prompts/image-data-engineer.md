---
description: Build a reproducible image parser with cleaned metadata and leakage-safe ML splits.
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

# Objective and boundaries

Build and run `parser.py` to associate images with their labels and attributes 
and export consistent train/test datasets. Preserve all source
information, including fields unrelated to the selected main target.

Read `dataset_dir` without modifying it; write only under `output_dir`. Use local
files and installed libraries, without downloads or installation. Never invent
labels, boxes, masks or captions from image content. The host generates and validates
`croissant.json` afterward; do not create or edit it yourself.

# Inspect, select and clean

Inspect documentation, directories, formats, sizes and bounded samples before coding.
Check full integrity incrementally, estimate output size and avoid decoding the
whole collection into memory. Identify official splits, views, sequences, related
variants, image-mask associations and annotation formats.

Use one sample row per primary image, retaining all of its annotations together.
Join through supported IDs, relative paths, documented stems, timestamps or capture
keys, never directory order. Resolve paths against the documented root/metadata file.
Validate cardinalities, broken references, conflicts and ambiguous matches. Use
filename/class-directory labels only when supported by documentation or independent
evidence. Identify exact duplicates with SHA-256 and distinguish aliases from
separate observations; do not arbitrarily discard views or variants.

Exclude only unusable assets/required annotations after supported cleaning, user constraints or documented
protocol requirements. Speed, smaller files and class balance do not justify exclusion. Explain chosen
subsets using local evidence and state when all samples are used.

Retain sample-linked fields, IDs and annotations, regardless of target relevance.
Leave raw metadata, documentation and unassigned/invalid records in their source
files; document paths and join keys in `report.md` instead of copying or indexing them. Document column
mappings, prefix conflicts and keep source measurements separate from derivative
properties. Do not automatically interpret other modalities as model inputs.

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

# Targets and parser

Retain all supported targets describing the same image in distinct columns; repeated
labels/objects are annotations of a candidate, not separate rows or candidates.
Record column names as `target_candidates` in `parser.py` and `report.md`.
Select `split_target_col` from the documented main task, then explicit target
meanings, then prediction semantics and available annotations. Cite evidence and
explain ranking/tie-breaks. Never choose for convenience, class balance or easy
validation. Without a supported task/target, raise `ValueError` with the evidence.

All candidates share the same split. Missing alternatives do not remove otherwise
eligible images. Report completeness/distributions; only the selected task gets
coverage guarantees. A different target requires reviewing inputs, eligibility
and leakage. Retain metadata and target columns but exclude targets and leakage
fields from model inputs; document roles rather than deleting information.

Expose `parse(data_dir: str, target_col: str) -> pandas.DataFrame`, returning a
cleaned table keyed by unique internal `source_file_name` with all linked source
fields. Determine task type from documentation and reject unsupported target names.

`parser.py` contains only `parse()` and the imports, constants and small helpers
needed to read, associate and clean this dataset. Keep it specific to the actual
task; do not build a generic framework or implement unrelated task modes. Parsing
is deterministic, independent of the working directory and has no write side effects.
Basic checks needed to parse correctly belong here; split/export orchestration,
augmentation, validation suites, report generation and a pipeline CLI do not.
Perform those steps separately using the available tools and unsaved execution code.
Write `report.md` directly after checks pass; do not generate report-writing code.

Use stable source-based, collision-checked filenames and portable paths independent
of working directory. Never encode targets in output names. Replace only owned
outputs, clear stale owned model assets and write CSVs atomically where practical.
A rerun must not depend on leftovers.

# Split

Freeze and count eligible originals. Train/test must be disjoint and cover that
population exactly once; count augmentation separately. Keep related views, frames,
exact duplicates and evidenced derivatives together. Never split annotation records
independently or discard valid images merely to improve proportions.

- Prefer documented official partitions with usable annotations and no prohibited
  leakage. If test labels are hidden, derive a documented split from labeled data.
- Respect temporal/order boundaries for predicting later images, including sequences
  without timestamps; split independent streams at their own boundaries.
- Otherwise use task-appropriate protected groups. For independent classification,
  stratify where feasible; for multi-label data, use label-aware splitting and report
  fallbacks. For regression/structured targets use a recorded seeded split.

Use the user/protocol ratio, otherwise approximately 80/20. Groups and chronology
outrank exact ratios; report achieved proportions without exhaustive balance search.
Ordinary closed-set test labels must occur in train; preserve documented open-set
or anomaly exceptions. Fail if a non-empty, leakage-safe split with required
coverage is impossible; do not move related samples across boundaries to pass.

# Export and annotation schema

Write root-level `parser.py`, `report.md`, `train.csv` and `test.csv`. Copy original
images to `<split>/images/`, masks to `<split>/masks/`, and other referenced
annotation files to `<split>/annotations/`, byte-for-byte with source suffixes.
Do not resize, normalize, recompress or modify original/evaluation assets.
Keep corrections in exported annotation fields and their audit while preserving
raw annotation files; clearly identify which representation is authoritative.

CSV manifests are UTF-8 with identical columns and deterministic row order.
`file_name` is the unique output key: a safe POSIX path relative to `output_dir`.
Include all sample-linked source fields and targets. Keep source IDs under distinct
names; do not generate redundant aliases or embed image bytes. Preserve source paths
as portable references. Generated augmentation audit fields belong in a sidecar.

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

For object lists, every index describes the same object across all columns. Preserve
repeated labels/order, use `None` for missing optional object values, and `[]` only
for confirmed absence of objects. Never sort/deduplicate columns independently.
Retain boxes and masks when both exist; do not derive absent boxes or replace source
`area` with computed area. Existing object IDs must be unique within their image.

If lossless, express boxes in source-image pixels with half-open bounds:
`0 <= xmin < xmax <= width` and `0 <= ymin < ymax <= height`. Otherwise retain
and validate the source convention. Preserve polygons, RLE, keypoints, masks and
nested attributes without simplification. Use source structures or unchanged
annotation files when aligned columns cannot represent them losslessly.

Semantic masks retain source values and class mappings; validate dimensions and
all image/object associations. Keep every required mask, not just a single-mask
convenience view. References must resolve inside the image's split. Shared source
annotation files may contain other splits: validate that each row selects only its
own annotations; retained raw content is provenance, not extra training samples.

Use canonical Python `repr()` for nested CSV cells and safe parsing, never `eval()`.
Do not create separate split annotation tables. Document field names, object order,
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
and duplicates. Add checks for cleaned source assignments, all annotation semantics,
object-list lengths/order, coordinate bounds, masks/class mappings, group/sequence
leakage, and exact decoded-pixel duplicates. Every referenced asset must exist in
its split and match its expected source hash, except audited training derivatives.
Validate corrections against source evidence and the cleaning audit.

Account for all source fields in the export or metadata archive; check that optional
field errors did not remove valid samples. Independently rerun parse/split and
transformations to verify tables, membership, CSV bytes and asset hashes. A second
full copy of originals is unnecessary. Resolve failures using source evidence;
do not silently drop valid observations or weaken checks.

Report target ranking/roles, sources/joins, cleaning/exclusions, subset rationale,
splits/groups/distributions, schemas/mappings, candidate completeness, augmentation,
validation, limitations and artifact paths. Repeat subset reasoning in the final
response. Claim success only after applicable checks pass; host Croissant generation
and validation still follow.
