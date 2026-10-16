### Objective and scope

Follow the main output scope. Keep `parser.py` limited to parsing and splitting;
run augmentation, audit, previews and checks separately using unsaved execution
code. Use local files and installed libraries only; do not create helper modules.

Loading this subprompt requires actual additional training observations in
`train.csv`. Implement only chosen methods using installed libraries; no model
training is required.

Support image-level single-label or multi-label classification. All retained
alternative targets and annotations must remain valid too. Reject methods that would require unsupported
box, mask, keypoint or caption transformations; never discard annotations or switch
tasks to enable augmentation. If no method is justified, stop with a clear
explanation instead of inventing variants or claiming augmentation succeeded.

### Choose methods

Derive methods, strengths, parameter ranges and variant counts from task evidence
and original training data. Do not use universal numeric defaults or tune on test.
Explain target preservation and record exact settings. Use a selected method per
variant rather than arbitrary chains.

| Method | Operation and requirements |
| --- | --- |
| Horizontal flip | Mirror left/right only when direction, handedness and text readability are irrelevant. |
| Vertical flip | Mirror top/bottom only when the resulting orientation is meaningful and target-preserving. |
| Brightness | Scale brightness only when absolute intensity is not decisive. |
| Contrast | Adjust contrast without removing subtle class evidence. |
| Small rotation | Rotate only when orientation is irrelevant; document interpolation and fill. On the fixed-size canvas, ensure no task-relevant content is cropped or obscured; otherwise skip rotation. |
| Gaussian noise | Add zero-mean noise relative to intensity scale only when noise and fine structure do not define the target. |
| Gaussian blur | Choose blur scale relative to relevant feature sizes; avoid tasks defined by fine edges or small defects. |

Do not apply every method automatically. When labels depend on fine structure,
weak contrast or defects, exclude noise and blur unless evidence supports them.
Cropping, arbitrary resizing, channel swaps, perspective warps, cutout, Mixup/CutMix and generative
synthesis are outside this policy.

Keep dimensions, mode, bit depth and color/alpha semantics. Use compatible operations
and a lossless output format; prefer PNG when it represents the data faithfully,
otherwise use a supported lossless format or report the limitation. Never silently
convert scientific, palette or high-bit-depth images to ordinary RGB. Avoid clipping
or lost task evidence. Photometric methods preserve alpha and non-image channels;
geometric methods keep all spatial channels aligned. Display-only thumbnails may be resized/converted.

### Generate and audit

Freeze original splits, retain baseline images and add distinct non-identity training
variants using a documented count policy for comparable sources. Explain ineligible
cases; do not add duplicate copies or claim success with no usable variants.
Keep the main manifest schema. Preserve source metadata as provenance and distinguish
it from recomputed derivative properties; never claim changed acquisition/intensity
properties are unchanged. Preserve targets and annotation associations. Update
existing fields that describe a changed derivative; do not add newly measured metadata columns.

Generate from original training observations only, never from earlier derivatives.
Use a documented bounded retry policy for invalid or identity results. Check the
saved/reloaded result: reject unchanged content and duplicates of existing training
observations or accepted variants. Evaluation data is for final leakage checks only,
never for method selection, parameter estimates or choosing replacements.

One derivative row represents one complete observation. For multiple associated
images, use consistent geometry where their relationship requires it. Preserve
aligned path lists and all annotations; skip a method if synchronized changes
cannot be implemented safely. Record each changed model-input image in the audit.
Report original/derived counts per class and relevant group; explain any change
caused by eligibility or count rules. Do not silently rebalance classes.

Use deterministic collision-checked filenames and a recorded seed.
Derive per-image/per-variant seeds with `hashlib` from stable identity and variant
index, independent of traversal order. Write `augmentation.csv` with one row per
changed model-input image:

```text
file_name,original_file_name,technique,seed,parameters,brightness_factor,contrast_factor,rotation_degrees,noise_std,blur_radius,original_sha256,augmented_sha256
```

Paths are relative to `output_dir` and resolve to model-input references in training
rows, including list-valued references. Leave unused parameter cells empty. Store
`parameters` as a Python list of `[name, value]` pairs, using nested lists
when needed. Serialize with canonical `repr()` and read with `ast.literal_eval`;
never use dictionaries or JSON in table cells.
Record interpolation, fill, channel handling and changed attributes. Keep named
parameter columns for the existing preview helper; use `parameters` for additional
settings rather than duplicating those values. Keep audit fields out of manifests.
Original image copies retain their source bytes.

Before image validation, generate `augmentation_preview.png` showing a baseline
and each used technique once, preferably with a common suitable reference. Caption
actual settings and show the saved derivatives; display-only resizing/conversion
is allowed, but do not generate separate example transformations. Do not add training
rows solely for preview coverage. Use `create_augmentation_preview` from
`agentic_data_engineer.image_preview` only when it satisfies that coverage; otherwise
render the preview separately. The helper shows a reference's audited variants
but does not guarantee distinct-method coverage. Exclude the preview from manifests.

### Validate and report

Beyond main checks, verify count-policy/audit agreement, readable derivatives with
correct dimensions/mode, target/annotation validity, accurate metadata and lineage.
Require changed decoded pixels, not just re-encoding, and check for duplicate variants
and cross-split pixel leakage. Independently reproduce pixels and file hashes from
recorded source/settings/seed. Check every used method appears in the preview;
PNG readability alone is insufficient.

Report chosen/rejected methods and evidence, settings/count policy, seed, library
versions, original/derived/evaluation counts, validation and audit/preview paths.
Print training counts. Claim success only after real variants exist and checks pass.
