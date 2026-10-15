### Objective and scope

Follow the main output allowlist. Use unsaved execution code; do not create an
augmentation module, helper scripts or additional audit/configuration files.

Implement offline image augmentation outside `parser.py` using installed Pillow/NumPy
or suitable installed libraries. This subprompt requires actual additional training
images in `train.csv`. Follow the main parser/export rules; no model training.

Support a single-label classification split target. All retained alternative targets
and annotations must remain valid too. Reject methods that would require unsupported
box, mask, keypoint or caption transformations; never discard annotations or switch
tasks to enable augmentation. Fail clearly if no justified method is available.

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
| Small rotation | Rotate only when orientation is irrelevant; document interpolation/padding and preserve relevant content. |
| Gaussian noise | Add zero-mean noise relative to intensity scale only when noise and fine structure do not define the target. |
| Gaussian blur | Choose blur scale relative to relevant feature sizes; avoid tasks defined by fine edges or small defects. |

Do not apply every method automatically. For scratch/crack/small-defect tasks,
exclude noise and blur unless specific evidence supports them. Cropping, arbitrary
resizing, channel swaps, perspective warps, cutout, Mixup/CutMix and generative
synthesis are outside this policy.

Keep dimensions, mode, bit depth and color/alpha semantics. Use compatible operations
and a lossless output format; prefer PNG when it represents the data faithfully,
otherwise use a supported lossless format or report the limitation. Never silently
convert scientific, palette or high-bit-depth images to ordinary RGB. Avoid clipping
or lost task evidence. Display-only thumbnails may be resized/converted.

### Generate and audit

Freeze original splits, retain baseline images and add distinct non-identity training
variants using a documented count policy for comparable sources. Explain ineligible
cases; do not add duplicate copies or claim success with no usable variants.
Keep the main manifest schema. Preserve source metadata as provenance and distinguish
it from recomputed derivative properties; never claim changed acquisition/intensity
properties are unchanged. Preserve target values and valid annotation associations.

Use deterministic collision-checked filenames and a configurable recorded seed.
Derive per-image/per-variant seeds with `hashlib` from stable identity and variant
index, independent of traversal order. Write `augmentation.csv` with:

```text
file_name,original_file_name,technique,seed,parameters,brightness_factor,contrast_factor,rotation_degrees,noise_std,blur_radius,original_sha256,augmented_sha256
```

Paths are relative to `output_dir` and resolve to training rows. Leave unused
parameter cells empty; use canonical JSON `parameters` for full settings including
interpolation, padding, mode handling and affected metadata. Keep audit columns
out of model inputs. Original image copies retain their source bytes.

Before image validation, generate `augmentation_preview.png` showing a baseline
and each used technique once, preferably with a common suitable reference. Caption
actual settings and identify any display-only reproductions. Do not add training
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
