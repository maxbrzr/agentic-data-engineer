### Objective: demonstrate agent-implemented offline image augmentation

Implement and run offline augmentation in `parser.py` with Pillow and, where
needed, NumPy. Export real derivatives into `train.csv` for inspection without
model training. Missing libraries must cause a clear dependency error.

Support single-label classification only. For other tasks, raise `ValueError`
explaining the need for task-specific annotation transformations and invariance
rules; never discard annotations or switch tasks to enable augmentation.

### Select label-preserving techniques

Justify each technique's label invariance using the task and local documentation.
Preserve the authoritative source label on every derivative.

Candidate techniques and conservative parameter bounds:

| Technique | Bounds | When it may be used |
| --- | --- | --- |
| Horizontal flip | Left-right mirror | Only when handedness, text readability or direction is irrelevant to the target |
| Vertical flip | Top-bottom mirror | Only when upside-down appearance is meaningful and label-preserving |
| Brightness | Factor 0.95 to 1.05 | Only when absolute intensity is not the target or decisive evidence |
| Contrast | Factor 0.95 to 1.05 | Only when subtle contrast changes cannot erase class evidence |
| Small rotation | Angle -5 to +5 degrees | Only when orientation is not a class distinction and padding does not invalidate the task |
| Gaussian noise | Mean 0, standard deviation at most 2 in 8-bit intensity units | Only for 8-bit images when fine structure and noise level are not decisive |
| Gaussian blur | Radius at most 0.4 pixels | Only when the task does not depend on fine edges or small defects |

Do not apply all candidates automatically. In particular, for scratch, crack
or small-defect classification, exclude blur and added noise by default and
assess brightness/contrast conservatively. Prefer flips when orientation is
irrelevant; do not assume that every image classification task permits them.
Exclude cropping, arbitrary resizing, color-channel swaps, perspective warps,
cutout, MixUp, CutMix and generative image synthesis in this experiment.

Use one selected technique per derivative, not a chain of transformations.
Keep width and height unchanged. Preserve image mode, bit depth and color
semantics; do not silently convert scientific, palette, alpha or high-bit-depth
images to 8-bit RGB. Restrict techniques accordingly or fail with a specific
unsupported-mode error. For rotations, explicitly document deterministic
interpolation and padding and validate their suitability.

### Generate training variants

Call a separate augmentation function during export after fixing original splits.
Retain all training originals; leave evaluation membership and counts unchanged.
Produce exactly two distinct, non-identity derivatives per training original,
tripling training size. Prefer two justified techniques, or use one with two
distinct valid parameter values. If impossible, fail with the source identified;
never add duplicates, omit originals or reduce the count.

Save derivatives as lossless PNGs under train/images/ with deterministic,
collision-checked names. Originals keep their normal export paths. Use the
main prompt's classification manifest schema; place augmentation audit fields
only in the separate audit CSV below.

Use global seed 42. Derive per-image/per-variant seeds from a stable source
identity and variant index with hashlib, not Python's randomized hash() or
iteration-order-dependent global draws.

### Make execution inspectable

Write augmentation.csv in output_dir with one row per derivative and these
columns:

file_name,original_file_name,technique,seed,brightness_factor,contrast_factor,
rotation_degrees,noise_std,blur_radius,original_sha256,augmented_sha256

Both file paths refer to exported files relative to output_dir. Store unused
parameters as empty CSV cells. Every derivative and its original must resolve
to train.csv and have identical labels.
The audit CSV is separate from the model input and does not require a change
to the host's Croissant generator.

After writing all derivative files and `augmentation.csv`, generate
`augmentation_preview.png` using exactly one original training image as the
common reference. Show that original once, followed by exactly one example of
each distinct technique actually used anywhere in this run. Do not show unused
candidate techniques, repeated examples of one technique or additional originals.
All displayed transformations must use the same reference image, never different
source images. Caption each example with its technique and actual parameters.

Select a reference deterministically on which every used technique is valid.
Choose one actual parameter configuration per technique deterministically from
the audit. Prefer existing derivatives of the reference image. If a technique
was used only on other images, reproduce it on the reference in memory for
display, with recorded parameters and seed; do not add a training derivative or
audit row solely for the preview. Report any display-only examples separately.
Do not restrict the run's augmentation techniques merely to fit the preview.

Use the installed helper only if its output meets the complete, once-per-technique
contract above:

```python
from agentic_data_engineer.image_preview import create_augmentation_preview

create_augmentation_preview(output_dir, max_originals=1)
```

The helper shows the selected original and all its audited derivatives; it does
not itself guarantee coverage of all techniques or remove repeated techniques.
When necessary, implement preview rendering in the generated `parser.py` with
Pillow to satisfy this contract and write the PNG atomically. The final preview
has exactly `1 + number_of_distinct_used_techniques` image tiles.
Mode conversion and resizing affect only in-memory preview thumbnails, never
training assets. Exclude the preview from `train.csv`; its one-sample limit does
not limit augmentation of every training original.

### Validate real outputs before reporting success

In addition to the main checks, validate:

- Unchanged evaluation membership; readable PNG derivatives with correct dimensions,
  labels and original-to-derivative lineage.
- Different source/derivative hashes and decoded pixels: PNG re-encoding alone is
  not augmentation, and hash inequality does not prove semantic correctness.
- No exact cross-split pixel duplicates; retain the group-leakage audit.
- Independent recomputation with identical source, operation, parameters and seed
  reproduces decoded pixels and exported SHA-256 hashes.
- Tripled training count, two audit rows per original, and consistent manifests.

Generate the preview before `test_image_validation()`: the gatekeeper rejects
`augmentation.csv` without a readable `augmentation_preview.png`.
Additionally assert that every actually used technique is displayed exactly once
and every example uses the same single reference original. The PNG readability
gate alone does not validate technique coverage.

Add a Data augmentation section to report.md describing selected and rejected
techniques, label-invariance reasoning and limitations, parameters, seed,
library versions, original/derived/evaluation counts and validation results.
Print original and augmented training counts during parser execution. Report
the paths to augmentation.csv and augmentation_preview.png. Do not report
success unless derivatives were actually generated and all checks passed.
