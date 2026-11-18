### Objective and scope

Follow the main output scope. Keep `parser.py` limited to parsing and splitting;
run augmentation, audit, previews and checks separately using in-memory execution
or temporary scripts/helper modules under `output_dir`. Delete these temporary
files before completing. Use local files and installed libraries only.

Loading this subprompt requires actual additional training observations in
`train.csv`. Keep original train/test membership fixed and retain all originals.
Use waveform methods on audio or feature methods on existing Mel/log-Mel matrices;
never compute new spectrograms from audio. No model training is required.

Support recording-level classification and whole-recording transcripts when the
chosen method preserves their meaning. Preserve every known target and annotation;
reject methods that would change timed annotations, speaker identity or other
retained targets. Missing annotations are not evidence that a method is safe.
If no method is justified, stop with a clear explanation rather than inventing
variants, switching tasks or claiming augmentation succeeded.

### Choose methods

Select methods, strengths and variant counts from task evidence and original
training data, without universal numeric defaults or test-data tuning. Explain
label preservation, parameter ranges and sampling rules. Apply only justified
methods. Use a selected method per derivative; SpecAugment is a named composite.

**Time domain: waveform samples**

| Method | Operation and requirements |
| --- | --- |
| Noise injection | Add zero-mean Gaussian noise scaled to a chosen SNR from signal/noise RMS; record the RMS convention and achieved SNR. Avoid tasks defined by noise or weak events; skip silent inputs. |
| Time shifting | Shift with silence padding, never wrap. Specify fixed or extended duration; a fixed-length shift may discard only verified irrelevant silence. Reject when timing annotations would become invalid. |
| Pitch shifting | Change pitch while preserving duration; only when pitch does not define any target. |
| Changing speed | Pitch-preserving time stretching; only when tempo/duration are not target-defining and retained annotations remain valid. Do not crop to force the original length. |
| Amplitude scaling | Multiply by a gain factor without clipping; only when absolute level is irrelevant. |
| Reverberation | Convolve with a local or reproducibly simulated RIR. Justify room conditions and record RIR provenance, normalization, wet/dry balance and tail handling. |
| Waveform Mixup | Weighted sum of compatible, aligned waveforms; follow the shared Mixup rules below. |

**Time-frequency domain: Mel/log-Mel matrices**

| Method | Operation and requirements |
| --- | --- |
| Time masking | Replace consecutive frames across all bands; retain enough task evidence. |
| Frequency masking | Replace consecutive bands across all frames; preserve decisive frequency evidence. |
| Spectrogram Mixup | Weighted sum of matrices with matching feature definitions and documented time alignment. |
| SpecAugment | Combine time/frequency masking and optionally time warping. State the components; the original method includes warping. |

Apply waveform methods only to audio inputs and time-frequency methods only to existing
Mel/log-Mel matrices. Do not extract new spectrograms from audio. Keep channel
operations synchronized. Record mask positions/fill values and warp mappings;
zero in a log-spectrogram need not mean silence. Never reconstruct augmented Mel
matrices into audio. For Mel methods, establish frequency/time/channel axes and
value scaling from source evidence first. Skip methods needing unknown settings;
do not guess axes or treat padding as signal. Bound masks by valid frames/bands
and retain enough evidence for the task. State whether SpecAugment includes warping.

For either Mixup, restrict use to single-label classification with distinct original
training partners sharing that label and compatible known annotations. This is a
same-class mixing variant; ordinary Mixup also interpolates labels. Whole-recording
transcripts cannot be preserved merely by mixing their recordings. Equal labels
alone do not prove that a mixture preserves the task. Missing targets are not evidence of compatibility. Record
non-negative weights summing to one, with both partners contributing, and any
alignment/padding. Require matching sample rates/channel meanings for waveform
mixing and matching feature definitions for Mel mixing; never mix splits.
Different-class weighted labels require a different target schema and are excluded. Mixing log-Mel matrices is not
equivalent to waveform mixing.

### Generate and audit

Retain baseline observations and add distinct non-identity variants under the main
prompt's training directory for the selected representation. Use supported lossless
audio or numeric `.npy`. Check finite values and clipping; allow duration/shape
changes only as defined by the method. Record a count policy, apply it consistently
to comparable sources and explain ineligible cases. Do not claim augmentation
success if no usable variants were produced. Preserve the true sample rate and
channel layout. Changing playback rate is not pitch-preserving time stretching;
do not relabel a sample rate to implement either operation.

Generate from original training observations only, never from earlier derivatives.
Use a documented bounded retry policy for invalid or identity results. Check the
saved/reloaded result: reject unchanged content and duplicates of existing training
observations or accepted variants. Evaluation data is for final leakage checks only,
never for method selection, parameter estimates or choosing replacements.

One derivative row represents one complete observation. If an observation uses
several synchronized files, apply a coherent transformation to all affected files
and their references; skip the method if this cannot be done correctly. Keep file
associations in aligned lists and record each changed asset in the audit.

Retain source metadata as provenance, clearly distinguished from derivative
properties. Update existing attributes describing the derivative, such as duration;
do not add new measurement columns solely for augmentation. For Mixup, preserve
both parents' metadata in `augmentation.csv` parameters as nested lists of named
values. Do not copy one parent's acquisition or identity metadata into the manifest
as though it described the mixture. Retain a shared value only when it remains
meaningful for the derivative; otherwise set the existing optional field to missing
and explain the rule in `report.md`. Update derivable properties such as duration
from the saved derivative. Never clear targets, required annotations or protected
group keys to make mixing possible: reject partners with incompatible values or
when no valid value can be retained under the task's group/split policy. Targets and
annotations must remain valid under the scope above. Keep train/test column schemas
identical and original rows unchanged.

Report counts by class and relevant group before/after augmentation; explain any
change caused by eligibility or count rules. Do not silently rebalance classes.

Use a recorded global seed and stable `hashlib`-derived seeds per
source/variant. Choose partners deterministically, independent of traversal order.
Write `augmentation.csv`, with a row per changed model-input file:

```text
file_name,original_file_name,partner_file_name,technique,seed,parameters,original_sha256,partner_sha256,augmented_sha256
```

Paths are relative to `output_dir` and resolve to model-input references in training
rows, including list-valued references. In Mel mode the original is its baseline matrix. Leave unused partner fields empty.
Store `parameters` as a Python list of `[name, value]` pairs, using nested lists
when needed. Serialize as valid Python list literals and read with `ast.literal_eval`;
never use dictionaries or JSON in table cells.
Record exact operation settings, source-file hashes, known feature settings and
RIR provenance where applicable. For local RIRs, record the source path and hash;
for simulated RIRs, record the algorithm, settings and seed. Record sample-rate
alignment, normalization, wet/dry balance and tail handling; use no evaluation
audio to estimate RIRs or noise. Do not create extra RIR files.

Create `augmentation_preview.png` with a baseline and an example of each used
method, preferably on a common suitable reference. Use consistent waveform/Mel
axes and color scales and label actual settings/partners. Plot saved derivatives;
do not generate separate example transformations or new spectrograms from audio.
Do not create training rows solely for preview coverage. Keep the preview outside
model-input directories and exclude it from manifests. Verify that it is a readable
PNG and actually shows each used method with its baseline and recorded settings.

### Validate and report

Run all four augmentation checks from `agentic_data_engineer.validation` explicitly
for both waveform and Mel exports; `test_audio_validation` does not call them.
Before augmentation, retain independent copies of both manifests and SHA-256
mappings for all original training and test model-input files. Compare them with
the reloaded final exports:

- `test_augmentation_preserves_originals`: pass the before/after manifests, stable
  unique observation `key_columns`, and both complete test-file hash mappings.
- `test_augmentation_audit_coverage`: derive expected changed/new file paths from
  the before/after exports independently of the audit; compare them with audited
  paths and all exported training model-input paths. Flatten list-valued references.
- `test_augmentation_lineage`: map each audited derivative to every original
  training parent, including Mixup partners and additional generating parents in
  parameters, with recorded hashes in matching order. Supply recorded result hashes
  and independently computed original/final training-file hashes, including originals.

- `test_augmentation_targets`: index original and final training tables by stable
  observation keys and map each variant to all generating original observations.
  Pass every target column that must be preserved, including protected alternative
  targets, and identify multi-label columns. All parents and the variant must agree
  on known target values; multi-label targets are compared as label sets.

Beyond the main checks, verify count-policy/audit agreement, source/partner lineage,
target and annotation validity, accurate metadata, operation-specific behavior and
changed exported samples/features, not just different encodings. Independently
reproduce values and hashes from recorded sources/settings/seeds. Report used and
rejected methods, parameter/count decisions, seed, library versions, original and
augmented counts, validation and audit/preview paths. Print training counts and
claim success only after actual variants exist and all applicable checks pass.

### Mandatory final augmentation-prompt check

Immediately before the final response, explicitly re-read the Data augmentation
section of the supplied system instructions and verify that this augmentation
subprompt is present and enabled. This paragraph is part of that subprompt: its
presence means augmentation instructions were supplied. Do not infer that the
subprompt was absent from the lack of generated variants, and do not report
"no augmentation subprompt was present" when these instructions are included.

When this subprompt is present, confirm that additional training observations were
actually exported in `train.csv`, `augmentation.csv` exists with at least one audit
entry, and `augmentation_preview.png` exists and passes the preview checks above.
If any requirement is unmet, continue the augmentation work or report a concrete task-specific failure; do not claim
successful completion of an original-only export. State explicitly in `report.md`
and the final response that the augmentation subprompt was present and enabled,
and report original and added training counts with the final check results.
