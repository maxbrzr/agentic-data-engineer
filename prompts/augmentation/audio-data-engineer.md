### Objective and scope

Follow the main output scope. Keep `parser.py` limited to parsing and splitting;
run augmentation, audit, previews and checks separately using unsaved execution
code. Use local files and installed libraries only; do not create helper modules.

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
do not add new measurement columns solely for augmentation. Do not claim original room/device conditions describe a synthetic mixture. For
Mixup, preserve both sources' metadata in the audit. Targets and annotations must
remain valid under the scope above. Keep train/test column schemas identical.

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
when needed. Serialize with canonical `repr()` and read with `ast.literal_eval`;
never use dictionaries or JSON in table cells.
Record exact operation settings, source-file hashes, known feature settings and
RIR provenance where applicable. For local RIRs, record the source path and hash;
for simulated RIRs, record the algorithm, settings and seed. Record sample-rate
alignment, normalization, wet/dry balance and tail handling; use no evaluation
audio to estimate RIRs or noise. Do not create extra RIR files.

Create `augmentation_preview.png` with a baseline and an example of each used
method, preferably on a common suitable reference. Use consistent waveform/Mel
axes and color scales and label actual settings/partners. Plot saved derivatives;
do not generate separate example transformations. Do not create training rows solely for preview coverage.

### Validate and report

Beyond the main checks, verify count-policy/audit agreement, source/partner lineage,
target and annotation validity, accurate metadata, operation-specific behavior and
changed exported samples/features, not just different encodings. Independently
reproduce values and hashes from recorded sources/settings/seeds. Report used and
rejected methods, parameter/count decisions, seed, library versions, original and
augmented counts, validation and audit/preview paths. Print training counts and
claim success only after actual variants exist and all applicable checks pass.
