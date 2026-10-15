### Objective and scope

Follow the main output allowlist. Use unsaved execution code; do not create an
augmentation module, helper scripts or additional audit/configuration files.

Implement offline augmentation outside `parser.py`: this subprompt requires actual
additional training variants in `train.csv`. Follow the main prompt's parser,
representation, source-preservation and validation rules. No model training is needed.

Support a single-label classification split target. Every available alternative
target and annotation must also remain valid. If a method would change one, reject
that method; never drop targets or annotations to enable augmentation. Fail clearly
when the task or available evidence cannot support actual augmentation.

### Choose methods

Select methods, strengths and variant counts from task evidence and original
training data, without universal numeric defaults or test-data tuning. Explain
label preservation, parameter ranges and sampling rules. Apply only justified
methods. Use a selected method per derivative; SpecAugment is a named composite.

**Time domain: waveform samples**

| Method | Operation and requirements |
| --- | --- |
| Noise injection | Add zero-mean Gaussian noise relative to signal RMS; record SNR. Avoid tasks defined by noise or weak events; skip this method on silent inputs. |
| Time shifting | Shift with silence padding, never wrap. Preserve event evidence and required timing annotations; do not crop relevant content. |
| Pitch shifting | Change pitch while preserving duration; only when pitch does not define any target. |
| Changing speed | Pitch-preserving time stretching; only when tempo/duration and retained annotations remain valid. Do not crop to force the original length. |
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

Apply waveform methods only to audio inputs and feature methods only to existing
Mel/log-Mel matrices. Do not extract new spectrograms from audio. Keep channel
operations synchronized. Record mask positions/fill values and warp mappings;
zero in a log-spectrogram need not mean silence. Never reconstruct augmented Mel
matrices into audio.

For either Mixup, use distinct original training partners with matching known
targets and compatible annotations. Equal labels alone do not prove that a mixture
preserves the task. Missing targets are not evidence of compatibility. Record
weights and alignment/padding; never mix splits. Different-class weighted labels
are outside this unchanged single-label schema. Mixing log-Mel matrices is not
equivalent to waveform mixing.

### Generate and audit

Retain baseline observations and add distinct non-identity variants under the main
prompt's training directory for the selected representation. Use supported lossless
audio or numeric `.npy`. Check finite values and clipping; allow duration/shape
changes only as defined by the method. Record a count policy, apply it consistently
to comparable sources and explain ineligible cases. Do not claim augmentation
success if no usable variants were produced.

Retain source metadata as provenance, clearly distinguished from derivative
properties. Recompute affected properties such as duration or measured level;
do not claim original room/device conditions describe a synthetic mixture. For
Mixup, preserve both sources' metadata in the audit. Targets and annotations must
remain valid under the scope above. Keep train/test column schemas identical.

Use a configurable recorded global seed and stable `hashlib`-derived seeds per
source/variant. Choose partners deterministically, independent of traversal order.
Write `augmentation.csv`, with a row per derivative:

```text
file_name,original_file_name,partner_file_name,technique,seed,parameters,original_sha256,partner_sha256,augmented_sha256
```

Paths are relative to `output_dir` and reference training model-input rows; in Mel
mode the original is its baseline matrix. Leave unused partner fields empty.
Store exact settings and needed source metadata in canonical JSON `parameters`,
including source-file hashes, known feature settings and RIR provenance where applicable. Record local RIR source paths or reproducible generation instructions in
`augmentation.csv`; do not create additional RIR files.

Create `augmentation_preview.png` with a baseline and an example of each used
method, preferably on a common suitable reference. Use consistent waveform/Mel
axes and color scales, label actual settings/partners and identify display-only
reproductions. Do not create training rows solely for preview coverage.

### Validate and report

Beyond the main checks, verify count-policy/audit agreement, source/partner lineage,
target and annotation validity, accurate metadata, operation-specific behavior and
changed exported samples/features, not just different encodings. Independently
reproduce values and hashes from recorded sources/settings/seeds. Report used and
rejected methods, parameter/count decisions, seed, library versions, original and
augmented counts, validation and audit/preview paths. Print training counts and
claim success only after actual variants exist and all applicable checks pass.
