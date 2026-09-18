# Augmentation subprompts

Write an augmentation subprompt in the file with the same name as its main
prompt, for example:

```text
prompts/audio-data-engineer.md
prompts/augmentation/audio-data-engineer.md
```

The supplied subprompt files are intentionally empty. Define the techniques,
parameters, task-specific annotation handling and validations yourself.
Do not repeat the `# Data augmentation` heading or template marker in a
subprompt: the loader inserts its entire contents into that section.

Augmentation is disabled by default. Enable it with `--data-augmentation`;
explicitly disable it with `--no-data-augmentation`. A missing or empty
subprompt causes an error when enabled. Disabled runs do not read subprompts.
The flag takes no value: do not pass the strings `true` or `false`.

For a custom main prompt, place its subprompt in an adjacent `augmentation/`
directory using the exact same filename. Include `{{DATA_AUGMENTATION}}`
under one `# Data augmentation` heading in the main prompt to choose the
insertion point. If the marker is absent, the loader appends the section.

Only training data may be augmented, after original split membership is
determined. Never modify source files or evaluation data. Keep derivatives
with their originals, audit lineage, preserve task semantics and document
the technique, parameters, seed and counts. The host inserts these shared
rules alongside your subprompt; implementing and validating the actual
augmentation remains the processing agent's responsibility.
