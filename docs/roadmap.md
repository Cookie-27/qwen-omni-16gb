# Roadmap and contribution ideas

| Status | Work | Completion evidence |
|---|---|---|
| Available | Candidate scoring and historical numerical checks | Archived vectors, operator tests and bounded GPU validation |
| Available | Synthetic input generator and walkthrough | Decodable WAV/PNG, complete processor input, CI checks |
| Open: small documentation task | Explain one result JSON from a contributor's short input | Actual output, command and environment; distinguish mass from conditional probability |
| Open: CPU tooling | Optional frame extraction helper | Explicit 1fps timestamps, deterministic ordering, preserved complete audio and synthetic test input |
| Open: runtime validation | Test another device or supported efficient backend | Same input/model/readout, numerical deltas and measured memory boundary |
| Research follow-up | Evaluate a newer Transformers version | Registry compatibility and all numerical gates pass; retain the existing version's evidence |

No release dates or device support are promised. Keep current evidence immutable, report failures as well as successes, and discuss new runtime/backend support before changing the compatibility gate. See [CONTRIBUTING.md](../CONTRIBUTING.md).
