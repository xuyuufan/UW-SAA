# AMCTS Author Dataset Migration Plan

## Objective
Prepare an isolated, reproducible package for the 1,680-case Mendeley v3 author dataset in this repository. Preserve the frozen AMCTS method, public-frame environment, dataset bytes, and all existing complete and partial results. The package runs original-RPM AMCTS only; it does not use H20, UWSAA, or train an MLP.

## Scope
- Add only `experiments/amcts_author_dataset/`; leave existing repository baselines, environment, results, and manifests unchanged.
- Include the three frozen method sources byte-for-byte, their minimal runtime dependency closure, the original archive, verified 1,680 inputs, metadata, task list, current results and interruption receipts, Linux resume coordinator, exact tested requirements, and cloud instructions.
- Resume only complete cases whose input and method hashes match exactly. Preserve partial attempts and restart those cases from the initial state.
- Do not create cloud resources, push, open a PR, or launch a new experiment in this preparation stage.

## Acceptance
- Git working tree is based on `origin/main` in the dedicated branch; the existing checkout and branch remain untouched.
- Frozen Python sources, dataset archive, input JSONs, metadata, task list, and existing terminal logs match their recorded SHA-256 values after copy/checkout.
- Dataset audit still reports 1,680 scored cases (56 configurations × 30) and validates the non-scored example and probability file separately.
- Portable runner/coordinator passes static/prepare checks, one six-container full-budget case result replays with legality, conservation, RR, and no-future-input checks, and a matching complete case is skipped on resume without a search.
- Package dependencies and launch/resume commands are documented for Linux; the cloud job remains gated on an explicit root release file and user-provided environment ID.

## Stop conditions
Stop and report if any frozen hash, dataset count/schema, complete-case terminal, source closure, or hidden-information boundary fails. After the six-container replay and complete-skip check pass, stop; do not run the remaining 1,643 cases or alter algorithm/search budgets. Deliver the local branch diff, file inventory, and hashes for review. Push and PR creation require the separate review/release step.
