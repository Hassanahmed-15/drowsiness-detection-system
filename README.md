# Drowsiness Detection System

## Repository Structure
```
project/
├── interfaces/     # schemas, examples, mock inputs/outputs
├── datasets/       # manifests, labels, acquisition/preparation scripts
├── module_sg1/
├── module_sg2/
├── module_sg3/
├── module_sg4/
├── module_sg5/
├── embedded_sg6/
├── integration/
├── evaluation/
└── documentation/
```
Each folder has a README covering its purpose, dependencies, input/output interface, how to run it, test data and current performance.

## Git Working Practice
- `main` is the stable, integrated branch. Do not push to it directly.
- Each sub-group works on a feature/module branch (e.g. `sg1/face-detection`) and merges through a reviewed pull request.
- Commit often, with messages that describe the technical change.
- Do not commit raw datasets, trained-model caches or environment folders unless approved. Put download/setup instructions in `datasets/` instead.
- Tag working versions at each integration gate (e.g. `gate-1`, `gate-2`) so the last known-good system is always recoverable.
