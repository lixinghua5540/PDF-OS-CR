# PDF-OS-CR Repository Refactor Design

## Goal

Prepare the research code for a clear, professional public release while preserving the existing `python train.py` and `python test.py` entry points and compatibility with both supplied `best_model.pt` checkpoints.

## Scope

The refactor covers repository organization, duplicated data-loading code, path handling, command-line configuration, unused code, generated cache files, dependency documentation, paper-derived figures, and a new English README.

The refactor does not change the published network architecture, module behavior, state-dict parameter names, loss weights, normalization constants, dataset splits, pretrained checkpoint contents, or reported experimental values. It does not rerun training, evaluation, tests, or benchmarks.

## Compatibility Requirements

- `python train.py ...` remains the training command.
- `python test.py ...` remains the evaluation command.
- Existing command-line option names remain accepted where they are already part of the public workflow.
- `ckpt/SEN12MS-CR/best_model.pt` and `ckpt/LuojiaSET-OSFCR/best_model.pt` remain loadable without checkpoint conversion.
- Generator and discriminator attribute names that contribute to checkpoint keys remain unchanged.
- Existing dataset CSV files remain usable.
- Existing numerical preprocessing constants and tensor channel ordering remain unchanged.

## Chosen Approach

Use a compatibility-focused internal refactor rather than converting the project into an installable `src/` package. The two root entry scripts remain stable while shared logic moves into focused internal modules. Compatibility shims are retained only where an old import or path is required by the existing workflow.

This approach provides most of the maintainability and presentation benefits of a package rewrite without forcing users to learn new commands or invalidating the released weights.

## Target Repository Structure

```text
PDF-OS-CR-GitHub/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- train.py
|-- test.py
|-- trainer.py
|-- options/
|   |-- train_options.py
|   `-- test_options.py
|-- datasets/
|   |-- common.py
|   |-- registry.py
|   |-- sen12ms_cr.py
|   `-- luojiaset_osfcr.py
|-- models/
|   |-- generator/
|   |   |-- generator.py
|   |   |-- gated_conv.py
|   |   |-- fusion.py
|   |   `-- attention.py
|   `-- discriminator/
|       |-- discriminator.py
|       |-- edge_branch.py
|       `-- block_branch.py
|-- losses/
|   `-- losses.py
|-- utils/
|   |-- distributed.py
|   |-- metrics.py
|   |-- model_utils.py
|   `-- rtv.py
|-- tools/
|   |-- generate_priors.py
|   `-- model_complexity.py
|-- data/
|-- ckpt/
|   |-- SEN12MS-CR/best_model.pt
|   `-- LuojiaSET-OSFCR/best_model.pt
|-- assets/
|   |-- architecture.png
|   |-- qualitative_results.png
|   |-- cloud_coverage_results.png
|   `-- large_scale_results.png
`-- docs/superpowers/
    |-- specs/
    `-- plans/
```

The existing `ckpt/` name is retained instead of renaming it to `checkpoints/`, because path compatibility is an explicit requirement. Internal filenames may be normalized only when all root commands and compatibility imports continue to work.

## Code Organization

### Entry Points

`train.py` and `test.py` become thin orchestration scripts. They parse options, select a dataset through the registry, construct models, load checkpoints, and call training or evaluation functions. They do not contain dataset-specific imports, personal GPU selection, repeated path setup, or large processing loops that belong in reusable modules.

### Dataset Layer

The SEN12MS-CR and LuojiaSET-OSFCR loaders currently duplicate CSV parsing, normalization, preview generation, metric-output helpers, and filesystem utilities. Shared behavior moves to `datasets/common.py`. Dataset-specific path assembly and band behavior remain in `datasets/sen12ms_cr.py` and `datasets/luojiaset_osfcr.py`.

`datasets/registry.py` maps the existing dataset names to loader classes and file-list functions. This removes conditional imports from both root entry points without changing accepted dataset names.

### Models

Generator and discriminator code is grouped by responsibility. OSGC, CCFS, and MSFA may share `models/generator/fusion.py` because they form the feature-fusion stage described in the paper. CAM remains an attention component, and gated convolution remains a reusable building block.

Class internals and registered submodule attribute names remain unchanged. File moves and import updates must not alter state-dict keys.

### Training, Losses, Metrics, and Utilities

`trainer_eval.py` becomes `trainer.py`. Loss code moves from singular `loss/` to `losses/`. Utility files retain only functions with live callers. Dead helpers, unused imports, debug prints, commented-out image-generation blocks, and dataset demo entry points are removed.

Model-complexity reporting remains an optional tool rather than running automatically on every training invocation. It must not mutate checkpoint state or become a required training dependency.

## Path and Configuration Policy

- No personal drive paths, server paths, or hard-coded GPU indices remain.
- User-controlled paths are command-line arguments.
- Repository defaults are resolved from the repository root with `pathlib.Path`, not from the caller's current working directory.
- Output directories are created only by the command that writes to them.
- Training checkpoints continue to default to `ckpt/<dataset-name>/`.
- Evaluation metrics and optional images share the configured result directory rather than separate hard-coded `results` paths.
- GPU selection is delegated to standard `CUDA_VISIBLE_DEVICES` environment configuration or PyTorch launcher behavior; the scripts do not overwrite it.

## Cleanup Policy

Delete:

- All committed/generated `__pycache__/` directories and `.pyc` files.
- Unused imports and functions with no callers.
- Dataset-loader demo `main()` blocks that duplicate the real entry points.
- Commented-out incomplete output code that cannot be invoked.
- Debug prints unrelated to normal progress reporting.
- Duplicate implementations after their callers use the shared implementation.

Keep:

- Both pretrained checkpoints.
- Both dataset CSV manifests.
- Prior-generation and model-complexity tools after path cleanup.
- Distributed-training helpers that are called by the training flow.
- Paper-specific architecture components even when they are small, because their names help connect code to the methodology.

Deletion decisions are based on static call-site inspection. Ambiguous research code is preserved when removal could affect reproducibility.

## README Design

The README is written in clear English for an international research audience. Its sections are:

1. Paper title, concise project summary, authors, and repository status.
2. Architecture figure and a short end-to-end explanation.
3. Method highlights: PSF-SNet, prior knowledge, OSGC, CCFS, MSFA, CAM, and the dual-branch discriminator.
4. Quantitative results for SEN12MS-CR and LuojiaSET-OSFCR using values transcribed from the manuscript.
5. Qualitative and large-scale visual results.
6. Environment and installation instructions with version notes matching the manuscript where known.
7. Dataset preparation, expected directory examples, CSV manifest meaning, normalization, and channel conventions.
8. Pretrained checkpoint locations and dataset association.
9. Copyable training and evaluation commands with the important arguments explained.
10. Repository structure.
11. Citation status. Because the paper is not online, the README states that formal citation information is coming soon and does not invent a DOI, venue, or BibTeX entry.
12. Acknowledgements, contact information from the manuscript, and a license-status note.

The README does not claim that commands were rerun during this refactor.

## Paper-Derived Assets

Figures are cropped from the author-provided `Manuscript.pdf` at sufficient resolution for GitHub display. Only figures needed to explain the method and results are included, rather than copying complete manuscript pages.

- `assets/architecture.png`: Fig. 2, the full PDF-OS-CR architecture.
- `assets/qualitative_results.png`: Fig. 12, comparisons on SEN12MS-CR and LuojiaSET-OSFCR.
- `assets/cloud_coverage_results.png`: Fig. 13, metric trends over cloud-coverage levels.
- `assets/large_scale_results.png`: Figs. 14-16, mountainous, urban, and farmland scenes.

Captions identify the manuscript figure number and avoid claiming that the manuscript is already published online.

## Reported Results

The headline values shown in the README are copied from the manuscript:

- SEN12MS-CR: MAE 0.0160, LPIPS 0.1874, PSNR 35.2691 dB, SSIM 0.9322.
- LuojiaSET-OSFCR: MAE 0.0099, LPIPS 0.2142, PSNR 38.7638 dB, SSIM 0.9455.
- Complexity: 18.93 M parameters, 56.38 G FLOPs, and 61.69 ms inference time under the manuscript's stated setup.

These are documented results, not newly measured results.

## Error Handling

- Unknown dataset names raise a concise error listing supported names.
- A missing checkpoint produces a clear message with the resolved path.
- A missing dataset manifest or data root fails before model execution with the resolved path shown.
- Empty train, validation, or test splits produce a clear error instead of later division-by-zero or empty-loader failures.
- Directory creation is limited to explicit output locations.

No new abstraction layer or configuration framework is introduced solely for error handling.

## Verification Constraint

At the user's request, no tests, training runs, evaluation runs, inference runs, or performance benchmarks are executed. Verification is limited to static inspection of imports, call sites, command-line option wiring, path construction, state-dict-sensitive attribute names, README links, asset existence, and the final repository tree.

The final handoff must explicitly state that runtime behavior was not executed or tested.

## Out of Scope

- Changing model accuracy or architecture.
- Retraining or regenerating checkpoints.
- Recomputing manuscript tables.
- Adding packaging or publishing automation.
- Uploading datasets or weights to external services.
- Creating a DOI, publication venue, or final citation before those details exist.
- Selecting a software license on the author's behalf.
