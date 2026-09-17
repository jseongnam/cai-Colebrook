# Manuscript update map

Do not insert numbers into the manuscript until the full run has completed and the generated CSV files have been checked.

## Baseline-input ablation

Use `paper_evaluation_summary.csv`, filtered to branches=2, regime=iid, and method=neural_plus_newton. Compare `full`, `no_state`, and `physics_only` with mean ± standard deviation over the three training seeds.

Report direct residual, Newton iterations, convergence ratio, q-relative RMSE, and x RMSE. The clean conclusion is whether encoder access to baseline-derived information adds value while the correction target and decoder remain fixed.

## Coupled runtime

Use `paper_runtime_summary.csv`. Report device, CPU model or GPU model, thread settings, sample count, repeats, median ms/sample, IQR, and speedup. State explicitly that model loading and disk I/O are excluded and that both pipelines include baseline construction.

## OOD

Use the same IID-trained checkpoint for every regime. A compact table should have one row per regime and show baseline+Newton and neural+Newton convergence ratio, iterations, and residual. Report degradation relative to IID rather than only absolute OOD performance.

## Multi-branch

Compare branches=2, 3, and 5 under the `full` input mode. This establishes scaling across parallel-branch systems. It does not establish performance on looped or arbitrary hydraulic networks.

## Required manuscript correction before using new results

The repository version inspected for this extension constructs some legacy Taylor centers from the target state. Re-run the affected main experiments with leakage-safe features before retaining the associated headline performance claims. The `audit-legacy` command records the evidence for each NPZ file.

## Target and parameterization factorial ablation

Use `run_additional_experiments.sh` and filter the summary by `target_mode` and
`flow_parameterization`. The primary comparisons are:

- correction/logit versus direct/logit: baseline-relative target effect;
- correction/logit versus correction/raw_ratio: logit-space effect;
- all four cells: disclose the interaction rather than selecting only a favorable pair.

All cells must use the same leakage-safe split, architecture, training budget, and seeds.
For runtime, report batch 1 as online end-to-end latency and batch 500 as throughput;
do not merge them into one speedup value.
