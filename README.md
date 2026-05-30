# Hybrid Deep Learning and Newton Refinement for Nonlinear Pipe-Flow Equations

Research repository for the manuscript:

**HYBRID DEEP LEARNING AND NEWTON REFINEMENT: A BASELINE-AWARE CORRECTION FRAMEWORK FOR NONLINEAR PIPE-FLOW EQUATIONS**  
SeokJun Jeong and Yoosoo Oh  
Department of Computer and Information Engineering, Daegu University  
Status: SCI manuscript under review

---

## Overview

This repository contains the research code and reproducibility materials for a hybrid neural-numerical framework for nonlinear pipe-flow equations governed by Colebrook-White residuals and hydraulic coupling constraints.

The main idea is not to replace classical Newton solvers with neural networks. Instead, a neural correction model improves a physically motivated baseline initializer, and Newton refinement remains the final high-precision solver.

In this framework, the neural model acts as a **warm-start correction module** for nonlinear equation solving.

---

## Research Motivation

Newton-type methods can achieve high-precision solutions for nonlinear equations, but their convergence speed, iteration count, and robustness depend strongly on the quality of the initial point.

In practical engineering problems, heuristic or explicit initializers often already exist. Therefore, directly replacing a reliable numerical solver may be less effective than learning a small correction to improve the initializer before refinement.

This project follows the principle:

> Neural models should improve numerical initialization or correction, while classical Newton refinement preserves final numerical reliability.

---

## Core Idea

Given a baseline initializer, the proposed framework learns a correction rather than directly predicting the final solution.

```text
corrected initializer = baseline initializer + neural correction
```

The corrected initializer is then passed to Newton refinement.

This design has three practical advantages:

- It reduces the scale of the learning target.
- It keeps the classical numerical solver as the final accuracy-preserving stage.
- It improves Newton warm-start quality without treating the neural network as a standalone solver.

## Method Overview

The proposed framework consists of five stages.

1. Problem Instance Generation

The framework supports two settings:

- one-dimensional scalar Colebrook-White validation problem
- coupled two-branch parallel pipe-flow system

The scalar setting is used as a controlled diagnostic problem, while the two-branch coupled system is the main evaluation target.

2. Taylor-Based Input Construction

Local Taylor coefficient features are computed around baseline-related expansion centers.

For a scalar residual function, the local approximation can be written as:

```text
f(c + t; a, b) ≈ sum_k gamma_k(a, b, c) t^k
```

The input representation includes:

- Taylor coefficients
- expansion center
- baseline state
- global pipe-flow parameters

3. Baseline Initialization

The method starts from nonlearned baseline initializers.

The scalar Colebrook setting includes:

- fixed baseline
- scale-based heuristic baseline
- Haaland explicit initializer
- Swamee-Jain explicit initializer
- Serghides explicit initializer

The coupled pipe-flow setting includes:

- conductance-based flow split
- diameter-proportional flow split
- equal flow split
- branch-wise explicit Colebrook initializers

4. Neural Correction Prediction

A neural model predicts a normalized correction relative to the baseline initializer.

Evaluated correction backbones:

- MLP
- LSTM
- GRU
- Transformer

All backbones use the same correction target and are compared under a shared evaluation protocol.

5. Newton Refinement

The corrected initializer is used as the starting point for Newton-type refinement.

Newton refinement remains the final high-precision solver.

## Coupled Pipe-Flow System

The main experimental setting is a two-branch parallel pipe-flow system.

The unknown vector is:

```text
z = [Q1, x1, x2]^T
```

where:

- Q1 is the branch-1 flow rate
- Q2 = QT - Q1
- x1 and x2 are branch-wise Colebrook state variables
- QT is the total flow rate

The coupled system combines:

- branch-wise Colebrook-White residual equations
- flow-distribution constraint
- hydraulic head-loss balance constraint

This formulation should be understood as a coupled pipe-flow system composed of branch-wise Colebrook-White equations and hydraulic network constraints, not as a standalone multidimensional Colebrook equation.

## Baseline-Aware Correction Target

Instead of directly predicting the full solution vector, the neural model predicts a correction to the baseline initializer.

```text
z_hat0 = z_base + delta_z
```

For the coupled two-branch system, the correction is parameterized as:

```text
delta_z_norm = [delta_logit_r, delta_x1, delta_x2]^T
```

where:

- the flow ratio is corrected in logit space
- branch-wise Colebrook variables are corrected additively

This design reduces target-scale imbalance and produces Newton-friendly warm starts.

## Neural Correction Backbones

Four neural correction modules are evaluated.

| Model       | Role                                                                            |
| ----------- | ------------------------------------------------------------------------------- |
| MLP         | Fully connected baseline that treats Taylor features as a flattened vector      |
| LSTM        | Sequence-aware model for ordered Taylor coefficient features                    |
| GRU         | Recurrent sequence model with fewer parameters than LSTM                        |
| Transformer | Self-attention model for global interactions among Taylor coefficient positions |

The coupled pipe-flow results suggest that sequence-aware models are particularly useful when Taylor coefficient sequences and global pipe-flow parameters are combined.

## Main Results

In two-branch coupled pipe-flow experiments, learned correction reduces Newton refinement cost while preserving high-precision final accuracy.

Representative result:

```text
Heuristic baseline mean Newton iterations: 5.9251
MLP + Newton mean iterations: 2.7281
LSTM + Newton mean iterations: 2.6119
GRU + Newton mean iterations: 2.6395
Transformer + Newton mean iterations: 2.6150

Final error order: approximately 1e-9
```

The neural model should therefore be interpreted as a warm-start accelerator, not as a solver replacement.

## Coupled Warm-Start Gain

Compared with the heuristic baseline, learned correction models reduce Newton iteration count by approximately 54 to 56 percent.

```text
Heuristic + Newton: 5.9251 iterations
MLP + Newton: 2.7281 iterations
LSTM + Newton: 2.6119 iterations
GRU + Newton: 2.6395 iterations
Transformer + Newton: 2.6150 iterations
```

This supports the central interpretation of the paper:

The learned correction improves the initial point, and Newton refinement recovers high-precision final solutions with fewer iterations.

## Ablation Studies

The manuscript evaluates the following ablation settings.

| Ablation                     | Purpose                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| Direct solution regression   | Tests whether predicting the full solution is less stable than correction learning |
| No Taylor features           | Measures the contribution of local residual-shape information                      |
| Raw-ratio correction         | Compares against logit-space flow-ratio correction                                 |
| Backbone comparison          | Compares MLP, LSTM, GRU, and Transformer under the same correction target          |
| Explicit baseline comparison | Tests whether neural correction improves strong engineering initializers           |

Main ablation findings:

- Direct solution regression is less stable than baseline-aware correction.
- Raw-ratio flow correction weakens strict post-Newton convergence.
- Taylor-feature benefits are distribution-dependent.
- Direct prediction error and Newton iteration efficiency are related but not identical.
- A neural warm start should be evaluated by both direct-stage quality and post-refinement cost.

## Scalar Colebrook Validation

The scalar Colebrook-White equation is used as a controlled validation setting.

The transformed scalar residual is:

```text
f(x; a, b) = x + 2 log10(a + bx) = 0
```

where:

```text
a = (epsilon / D) / 3.7
b = 2.51 / Re
x = 1 / sqrt(lambda)
```

The scalar experiment is used to evaluate:

- Taylor-degree sensitivity
- direct correction accuracy
- Newton iteration reduction
- backbone behavior under a monotone one-dimensional equation

## Dataset and Reproducibility Protocol

The manuscript reports a controlled synthetic-data protocol.

Main coupled-system setting:

```text
Problem: two-branch parallel pipe-flow system
Unknowns: z = (Q1, x1, x2)
Taylor degrees: 10, 15, 20, 25, 30, 35
Train / validation / test: 20,000 / 4,000 / 4,000
Random seeds: 42 / 43 / 44
Backbones: MLP, LSTM, GRU, Transformer
Optimizer: AdamW
Newton tolerance: 1e-12
Maximum Newton iterations: 20
Training hardware: NVIDIA H100 NVL GPU
```

Validation data are used for hyperparameter and checkpoint selection. Test data are reserved for final reporting only.

## Repository Structure

The cleaned repository is organized as follows.

```text
cai-Colebrook/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── scalar/
│   └── coupled/
│
├── src/
│   ├── equations/
│   ├── baselines/
│   ├── features/
│   ├── models/
│   ├── solvers/
│   ├── training/
│   └── utils/
│
├── scripts/
│   ├── generate_scalar_dataset.py
│   ├── generate_coupled_dataset.py
│   ├── run_scalar_degree_sweep.sh
│   ├── run_coupled_models.sh
│   ├── run_ablation_no_taylor.sh
│   ├── run_ablation_nologit.sh
│   └── run_cpu_runtime_benchmark.sh
│
├── results/
│   ├── tables/
│   └── figures/
│
├── data/
│   └── README.md
│
├── checkpoints/
│   └── README.md
│
└── docs/
    ├── method_overview.md
    ├── reproducibility.md
    ├── ablation_studies.md
    └── troubleshooting.md
```

## Data and Checkpoints

Large generated datasets and trained model checkpoints are not included in this repository.

Expected local dataset structure:

```text
data/
├── scalar/
└── coupled/
```

Expected checkpoint structure:

```text
checkpoints/
├── mlp/
├── lstm/
├── gru/
└── transformer/
```

See:

```text
data/README.md
checkpoints/README.md
```

for details.

## Results to Reproduce

The cleaned reproducibility package is intended to reproduce the following types of results.

### Coupled Pipe-Flow Results
- direct initialization performance of nonlearned initializers
- Newton refinement performance of nonlearned initializers
- direct initialization performance of learned correction models
- post-refinement performance of learned correction models
- Newton iteration savings relative to heuristic baseline
- direct residual reduction before Newton refinement
### Scalar Colebrook Results
- Taylor-degree sensitivity
- backbone comparison
- direct RMSE comparison
- post-refinement Newton iteration comparison
- direct-regression ablation
### Runtime Results
- CPU-only scalar validation benchmark
- neural forward time
- neural correction plus Newton refinement time
- nonlearned initializer plus Newton refinement time

## Planned Commands

The full public reproducibility package will include commands similar to the following.

```text
# 1. Generate scalar Colebrook dataset
python scripts/generate_scalar_dataset.py \
  --degree 25 \
  --n-train 80000 \
  --n-val 10000 \
  --n-test 10000 \
  --seed 42 \
  --out-dir data/scalar/deg25

# 2. Generate coupled pipe-flow dataset
python scripts/generate_coupled_dataset.py \
  --degree 25 \
  --n-train 20000 \
  --n-val 4000 \
  --n-test 4000 \
  --seed 42 \
  --out-dir data/coupled/deg25

# 3. Train coupled LSTM correction model
python src/training/train_coupled.py \
  --config configs/coupled/coupled_lstm_deg25.yaml \
  --train data/coupled/deg25/train.npz \
  --val data/coupled/deg25/val.npz \
  --out-dir results/coupled_lstm_deg25

# 4. Evaluate coupled correction model
python src/training/evaluate.py \
  --config configs/coupled/eval_coupled.yaml \
  --test data/coupled/deg25/test.npz \
  --checkpoint results/coupled_lstm_deg25/best.pt \
  --out-dir results/eval_coupled_lstm_deg25
```

## Current Status

This repository is being cleaned and structured as a reproducibility package for the manuscript.

Current public status:

- README and documentation are being prepared.
- Large generated datasets are excluded.
- Model checkpoints are excluded.
- Curated result tables and figures will be added after repository cleanup.
- Full public release will be coordinated with the manuscript review and journal policy.

## Related Repository

This project is part of a broader research direction on hybrid neural-numerical AI systems.

Related work:

- Taylor-Root-Prediction

The Taylor Root Prediction project reformulates nonlinear root finding as a structured neural prediction problem using Transformer-based interval localization, local Taylor representations, coefficient-based neural regression, and residual validation.

## Citation

This manuscript is currently under review. Citation information will be updated after publication.

For now, please refer to this repository as:

```text
Jeong, S. and Oh, Y. Hybrid Deep Learning and Newton Refinement: A Baseline-Aware Correction Framework for Nonlinear Pipe-Flow Equations. Manuscript under review.
```

## Contact

- SeokJun Jeong
- Email: wjdtjrwns1109@gmail.com
- GitHub: jseongnam