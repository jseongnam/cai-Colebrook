#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Neural Initial Guess + Classical Baselines + Safeguarded Newton Benchmark Framework
==========================================================================================

목적
----
이 스크립트는 "신경망 기반 초기값 생성"을 고전 초기값들과 공정 비교하고,
Safeguarded Newton refinement, 실패 사례 수집, 분포 분석, 시간 측정,
OOD 평가까지 한 번에 수행하기 위한 통합 프레임워크이다.

핵심 기능
---------
1. NPZ 데이터셋 로드
2. MLP 기반 direct / residual predictor 학습
3. baseline predictor 비교
   - zero
   - train mean
   - other_root (있을 때)
   - linearized guess (-c0/c1)
   - quadratic formula (degree=2일 때 가능)
4. Safeguarded Newton refinement
   - derivative small fallback
   - step clipping
   - damping line search
   - domain clamp
5. 샘플 단위 CSV 로그 저장
6. 분포 요약(mean/median/std/p90/p95/p99/max)
7. hard case 자동 수집
8. inference / refine / total time 측정
9. optional OOD split 평가

가정
----
- coeffs는 다항식 p(x)=c0 + c1 x + c2 x^2 + ... + cn x^n 의 power basis 계수라고 가정한다.
- target root는 root1 이다.
- other root 정보가 있으면 root2_label 같은 키로 제공할 수 있다.
- 입력 feature는 기본적으로 [coeffs, other_root(optional)] 형태로 사용한다.

예시 실행
---------
학습:
python root_benchmark_framework.py train \
  --train_npz /path/to/train.npz \
  --val_npz /path/to/val.npz \
  --test_npz /path/to/test.npz \
  --save_dir /path/to/exp1 \
  --coeffs_key coeffs \
  --target_key root1 \
  --other_root_key root2_label \
  --mode residual \
  --epochs 3000 \
  --batch_size 512 \
  --hidden_dims 256 256 128 \
  --lr 1e-3 \
  --patience 200

평가만:
python root_benchmark_framework.py eval \
  --test_npz /path/to/test.npz \
  --save_dir /path/to/exp1 \
  --coeffs_key coeffs \
  --target_key root1 \
  --other_root_key root2_label

OOD 평가:
python root_benchmark_framework.py eval \
  --test_npz /path/to/test.npz \
  --ood_npz /path/to/ood_test.npz \
  --save_dir /path/to/exp1

주의
----
- quadratic_formula baseline은 coeffs 길이가 정확히 3일 때만 의미 있다.
- linearized baseline은 c1이 매우 작으면 fallback 처리한다.
- exact polynomial root solver(np.roots)를 baseline으로 넣고 싶다면 쉽게 확장 가능하지만,
  여기서는 초기값 전략 비교에 집중한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================
# Utility
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentile_dict(x: np.ndarray) -> Dict[str, float]:
    if len(x) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# =========================
# Polynomial helpers
# =========================

def poly_eval_scalar(coeffs: np.ndarray, x: float) -> float:
    # coeffs: [c0, c1, ..., cn]
    # Horner in ascending basis
    # convert to descending for Horner convenience
    y = 0.0
    for c in coeffs[::-1]:
        y = y * x + float(c)
    return y


def poly_derivative_coeffs(coeffs: np.ndarray) -> np.ndarray:
    if len(coeffs) <= 1:
        return np.array([0.0], dtype=np.float64)
    return np.array([k * coeffs[k] for k in range(1, len(coeffs))], dtype=np.float64)


def poly_eval_and_derivative(coeffs: np.ndarray, x: float) -> Tuple[float, float]:
    f = poly_eval_scalar(coeffs, x)
    dcoeffs = poly_derivative_coeffs(coeffs)
    df = poly_eval_scalar(dcoeffs, x)
    return f, df


def linearized_root_guess(coeffs: np.ndarray, eps: float = 1e-12) -> float:
    # p(x) ~= c0 + c1 x  =>  x = -c0/c1
    c0 = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    c1 = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    if abs(c1) < eps:
        return 0.0
    return -c0 / c1


def quadratic_formula_guesses(coeffs: np.ndarray) -> Optional[Tuple[float, float]]:
    # coeffs = [c0, c1, c2] for c0 + c1 x + c2 x^2 = 0
    if len(coeffs) != 3:
        return None
    a = float(coeffs[2])
    b = float(coeffs[1])
    c = float(coeffs[0])
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            return None
        r = -c / b
        return (r, r)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    r1 = (-b + s) / (2 * a)
    r2 = (-b - s) / (2 * a)
    return (r1, r2)


def choose_closest(candidates: List[float], target: float) -> float:
    if len(candidates) == 0:
        return 0.0
    d = [abs(c - target) for c in candidates]
    return float(candidates[int(np.argmin(d))])


def compute_difficulty_features(coeffs: np.ndarray, true_root: float) -> Dict[str, float]:
    f, df = poly_eval_and_derivative(coeffs, true_root)
    return {
        "abs_true_residual": abs(f),
        "abs_df_at_root": abs(df),
        "linearized_guess": linearized_root_guess(coeffs),
        "degree": len(coeffs) - 1,
    }


# =========================
# Dataset
# =========================

class RootDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        coeffs_key: str = "coeffs",
        target_key: str = "root1",
        other_root_key: Optional[str] = "root2_label",
        input_mean: Optional[np.ndarray] = None,
        input_std: Optional[np.ndarray] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        fit_normalizer: bool = False,
    ):
        self.npz_path = npz_path
        data = np.load(npz_path, allow_pickle=True)
        if coeffs_key not in data:
            raise KeyError(f"'{coeffs_key}' not found in {npz_path}. keys={list(data.keys())}")
        if target_key not in data:
            raise KeyError(f"'{target_key}' not found in {npz_path}. keys={list(data.keys())}")

        coeffs = np.asarray(data[coeffs_key], dtype=np.float32)
        targets = np.asarray(data[target_key], dtype=np.float32).reshape(-1, 1)

        if coeffs.ndim != 2:
            raise ValueError(f"coeffs must be 2D [N, D]. got shape={coeffs.shape}")

        self.has_other_root = other_root_key is not None and other_root_key in data
        other_root = None
        if self.has_other_root:
            other_root = np.asarray(data[other_root_key], dtype=np.float32).reshape(-1, 1)
            feats = np.concatenate([coeffs, other_root], axis=1)
        else:
            feats = coeffs.copy()

        self.coeffs = coeffs
        self.other_root = other_root
        self.targets = targets
        self.raw_feats = feats

        if fit_normalizer:
            self.input_mean = feats.mean(axis=0)
            self.input_std = feats.std(axis=0) + 1e-8
            self.target_mean = float(targets.mean())
            self.target_std = float(targets.std() + 1e-8)
        else:
            assert input_mean is not None and input_std is not None
            assert target_mean is not None and target_std is not None
            self.input_mean = input_mean.astype(np.float32)
            self.input_std = input_std.astype(np.float32)
            self.target_mean = float(target_mean)
            self.target_std = float(target_std)

        self.x = (feats - self.input_mean) / self.input_std
        self.y = (targets - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        item = {
            "x": torch.from_numpy(self.x[idx]).float(),
            "y": torch.from_numpy(self.y[idx]).float(),
            "coeffs": torch.from_numpy(self.coeffs[idx]).float(),
            "target": torch.from_numpy(self.targets[idx]).float(),
            "index": idx,
        }
        if self.other_root is not None:
            item["other_root"] = torch.from_numpy(self.other_root[idx]).float()
        return item


# =========================
# Model
# =========================

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }[activation]

        dims = [input_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Loss
# =========================

def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        return ((pred - target) ** 2).mean()
    return (weights * (pred - target) ** 2).mean()


# =========================
# Training config
# =========================

@dataclass
class TrainConfig:
    train_npz: str
    val_npz: str
    test_npz: Optional[str]
    save_dir: str
    coeffs_key: str = "coeffs"
    target_key: str = "root1"
    other_root_key: Optional[str] = "root2_label"
    seed: int = 42
    epochs: int = 3000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    dropout: float = 0.05
    activation: str = "gelu"
    patience: int = 200
    mode: str = "residual"  # direct or residual
    residual_anchor: str = "linearized"  # linearized / zero / mean / other_root
    hard_weight_alpha: float = 1.0
    residual_loss_lambda: float = 0.0
    grad_clip: float = 5.0
    num_workers: int = 0
    device: str = "cuda"


# =========================
# Predictor classes
# =========================

class BasePredictor:
    name = "base"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        raise NotImplementedError


class ZeroPredictor(BasePredictor):
    name = "zero"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return 0.0


class MeanPredictor(BasePredictor):
    name = "train_mean"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return float(meta["train_target_mean"])


class OtherRootPredictor(BasePredictor):
    name = "other_root"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return 0.0 if other_root is None else float(other_root)


class LinearizedPredictor(BasePredictor):
    name = "linearized"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return linearized_root_guess(coeffs)


class QuadraticFormulaPredictor(BasePredictor):
    name = "quadratic_formula"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        qs = quadratic_formula_guesses(coeffs)
        if qs is None:
            return 0.0
        if other_root is not None:
            # 다른 루트가 주어지면 그 반대쪽 root를 선택하려고 시도
            cands = list(qs)
            idx = int(np.argmax([abs(c - other_root) for c in cands]))
            return float(cands[idx])
        target_proxy = linearized_root_guess(coeffs)
        return choose_closest(list(qs), target_proxy)


class NeuralPredictor(BasePredictor):
    name = "neural"

    def __init__(self, ckpt_dir: Path, device: str = "cpu", mc_dropout_passes: int = 0):
        self.ckpt_dir = ckpt_dir
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.meta = load_json(ckpt_dir / "meta.json")
        self.norm = load_json(ckpt_dir / "normalizer.json")
        model_cfg = self.meta["model"]
        self.model = MLP(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            dropout=model_cfg["dropout"],
            activation=model_cfg["activation"],
        )
        state = torch.load(ckpt_dir / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.mc_dropout_passes = mc_dropout_passes

    def _build_input(self, coeffs: np.ndarray, other_root: Optional[float]) -> np.ndarray:
        feats = coeffs.astype(np.float32)
        if self.meta["has_other_root_feature"]:
            oroot = np.array([0.0 if other_root is None else other_root], dtype=np.float32)
            feats = np.concatenate([feats, oroot], axis=0)
        mean = np.asarray(self.norm["input_mean"], dtype=np.float32)
        std = np.asarray(self.norm["input_std"], dtype=np.float32)
        feats = (feats - mean) / std
        return feats

    def _denorm_target(self, y: np.ndarray) -> np.ndarray:
        tmean = float(self.norm["target_mean"])
        tstd = float(self.norm["target_std"])
        return y * tstd + tmean

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        x = self._build_input(coeffs, other_root)
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(xt).cpu().numpy().reshape(-1)
        pred = float(self._denorm_target(y)[0])
        if self.meta["train_config"]["mode"] == "residual":
            anchor_name = self.meta["train_config"]["residual_anchor"]
            anchor_pred = predictor_by_name(anchor_name).predict_one(coeffs, other_root, self.meta)
            pred = anchor_pred + pred
        return pred

    def predict_with_uncertainty(self, coeffs: np.ndarray, other_root: Optional[float]) -> Tuple[float, float]:
        x = self._build_input(coeffs, other_root)
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        if self.mc_dropout_passes <= 1:
            self.model.eval()
            with torch.no_grad():
                y = self.model(xt).cpu().numpy().reshape(-1)
            pred = float(self._denorm_target(y)[0])
            if self.meta["train_config"]["mode"] == "residual":
                anchor_name = self.meta["train_config"]["residual_anchor"]
                anchor_pred = predictor_by_name(anchor_name).predict_one(coeffs, other_root, self.meta)
                pred = anchor_pred + pred
            return pred, 0.0

        self.model.train()  # dropout on
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_passes):
                y = self.model(xt).cpu().numpy().reshape(-1)
                pred = float(self._denorm_target(y)[0])
                if self.meta["train_config"]["mode"] == "residual":
                    anchor_name = self.meta["train_config"]["residual_anchor"]
                    anchor_pred = predictor_by_name(anchor_name).predict_one(coeffs, other_root, self.meta)
                    pred = anchor_pred + pred
                preds.append(pred)
        self.model.eval()
        return float(np.mean(preds)), float(np.std(preds))


class HybridUncertaintyPredictor(BasePredictor):
    name = "hybrid_uncertainty"

    def __init__(
        self,
        neural_ckpt_dir: Path,
        device: str = "cpu",
        mc_dropout_passes: int = 10,
        uncertainty_threshold: float = 0.05,
        fallback_name: str = "linearized",
    ):
        self.neural = NeuralPredictor(neural_ckpt_dir, device=device, mc_dropout_passes=mc_dropout_passes)
        self.threshold = uncertainty_threshold
        self.fallback_name = fallback_name

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        pred, std = self.neural.predict_with_uncertainty(coeffs, other_root)
        if std > self.threshold:
            return predictor_by_name(self.fallback_name).predict_one(coeffs, other_root, meta)
        return pred


# =========================
# Predictor factory
# =========================

def predictor_by_name(name: str) -> BasePredictor:
    table = {
        "zero": ZeroPredictor,
        "train_mean": MeanPredictor,
        "other_root": OtherRootPredictor,
        "linearized": LinearizedPredictor,
        "quadratic_formula": QuadraticFormulaPredictor,
    }
    if name not in table:
        raise KeyError(f"Unknown predictor: {name}")
    return table[name]()


# =========================
# Safeguarded Newton
# =========================

@dataclass
class NewtonConfig:
    max_iters: int = 20
    tol_residual: float = 1e-10
    tol_step: float = 1e-12
    deriv_eps: float = 1e-12
    fallback_step: float = 1e-2
    max_abs_step: float = 5.0
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    line_search_alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass
class NewtonResult:
    refined: float
    iterations: int
    success: bool
    diverged: bool
    used_small_derivative_fallback: bool
    used_line_search: bool
    hit_clamp: bool
    start_residual: float
    end_residual: float
    trajectory: List[dict]


def safeguarded_newton(coeffs: np.ndarray, x0: float, cfg: NewtonConfig) -> NewtonResult:
    x = float(x0)
    traj: List[dict] = []
    used_small_derivative_fallback = False
    used_line_search = False
    hit_clamp = False
    diverged = False

    f0, _ = poly_eval_and_derivative(coeffs, x)
    prev_abs_f = abs(f0)

    for it in range(1, cfg.max_iters + 1):
        f, df = poly_eval_and_derivative(coeffs, x)
        abs_f = abs(f)
        traj.append({
            "iter": it,
            "x": float(x),
            "f": float(f),
            "df": float(df),
            "abs_f": float(abs_f),
        })

        if abs_f <= cfg.tol_residual:
            return NewtonResult(
                refined=float(x),
                iterations=it - 1,
                success=True,
                diverged=False,
                used_small_derivative_fallback=used_small_derivative_fallback,
                used_line_search=used_line_search,
                hit_clamp=hit_clamp,
                start_residual=float(abs(f0)),
                end_residual=float(abs_f),
                trajectory=traj,
            )

        if abs(df) < cfg.deriv_eps:
            used_small_derivative_fallback = True
            step = math.copysign(cfg.fallback_step, f if f != 0 else 1.0)
        else:
            step = f / df

        if not np.isfinite(step):
            diverged = True
            break

        step = float(np.clip(step, -cfg.max_abs_step, cfg.max_abs_step))

        accepted = False
        best_x = x
        best_abs_f = abs_f
        for alpha in cfg.line_search_alphas:
            cand = x - alpha * step
            if cfg.clamp_min is not None:
                cand = max(cand, cfg.clamp_min)
            if cfg.clamp_max is not None:
                cand = min(cand, cfg.clamp_max)
            if cand != x - alpha * step:
                hit_clamp = True
            f_cand, _ = poly_eval_and_derivative(coeffs, cand)
            abs_f_cand = abs(f_cand)
            if abs_f_cand < best_abs_f:
                best_x = cand
                best_abs_f = abs_f_cand
                accepted = True
                if alpha != 1.0:
                    used_line_search = True
                break

        if not accepted:
            # 더 나빠지면 fallback 한 스텝만 적용
            cand = x - math.copysign(min(abs(step), cfg.fallback_step), step)
            if cfg.clamp_min is not None:
                cand = max(cand, cfg.clamp_min)
            if cfg.clamp_max is not None:
                cand = min(cand, cfg.clamp_max)
            if cand != x - math.copysign(min(abs(step), cfg.fallback_step), step):
                hit_clamp = True
            best_x = cand
            f_cand, _ = poly_eval_and_derivative(coeffs, best_x)
            best_abs_f = abs(f_cand)

        if not np.isfinite(best_x) or best_abs_f > 1e30:
            diverged = True
            break

        if abs(best_x - x) <= cfg.tol_step:
            x = best_x
            break

        x = best_x
        prev_abs_f = best_abs_f

    f_end, _ = poly_eval_and_derivative(coeffs, x)
    success = abs(f_end) <= cfg.tol_residual and not diverged
    return NewtonResult(
        refined=float(x),
        iterations=len(traj),
        success=success,
        diverged=diverged,
        used_small_derivative_fallback=used_small_derivative_fallback,
        used_line_search=used_line_search,
        hit_clamp=hit_clamp,
        start_residual=float(abs(f0)),
        end_residual=float(abs(f_end)),
        trajectory=traj,
    )


# =========================
# Training
# =========================

def build_anchor_targets(
    coeffs_batch: np.ndarray,
    other_root_batch: Optional[np.ndarray],
    y_true_batch: np.ndarray,
    anchor_name: str,
    meta: dict,
) -> np.ndarray:
    anchor = predictor_by_name(anchor_name)
    preds = []
    for i in range(len(coeffs_batch)):
        other = None if other_root_batch is None else float(other_root_batch[i].reshape(-1)[0])
        preds.append(anchor.predict_one(coeffs_batch[i], other, meta))
    preds = np.asarray(preds, dtype=np.float32).reshape(-1, 1)
    return y_true_batch - preds


def train_model(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    save_dir = Path(cfg.save_dir)
    ensure_dir(save_dir)

    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    train_ds = RootDataset(
        cfg.train_npz,
        coeffs_key=cfg.coeffs_key,
        target_key=cfg.target_key,
        other_root_key=cfg.other_root_key,
        fit_normalizer=True,
    )
    val_ds = RootDataset(
        cfg.val_npz,
        coeffs_key=cfg.coeffs_key,
        target_key=cfg.target_key,
        other_root_key=cfg.other_root_key,
        input_mean=train_ds.input_mean,
        input_std=train_ds.input_std,
        target_mean=train_ds.target_mean,
        target_std=train_ds.target_std,
        fit_normalizer=False,
    )

    test_ds = None
    if cfg.test_npz is not None:
        test_ds = RootDataset(
            cfg.test_npz,
            coeffs_key=cfg.coeffs_key,
            target_key=cfg.target_key,
            other_root_key=cfg.other_root_key,
            input_mean=train_ds.input_mean,
            input_std=train_ds.input_std,
            target_mean=train_ds.target_mean,
            target_std=train_ds.target_std,
            fit_normalizer=False,
        )

    input_dim = train_ds.x.shape[1]
    model = MLP(input_dim, cfg.hidden_dims, cfg.dropout, cfg.activation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=40)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    meta = {
        "train_target_mean": float(train_ds.targets.mean()),
        "has_other_root_feature": bool(train_ds.has_other_root),
        "model": {
            "input_dim": input_dim,
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "activation": cfg.activation,
        },
        "train_config": asdict(cfg),
    }

    save_json(save_dir / "normalizer.json", {
        "input_mean": train_ds.input_mean.tolist(),
        "input_std": train_ds.input_std.tolist(),
        "target_mean": float(train_ds.target_mean),
        "target_std": float(train_ds.target_std),
    })
    save_json(save_dir / "meta.json", meta)

    best_val = float("inf")
    best_epoch = -1
    patience_count = 0
    history = []

    def infer_on_loader(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        losses = []
        ys_true = []
        ys_pred = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                raw_target = batch["target"].cpu().numpy().reshape(-1, 1).astype(np.float32)
                coeffs = batch["coeffs"].cpu().numpy().astype(np.float32)
                other_root = None
                if "other_root" in batch:
                    other_root = batch["other_root"].cpu().numpy().astype(np.float32)

                pred = model(x)

                if cfg.mode == "residual":
                    anchor_targets = build_anchor_targets(coeffs, other_root, raw_target, cfg.residual_anchor, meta)
                    anchor_targets_norm = (anchor_targets - train_ds.target_mean) / train_ds.target_std
                    y_ref = torch.from_numpy(anchor_targets_norm).float().to(device)
                else:
                    y_ref = y

                loss = weighted_mse_loss(pred, y_ref)
                losses.append(loss.item())

                pred_np = pred.cpu().numpy() * train_ds.target_std + train_ds.target_mean
                if cfg.mode == "residual":
                    anchor_pred = build_anchor_targets(coeffs, other_root, np.zeros_like(raw_target), cfg.residual_anchor, meta)
                    # build_anchor_targets returns y_true - anchor; for zero target, result = -anchor
                    anchor_pred = -anchor_pred
                    pred_np = pred_np + anchor_pred
                ys_pred.append(pred_np.reshape(-1))
                ys_true.append(raw_target.reshape(-1))

        ys_true = np.concatenate(ys_true)
        ys_pred = np.concatenate(ys_pred)
        mae = float(np.mean(np.abs(ys_pred - ys_true)))
        return float(np.mean(losses)), mae

    print(f"[{now_str()}] Training start | device={device} | save_dir={save_dir}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            raw_target = batch["target"].cpu().numpy().reshape(-1, 1).astype(np.float32)
            coeffs = batch["coeffs"].cpu().numpy().astype(np.float32)
            other_root = None
            if "other_root" in batch:
                other_root = batch["other_root"].cpu().numpy().astype(np.float32)

            opt.zero_grad(set_to_none=True)
            pred = model(x)

            if cfg.mode == "residual":
                anchor_targets = build_anchor_targets(coeffs, other_root, raw_target, cfg.residual_anchor, meta)
                target_for_loss = (anchor_targets - train_ds.target_mean) / train_ds.target_std
                y_ref = torch.from_numpy(target_for_loss).float().to(device)
            else:
                y_ref = y

            # hard weighting: anchor/direct 기반 오차 큰 샘플에 더 큰 weight 부여
            weights = None
            if cfg.hard_weight_alpha > 0:
                with torch.no_grad():
                    if cfg.mode == "residual":
                        hard_base = np.abs(anchor_targets).reshape(-1)
                    else:
                        hard_base = np.abs(raw_target.reshape(-1) - raw_target.mean())
                    hard_base = hard_base / (hard_base.mean() + 1e-8)
                    weights = torch.from_numpy((1.0 + cfg.hard_weight_alpha * hard_base).astype(np.float32)).to(device).view(-1, 1)

            loss = weighted_mse_loss(pred, y_ref, weights=weights)

            # optional residual-on-function loss
            if cfg.residual_loss_lambda > 0:
                pred_np = pred.detach().cpu().numpy() * train_ds.target_std + train_ds.target_mean
                if cfg.mode == "residual":
                    anchor_pred = build_anchor_targets(coeffs, other_root, np.zeros_like(raw_target), cfg.residual_anchor, meta)
                    anchor_pred = -anchor_pred
                    pred_np = pred_np + anchor_pred
                residuals = []
                for i in range(len(pred_np)):
                    residuals.append(poly_eval_scalar(coeffs[i], float(pred_np[i, 0])))
                residuals = np.asarray(residuals, dtype=np.float32).reshape(-1, 1)
                residual_loss = torch.from_numpy((residuals ** 2).astype(np.float32)).to(device).mean()
                loss = loss + cfg.residual_loss_lambda * residual_loss

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)

        train_loss = running_loss / max(1, count)
        val_loss, val_mae = infer_on_loader(val_loader)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Epoch {epoch:4d}] train_loss={train_loss:.6e} "
                f"val_loss={val_loss:.6e} val_mae={val_mae:.6e} lr={opt.param_groups[0]['lr']:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val={best_val:.6e})")
            break

    save_json(save_dir / "train_history.json", {"history": history, "best_epoch": best_epoch, "best_val": best_val})
    print(f"[{now_str()}] Training finished. best_epoch={best_epoch}, best_val={best_val:.6e}")

    if test_ds is not None:
        evaluate_all(
            test_npz=cfg.test_npz,
            save_dir=cfg.save_dir,
            coeffs_key=cfg.coeffs_key,
            target_key=cfg.target_key,
            other_root_key=cfg.other_root_key,
            ood_npz=None,
            device=cfg.device,
        )


# =========================
# Evaluation
# =========================

def denorm_target(y_norm: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    return y_norm * target_std + target_mean


def load_dataset_for_eval(npz_path: str, save_dir: Path, coeffs_key: str, target_key: str, other_root_key: Optional[str]):
    norm = load_json(save_dir / "normalizer.json")
    ds = RootDataset(
        npz_path=npz_path,
        coeffs_key=coeffs_key,
        target_key=target_key,
        other_root_key=other_root_key,
        input_mean=np.asarray(norm["input_mean"], dtype=np.float32),
        input_std=np.asarray(norm["input_std"], dtype=np.float32),
        target_mean=float(norm["target_mean"]),
        target_std=float(norm["target_std"]),
        fit_normalizer=False,
    )
    return ds


def evaluate_predictor_on_dataset(
    predictor_name: str,
    predictor_obj,
    dataset: RootDataset,
    meta: dict,
    save_subdir: Path,
    newton_cfg: NewtonConfig,
    uncertainty_info: bool = False,
) -> Dict[str, float]:
    ensure_dir(save_subdir)
    csv_path = save_subdir / f"sample_logs_{predictor_name}.csv"
    hard_path = save_subdir / f"hard_cases_{predictor_name}.json"
    summary_path = save_subdir / f"summary_{predictor_name}.json"

    rows = []
    hard_cases = []
    timing_predict = []
    timing_refine = []
    timing_total = []

    abs_err_directs = []
    abs_err_refineds = []
    residual_directs = []
    residual_refineds = []
    iterations = []
    success_flags = []
    diverged_flags = []

    # warmup for neural
    if isinstance(predictor_obj, NeuralPredictor):
        for i in range(min(8, len(dataset))):
            coeffs = dataset.coeffs[i]
            other = None if dataset.other_root is None else float(dataset.other_root[i, 0])
            _ = predictor_obj.predict_one(coeffs, other, meta)
    elif isinstance(predictor_obj, HybridUncertaintyPredictor):
        for i in range(min(8, len(dataset))):
            coeffs = dataset.coeffs[i]
            other = None if dataset.other_root is None else float(dataset.other_root[i, 0])
            _ = predictor_obj.predict_one(coeffs, other, meta)

    for i in range(len(dataset)):
        coeffs = dataset.coeffs[i].astype(np.float64)
        y_true = float(dataset.targets[i, 0])
        other = None if dataset.other_root is None else float(dataset.other_root[i, 0])

        t0 = time.perf_counter()
        pred_std = 0.0
        if uncertainty_info and isinstance(predictor_obj, HybridUncertaintyPredictor):
            pred_direct = predictor_obj.predict_one(coeffs, other, meta)
        elif uncertainty_info and isinstance(predictor_obj, NeuralPredictor):
            pred_direct, pred_std = predictor_obj.predict_with_uncertainty(coeffs, other)
        else:
            pred_direct = predictor_obj.predict_one(coeffs, other, meta)
        t1 = time.perf_counter()

        f_direct, _ = poly_eval_and_derivative(coeffs, pred_direct)
        abs_err_direct = abs(pred_direct - y_true)
        residual_direct = abs(f_direct)

        nr = safeguarded_newton(coeffs, pred_direct, newton_cfg)
        t2 = time.perf_counter()

        pred_refined = nr.refined
        f_refined, _ = poly_eval_and_derivative(coeffs, pred_refined)
        abs_err_refined = abs(pred_refined - y_true)
        residual_refined = abs(f_refined)

        timing_predict.append(t1 - t0)
        timing_refine.append(t2 - t1)
        timing_total.append(t2 - t0)
        abs_err_directs.append(abs_err_direct)
        abs_err_refineds.append(abs_err_refined)
        residual_directs.append(residual_direct)
        residual_refineds.append(residual_refined)
        iterations.append(nr.iterations)
        success_flags.append(1 if nr.success else 0)
        diverged_flags.append(1 if nr.diverged else 0)

        diff_feats = compute_difficulty_features(coeffs, y_true)

        row = {
            "sample_id": i,
            "method": predictor_name,
            "y_true": y_true,
            "other_root": other if other is not None else "",
            "pred_direct": safe_float(pred_direct),
            "pred_refined": safe_float(pred_refined),
            "pred_std": safe_float(pred_std),
            "abs_err_direct": safe_float(abs_err_direct),
            "abs_err_refined": safe_float(abs_err_refined),
            "residual_direct": safe_float(residual_direct),
            "residual_refined": safe_float(residual_refined),
            "newton_iters": int(nr.iterations),
            "success": int(nr.success),
            "diverged": int(nr.diverged),
            "used_small_derivative_fallback": int(nr.used_small_derivative_fallback),
            "used_line_search": int(nr.used_line_search),
            "hit_clamp": int(nr.hit_clamp),
            "predict_time_sec": safe_float(t1 - t0),
            "refine_time_sec": safe_float(t2 - t1),
            "total_time_sec": safe_float(t2 - t0),
            "degree": diff_feats["degree"],
            "abs_df_at_root": diff_feats["abs_df_at_root"],
            "linearized_guess": diff_feats["linearized_guess"],
        }
        rows.append(row)

        # hard case criteria
        if (
            residual_refined > 1e-8
            or nr.iterations >= max(8, newton_cfg.max_iters // 2)
            or nr.diverged
            or abs_err_refined > 1e-4
        ):
            hard_cases.append({
                "sample_id": i,
                "method": predictor_name,
                "coeffs": coeffs.tolist(),
                "y_true": y_true,
                "other_root": other,
                "pred_direct": pred_direct,
                "pred_refined": pred_refined,
                "abs_err_direct": abs_err_direct,
                "abs_err_refined": abs_err_refined,
                "residual_direct": residual_direct,
                "residual_refined": residual_refined,
                "newton_iters": nr.iterations,
                "success": nr.success,
                "diverged": nr.diverged,
                "trajectory": nr.trajectory,
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json(hard_path, {"num_hard_cases": len(hard_cases), "hard_cases": hard_cases[:500]})

    abs_err_directs = np.asarray(abs_err_directs)
    abs_err_refineds = np.asarray(abs_err_refineds)
    residual_directs = np.asarray(residual_directs)
    residual_refineds = np.asarray(residual_refineds)
    iterations = np.asarray(iterations)
    timing_predict = np.asarray(timing_predict)
    timing_refine = np.asarray(timing_refine)
    timing_total = np.asarray(timing_total)
    success_flags = np.asarray(success_flags)
    diverged_flags = np.asarray(diverged_flags)

    summary = {
        "method": predictor_name,
        "n_samples": int(len(dataset)),
        "success_rate": float(success_flags.mean()),
        "fail_rate": float(1.0 - success_flags.mean()),
        "diverged_rate": float(diverged_flags.mean()),
        "abs_err_direct": percentile_dict(abs_err_directs),
        "abs_err_refined": percentile_dict(abs_err_refineds),
        "residual_direct": percentile_dict(residual_directs),
        "residual_refined": percentile_dict(residual_refineds),
        "iterations": percentile_dict(iterations),
        "predict_time_sec": percentile_dict(timing_predict),
        "refine_time_sec": percentile_dict(timing_refine),
        "total_time_sec": percentile_dict(timing_total),
        "hard_case_count": int(len(hard_cases)),
    }
    save_json(summary_path, summary)
    return summary


def print_summary_table(title: str, summaries: List[dict]) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    header = f"{'method':<22} {'succ%':>8} {'dir_mae':>14} {'ref_mae':>14} {'iter_mean':>12} {'total_ms':>12} {'p95_iter':>10} {'hard':>8}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        line = (
            f"{s['method']:<22} "
            f"{100*s['success_rate']:>7.2f} "
            f"{s['abs_err_direct']['mean']:>14.6e} "
            f"{s['abs_err_refined']['mean']:>14.6e} "
            f"{s['iterations']['mean']:>12.4f} "
            f"{1000*s['total_time_sec']['mean']:>12.4f} "
            f"{s['iterations']['p95']:>10.2f} "
            f"{s['hard_case_count']:>8d}"
        )
        print(line)
    print("=" * 120)



def evaluate_all(
    test_npz: str,
    save_dir: str,
    coeffs_key: str = "coeffs",
    target_key: str = "root1",
    other_root_key: Optional[str] = "root2_label",
    ood_npz: Optional[str] = None,
    device: str = "cuda",
) -> None:
    save_dir_p = Path(save_dir)
    meta = load_json(save_dir_p / "meta.json")
    test_ds = load_dataset_for_eval(test_npz, save_dir_p, coeffs_key, target_key, other_root_key)

    newton_cfg = NewtonConfig(
        max_iters=20,
        tol_residual=1e-10,
        tol_step=1e-12,
        deriv_eps=1e-12,
        fallback_step=1e-3,
        max_abs_step=5.0,
        clamp_min=None,
        clamp_max=None,
        line_search_alphas=(1.0, 0.5, 0.25, 0.125, 0.0625),
    )

    methods = []
    methods.append(("zero", predictor_by_name("zero"), False))
    methods.append(("train_mean", predictor_by_name("train_mean"), False))
    methods.append(("linearized", predictor_by_name("linearized"), False))
    if test_ds.has_other_root:
        methods.append(("other_root", predictor_by_name("other_root"), False))
    if test_ds.coeffs.shape[1] == 3:
        methods.append(("quadratic_formula", predictor_by_name("quadratic_formula"), False))

    neural = NeuralPredictor(save_dir_p, device=device, mc_dropout_passes=10)
    methods.append(("neural", neural, True))
    hybrid = HybridUncertaintyPredictor(
        neural_ckpt_dir=save_dir_p,
        device=device,
        mc_dropout_passes=10,
        uncertainty_threshold=0.05,
        fallback_name="linearized",
    )
    methods.append(("hybrid_uncertainty", hybrid, False))

    eval_root = save_dir_p / "evaluation"
    ensure_dir(eval_root)

    summaries = []
    for name, pred_obj, use_uncertainty_info in methods:
        print(f"[{now_str()}] Evaluating method={name} on TEST ...")
        summary = evaluate_predictor_on_dataset(
            predictor_name=name,
            predictor_obj=pred_obj,
            dataset=test_ds,
            meta=meta,
            save_subdir=eval_root / "test",
            newton_cfg=newton_cfg,
            uncertainty_info=use_uncertainty_info,
        )
        summaries.append(summary)
    save_json(eval_root / "test_all_summaries.json", {"summaries": summaries})
    print_summary_table("TEST SUMMARY", summaries)

    if ood_npz is not None:
        ood_ds = load_dataset_for_eval(ood_npz, save_dir_p, coeffs_key, target_key, other_root_key)
        ood_summaries = []
        for name, pred_obj, use_uncertainty_info in methods:
            print(f"[{now_str()}] Evaluating method={name} on OOD ...")
            summary = evaluate_predictor_on_dataset(
                predictor_name=name,
                predictor_obj=pred_obj,
                dataset=ood_ds,
                meta=meta,
                save_subdir=eval_root / "ood",
                newton_cfg=newton_cfg,
                uncertainty_info=use_uncertainty_info,
            )
            ood_summaries.append(summary)
        save_json(eval_root / "ood_all_summaries.json", {"summaries": ood_summaries})
        print_summary_table("OOD SUMMARY", ood_summaries)


# =========================
# Argument parser
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural init guess + safeguarded Newton benchmark framework")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train_npz", type=str, required=True)
    p_train.add_argument("--val_npz", type=str, required=True)
    p_train.add_argument("--test_npz", type=str, default=None)
    p_train.add_argument("--save_dir", type=str, required=True)
    p_train.add_argument("--coeffs_key", type=str, default="coeffs")
    p_train.add_argument("--target_key", type=str, default="root1")
    p_train.add_argument("--other_root_key", type=str, default="root2_label")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=3000)
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-5)
    p_train.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 128])
    p_train.add_argument("--dropout", type=float, default=0.05)
    p_train.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "silu", "tanh"])
    p_train.add_argument("--patience", type=int, default=200)
    p_train.add_argument("--mode", type=str, default="residual", choices=["direct", "residual"])
    p_train.add_argument("--residual_anchor", type=str, default="linearized", choices=["zero", "train_mean", "linearized", "other_root"])
    p_train.add_argument("--hard_weight_alpha", type=float, default=1.0)
    p_train.add_argument("--residual_loss_lambda", type=float, default=0.0)
    p_train.add_argument("--grad_clip", type=float, default=5.0)
    p_train.add_argument("--num_workers", type=int, default=0)
    p_train.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--test_npz", type=str, required=True)
    p_eval.add_argument("--ood_npz", type=str, default=None)
    p_eval.add_argument("--save_dir", type=str, required=True)
    p_eval.add_argument("--coeffs_key", type=str, default="coeffs")
    p_eval.add_argument("--target_key", type=str, default="root1")
    p_eval.add_argument("--other_root_key", type=str, default="root2_label")
    p_eval.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return p


# =========================
# Main
# =========================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainConfig(
            train_npz=args.train_npz,
            val_npz=args.val_npz,
            test_npz=args.test_npz,
            save_dir=args.save_dir,
            coeffs_key=args.coeffs_key,
            target_key=args.target_key,
            other_root_key=args.other_root_key,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            activation=args.activation,
            patience=args.patience,
            mode=args.mode,
            residual_anchor=args.residual_anchor,
            hard_weight_alpha=args.hard_weight_alpha,
            residual_loss_lambda=args.residual_loss_lambda,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            device=args.device,
        )
        train_model(cfg)

    elif args.command == "eval":
        evaluate_all(
            test_npz=args.test_npz,
            save_dir=args.save_dir,
            coeffs_key=args.coeffs_key,
            target_key=args.target_key,
            other_root_key=args.other_root_key,
            ood_npz=args.ood_npz,
            device=args.device,
        )


if __name__ == "__main__":
    main()
"""
python root_benchmark_framework.py train \
   --train_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_train.npz \
   --val_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_val.npz \
   --test_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_test.npz \
   --save_dir /home/seokjun/math_03_14/save_path \
   --coeffs_key coeffs \
   --target_key root \
   --other_root_key root2_label \
   --mode residual \
   --epochs 3000 \
   --batch_size 512 \
   --hidden_dims 256 256 128 \
   --lr 1e-3 \
   --patience 200

python root_benchmark_framework.py eval \
  --test_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_test.npz \
  --save_dir /home/seokjun/math_03_14/colebrook_data/save_path \
  --coeffs_key coeffs \
  --target_key root \
  --other_root_key root2_label
"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Neural Initial Guess + Classical Baselines + Safeguarded Newton Benchmark Framework
==========================================================================================

목적
----
이 스크립트는 "신경망 기반 초기값 생성"을 고전 초기값들과 공정 비교하고,
Safeguarded Newton refinement, 실패 사례 수집, 분포 분석, 시간 측정,
OOD 평가까지 한 번에 수행하기 위한 통합 프레임워크이다.

핵심 기능
---------
1. NPZ 데이터셋 로드
2. MLP 기반 direct / residual predictor 학습
3. baseline predictor 비교
   - zero
   - train mean
   - other_root (있을 때)
   - linearized guess (-c0/c1)
   - quadratic formula (degree=2일 때 가능)
4. Safeguarded Newton refinement
   - derivative small fallback
   - step clipping
   - damping line search
   - domain clamp
5. 샘플 단위 CSV 로그 저장
6. 분포 요약(mean/median/std/p90/p95/p99/max)
7. hard case 자동 수집
8. inference / refine / total time 측정
9. optional OOD split 평가

가정
----
- coeffs는 다항식 p(x)=c0 + c1 x + c2 x^2 + ... + cn x^n 의 power basis 계수라고 가정한다.
- target root는 root1 이다.
- other root 정보가 있으면 root2_label 같은 키로 제공할 수 있다.
- 입력 feature는 기본적으로 [coeffs, other_root(optional)] 형태로 사용한다.

예시 실행
---------
학습:
python root_benchmark_framework.py train \
  --train_npz /path/to/train.npz \
  --val_npz /path/to/val.npz \
  --test_npz /path/to/test.npz \
  --save_dir /path/to/exp1 \
  --coeffs_key coeffs \
  --target_key root1 \
  --other_root_key root2_label \
  --mode residual \
  --epochs 3000 \
  --batch_size 512 \
  --hidden_dims 256 256 128 \
  --lr 1e-3 \
  --patience 200

평가만:
python root_benchmark_framework.py eval \
  --test_npz /path/to/test.npz \
  --save_dir /path/to/exp1 \
  --coeffs_key coeffs \
  --target_key root1 \
  --other_root_key root2_label

OOD 평가:
python root_benchmark_framework.py eval \
  --test_npz /path/to/test.npz \
  --ood_npz /path/to/ood_test.npz \
  --save_dir /path/to/exp1

주의
----
- quadratic_formula baseline은 coeffs 길이가 정확히 3일 때만 의미 있다.
- linearized baseline은 c1이 매우 작으면 fallback 처리한다.
- exact polynomial root solver(np.roots)를 baseline으로 넣고 싶다면 쉽게 확장 가능하지만,
  여기서는 초기값 전략 비교에 집중한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================
# Utility
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentile_dict(x: np.ndarray) -> Dict[str, float]:
    if len(x) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# =========================
# Polynomial helpers
# =========================

def poly_eval_scalar(coeffs: np.ndarray, x: float) -> float:
    # coeffs: [c0, c1, ..., cn]
    # Horner in ascending basis
    # convert to descending for Horner convenience
    y = 0.0
    for c in coeffs[::-1]:
        y = y * x + float(c)
    return y


def poly_derivative_coeffs(coeffs: np.ndarray) -> np.ndarray:
    if len(coeffs) <= 1:
        return np.array([0.0], dtype=np.float64)
    return np.array([k * coeffs[k] for k in range(1, len(coeffs))], dtype=np.float64)


def poly_eval_and_derivative(coeffs: np.ndarray, x: float) -> Tuple[float, float]:
    f = poly_eval_scalar(coeffs, x)
    dcoeffs = poly_derivative_coeffs(coeffs)
    df = poly_eval_scalar(dcoeffs, x)
    return f, df


def linearized_root_guess(coeffs: np.ndarray, eps: float = 1e-12) -> float:
    # p(x) ~= c0 + c1 x  =>  x = -c0/c1
    c0 = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    c1 = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    if abs(c1) < eps:
        return 0.0
    return -c0 / c1


def quadratic_formula_guesses(coeffs: np.ndarray) -> Optional[Tuple[float, float]]:
    # coeffs = [c0, c1, c2] for c0 + c1 x + c2 x^2 = 0
    if len(coeffs) != 3:
        return None
    a = float(coeffs[2])
    b = float(coeffs[1])
    c = float(coeffs[0])
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            return None
        r = -c / b
        return (r, r)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    r1 = (-b + s) / (2 * a)
    r2 = (-b - s) / (2 * a)
    return (r1, r2)


def choose_closest(candidates: List[float], target: float) -> float:
    if len(candidates) == 0:
        return 0.0
    d = [abs(c - target) for c in candidates]
    return float(candidates[int(np.argmin(d))])


def compute_difficulty_features(coeffs: np.ndarray, true_root: float) -> Dict[str, float]:
    f, df = poly_eval_and_derivative(coeffs, true_root)
    return {
        "abs_true_residual": abs(f),
        "abs_df_at_root": abs(df),
        "linearized_guess": linearized_root_guess(coeffs),
        "degree": len(coeffs) - 1,
    }


# =========================
# Dataset
# =========================

class RootDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        coeffs_key: str = "coeffs",
        target_key: str = "root1",
        other_root_key: Optional[str] = "root2_label",
        input_mean: Optional[np.ndarray] = None,
        input_std: Optional[np.ndarray] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        fit_normalizer: bool = False,
    ):
        self.npz_path = npz_path
        data = np.load(npz_path, allow_pickle=True)
        if coeffs_key not in data:
            raise KeyError(f"'{coeffs_key}' not found in {npz_path}. keys={list(data.keys())}")
        if target_key not in data:
            raise KeyError(f"'{target_key}' not found in {npz_path}. keys={list(data.keys())}")

        coeffs = np.asarray(data[coeffs_key], dtype=np.float32)
        targets = np.asarray(data[target_key], dtype=np.float32).reshape(-1, 1)

        if coeffs.ndim != 2:
            raise ValueError(f"coeffs must be 2D [N, D]. got shape={coeffs.shape}")

        self.has_other_root = other_root_key is not None and other_root_key in data
        other_root = None
        if self.has_other_root:
            other_root = np.asarray(data[other_root_key], dtype=np.float32).reshape(-1, 1)
            feats = np.concatenate([coeffs, other_root], axis=1)
        else:
            feats = coeffs.copy()

        self.coeffs = coeffs
        self.other_root = other_root
        self.targets = targets
        self.raw_feats = feats

        if fit_normalizer:
            self.input_mean = feats.mean(axis=0)
            self.input_std = feats.std(axis=0) + 1e-8
            self.target_mean = float(targets.mean())
            self.target_std = float(targets.std() + 1e-8)
        else:
            assert input_mean is not None and input_std is not None
            assert target_mean is not None and target_std is not None
            self.input_mean = input_mean.astype(np.float32)
            self.input_std = input_std.astype(np.float32)
            self.target_mean = float(target_mean)
            self.target_std = float(target_std)

        self.x = (feats - self.input_mean) / self.input_std
        self.y = (targets - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        item = {
            "x": torch.from_numpy(self.x[idx]).float(),
            "y": torch.from_numpy(self.y[idx]).float(),
            "coeffs": torch.from_numpy(self.coeffs[idx]).float(),
            "target": torch.from_numpy(self.targets[idx]).float(),
            "index": idx,
        }
        if self.other_root is not None:
            item["other_root"] = torch.from_numpy(self.other_root[idx]).float()
        return item


# =========================
# Model
# =========================

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }[activation]

        dims = [input_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Loss
# =========================

def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        return ((pred - target) ** 2).mean()
    return (weights * (pred - target) ** 2).mean()


# =========================
# Training config
# =========================

@dataclass
class TrainConfig:
    train_npz: str
    val_npz: str
    test_npz: Optional[str]
    save_dir: str
    coeffs_key: str = "coeffs"
    target_key: str = "root1"
    other_root_key: Optional[str] = "root2_label"
    seed: int = 42
    epochs: int = 3000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    dropout: float = 0.05
    activation: str = "gelu"
    patience: int = 200
    mode: str = "residual"  # direct or residual
    residual_anchor: str = "linearized"  # linearized / zero / mean / other_root
    hard_weight_alpha: float = 1.0
    residual_loss_lambda: float = 0.0
    grad_clip: float = 5.0
    num_workers: int = 0
    device: str = "cuda"


# =========================
# Predictor classes
# =========================

class BasePredictor:
    name = "base"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        raise NotImplementedError


class ZeroPredictor(BasePredictor):
    name = "zero"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return 0.0


class MeanPredictor(BasePredictor):
    name = "train_mean"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return float(meta["train_target_mean"])


class OtherRootPredictor(BasePredictor):
    name = "other_root"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return 0.0 if other_root is None else float(other_root)


class LinearizedPredictor(BasePredictor):
    name = "linearized"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        return linearized_root_guess(coeffs)


class QuadraticFormulaPredictor(BasePredictor):
    name = "quadratic_formula"

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        qs = quadratic_formula_guesses(coeffs)
        if qs is None:
            return 0.0
        if other_root is not None:
            # 다른 루트가 주어지면 그 반대쪽 root를 선택하려고 시도
            cands = list(qs)
            idx = int(np.argmax([abs(c - other_root) for c in cands]))
            return float(cands[idx])
        target_proxy = linearized_root_guess(coeffs)
        return choose_closest(list(qs), target_proxy)


class NeuralPredictor(BasePredictor):
    name = "neural"

    def __init__(self, ckpt_dir: Path, device: str = "cpu", mc_dropout_passes: int = 0):
        self.ckpt_dir = ckpt_dir
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.meta = load_json(ckpt_dir / "meta.json")
        self.norm = load_json(ckpt_dir / "normalizer.json")
        model_cfg = self.meta["model"]
        self.model = MLP(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            dropout=model_cfg["dropout"],
            activation=model_cfg["activation"],
        )
        state = torch.load(ckpt_dir / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.mc_dropout_passes = mc_dropout_passes

    def _build_input(self, coeffs: np.ndarray, other_root: Optional[float]) -> np.ndarray:
        feats = coeffs.astype(np.float32)
        if self.meta["has_other_root_feature"]:
            oroot = np.array([0.0 if other_root is None else other_root], dtype=np.float32)
            feats = np.concatenate([feats, oroot], axis=0)
        mean = np.asarray(self.norm["input_mean"], dtype=np.float32)
        std = np.asarray(self.norm["input_std"], dtype=np.float32)
        feats = (feats - mean) / std
        return feats

    def _denorm_target(self, y: np.ndarray) -> np.ndarray:
        tmean = float(self.norm["target_mean"])
        tstd = float(self.norm["target_std"])
        return y * tstd + tmean

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        x = self._build_input(coeffs, other_root)
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(xt).cpu().numpy().reshape(-1)
        pred = float(self._denorm_target(y)[0])
        if self.meta["train_config"]["mode"] == "residual":
            anchor_name = self.meta["train_config"]["residual_anchor"]
            anchor_pred = predictor_by_name(anchor_name).predict_one(coeffs, other_root, self.meta)
            pred = anchor_pred + pred
        return pred

    def predict_with_uncertainty(self, coeffs: np.ndarray, other_root: Optional[float]) -> Tuple[float, float]:
        x = self._build_input(coeffs, other_root)
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        if self.mc_dropout_passes <= 1:
            self.model.eval()
            with torch.no_grad():
                y = self.model(xt).cpu().numpy().reshape(-1)
            pred = float(self._denorm_target(y)[0])
            if self.meta["train_config"]["mode"] == "residual":
                anchor_name = self.meta["train_config"]["residual_anchor"]
                anchor_pred = predictor_by_name(anchor_name).predict_one(coeffs, other_root, self.meta)
                pred = anchor_pred + pred
            return pred, 0.0

        self.model.train()  # dropout on
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_passes):
                y = self.model(xt).cpu().numpy().reshape(-1)
                pred = float(self._denorm_target(y)[0])
                if self.meta["train_config"]["mode"] == "residual":
                    anchor_name = self.meta["train_config"]["residual_anchor"]
                    anchor_pred = predictor_by_name(anchor_name).predict_one(coeffs, other_root, self.meta)
                    pred = anchor_pred + pred
                preds.append(pred)
        self.model.eval()
        return float(np.mean(preds)), float(np.std(preds))


class HybridUncertaintyPredictor(BasePredictor):
    name = "hybrid_uncertainty"

    def __init__(
        self,
        neural_ckpt_dir: Path,
        device: str = "cpu",
        mc_dropout_passes: int = 10,
        uncertainty_threshold: float = 0.05,
        fallback_name: str = "linearized",
    ):
        self.neural = NeuralPredictor(neural_ckpt_dir, device=device, mc_dropout_passes=mc_dropout_passes)
        self.threshold = uncertainty_threshold
        self.fallback_name = fallback_name

    def predict_one(self, coeffs: np.ndarray, other_root: Optional[float], meta: dict) -> float:
        pred, std = self.neural.predict_with_uncertainty(coeffs, other_root)
        if std > self.threshold:
            return predictor_by_name(self.fallback_name).predict_one(coeffs, other_root, meta)
        return pred


# =========================
# Predictor factory
# =========================

def predictor_by_name(name: str) -> BasePredictor:
    table = {
        "zero": ZeroPredictor,
        "train_mean": MeanPredictor,
        "other_root": OtherRootPredictor,
        "linearized": LinearizedPredictor,
        "quadratic_formula": QuadraticFormulaPredictor,
    }
    if name not in table:
        raise KeyError(f"Unknown predictor: {name}")
    return table[name]()


# =========================
# Safeguarded Newton
# =========================

@dataclass
class NewtonConfig:
    max_iters: int = 20
    tol_residual: float = 1e-10
    tol_step: float = 1e-12
    deriv_eps: float = 1e-12
    fallback_step: float = 1e-2
    max_abs_step: float = 5.0
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    line_search_alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass
class NewtonResult:
    refined: float
    iterations: int
    success: bool
    diverged: bool
    used_small_derivative_fallback: bool
    used_line_search: bool
    hit_clamp: bool
    start_residual: float
    end_residual: float
    trajectory: List[dict]


def safeguarded_newton(coeffs: np.ndarray, x0: float, cfg: NewtonConfig) -> NewtonResult:
    x = float(x0)
    traj: List[dict] = []
    used_small_derivative_fallback = False
    used_line_search = False
    hit_clamp = False
    diverged = False

    f0, _ = poly_eval_and_derivative(coeffs, x)
    prev_abs_f = abs(f0)

    for it in range(1, cfg.max_iters + 1):
        f, df = poly_eval_and_derivative(coeffs, x)
        abs_f = abs(f)
        traj.append({
            "iter": it,
            "x": float(x),
            "f": float(f),
            "df": float(df),
            "abs_f": float(abs_f),
        })

        if abs_f <= cfg.tol_residual:
            return NewtonResult(
                refined=float(x),
                iterations=it - 1,
                success=True,
                diverged=False,
                used_small_derivative_fallback=used_small_derivative_fallback,
                used_line_search=used_line_search,
                hit_clamp=hit_clamp,
                start_residual=float(abs(f0)),
                end_residual=float(abs_f),
                trajectory=traj,
            )

        if abs(df) < cfg.deriv_eps:
            used_small_derivative_fallback = True
            step = math.copysign(cfg.fallback_step, f if f != 0 else 1.0)
        else:
            step = f / df

        if not np.isfinite(step):
            diverged = True
            break

        step = float(np.clip(step, -cfg.max_abs_step, cfg.max_abs_step))

        accepted = False
        best_x = x
        best_abs_f = abs_f
        for alpha in cfg.line_search_alphas:
            cand = x - alpha * step
            if cfg.clamp_min is not None:
                cand = max(cand, cfg.clamp_min)
            if cfg.clamp_max is not None:
                cand = min(cand, cfg.clamp_max)
            if cand != x - alpha * step:
                hit_clamp = True
            f_cand, _ = poly_eval_and_derivative(coeffs, cand)
            abs_f_cand = abs(f_cand)
            if abs_f_cand < best_abs_f:
                best_x = cand
                best_abs_f = abs_f_cand
                accepted = True
                if alpha != 1.0:
                    used_line_search = True
                break

        if not accepted:
            # 더 나빠지면 fallback 한 스텝만 적용
            cand = x - math.copysign(min(abs(step), cfg.fallback_step), step)
            if cfg.clamp_min is not None:
                cand = max(cand, cfg.clamp_min)
            if cfg.clamp_max is not None:
                cand = min(cand, cfg.clamp_max)
            if cand != x - math.copysign(min(abs(step), cfg.fallback_step), step):
                hit_clamp = True
            best_x = cand
            f_cand, _ = poly_eval_and_derivative(coeffs, best_x)
            best_abs_f = abs(f_cand)

        if not np.isfinite(best_x) or best_abs_f > 1e30:
            diverged = True
            break

        if abs(best_x - x) <= cfg.tol_step:
            x = best_x
            break

        x = best_x
        prev_abs_f = best_abs_f

    f_end, _ = poly_eval_and_derivative(coeffs, x)
    success = abs(f_end) <= cfg.tol_residual and not diverged
    return NewtonResult(
        refined=float(x),
        iterations=len(traj),
        success=success,
        diverged=diverged,
        used_small_derivative_fallback=used_small_derivative_fallback,
        used_line_search=used_line_search,
        hit_clamp=hit_clamp,
        start_residual=float(abs(f0)),
        end_residual=float(abs(f_end)),
        trajectory=traj,
    )


# =========================
# Training
# =========================

def build_anchor_targets(
    coeffs_batch: np.ndarray,
    other_root_batch: Optional[np.ndarray],
    y_true_batch: np.ndarray,
    anchor_name: str,
    meta: dict,
) -> np.ndarray:
    anchor = predictor_by_name(anchor_name)
    preds = []
    for i in range(len(coeffs_batch)):
        other = None if other_root_batch is None else float(other_root_batch[i].reshape(-1)[0])
        preds.append(anchor.predict_one(coeffs_batch[i], other, meta))
    preds = np.asarray(preds, dtype=np.float32).reshape(-1, 1)
    return y_true_batch - preds


def train_model(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    save_dir = Path(cfg.save_dir)
    ensure_dir(save_dir)

    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    train_ds = RootDataset(
        cfg.train_npz,
        coeffs_key=cfg.coeffs_key,
        target_key=cfg.target_key,
        other_root_key=cfg.other_root_key,
        fit_normalizer=True,
    )
    val_ds = RootDataset(
        cfg.val_npz,
        coeffs_key=cfg.coeffs_key,
        target_key=cfg.target_key,
        other_root_key=cfg.other_root_key,
        input_mean=train_ds.input_mean,
        input_std=train_ds.input_std,
        target_mean=train_ds.target_mean,
        target_std=train_ds.target_std,
        fit_normalizer=False,
    )

    test_ds = None
    if cfg.test_npz is not None:
        test_ds = RootDataset(
            cfg.test_npz,
            coeffs_key=cfg.coeffs_key,
            target_key=cfg.target_key,
            other_root_key=cfg.other_root_key,
            input_mean=train_ds.input_mean,
            input_std=train_ds.input_std,
            target_mean=train_ds.target_mean,
            target_std=train_ds.target_std,
            fit_normalizer=False,
        )

    input_dim = train_ds.x.shape[1]
    model = MLP(input_dim, cfg.hidden_dims, cfg.dropout, cfg.activation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=40)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    meta = {
        "train_target_mean": float(train_ds.targets.mean()),
        "has_other_root_feature": bool(train_ds.has_other_root),
        "model": {
            "input_dim": input_dim,
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "activation": cfg.activation,
        },
        "train_config": asdict(cfg),
    }

    save_json(save_dir / "normalizer.json", {
        "input_mean": train_ds.input_mean.tolist(),
        "input_std": train_ds.input_std.tolist(),
        "target_mean": float(train_ds.target_mean),
        "target_std": float(train_ds.target_std),
    })
    save_json(save_dir / "meta.json", meta)

    best_val = float("inf")
    best_epoch = -1
    patience_count = 0
    history = []

    def infer_on_loader(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        losses = []
        ys_true = []
        ys_pred = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                raw_target = batch["target"].cpu().numpy().reshape(-1, 1).astype(np.float32)
                coeffs = batch["coeffs"].cpu().numpy().astype(np.float32)
                other_root = None
                if "other_root" in batch:
                    other_root = batch["other_root"].cpu().numpy().astype(np.float32)

                pred = model(x)

                if cfg.mode == "residual":
                    anchor_targets = build_anchor_targets(coeffs, other_root, raw_target, cfg.residual_anchor, meta)
                    anchor_targets_norm = (anchor_targets - train_ds.target_mean) / train_ds.target_std
                    y_ref = torch.from_numpy(anchor_targets_norm).float().to(device)
                else:
                    y_ref = y

                loss = weighted_mse_loss(pred, y_ref)
                losses.append(loss.item())

                pred_np = pred.cpu().numpy() * train_ds.target_std + train_ds.target_mean
                if cfg.mode == "residual":
                    anchor_pred = build_anchor_targets(coeffs, other_root, np.zeros_like(raw_target), cfg.residual_anchor, meta)
                    # build_anchor_targets returns y_true - anchor; for zero target, result = -anchor
                    anchor_pred = -anchor_pred
                    pred_np = pred_np + anchor_pred
                ys_pred.append(pred_np.reshape(-1))
                ys_true.append(raw_target.reshape(-1))

        ys_true = np.concatenate(ys_true)
        ys_pred = np.concatenate(ys_pred)
        mae = float(np.mean(np.abs(ys_pred - ys_true)))
        return float(np.mean(losses)), mae

    print(f"[{now_str()}] Training start | device={device} | save_dir={save_dir}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            raw_target = batch["target"].cpu().numpy().reshape(-1, 1).astype(np.float32)
            coeffs = batch["coeffs"].cpu().numpy().astype(np.float32)
            other_root = None
            if "other_root" in batch:
                other_root = batch["other_root"].cpu().numpy().astype(np.float32)

            opt.zero_grad(set_to_none=True)
            pred = model(x)

            if cfg.mode == "residual":
                anchor_targets = build_anchor_targets(coeffs, other_root, raw_target, cfg.residual_anchor, meta)
                target_for_loss = (anchor_targets - train_ds.target_mean) / train_ds.target_std
                y_ref = torch.from_numpy(target_for_loss).float().to(device)
            else:
                y_ref = y

            # hard weighting: anchor/direct 기반 오차 큰 샘플에 더 큰 weight 부여
            weights = None
            if cfg.hard_weight_alpha > 0:
                with torch.no_grad():
                    if cfg.mode == "residual":
                        hard_base = np.abs(anchor_targets).reshape(-1)
                    else:
                        hard_base = np.abs(raw_target.reshape(-1) - raw_target.mean())
                    hard_base = hard_base / (hard_base.mean() + 1e-8)
                    weights = torch.from_numpy((1.0 + cfg.hard_weight_alpha * hard_base).astype(np.float32)).to(device).view(-1, 1)

            loss = weighted_mse_loss(pred, y_ref, weights=weights)

            # optional residual-on-function loss
            if cfg.residual_loss_lambda > 0:
                pred_np = pred.detach().cpu().numpy() * train_ds.target_std + train_ds.target_mean
                if cfg.mode == "residual":
                    anchor_pred = build_anchor_targets(coeffs, other_root, np.zeros_like(raw_target), cfg.residual_anchor, meta)
                    anchor_pred = -anchor_pred
                    pred_np = pred_np + anchor_pred
                residuals = []
                for i in range(len(pred_np)):
                    residuals.append(poly_eval_scalar(coeffs[i], float(pred_np[i, 0])))
                residuals = np.asarray(residuals, dtype=np.float32).reshape(-1, 1)
                residual_loss = torch.from_numpy((residuals ** 2).astype(np.float32)).to(device).mean()
                loss = loss + cfg.residual_loss_lambda * residual_loss

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)

        train_loss = running_loss / max(1, count)
        val_loss, val_mae = infer_on_loader(val_loader)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Epoch {epoch:4d}] train_loss={train_loss:.6e} "
                f"val_loss={val_loss:.6e} val_mae={val_mae:.6e} lr={opt.param_groups[0]['lr']:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val={best_val:.6e})")
            break

    save_json(save_dir / "train_history.json", {"history": history, "best_epoch": best_epoch, "best_val": best_val})
    print(f"[{now_str()}] Training finished. best_epoch={best_epoch}, best_val={best_val:.6e}")

    if test_ds is not None:
        evaluate_all(
            test_npz=cfg.test_npz,
            save_dir=cfg.save_dir,
            coeffs_key=cfg.coeffs_key,
            target_key=cfg.target_key,
            other_root_key=cfg.other_root_key,
            ood_npz=None,
            device=cfg.device,
        )


# =========================
# Evaluation
# =========================

def denorm_target(y_norm: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    return y_norm * target_std + target_mean


def load_dataset_for_eval(npz_path: str, save_dir: Path, coeffs_key: str, target_key: str, other_root_key: Optional[str]):
    norm = load_json(save_dir / "normalizer.json")
    ds = RootDataset(
        npz_path=npz_path,
        coeffs_key=coeffs_key,
        target_key=target_key,
        other_root_key=other_root_key,
        input_mean=np.asarray(norm["input_mean"], dtype=np.float32),
        input_std=np.asarray(norm["input_std"], dtype=np.float32),
        target_mean=float(norm["target_mean"]),
        target_std=float(norm["target_std"]),
        fit_normalizer=False,
    )
    return ds


def evaluate_predictor_on_dataset(
    predictor_name: str,
    predictor_obj,
    dataset: RootDataset,
    meta: dict,
    save_subdir: Path,
    newton_cfg: NewtonConfig,
    uncertainty_info: bool = False,
) -> Dict[str, float]:
    ensure_dir(save_subdir)
    csv_path = save_subdir / f"sample_logs_{predictor_name}.csv"
    hard_path = save_subdir / f"hard_cases_{predictor_name}.json"
    summary_path = save_subdir / f"summary_{predictor_name}.json"

    rows = []
    hard_cases = []
    timing_predict = []
    timing_refine = []
    timing_total = []

    abs_err_directs = []
    abs_err_refineds = []
    residual_directs = []
    residual_refineds = []
    iterations = []
    success_flags = []
    diverged_flags = []

    # warmup for neural
    if isinstance(predictor_obj, NeuralPredictor):
        for i in range(min(8, len(dataset))):
            coeffs = dataset.coeffs[i]
            other = None if dataset.other_root is None else float(dataset.other_root[i, 0])
            _ = predictor_obj.predict_one(coeffs, other, meta)
    elif isinstance(predictor_obj, HybridUncertaintyPredictor):
        for i in range(min(8, len(dataset))):
            coeffs = dataset.coeffs[i]
            other = None if dataset.other_root is None else float(dataset.other_root[i, 0])
            _ = predictor_obj.predict_one(coeffs, other, meta)

    for i in range(len(dataset)):
        coeffs = dataset.coeffs[i].astype(np.float64)
        y_true = float(dataset.targets[i, 0])
        other = None if dataset.other_root is None else float(dataset.other_root[i, 0])

        t0 = time.perf_counter()
        pred_std = 0.0
        if uncertainty_info and isinstance(predictor_obj, HybridUncertaintyPredictor):
            pred_direct = predictor_obj.predict_one(coeffs, other, meta)
        elif uncertainty_info and isinstance(predictor_obj, NeuralPredictor):
            pred_direct, pred_std = predictor_obj.predict_with_uncertainty(coeffs, other)
        else:
            pred_direct = predictor_obj.predict_one(coeffs, other, meta)
        t1 = time.perf_counter()

        f_direct, _ = poly_eval_and_derivative(coeffs, pred_direct)
        abs_err_direct = abs(pred_direct - y_true)
        residual_direct = abs(f_direct)

        nr = safeguarded_newton(coeffs, pred_direct, newton_cfg)
        t2 = time.perf_counter()

        pred_refined = nr.refined
        f_refined, _ = poly_eval_and_derivative(coeffs, pred_refined)
        abs_err_refined = abs(pred_refined - y_true)
        residual_refined = abs(f_refined)

        timing_predict.append(t1 - t0)
        timing_refine.append(t2 - t1)
        timing_total.append(t2 - t0)
        abs_err_directs.append(abs_err_direct)
        abs_err_refineds.append(abs_err_refined)
        residual_directs.append(residual_direct)
        residual_refineds.append(residual_refined)
        iterations.append(nr.iterations)
        success_flags.append(1 if nr.success else 0)
        diverged_flags.append(1 if nr.diverged else 0)

        diff_feats = compute_difficulty_features(coeffs, y_true)

        row = {
            "sample_id": i,
            "method": predictor_name,
            "y_true": y_true,
            "other_root": other if other is not None else "",
            "pred_direct": safe_float(pred_direct),
            "pred_refined": safe_float(pred_refined),
            "pred_std": safe_float(pred_std),
            "abs_err_direct": safe_float(abs_err_direct),
            "abs_err_refined": safe_float(abs_err_refined),
            "residual_direct": safe_float(residual_direct),
            "residual_refined": safe_float(residual_refined),
            "newton_iters": int(nr.iterations),
            "success": int(nr.success),
            "diverged": int(nr.diverged),
            "used_small_derivative_fallback": int(nr.used_small_derivative_fallback),
            "used_line_search": int(nr.used_line_search),
            "hit_clamp": int(nr.hit_clamp),
            "predict_time_sec": safe_float(t1 - t0),
            "refine_time_sec": safe_float(t2 - t1),
            "total_time_sec": safe_float(t2 - t0),
            "degree": diff_feats["degree"],
            "abs_df_at_root": diff_feats["abs_df_at_root"],
            "linearized_guess": diff_feats["linearized_guess"],
        }
        rows.append(row)

        # hard case criteria
        if (
            residual_refined > 1e-8
            or nr.iterations >= max(8, newton_cfg.max_iters // 2)
            or nr.diverged
            or abs_err_refined > 1e-4
        ):
            hard_cases.append({
                "sample_id": i,
                "method": predictor_name,
                "coeffs": coeffs.tolist(),
                "y_true": y_true,
                "other_root": other,
                "pred_direct": pred_direct,
                "pred_refined": pred_refined,
                "abs_err_direct": abs_err_direct,
                "abs_err_refined": abs_err_refined,
                "residual_direct": residual_direct,
                "residual_refined": residual_refined,
                "newton_iters": nr.iterations,
                "success": nr.success,
                "diverged": nr.diverged,
                "trajectory": nr.trajectory,
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json(hard_path, {"num_hard_cases": len(hard_cases), "hard_cases": hard_cases[:500]})

    abs_err_directs = np.asarray(abs_err_directs)
    abs_err_refineds = np.asarray(abs_err_refineds)
    residual_directs = np.asarray(residual_directs)
    residual_refineds = np.asarray(residual_refineds)
    iterations = np.asarray(iterations)
    timing_predict = np.asarray(timing_predict)
    timing_refine = np.asarray(timing_refine)
    timing_total = np.asarray(timing_total)
    success_flags = np.asarray(success_flags)
    diverged_flags = np.asarray(diverged_flags)

    summary = {
        "method": predictor_name,
        "n_samples": int(len(dataset)),
        "success_rate": float(success_flags.mean()),
        "fail_rate": float(1.0 - success_flags.mean()),
        "diverged_rate": float(diverged_flags.mean()),
        "abs_err_direct": percentile_dict(abs_err_directs),
        "abs_err_refined": percentile_dict(abs_err_refineds),
        "residual_direct": percentile_dict(residual_directs),
        "residual_refined": percentile_dict(residual_refineds),
        "iterations": percentile_dict(iterations),
        "predict_time_sec": percentile_dict(timing_predict),
        "refine_time_sec": percentile_dict(timing_refine),
        "total_time_sec": percentile_dict(timing_total),
        "hard_case_count": int(len(hard_cases)),
    }
    save_json(summary_path, summary)
    return summary


def print_summary_table(title: str, summaries: List[dict]) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    header = f"{'method':<22} {'succ%':>8} {'dir_mae':>14} {'ref_mae':>14} {'iter_mean':>12} {'total_ms':>12} {'p95_iter':>10} {'hard':>8}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        line = (
            f"{s['method']:<22} "
            f"{100*s['success_rate']:>7.2f} "
            f"{s['abs_err_direct']['mean']:>14.6e} "
            f"{s['abs_err_refined']['mean']:>14.6e} "
            f"{s['iterations']['mean']:>12.4f} "
            f"{1000*s['total_time_sec']['mean']:>12.4f} "
            f"{s['iterations']['p95']:>10.2f} "
            f"{s['hard_case_count']:>8d}"
        )
        print(line)
    print("=" * 120)



def evaluate_all(
    test_npz: str,
    save_dir: str,
    coeffs_key: str = "coeffs",
    target_key: str = "root1",
    other_root_key: Optional[str] = "root2_label",
    ood_npz: Optional[str] = None,
    device: str = "cuda",
) -> None:
    save_dir_p = Path(save_dir)
    meta = load_json(save_dir_p / "meta.json")
    test_ds = load_dataset_for_eval(test_npz, save_dir_p, coeffs_key, target_key, other_root_key)

    newton_cfg = NewtonConfig(
        max_iters=20,
        tol_residual=1e-10,
        tol_step=1e-12,
        deriv_eps=1e-12,
        fallback_step=1e-3,
        max_abs_step=5.0,
        clamp_min=None,
        clamp_max=None,
        line_search_alphas=(1.0, 0.5, 0.25, 0.125, 0.0625),
    )

    methods = []
    methods.append(("zero", predictor_by_name("zero"), False))
    methods.append(("train_mean", predictor_by_name("train_mean"), False))
    methods.append(("linearized", predictor_by_name("linearized"), False))
    if test_ds.has_other_root:
        methods.append(("other_root", predictor_by_name("other_root"), False))
    if test_ds.coeffs.shape[1] == 3:
        methods.append(("quadratic_formula", predictor_by_name("quadratic_formula"), False))

    neural = NeuralPredictor(save_dir_p, device=device, mc_dropout_passes=10)
    methods.append(("neural", neural, True))
    hybrid = HybridUncertaintyPredictor(
        neural_ckpt_dir=save_dir_p,
        device=device,
        mc_dropout_passes=10,
        uncertainty_threshold=0.05,
        fallback_name="linearized",
    )
    methods.append(("hybrid_uncertainty", hybrid, False))

    eval_root = save_dir_p / "evaluation"
    ensure_dir(eval_root)

    summaries = []
    for name, pred_obj, use_uncertainty_info in methods:
        print(f"[{now_str()}] Evaluating method={name} on TEST ...")
        summary = evaluate_predictor_on_dataset(
            predictor_name=name,
            predictor_obj=pred_obj,
            dataset=test_ds,
            meta=meta,
            save_subdir=eval_root / "test",
            newton_cfg=newton_cfg,
            uncertainty_info=use_uncertainty_info,
        )
        summaries.append(summary)
    save_json(eval_root / "test_all_summaries.json", {"summaries": summaries})
    print_summary_table("TEST SUMMARY", summaries)

    if ood_npz is not None:
        ood_ds = load_dataset_for_eval(ood_npz, save_dir_p, coeffs_key, target_key, other_root_key)
        ood_summaries = []
        for name, pred_obj, use_uncertainty_info in methods:
            print(f"[{now_str()}] Evaluating method={name} on OOD ...")
            summary = evaluate_predictor_on_dataset(
                predictor_name=name,
                predictor_obj=pred_obj,
                dataset=ood_ds,
                meta=meta,
                save_subdir=eval_root / "ood",
                newton_cfg=newton_cfg,
                uncertainty_info=use_uncertainty_info,
            )
            ood_summaries.append(summary)
        save_json(eval_root / "ood_all_summaries.json", {"summaries": ood_summaries})
        print_summary_table("OOD SUMMARY", ood_summaries)


# =========================
# Argument parser
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural init guess + safeguarded Newton benchmark framework")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train_npz", type=str, required=True)
    p_train.add_argument("--val_npz", type=str, required=True)
    p_train.add_argument("--test_npz", type=str, default=None)
    p_train.add_argument("--save_dir", type=str, required=True)
    p_train.add_argument("--coeffs_key", type=str, default="coeffs")
    p_train.add_argument("--target_key", type=str, default="root1")
    p_train.add_argument("--other_root_key", type=str, default="root2_label")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=3000)
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-5)
    p_train.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 128])
    p_train.add_argument("--dropout", type=float, default=0.05)
    p_train.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "silu", "tanh"])
    p_train.add_argument("--patience", type=int, default=200)
    p_train.add_argument("--mode", type=str, default="residual", choices=["direct", "residual"])
    p_train.add_argument("--residual_anchor", type=str, default="linearized", choices=["zero", "train_mean", "linearized", "other_root"])
    p_train.add_argument("--hard_weight_alpha", type=float, default=1.0)
    p_train.add_argument("--residual_loss_lambda", type=float, default=0.0)
    p_train.add_argument("--grad_clip", type=float, default=5.0)
    p_train.add_argument("--num_workers", type=int, default=0)
    p_train.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--test_npz", type=str, required=True)
    p_eval.add_argument("--ood_npz", type=str, default=None)
    p_eval.add_argument("--save_dir", type=str, required=True)
    p_eval.add_argument("--coeffs_key", type=str, default="coeffs")
    p_eval.add_argument("--target_key", type=str, default="root1")
    p_eval.add_argument("--other_root_key", type=str, default="root2_label")
    p_eval.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return p


# =========================
# Main
# =========================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainConfig(
            train_npz=args.train_npz,
            val_npz=args.val_npz,
            test_npz=args.test_npz,
            save_dir=args.save_dir,
            coeffs_key=args.coeffs_key,
            target_key=args.target_key,
            other_root_key=args.other_root_key,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            activation=args.activation,
            patience=args.patience,
            mode=args.mode,
            residual_anchor=args.residual_anchor,
            hard_weight_alpha=args.hard_weight_alpha,
            residual_loss_lambda=args.residual_loss_lambda,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            device=args.device,
        )
        train_model(cfg)

    elif args.command == "eval":
        evaluate_all(
            test_npz=args.test_npz,
            save_dir=args.save_dir,
            coeffs_key=args.coeffs_key,
            target_key=args.target_key,
            other_root_key=args.other_root_key,
            ood_npz=args.ood_npz,
            device=args.device,
        )


if __name__ == "__main__":
    main()
"""
python root_benchmark_framework.py train \
   --train_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_train.npz \
   --val_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_val.npz \
   --test_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_test.npz \
   --save_dir /home/seokjun/math_03_14/save_path \
   --coeffs_key coeffs \
   --target_key root \
   --other_root_key root2_label \
   --mode residual \
   --epochs 3000 \
   --batch_size 512 \
   --hidden_dims 256 256 128 \
   --lr 1e-3 \
   --patience 200

python root_benchmark_framework.py eval \
  --test_npz /home/seokjun/math_03_14/colebrook_data/colebrook_deg25_test.npz \
  --save_dir /home/seokjun/math_03_14/colebrook_data/save_path \
  --coeffs_key coeffs \
  --target_key root \
  --other_root_key root2_label
"""