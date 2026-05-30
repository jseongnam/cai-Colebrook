#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colebrook_like_dataset.py

방정식:
    F(x; a, b) = x + 2*log10(a + b*x) = 0,   with a > 0, b > 0

이 스크립트는 위 방정식의 데이터셋을 생성한다.
핵심 아이디어:
- a > 0, b > 0 이면 정의역은 x > -a/b
- F'(x) = 1 + 2b / ((a + bx) ln 10) > 0
  -> F는 정의역에서 단조증가
- x -> (-a/b)^+ 일 때 F(x) -> -∞
- x -> +∞ 일 때 F(x) -> +∞
  -> 해는 항상 유일하게 존재

저장 내용:
- a, b
- root x*
- center x0
- Taylor coefficients c0..c_deg at x0
- residual at root
- optional train/val/test split

예시:
    python colebrook_like_dataset.py \
        --n_samples 30000 \
        --degree 10 \
        --center_mode zero \
        --a_min 1e-3 --a_max 1e3 \
        --b_min 1e-3 --b_max 1e2 \
        --out_dir ./colebrook_data

출력:
- out_dir/colebrook_deg{degree}_all.npz
- out_dir/colebrook_deg{degree}_train.npz
- out_dir/colebrook_deg{degree}_val.npz
- out_dir/colebrook_deg{degree}_test.npz
- out_dir/summary.txt
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


LN10 = math.log(10.0)


def F(x: float, a: float, b: float) -> float:
    return x + 2.0 * math.log10(a + b * x)


def brentq(f, lo: float, hi: float, args=(), max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    scipy 없이 쓰는 간단한 Brent/bisection 혼합형 root finder.
    bracket [lo, hi] 내에서 f(lo)*f(hi) < 0 가정.
    """
    flo = f(lo, *args)
    fhi = f(hi, *args)

    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise ValueError("Non-finite function value at bracket endpoints.")
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("Root is not bracketed.")

    a, b = lo, hi
    fa, fb = flo, fhi
    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if fb == 0.0:
            return b

        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a

        if abs(fa) < abs(fb):
            c, b, a = b, a, b
            fc, fb, fa = fb, fa, fb

        tol1 = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        xm = 0.5 * (a - b)

        if abs(xm) <= tol1 or fb == 0.0:
            return b

        if abs(e) >= tol1 and abs(fc) > abs(fb):
            s = fb / fc
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q_ = fc / fa
                r = fb / fa
                p = s * (2.0 * xm * q_ * (q_ - r) - (b - c) * (r - 1.0))
                q = (q_ - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            p = abs(p)

            min1 = 3.0 * xm * q - abs(tol1 * q)
            min2 = abs(e * q)

            if 2.0 * p < min(min1, min2):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        c, fc = b, fb
        if abs(d) > tol1:
            b = b + d
        else:
            b = b + tol1 if xm >= 0 else b - tol1
        fb = f(b, *args)

    return b


def solve_unique_root(a: float, b: float, tol: float = 1e-12) -> float:
    """
    a>0, b>0인 경우 F는 정의역 x > -a/b 에서 단조증가하고 유일근 존재.
    """
    if not (a > 0 and b > 0):
        raise ValueError("a and b must be positive.")

    # left endpoint 바로 오른쪽
    domain_left = -a / b
    eps = max(1e-12, 1e-12 * (1.0 + abs(domain_left)))
    lo = domain_left + eps

    # 충분히 큰 high를 찾기 위한 exponential search
    hi = max(1.0, abs(domain_left) + 1.0)
    fhi = F(hi, a, b)
    grow_steps = 0
    while fhi <= 0.0:
        hi = 2.0 * hi + 1.0
        fhi = F(hi, a, b)
        grow_steps += 1
        if grow_steps > 200:
            raise RuntimeError(f"Failed to find upper bracket for a={a}, b={b}")

    return brentq(F, lo, hi, args=(a, b), tol=tol)


def taylor_coeffs_at_x0(a: float, b: float, x0: float, degree: int) -> np.ndarray:
    """
    F(x) = x + 2 log10(a + bx)
    Taylor coefficients c_n around x0:
        F(x) = sum_{n=0}^degree c_n (x-x0)^n + ...
    """
    if a + b * x0 <= 0:
        raise ValueError("x0 is outside the log domain.")

    coeffs = np.zeros(degree + 1, dtype=np.float64)
    u = a + b * x0

    # n = 0
    coeffs[0] = x0 + 2.0 * math.log10(u)

    # n = 1
    coeffs[1] = 1.0 + 2.0 * b / (u * LN10)

    # n >= 2
    # d^n/dx^n [2 log10(a+bx)] = 2/ln(10) * (-1)^(n-1) * (n-1)! * b^n / (a+bx)^n
    # Taylor coefficient c_n = F^(n)(x0) / n!
    #                       = 2/ln(10) * (-1)^(n-1) * b^n / (n * u^n)
    for n in range(2, degree + 1):
        coeffs[n] = (2.0 / LN10) * ((-1) ** (n - 1)) * (b ** n) / (n * (u ** n))

    return coeffs


def sample_log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    if not (low > 0 and high > 0 and high > low):
        raise ValueError("log-uniform range must satisfy 0 < low < high.")
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def choose_center(
    rng: np.random.Generator,
    center_mode: str,
    a: float,
    b: float,
    root: float,
    root_margin_scale: float,
) -> float:
    """
    center_mode:
    - zero: always x0 = 0 (a>0이면 항상 정의역 안)
    - root: x0 = root
    - near_root: root 주변 랜덤
    - domain_random: 정의역 안에서 root 근방 랜덤
    """
    if center_mode == "zero":
        return 0.0

    if center_mode == "root":
        return root

    if center_mode == "near_root":
        # root 주변에서 약간 이동
        radius = root_margin_scale * max(1.0, abs(root))
        x0 = root + rng.uniform(-radius, radius)
        left = -a / b + 1e-10
        return max(x0, left)

    if center_mode == "domain_random":
        left = -a / b + 1e-8
        low = max(left, root - root_margin_scale * max(1.0, abs(root)))
        high = root + root_margin_scale * max(1.0, abs(root))
        if low >= high:
            return root
        return rng.uniform(low, high)

    raise ValueError(f"Unknown center_mode: {center_mode}")


@dataclass
class DatasetArrays:
    a: np.ndarray
    b: np.ndarray
    root: np.ndarray
    center: np.ndarray
    residual: np.ndarray
    coeffs: np.ndarray


def build_dataset(
    n_samples: int,
    degree: int,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    center_mode: str,
    root_margin_scale: float,
    seed: int,
) -> DatasetArrays:
    rng = np.random.default_rng(seed)

    a_vals = sample_log_uniform(rng, a_min, a_max, n_samples)
    b_vals = sample_log_uniform(rng, b_min, b_max, n_samples)

    roots = np.zeros(n_samples, dtype=np.float64)
    centers = np.zeros(n_samples, dtype=np.float64)
    residuals = np.zeros(n_samples, dtype=np.float64)
    coeffs = np.zeros((n_samples, degree + 1), dtype=np.float64)

    for i in range(n_samples):
        a = float(a_vals[i])
        b = float(b_vals[i])

        root = solve_unique_root(a, b)
        x0 = choose_center(rng, center_mode, a, b, root, root_margin_scale)
        c = taylor_coeffs_at_x0(a, b, x0, degree)
        r = abs(F(root, a, b))

        roots[i] = root
        centers[i] = x0
        residuals[i] = r
        coeffs[i] = c

    return DatasetArrays(
        a=a_vals,
        b=b_vals,
        root=roots,
        center=centers,
        residual=residuals,
        coeffs=coeffs,
    )


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_npz(path: Path, arrays: DatasetArrays, idx: np.ndarray) -> None:
    np.savez_compressed(
        path,
        a=arrays.a[idx],
        b=arrays.b[idx],
        root=arrays.root[idx],
        center=arrays.center[idx],
        residual=arrays.residual[idx],
        coeffs=arrays.coeffs[idx],
    )


def write_summary(
    path: Path,
    arrays: DatasetArrays,
    splits: Dict[str, np.ndarray],
    degree: int,
    args: argparse.Namespace,
) -> None:
    roots = arrays.root
    residuals = arrays.residual
    centers = arrays.center
    coeffs = arrays.coeffs

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Colebrook-like dataset summary ===\n")
        f.write(f"Equation: x + 2*log10(a + b*x) = 0\n")
        f.write(f"Condition: a > 0, b > 0, domain a + b*x > 0\n\n")

        f.write("[Generation args]\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("\n[Dataset stats]\n")
        f.write(f"n_samples: {len(roots)}\n")
        f.write(f"degree: {degree}\n")
        f.write(f"coeffs shape: {coeffs.shape}\n")
        f.write(f"root min/max/mean: {roots.min():.8f} / {roots.max():.8f} / {roots.mean():.8f}\n")
        f.write(f"center min/max/mean: {centers.min():.8f} / {centers.max():.8f} / {centers.mean():.8f}\n")
        f.write(
            "residual mean/max: "
            f"{residuals.mean():.3e} / {residuals.max():.3e}\n"
        )

        f.write("\n[Parameter stats]\n")
        f.write(f"a min/max/mean: {arrays.a.min():.8e} / {arrays.a.max():.8e} / {arrays.a.mean():.8e}\n")
        f.write(f"b min/max/mean: {arrays.b.min():.8e} / {arrays.b.max():.8e} / {arrays.b.mean():.8e}\n")

        f.write("\n[Splits]\n")
        for k, idx in splits.items():
            f.write(f"{k}: {len(idx)}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=30000)
    p.add_argument("--degree", type=int, default=10)
    p.add_argument("--a_min", type=float, default=1e-3)
    p.add_argument("--a_max", type=float, default=1e3)
    p.add_argument("--b_min", type=float, default=1e-3)
    p.add_argument("--b_max", type=float, default=1e2)
    p.add_argument("--center_mode", type=str, default="zero",
                   choices=["zero", "root", "near_root", "domain_random"])
    p.add_argument("--root_margin_scale", type=float, default=0.5)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./colebrook_data")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_dataset(
        n_samples=args.n_samples,
        degree=args.degree,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        center_mode=args.center_mode,
        root_margin_scale=args.root_margin_scale,
        seed=args.seed,
    )

    splits = split_indices(
        n=len(arrays.root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    base = f"colebrook_deg{args.degree}"
    save_npz(out_dir / f"{base}_all.npz", arrays, np.arange(len(arrays.root)))
    save_npz(out_dir / f"{base}_train.npz", arrays, splits["train"])
    save_npz(out_dir / f"{base}_val.npz", arrays, splits["val"])
    save_npz(out_dir / f"{base}_test.npz", arrays, splits["test"])
    write_summary(out_dir / "summary.txt", arrays, splits, args.degree, args)

    print(f"[DONE] Saved dataset to: {out_dir.resolve()}")
    print(f"  - {base}_all.npz")
    print(f"  - {base}_train.npz")
    print(f"  - {base}_val.npz")
    print(f"  - {base}_test.npz")
    print(f"  - summary.txt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colebrook_like_dataset.py

방정식:
    F(x; a, b) = x + 2*log10(a + b*x) = 0,   with a > 0, b > 0

이 스크립트는 위 방정식의 데이터셋을 생성한다.
핵심 아이디어:
- a > 0, b > 0 이면 정의역은 x > -a/b
- F'(x) = 1 + 2b / ((a + bx) ln 10) > 0
  -> F는 정의역에서 단조증가
- x -> (-a/b)^+ 일 때 F(x) -> -∞
- x -> +∞ 일 때 F(x) -> +∞
  -> 해는 항상 유일하게 존재

저장 내용:
- a, b
- root x*
- center x0
- Taylor coefficients c0..c_deg at x0
- residual at root
- optional train/val/test split

예시:
    python colebrook_like_dataset.py \
        --n_samples 30000 \
        --degree 10 \
        --center_mode zero \
        --a_min 1e-3 --a_max 1e3 \
        --b_min 1e-3 --b_max 1e2 \
        --out_dir ./colebrook_data

출력:
- out_dir/colebrook_deg{degree}_all.npz
- out_dir/colebrook_deg{degree}_train.npz
- out_dir/colebrook_deg{degree}_val.npz
- out_dir/colebrook_deg{degree}_test.npz
- out_dir/summary.txt
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


LN10 = math.log(10.0)


def F(x: float, a: float, b: float) -> float:
    return x + 2.0 * math.log10(a + b * x)


def brentq(f, lo: float, hi: float, args=(), max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    scipy 없이 쓰는 간단한 Brent/bisection 혼합형 root finder.
    bracket [lo, hi] 내에서 f(lo)*f(hi) < 0 가정.
    """
    flo = f(lo, *args)
    fhi = f(hi, *args)

    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise ValueError("Non-finite function value at bracket endpoints.")
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("Root is not bracketed.")

    a, b = lo, hi
    fa, fb = flo, fhi
    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if fb == 0.0:
            return b

        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a

        if abs(fa) < abs(fb):
            c, b, a = b, a, b
            fc, fb, fa = fb, fa, fb

        tol1 = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        xm = 0.5 * (a - b)

        if abs(xm) <= tol1 or fb == 0.0:
            return b

        if abs(e) >= tol1 and abs(fc) > abs(fb):
            s = fb / fc
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q_ = fc / fa
                r = fb / fa
                p = s * (2.0 * xm * q_ * (q_ - r) - (b - c) * (r - 1.0))
                q = (q_ - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            p = abs(p)

            min1 = 3.0 * xm * q - abs(tol1 * q)
            min2 = abs(e * q)

            if 2.0 * p < min(min1, min2):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        c, fc = b, fb
        if abs(d) > tol1:
            b = b + d
        else:
            b = b + tol1 if xm >= 0 else b - tol1
        fb = f(b, *args)

    return b


def solve_unique_root(a: float, b: float, tol: float = 1e-12) -> float:
    """
    a>0, b>0인 경우 F는 정의역 x > -a/b 에서 단조증가하고 유일근 존재.
    """
    if not (a > 0 and b > 0):
        raise ValueError("a and b must be positive.")

    # left endpoint 바로 오른쪽
    domain_left = -a / b
    eps = max(1e-12, 1e-12 * (1.0 + abs(domain_left)))
    lo = domain_left + eps

    # 충분히 큰 high를 찾기 위한 exponential search
    hi = max(1.0, abs(domain_left) + 1.0)
    fhi = F(hi, a, b)
    grow_steps = 0
    while fhi <= 0.0:
        hi = 2.0 * hi + 1.0
        fhi = F(hi, a, b)
        grow_steps += 1
        if grow_steps > 200:
            raise RuntimeError(f"Failed to find upper bracket for a={a}, b={b}")

    return brentq(F, lo, hi, args=(a, b), tol=tol)


def taylor_coeffs_at_x0(a: float, b: float, x0: float, degree: int) -> np.ndarray:
    """
    F(x) = x + 2 log10(a + bx)
    Taylor coefficients c_n around x0:
        F(x) = sum_{n=0}^degree c_n (x-x0)^n + ...
    """
    if a + b * x0 <= 0:
        raise ValueError("x0 is outside the log domain.")

    coeffs = np.zeros(degree + 1, dtype=np.float64)
    u = a + b * x0

    # n = 0
    coeffs[0] = x0 + 2.0 * math.log10(u)

    # n = 1
    coeffs[1] = 1.0 + 2.0 * b / (u * LN10)

    # n >= 2
    # d^n/dx^n [2 log10(a+bx)] = 2/ln(10) * (-1)^(n-1) * (n-1)! * b^n / (a+bx)^n
    # Taylor coefficient c_n = F^(n)(x0) / n!
    #                       = 2/ln(10) * (-1)^(n-1) * b^n / (n * u^n)
    for n in range(2, degree + 1):
        coeffs[n] = (2.0 / LN10) * ((-1) ** (n - 1)) * (b ** n) / (n * (u ** n))

    return coeffs


def sample_log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    if not (low > 0 and high > 0 and high > low):
        raise ValueError("log-uniform range must satisfy 0 < low < high.")
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def choose_center(
    rng: np.random.Generator,
    center_mode: str,
    a: float,
    b: float,
    root: float,
    root_margin_scale: float,
) -> float:
    """
    center_mode:
    - zero: always x0 = 0 (a>0이면 항상 정의역 안)
    - root: x0 = root
    - near_root: root 주변 랜덤
    - domain_random: 정의역 안에서 root 근방 랜덤
    """
    if center_mode == "zero":
        return 0.0

    if center_mode == "root":
        return root

    if center_mode == "near_root":
        # root 주변에서 약간 이동
        radius = root_margin_scale * max(1.0, abs(root))
        x0 = root + rng.uniform(-radius, radius)
        left = -a / b + 1e-10
        return max(x0, left)

    if center_mode == "domain_random":
        left = -a / b + 1e-8
        low = max(left, root - root_margin_scale * max(1.0, abs(root)))
        high = root + root_margin_scale * max(1.0, abs(root))
        if low >= high:
            return root
        return rng.uniform(low, high)

    raise ValueError(f"Unknown center_mode: {center_mode}")


@dataclass
class DatasetArrays:
    a: np.ndarray
    b: np.ndarray
    root: np.ndarray
    center: np.ndarray
    residual: np.ndarray
    coeffs: np.ndarray


def build_dataset(
    n_samples: int,
    degree: int,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    center_mode: str,
    root_margin_scale: float,
    seed: int,
) -> DatasetArrays:
    rng = np.random.default_rng(seed)

    a_vals = sample_log_uniform(rng, a_min, a_max, n_samples)
    b_vals = sample_log_uniform(rng, b_min, b_max, n_samples)

    roots = np.zeros(n_samples, dtype=np.float64)
    centers = np.zeros(n_samples, dtype=np.float64)
    residuals = np.zeros(n_samples, dtype=np.float64)
    coeffs = np.zeros((n_samples, degree + 1), dtype=np.float64)

    for i in range(n_samples):
        a = float(a_vals[i])
        b = float(b_vals[i])

        root = solve_unique_root(a, b)
        x0 = choose_center(rng, center_mode, a, b, root, root_margin_scale)
        c = taylor_coeffs_at_x0(a, b, x0, degree)
        r = abs(F(root, a, b))

        roots[i] = root
        centers[i] = x0
        residuals[i] = r
        coeffs[i] = c

    return DatasetArrays(
        a=a_vals,
        b=b_vals,
        root=roots,
        center=centers,
        residual=residuals,
        coeffs=coeffs,
    )


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_npz(path: Path, arrays: DatasetArrays, idx: np.ndarray) -> None:
    np.savez_compressed(
        path,
        a=arrays.a[idx],
        b=arrays.b[idx],
        root=arrays.root[idx],
        center=arrays.center[idx],
        residual=arrays.residual[idx],
        coeffs=arrays.coeffs[idx],
    )


def write_summary(
    path: Path,
    arrays: DatasetArrays,
    splits: Dict[str, np.ndarray],
    degree: int,
    args: argparse.Namespace,
) -> None:
    roots = arrays.root
    residuals = arrays.residual
    centers = arrays.center
    coeffs = arrays.coeffs

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Colebrook-like dataset summary ===\n")
        f.write(f"Equation: x + 2*log10(a + b*x) = 0\n")
        f.write(f"Condition: a > 0, b > 0, domain a + b*x > 0\n\n")

        f.write("[Generation args]\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("\n[Dataset stats]\n")
        f.write(f"n_samples: {len(roots)}\n")
        f.write(f"degree: {degree}\n")
        f.write(f"coeffs shape: {coeffs.shape}\n")
        f.write(f"root min/max/mean: {roots.min():.8f} / {roots.max():.8f} / {roots.mean():.8f}\n")
        f.write(f"center min/max/mean: {centers.min():.8f} / {centers.max():.8f} / {centers.mean():.8f}\n")
        f.write(
            "residual mean/max: "
            f"{residuals.mean():.3e} / {residuals.max():.3e}\n"
        )

        f.write("\n[Parameter stats]\n")
        f.write(f"a min/max/mean: {arrays.a.min():.8e} / {arrays.a.max():.8e} / {arrays.a.mean():.8e}\n")
        f.write(f"b min/max/mean: {arrays.b.min():.8e} / {arrays.b.max():.8e} / {arrays.b.mean():.8e}\n")

        f.write("\n[Splits]\n")
        for k, idx in splits.items():
            f.write(f"{k}: {len(idx)}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=30000)
    p.add_argument("--degree", type=int, default=10)
    p.add_argument("--a_min", type=float, default=1e-3)
    p.add_argument("--a_max", type=float, default=1e3)
    p.add_argument("--b_min", type=float, default=1e-3)
    p.add_argument("--b_max", type=float, default=1e2)
    p.add_argument("--center_mode", type=str, default="zero",
                   choices=["zero", "root", "near_root", "domain_random"])
    p.add_argument("--root_margin_scale", type=float, default=0.5)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./colebrook_data")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_dataset(
        n_samples=args.n_samples,
        degree=args.degree,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        center_mode=args.center_mode,
        root_margin_scale=args.root_margin_scale,
        seed=args.seed,
    )

    splits = split_indices(
        n=len(arrays.root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    base = f"colebrook_deg{args.degree}"
    save_npz(out_dir / f"{base}_all.npz", arrays, np.arange(len(arrays.root)))
    save_npz(out_dir / f"{base}_train.npz", arrays, splits["train"])
    save_npz(out_dir / f"{base}_val.npz", arrays, splits["val"])
    save_npz(out_dir / f"{base}_test.npz", arrays, splits["test"])
    write_summary(out_dir / "summary.txt", arrays, splits, args.degree, args)

    print(f"[DONE] Saved dataset to: {out_dir.resolve()}")
    print(f"  - {base}_all.npz")
    print(f"  - {base}_train.npz")
    print(f"  - {base}_val.npz")
    print(f"  - {base}_test.npz")
    print(f"  - summary.txt")


if __name__ == "__main__":
    main()
