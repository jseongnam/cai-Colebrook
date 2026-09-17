from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

PI = np.pi
LN10 = np.log(10.0)


@dataclass(frozen=True)
class NewtonResult:
    q: np.ndarray
    x: np.ndarray
    iterations: int
    converged: bool
    residual_inf: float


def reynolds(q: np.ndarray, rho: float, mu: float, diameter: np.ndarray) -> np.ndarray:
    return 4.0 * rho * np.asarray(q) / (PI * mu * np.asarray(diameter))


def colebrook_residual(x: np.ndarray, re: np.ndarray, rel_rough: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    re = np.asarray(re, dtype=np.float64)
    arg = np.asarray(rel_rough, dtype=np.float64) / 3.7 + 2.51 * x / re
    return x + 2.0 * np.log10(np.maximum(arg, 1e-300))


def haaland_x(re: np.ndarray, rel_rough: np.ndarray) -> np.ndarray:
    """Return x=1/sqrt(f) from the Haaland explicit approximation."""
    re = np.maximum(np.asarray(re, dtype=np.float64), 4_000.0)
    rr = np.maximum(np.asarray(rel_rough, dtype=np.float64), 0.0)
    return -1.8 * np.log10((rr / 3.7) ** 1.11 + 6.9 / re)


def head_loss(q: np.ndarray, x: np.ndarray, length: np.ndarray,
              diameter: np.ndarray, g: float) -> np.ndarray:
    return 8.0 * np.asarray(length) * np.asarray(q) ** 2 / (
        g * PI**2 * np.asarray(diameter) ** 5 * np.asarray(x) ** 2
    )


def fractions_to_alr(r: np.ndarray) -> np.ndarray:
    """Additive log-ratio coordinates using the last branch as reference."""
    r = np.clip(np.asarray(r, dtype=np.float64), 1e-10, 1.0)
    r = r / r.sum()
    return np.log(r[:-1] / r[-1])


def alr_to_fractions(alr: np.ndarray) -> np.ndarray:
    logits = np.concatenate([np.asarray(alr, dtype=np.float64), np.zeros(1)])
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def pack_state(q: np.ndarray, x: np.ndarray, q_total: float) -> np.ndarray:
    return np.concatenate([fractions_to_alr(np.asarray(q) / q_total), np.asarray(x)])


def unpack_state(state: np.ndarray, q_total: float, branches: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    r = alr_to_fractions(state[: branches - 1])
    x = np.maximum(state[branches - 1 :], 1e-3)
    return q_total * r, x


def baseline_initializer(params: Mapping[str, object], iterations: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Conductance/Haaland initializer; uses no target information."""
    qt = float(params["Q_total"])
    d = np.asarray(params["D"], dtype=np.float64)
    eps = np.asarray(params["eps"], dtype=np.float64)
    length = np.asarray(params["L"], dtype=np.float64)
    rho, mu = float(params["rho"]), float(params["mu"])
    weights = d**2.5 / np.sqrt(length)
    q = qt * weights / weights.sum()
    for _ in range(iterations):
        re = reynolds(q, rho, mu, d)
        x = haaland_x(re, eps / d)
        weights = x * d**2.5 / np.sqrt(length)
        q = qt * weights / weights.sum()
    re = reynolds(q, rho, mu, d)
    return q, haaland_x(re, eps / d)


def physical_residual(q: np.ndarray, x: np.ndarray,
                      params: Mapping[str, object]) -> np.ndarray:
    d = np.asarray(params["D"], dtype=np.float64)
    eps = np.asarray(params["eps"], dtype=np.float64)
    length = np.asarray(params["L"], dtype=np.float64)
    rho, mu, g = float(params["rho"]), float(params["mu"]), float(params["g"])
    cb = colebrook_residual(x, reynolds(q, rho, mu, d), eps / d)
    h = head_loss(q, x, length, d, g)
    return np.concatenate([cb, h[:-1] - h[-1]])


def residual_scale(params: Mapping[str, object], q0: np.ndarray, x0: np.ndarray) -> np.ndarray:
    b = len(q0)
    h = head_loss(q0, x0, np.asarray(params["L"]), np.asarray(params["D"]), float(params["g"]))
    h_scale = max(float(np.max(np.abs(h))), 1e-6)
    return np.concatenate([np.ones(b), np.full(b - 1, h_scale)])


def scaled_state_residual(state: np.ndarray, params: Mapping[str, object],
                          scale: np.ndarray) -> np.ndarray:
    q, x = unpack_state(state, float(params["Q_total"]), len(np.asarray(params["D"])))
    return physical_residual(q, x, params) / scale


def solve_reference(params: Mapping[str, object], tol: float = 1e-12) -> NewtonResult:
    """High-accuracy trust-region solve used only to create labels."""
    q0, x0 = baseline_initializer(params)
    b = len(q0)
    s0 = pack_state(q0, x0, float(params["Q_total"]))
    scale = residual_scale(params, q0, x0)
    lower = np.concatenate([np.full(b - 1, -25.0), np.full(b, 1e-3)])
    upper = np.concatenate([np.full(b - 1, 25.0), np.full(b, 100.0)])
    sol = least_squares(
        scaled_state_residual, s0, args=(params, scale), bounds=(lower, upper),
        xtol=tol, ftol=tol, gtol=tol, max_nfev=500, method="trf",
    )
    q, x = unpack_state(sol.x, float(params["Q_total"]), b)
    res = float(np.max(np.abs(scaled_state_residual(sol.x, params, scale))))
    return NewtonResult(q, x, int(sol.nfev), bool(sol.success and res < 1e-8), res)


def numerical_jacobian(fun, state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    f0 = fun(state)
    jac = np.empty((f0.size, state.size), dtype=np.float64)
    for j in range(state.size):
        step = 1e-6 * max(1.0, abs(float(state[j])))
        plus, minus = state.copy(), state.copy()
        plus[j] += step
        minus[j] -= step
        jac[:, j] = (fun(plus) - fun(minus)) / (2.0 * step)
    return jac


def newton_refine(q_init: np.ndarray, x_init: np.ndarray,
                  params: Mapping[str, object], tol: float = 1e-10,
                  max_iter: int = 30) -> NewtonResult:
    """Damped Newton in ALR coordinates; every flow remains positive and sums to QT."""
    b = len(q_init)
    state = pack_state(q_init, x_init, float(params["Q_total"]))
    q0, x0 = baseline_initializer(params)
    scale = residual_scale(params, q0, x0)
    fun = lambda s: scaled_state_residual(s, params, scale)
    used = 0
    converged = False
    for k in range(1, max_iter + 1):
        f = fun(state)
        norm = float(np.max(np.abs(f)))
        if norm <= tol:
            converged = True
            break
        jac = numerical_jacobian(fun, state)
        try:
            step = np.linalg.solve(jac, f)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, f, rcond=None)[0]
        step = np.clip(step, -5.0, 5.0)
        accepted = False
        alpha = 1.0
        for _ in range(12):
            candidate = state - alpha * step
            candidate[b - 1 :] = np.clip(candidate[b - 1 :], 1e-3, 100.0)
            if np.max(np.abs(fun(candidate))) < norm:
                state = candidate
                accepted = True
                break
            alpha *= 0.5
        used = k
        if not accepted:
            break
    final = float(np.max(np.abs(fun(state))))
    converged = converged or final <= tol
    q, x = unpack_state(state, float(params["Q_total"]), b)
    return NewtonResult(q, x, used, converged, final)


def minimum_reynolds(q: np.ndarray, params: Mapping[str, object]) -> float:
    return float(np.min(reynolds(q, float(params["rho"]), float(params["mu"]), np.asarray(params["D"]))))
