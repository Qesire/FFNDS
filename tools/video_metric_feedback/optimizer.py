"""Small, deterministic video-metric feedback optimizer.

The evaluator owns rendering, metric definitions, fixed source/seed pairing, and
persistent result caching.  This module only requests coefficient vectors.  It
uses coordinate finite differences, a convex diagonal quadratic model inside
an L-infinity trust region, and acceptance on an actually evaluated candidate.
It never imports or runs a video model.

``loss`` is minimized.  Every ``guard_margin`` must be nonnegative.  Guards
must refer to a fixed initial reference, rather than being reset after each
accepted step.  ``unit_losses`` identify independent image/source clusters;
their keys and the guard keys must remain identical throughout an optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Evaluation:
    loss: float
    unit_losses: Mapping[str, float]
    guard_margins: Mapping[str, float]
    fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerConfig:
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]
    finite_difference_step: float | Sequence[float] = 0.1
    trust_radius: float | Sequence[float] = 0.25
    max_iterations: int = 1
    max_evaluations: int = 6
    damping: float = 1.0
    min_improvement: float = 0.0
    acceptance_ratio: float = 0.1
    prior_strength: float = 0.0
    shrink_factor: float = 0.5
    projection_tol: float = 1e-9
    projection_max_iterations: int = 10000
    flat_tolerance: float = 1e-12


@dataclass
class OptimizationResult:
    theta: list[float]
    evaluation: Evaluation
    history: list[dict[str, Any]]
    evaluations_used: int
    status: str
    accepted_steps: int = 0


def _vector(value: Any, size: int, name: str, *, broadcast: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if broadcast and array.ndim == 0:
        array = np.full(size, float(array))
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


def _validate_evaluation(value: Evaluation) -> Evaluation:
    if not isinstance(value, Evaluation):
        raise TypeError("evaluate(theta) must return Evaluation")
    if not np.isfinite(value.loss):
        raise ValueError("evaluation loss must be finite")
    if not value.unit_losses:
        raise ValueError("unit_losses must identify at least one source cluster")
    for label, values in (("unit_losses", value.unit_losses), ("guard_margins", value.guard_margins)):
        if any(not isinstance(key, str) or not key for key in values):
            raise ValueError(f"{label} keys must be nonempty strings")
        if any(not np.isfinite(number) for number in values.values()):
            raise ValueError(f"{label} values must be finite")
    if value.fingerprint is not None and not isinstance(value.fingerprint, str):
        raise ValueError("fingerprint must be a string or None")
    return Evaluation(
        float(value.loss),
        {key: float(number) for key, number in value.unit_losses.items()},
        {key: float(number) for key, number in value.guard_margins.items()},
        value.fingerprint,
        dict(value.metadata),
    )


def _finite_difference(f0: float, fm: float, fp: float, minus: float, plus: float) -> tuple[float, float]:
    """Derivative/curvature at the center using actual, possibly unequal gaps.

    An unequal three-point stencil is exact for a quadratic.  At a box boundary
    only one side exists, so curvature is unknown and the slope is one-sided.
    """
    if minus > 0.0 and plus > 0.0:
        forward = (fp - f0) / plus
        backward = (f0 - fm) / minus
        return (
            (minus * forward + plus * backward) / (minus + plus),
            2.0 * (forward - backward) / (minus + plus),
        )
    if plus > 0.0:
        return (fp - f0) / plus, 0.0
    if minus > 0.0:
        return (f0 - fm) / minus, 0.0
    return 0.0, 0.0


def _project_quadratic(
    gradient: np.ndarray,
    diagonal: np.ndarray,
    lower_step: np.ndarray,
    upper_step: np.ndarray,
    guard_jacobian: np.ndarray,
    guard_margins: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray | None, int]:
    """Minimize g.d + .5 d.H.d over a box and linearized guards.

    Whitening by sqrt(H) turns this into Euclidean projection onto halfspaces.
    Dykstra's corrections, unlike naive alternating projections, recover the
    projection of the original point.  An uncertified iterate is not returned.
    """
    size = gradient.size
    matrix = np.vstack((np.eye(size), -np.eye(size), -guard_jacobian))
    bounds = np.concatenate((upper_step, -lower_step, guard_margins))
    if np.any(bounds < 0.0):
        return None, 0  # The required feasible starting point d=0 is absent.
    root = np.sqrt(diagonal)
    whitened = matrix / root[None, :]
    norms = np.linalg.norm(whitened, axis=1)
    nonzero = norms > 0.0
    normals = whitened[nonzero] / norms[nonzero, None]
    levels = bounds[nonzero] / norms[nonzero]
    point = -gradient / root
    corrections = np.zeros_like(normals)
    converged = False
    for iteration in range(1, max_iterations + 1):
        previous = point.copy()
        for index, normal in enumerate(normals):
            shifted = point + corrections[index]
            excess = max(float(normal @ shifted - levels[index]), 0.0)
            point = shifted - excess * normal
            corrections[index] = shifted - point
        if not np.all(np.isfinite(point)):
            return None, iteration
        displacement = np.max(np.abs(point - previous))
        violation = np.max(normals @ point - levels, initial=0.0)
        if displacement <= tolerance and violation <= tolerance:
            converged = True
            break
    if not converged:
        return None, max_iterations
    step = np.clip(point / root, lower_step, upper_step)
    if np.max(matrix @ step - bounds, initial=0.0) > tolerance:
        return None, iteration
    # Move into the feasible segment from d=0.  This removes roundoff at an
    # interior guard boundary and keeps strict real-score acceptance meaningful.
    activity = matrix @ step
    positive = activity > 0.0
    factor = min(1.0, float(np.min(bounds[positive] / activity[positive], initial=1.0)))
    step *= max(factor, 0.0) * (1.0 - 1e-10)
    if np.any(matrix @ step - bounds > tolerance):
        return None, iteration
    return step, iteration


def _normalize_config(initial_theta: Sequence[float], config: OptimizerConfig) -> tuple[np.ndarray, ...]:
    theta = np.asarray(initial_theta, dtype=np.float64)
    if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
        raise ValueError("initial_theta must be a nonempty finite vector")
    theta = theta.copy()
    size = theta.size
    lower = _vector(config.lower_bounds, size, "lower_bounds")
    upper = _vector(config.upper_bounds, size, "upper_bounds")
    steps = _vector(config.finite_difference_step, size, "finite_difference_step", broadcast=True)
    radius = _vector(config.trust_radius, size, "trust_radius", broadcast=True)
    if np.any(lower > upper) or np.any(theta < lower) or np.any(theta > upper):
        raise ValueError("initial_theta must lie within valid parameter bounds")
    if np.any(steps <= 0.0) or np.any(radius <= 0.0):
        raise ValueError("finite_difference_step and trust_radius must be positive")
    for name in ("max_iterations", "max_evaluations", "projection_max_iterations"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("damping", "projection_tol"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("min_improvement", "prior_strength", "flat_tolerance"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 <= config.acceptance_ratio < 1.0 or not 0.0 < config.shrink_factor < 1.0:
        raise ValueError("acceptance_ratio must be in [0,1), shrink_factor in (0,1)")
    return theta, lower, upper, steps, radius


def validate_optimizer_config(initial_theta: Sequence[float], config: OptimizerConfig) -> None:
    """Validate all numerical options before an expensive initial evaluation."""
    _normalize_config(initial_theta, config)


def optimize(
    initial_theta: Sequence[float],
    evaluate: Callable[[np.ndarray], Evaluation],
    config: OptimizerConfig,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> OptimizationResult:
    """Run bounded metric feedback with a strict oracle-call budget.

    With p interior coordinates, a first iteration needs at most 2p+2 distinct
    evaluations: center, paired probes, and candidate.  Later centers are reused.
    Boundary/cache coincidences reduce that count.  A complete stencil plus one
    candidate is reserved before probing.  No hidden line search or fallback
    candidate is evaluated.  Rejected steps shrink the next trust region.

    The evaluator may persist and reuse results for deterministic restart; its
    cache key must include source/seed cohort, generator, metric, and parameter
    identities.  ``evaluations_used`` counts distinct vectors requested in this
    call, including results returned by such an external cache.
    """
    theta, lower, upper, steps, radius = _normalize_config(initial_theta, config)
    size = theta.size
    reference = theta.copy()
    history: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], Evaluation] = {}
    unit_keys: set[str] | None = None
    guard_keys: set[str] | None = None

    def emit(event: dict[str, Any]) -> None:
        history.append(event)
        if on_event is not None:
            on_event(event)

    def observe(point: np.ndarray, phase: str) -> Evaluation:
        nonlocal unit_keys, guard_keys
        key = tuple(float(value) for value in point)
        if key in cache:
            return cache[key]
        if len(cache) >= config.max_evaluations:
            raise RuntimeError("internal error: evaluation budget was not reserved")
        value = _validate_evaluation(evaluate(point.copy()))
        if unit_keys is None:
            unit_keys = set(value.unit_losses)
            guard_keys = set(value.guard_margins)
        elif set(value.unit_losses) != unit_keys or set(value.guard_margins) != guard_keys:
            raise ValueError("source-cluster keys and guard keys must remain identical")
        cache[key] = value
        emit({
            "type": "evaluation", "phase": phase, "theta": point.tolist(),
            "loss": value.loss, "unit_losses": dict(value.unit_losses),
            "guard_margins": dict(value.guard_margins), "fingerprint": value.fingerprint,
            "metadata": dict(value.metadata), "evaluation_index": len(cache),
        })
        return value

    def objective(value: Evaluation, point: np.ndarray) -> float:
        distance = point - reference
        result = value.loss + 0.5 * config.prior_strength * float(distance @ distance)
        if not np.isfinite(result):
            raise ValueError("parameter-prior objective must remain finite")
        return result

    current = observe(theta, "initial")
    if any(margin < 0.0 for margin in current.guard_margins.values()):
        return OptimizationResult(theta.tolist(), current, history, len(cache), "initial_guard_violation")
    accepted_steps = 0
    status = "no_proposal"
    names = sorted(current.guard_margins)
    for iteration in range(config.max_iterations):
        probe_points: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(size):
            displacement = min(steps[index], radius[index])
            negative, positive = theta.copy(), theta.copy()
            negative[index] = max(lower[index], theta[index] - displacement)
            positive[index] = min(upper[index], theta[index] + displacement)
            probe_points.append((negative, positive))
        new_keys = {tuple(float(v) for v in point) for pair in probe_points for point in pair} - cache.keys()
        if len(cache) + len(new_keys) + 1 > config.max_evaluations:
            emit({"type": "iteration", "iteration": iteration, "status": "budget_exhausted", "required_new_evaluations": len(new_keys) + 1})
            status = "budget_exhausted"
            break
        gradient, curvature = np.zeros(size), np.zeros(size)
        guard_jacobian = np.zeros((len(names), size))
        diagnostics = []
        f0 = objective(current, theta)
        for index, (negative, positive) in enumerate(probe_points):
            minus = float(theta[index] - negative[index])
            plus = float(positive[index] - theta[index])
            em = observe(negative, f"iteration_{iteration}_dimension_{index}_minus")
            ep = observe(positive, f"iteration_{iteration}_dimension_{index}_plus")
            gradient[index], curvature[index] = _finite_difference(f0, objective(em, negative), objective(ep, positive), minus, plus)
            for row, name in enumerate(names):
                guard_jacobian[row, index], _ = _finite_difference(current.guard_margins[name], em.guard_margins[name], ep.guard_margins[name], minus, plus)
            identical_execution = em.fingerprint is not None and ep.fingerprint is not None and em.fingerprint == ep.fingerprint
            diagnostics.append({
                "dimension": index, "minus_step": minus, "plus_step": plus,
                "stencil": "three_point" if minus > 0.0 and plus > 0.0 else "one_sided" if minus > 0.0 or plus > 0.0 else "fixed_coordinate",
                "flat_metric": max(abs(em.loss - current.loss), abs(ep.loss - current.loss)) <= config.flat_tolerance,
                "same_execution_fingerprint": identical_execution,
                "repeated_execution_score_disagreement": identical_execution and abs(em.loss - ep.loss) > config.flat_tolerance,
            })
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(curvature)) or not np.all(np.isfinite(guard_jacobian)):
            emit({"type": "iteration", "iteration": iteration, "status": "no_proposal", "reason": "nonfinite_difference_model", "probe_diagnostics": diagnostics})
            status = "no_proposal"
            break
        diagonal = np.maximum(curvature, 0.0) + config.damping
        lower_step, upper_step = np.maximum(lower - theta, -radius), np.minimum(upper - theta, radius)
        margins = np.asarray([current.guard_margins[name] for name in names])
        step, projection_iterations = _project_quadratic(gradient, diagonal, lower_step, upper_step, guard_jacobian, margins, config.projection_tol, config.projection_max_iterations)
        event: dict[str, Any] = {
            "type": "iteration", "iteration": iteration, "theta": theta.tolist(),
            "trust_radius": radius.tolist(), "gradient": gradient.tolist(),
            "curvature": curvature.tolist(), "model_diagonal": diagonal.tolist(),
            "probe_diagnostics": diagnostics, "projection_iterations": projection_iterations,
        }
        if step is None:
            event.update(status="no_proposal", reason="projection_not_certified")
            emit(event)
            status = "no_proposal"
            break
        predicted = -float(gradient @ step + 0.5 * np.dot(diagonal, step * step))
        if np.max(np.abs(step), initial=0.0) <= config.projection_tol or predicted <= 0.0:
            event.update(status="no_proposal", reason="no_predicted_descent", step=step.tolist(), predicted_reduction=predicted)
            emit(event)
            status = "no_proposal"
            break
        candidate_theta = np.clip(theta + step, lower, upper)
        candidate = observe(candidate_theta, f"iteration_{iteration}_candidate")
        actual = f0 - objective(candidate, candidate_theta)
        metric_reduction = current.loss - candidate.loss
        ratio = actual / predicted
        feasible = all(margin >= 0.0 for margin in candidate.guard_margins.values())
        same_execution = current.fingerprint is not None and candidate.fingerprint is not None and current.fingerprint == candidate.fingerprint
        accepted = feasible and not same_execution and metric_reduction > config.min_improvement and actual > 0.0 and ratio >= config.acceptance_ratio
        event.update(
            status="accepted" if accepted else "rejected", candidate_theta=candidate_theta.tolist(),
            step=step.tolist(), predicted_reduction=predicted, actual_reduction=actual,
            metric_reduction=metric_reduction, agreement_ratio=ratio, candidate_feasible=feasible,
            same_execution_as_center=same_execution,
            unit_loss_reductions={key: current.unit_losses[key] - candidate.unit_losses[key] for key in sorted(current.unit_losses)},
        )
        emit(event)
        status = event["status"]
        if accepted:
            theta, current = candidate_theta, candidate
            accepted_steps += 1
        else:
            radius *= config.shrink_factor
    return OptimizationResult(theta.tolist(), current, history, len(cache), status, accepted_steps)
