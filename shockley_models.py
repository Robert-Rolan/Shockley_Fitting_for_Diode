from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import brentq, least_squares, newton

Q_E = 1.602176634e-19
K_B = 1.380649e-23
DEFAULT_FIXED_VALUES = {"T": 300.0, "Vbi": 1.0}
TAT_EXP_LIMIT = 20.0
DEFAULT_REVERSE_BIAS_WEIGHT = 5.0


class ShockleyFitError(ValueError):
    """Raised when input data or fit settings cannot be used."""


@dataclass(frozen=True)
class IVData:
    voltage: np.ndarray
    current: np.ndarray
    source_path: str | None = None


@dataclass(frozen=True)
class ParameterRow:
    name: str
    label: str
    unit: str
    initial: float
    lower: float
    upper: float
    value: float | None = None

    def with_values(
        self,
        *,
        initial: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
        value: float | None = None,
    ) -> "ParameterRow":
        return replace(
            self,
            initial=self.initial if initial is None else float(initial),
            lower=self.lower if lower is None else float(lower),
            upper=self.upper if upper is None else float(upper),
            value=self.value if value is None else float(value),
        )


@dataclass(frozen=True)
class ModelParameters:
    rows: list[ParameterRow]
    fixed: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FIXED_VALUES))
    fit_range_weights: dict[str, float] = field(default_factory=lambda: default_fit_range_weights())


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class FitWeightRange:
    key: str
    label: str
    lower_v: float
    upper_v: float
    default_weight: float


@dataclass(frozen=True)
class FitResult:
    model_key: str
    model_label: str
    source_path: str | None
    voltage: np.ndarray
    measured_current: np.ndarray
    fitted_current: np.ndarray
    component_currents: dict[str, np.ndarray]
    residual: np.ndarray
    residual_scaled: np.ndarray
    parameters: dict[str, ParameterRow]
    fixed: dict[str, float]
    fit_range_weights: dict[str, float]
    success: bool
    message: str
    optimizer_status: int
    nfev: int
    metrics: dict[str, float]


MODEL_SPECS = {
    "ideal": ModelSpec("ideal", "Ideal Shockley diode equation", ("I0", "n", "Rs")),
    "tat": ModelSpec(
        "tat",
        "Non-ideal generalized Shockley diode equation (TAT)",
        ("I0", "n", "Rs", "Rsh", "A_TAT", "B"),
    ),
}

PARAMETER_INFO = {
    "I0": ("I0", "A"),
    "n": ("n", ""),
    "Rs": ("Rs", "ohm"),
    "Rsh": ("Rsh", "ohm"),
    "A_TAT": ("A_TAT", "A/V"),
    "B": ("B", "V"),
}

FIT_WEIGHT_RANGES = (
    FitWeightRange("neg_2_to_neg_0p5", "-2 to -0.5 V", -2.0, -0.5, 50.0),
    FitWeightRange("neg_0p5_to_0", "-0.5 to 0 V", -0.5, 0.0, 10.0),
    FitWeightRange("pos_0_to_0p5", "0 to 0.5 V", 0.0, 0.5, 10.0),
    FitWeightRange("pos_0p5_to_2", "0.5 to 2 V", 0.5, 2.0, 50.0),
)


def read_iv_csv(path: str | Path) -> IVData:
    rows: list[tuple[float, float]] = []
    csv_path = Path(path)
    if not csv_path.exists():
        raise ShockleyFitError(f"File does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if len(raw_row) < 2:
                continue
            try:
                voltage = float(raw_row[0].strip())
                current = float(raw_row[-1].strip())
            except ValueError:
                continue
            if math.isfinite(voltage) and math.isfinite(current):
                rows.append((voltage, current))

    if len(rows) < 3:
        raise ShockleyFitError("CSV must contain at least three numeric V/I rows.")

    data = np.array(rows, dtype=float)
    order = np.argsort(data[:, 0])
    sorted_data = data[order]
    return IVData(sorted_data[:, 0], sorted_data[:, 1], str(csv_path))


def prepare_log_preview(current: Iterable[float], floor: float | None = None) -> np.ndarray:
    values = np.asarray(current, dtype=float)
    preview = np.abs(values).astype(float, copy=True)
    finite_positive = preview[np.isfinite(preview) & (preview > 0)]
    if floor is None:
        if finite_positive.size:
            floor = max(float(np.nanmin(finite_positive)) * 0.1, 1e-18)
        else:
            floor = 1e-18
    preview[~np.isfinite(preview) | (preview <= 0)] = float(floor)
    return preview


def thermal_voltage(temperature_k: float) -> float:
    if temperature_k <= 0 or not math.isfinite(temperature_k):
        raise ShockleyFitError("Temperature must be a positive finite value.")
    return K_B * temperature_k / Q_E


def safe_exp(value: np.ndarray | float) -> np.ndarray | float:
    return np.exp(np.clip(value, -60.0, 60.0))


def ideal_current_explicit(
    voltage: np.ndarray,
    i0: float,
    n: float,
    temperature_k: float,
) -> np.ndarray:
    vt = thermal_voltage(temperature_k)
    exponent = np.asarray(voltage, dtype=float) / (n * vt)
    return i0 * (safe_exp(exponent) - 1.0)


def model_current(
    model_key: str,
    voltage: np.ndarray,
    params: dict[str, float],
    fixed: dict[str, float] | None = None,
) -> np.ndarray:
    fixed_values = dict(DEFAULT_FIXED_VALUES)
    if fixed:
        fixed_values.update({key: float(value) for key, value in fixed.items()})
    if model_key not in MODEL_SPECS:
        raise ShockleyFitError(f"Unknown model: {model_key}")

    voltage_arr = np.asarray(voltage, dtype=float)
    i0 = max(float(params["I0"]), 0.0)
    n = max(float(params["n"]), 1e-12)
    rs = max(float(params.get("Rs", 0.0)), 0.0)
    temperature_k = float(fixed_values["T"])

    rsh = math.inf
    a_tat = 0.0
    b_tat = 0.0
    vbi = float(fixed_values.get("Vbi", 1.0))
    if model_key == "tat":
        rsh = max(float(params.get("Rsh", math.inf)), 1e-30)
        a_tat = max(float(params.get("A_TAT", 0.0)), 0.0)
        b_tat = float(params.get("B", 0.0))

    if rs <= 1e-15:
        return _explicit_terms(voltage_arr, i0, n, temperature_k, rsh, a_tat, b_tat, vbi)

    solved = np.empty_like(voltage_arr, dtype=float)
    previous = math.nan
    for index, value in enumerate(voltage_arr):
        previous = _solve_implicit_current(
            float(value),
            i0=i0,
            n=n,
            rs=rs,
            temperature_k=temperature_k,
            rsh=rsh,
            a_tat=a_tat,
            b_tat=b_tat,
            vbi=vbi,
            previous=previous,
        )
        solved[index] = previous
    return solved


def component_currents(
    model_key: str,
    voltage: np.ndarray,
    total_current: np.ndarray,
    params: dict[str, float],
    fixed: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    fixed_values = dict(DEFAULT_FIXED_VALUES)
    if fixed:
        fixed_values.update({key: float(value) for key, value in fixed.items()})
    if model_key not in MODEL_SPECS:
        raise ShockleyFitError(f"Unknown model: {model_key}")

    voltage_arr = np.asarray(voltage, dtype=float)
    current_arr = np.asarray(total_current, dtype=float)
    i0 = max(float(params["I0"]), 0.0)
    n = max(float(params["n"]), 1e-12)
    rs = max(float(params.get("Rs", 0.0)), 0.0)
    internal_voltage = voltage_arr - current_arr * rs

    ideal = ideal_current_explicit(internal_voltage, i0, n, float(fixed_values["T"]))
    ohmic = np.zeros_like(voltage_arr, dtype=float)
    tat = np.zeros_like(voltage_arr, dtype=float)

    if model_key == "tat":
        rsh = max(float(params.get("Rsh", math.inf)), 1e-30)
        if math.isfinite(rsh):
            ohmic = internal_voltage / rsh
        tat = _tat_term(
            voltage_arr,
            max(float(params.get("A_TAT", 0.0)), 0.0),
            float(params.get("B", 0.0)),
            float(fixed_values.get("Vbi", 1.0)),
        )

    return {
        "ideal_diode_A": ideal,
        "ohmic_A": ohmic,
        "tat_A": tat,
    }


def default_parameter_rows(
    model_key: str,
    voltage: np.ndarray,
    current: np.ndarray,
    fixed: dict[str, float] | None = None,
) -> list[ParameterRow]:
    if model_key not in MODEL_SPECS:
        raise ShockleyFitError(f"Unknown model: {model_key}")
    fixed_values = dict(DEFAULT_FIXED_VALUES)
    if fixed:
        fixed_values.update(fixed)

    voltage_arr = np.asarray(voltage, dtype=float)
    current_arr = np.asarray(current, dtype=float)
    max_abs_i = _positive_or_default(np.nanmax(np.abs(current_arr)), 1e-12)
    voltage_span = _positive_or_default(float(np.nanmax(voltage_arr) - np.nanmin(voltage_arr)), 1.0)

    i0_guess = _estimate_i0(voltage_arr, current_arr)
    n_guess = _estimate_n(voltage_arr, current_arr, float(fixed_values["T"]))
    rs_guess = _estimate_rs(voltage_arr, current_arr)
    rsh_guess = _estimate_rsh(voltage_arr, current_arr)

    i0_upper = max(max_abs_i * 10.0, 1e-9)
    rs_upper = max(voltage_span / max(max_abs_i, 1e-18) * 10.0, 1.0)
    rsh_upper = max(rs_upper * 1e6, 1e12)
    max_abs_v = max(float(np.nanmax(np.abs(voltage_arr))), 1.0)
    a_upper = max(max_abs_i / max_abs_v * 10.0, 1e-12)

    rows_by_name = {
        "I0": _make_row("I0", i0_guess, 1e-18, i0_upper),
        "n": _make_row("n", n_guess, 0.5, 10.0),
        "Rs": _make_row("Rs", rs_guess, 0.0, rs_upper),
        "Rsh": _make_row("Rsh", rsh_guess, 1.0, rsh_upper),
        "A_TAT": _make_row("A_TAT", 0.0, 0.0, a_upper),
        "B": _make_row("B", 0.1, -50.0, 50.0),
    }
    return [rows_by_name[name] for name in MODEL_SPECS[model_key].parameter_names]


def fit_iv_curve(
    model_key: str,
    voltage: np.ndarray,
    current: np.ndarray,
    parameters: ModelParameters,
    *,
    source_path: str | None = None,
    max_nfev: int = 1200,
) -> FitResult:
    if model_key not in MODEL_SPECS:
        raise ShockleyFitError(f"Unknown model: {model_key}")

    voltage_arr = np.asarray(voltage, dtype=float)
    current_arr = np.asarray(current, dtype=float)
    if voltage_arr.size != current_arr.size or voltage_arr.size < 3:
        raise ShockleyFitError("Voltage and current arrays must have the same length and at least 3 rows.")

    rows = _validate_rows(model_key, parameters.rows)
    fixed_values = dict(DEFAULT_FIXED_VALUES)
    fixed_values.update({key: float(value) for key, value in parameters.fixed.items()})
    fit_range_weights = validate_fit_range_weights(parameters.fit_range_weights)
    floor = current_floor(current_arr)
    objective_weights = residual_objective_weights(voltage_arr)
    log_floor = log_current_floor(current_arr)

    transforms = [_transform_for(row) for row in rows]
    x0 = np.array([transform.encode(row.initial) for row, transform in zip(rows, transforms)], dtype=float)
    lower = np.array([transform.encode(row.lower) for row, transform in zip(rows, transforms)], dtype=float)
    upper = np.array([transform.encode(row.upper) for row, transform in zip(rows, transforms)], dtype=float)
    x0 = _clip_initial(x0, lower, upper)

    def residuals(encoded: np.ndarray) -> np.ndarray:
        try:
            decoded = {
                row.name: transform.decode(value)
                for row, transform, value in zip(rows, transforms, encoded)
            }
            fitted = model_current(model_key, voltage_arr, decoded, fixed=fixed_values)
            scaled = (fitted - current_arr) / np.maximum(np.abs(current_arr), floor)
            if not np.all(np.isfinite(scaled)):
                return np.full_like(current_arr, 1e30, dtype=float)
            weighted_scaled = scaled * objective_weights
            range_log = range_log_magnitude_residuals(
                voltage_arr,
                fitted,
                current_arr,
                fit_range_weights,
                floor=log_floor,
            )
            if range_log.size:
                if not np.all(np.isfinite(range_log)):
                    return np.full(current_arr.size + range_log.size, 1e30, dtype=float)
                return np.concatenate([weighted_scaled, range_log])
            return weighted_scaled
        except (ArithmeticError, ValueError, RuntimeError, ShockleyFitError):
            return np.full_like(current_arr, 1e30, dtype=float)

    optimized = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    fitted_values = {
        row.name: transform.decode(value)
        for row, transform, value in zip(rows, transforms, optimized.x)
    }
    fitted_current = model_current(model_key, voltage_arr, fitted_values, fixed=fixed_values)
    components = component_currents(model_key, voltage_arr, fitted_current, fitted_values, fixed=fixed_values)
    residual = fitted_current - current_arr
    residual_scaled = residual / np.maximum(np.abs(current_arr), floor)
    log_residual = log_magnitude_residual(fitted_current, current_arr, log_floor)

    parameter_results = {
        row.name: row.with_values(value=fitted_values[row.name])
        for row in rows
    }
    metrics = {
        "point_count": float(voltage_arr.size),
        "rmse_A": float(math.sqrt(np.mean(np.square(residual)))),
        "max_abs_error_A": float(np.max(np.abs(residual))),
        "scaled_rmse": float(math.sqrt(np.mean(np.square(residual_scaled)))),
        "reverse_scaled_rmse": _region_rmse(residual_scaled, voltage_arr < 0.0),
        "nonnegative_scaled_rmse": _region_rmse(residual_scaled, voltage_arr >= 0.0),
        "reverse_log_mean_abs_decades": _region_mean_abs(log_residual, voltage_arr < 0.0),
        "reverse_log_max_abs_decades": _region_max_abs(log_residual, voltage_arr < 0.0),
        "reverse_bias_objective_weight": DEFAULT_REVERSE_BIAS_WEIGHT,
    }

    return FitResult(
        model_key=model_key,
        model_label=MODEL_SPECS[model_key].label,
        source_path=source_path,
        voltage=voltage_arr.copy(),
        measured_current=current_arr.copy(),
        fitted_current=fitted_current,
        component_currents=components,
        residual=residual,
        residual_scaled=residual_scaled,
        parameters=parameter_results,
        fixed=fixed_values,
        fit_range_weights=fit_range_weights,
        success=bool(optimized.success and np.all(np.isfinite(fitted_current))),
        message=str(optimized.message),
        optimizer_status=int(optimized.status),
        nfev=int(optimized.nfev),
        metrics=metrics,
    )


def export_fit_result(result: FitResult, directory: str | Path) -> tuple[str, str]:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "fitted_curve.csv"
    parameter_path = output_dir / "fit_parameters.json"

    curve = np.column_stack(
        [
            result.voltage,
            result.measured_current,
            result.fitted_current,
            result.component_currents["ideal_diode_A"],
            result.component_currents["ohmic_A"],
            result.component_currents["tat_A"],
            result.residual,
            result.residual_scaled,
        ]
    )
    np.savetxt(
        curve_path,
        curve,
        delimiter=",",
        header=(
            "V_V,I_measured_A,I_fitted_total_A,I_fitted_ideal_diode_A,"
            "I_fitted_ohmic_A,I_fitted_TAT_A,residual_A,residual_scaled"
        ),
        comments="",
        fmt="%.12g",
    )

    payload = {
        "source_path": result.source_path,
        "model": result.model_key,
        "model_label": result.model_label,
        "fixed": {key: float(value) for key, value in result.fixed.items()},
        "fit_range_weights": result.fit_range_weights,
        "component_currents": {
            "total": "I_fitted_total_A",
            "ideal_diode": "I_fitted_ideal_diode_A",
            "ohmic": "I_fitted_ohmic_A",
            "tat": "I_fitted_TAT_A",
        },
        "parameters": {
            name: {
                "label": row.label,
                "unit": row.unit,
                "initial": row.initial,
                "lower": row.lower,
                "upper": row.upper,
                "value": row.value,
            }
            for name, row in result.parameters.items()
        },
        "metrics": result.metrics,
        "optimizer": {
            "success": result.success,
            "status": result.optimizer_status,
            "message": result.message,
            "nfev": result.nfev,
        },
    }
    parameter_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(curve_path), str(parameter_path)


def current_floor(current: np.ndarray) -> float:
    finite = np.abs(np.asarray(current, dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return 1e-12
    return max(float(np.percentile(finite, 5)) * 0.1, 1e-12)


def log_current_floor(current: np.ndarray) -> float:
    finite = np.abs(np.asarray(current, dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return 1e-18
    return max(float(np.min(finite)) * 0.1, 1e-18)


def log_magnitude_residual(fitted: np.ndarray, measured: np.ndarray, floor: float) -> np.ndarray:
    fitted_mag = np.maximum(np.abs(np.asarray(fitted, dtype=float)), floor)
    measured_mag = np.maximum(np.abs(np.asarray(measured, dtype=float)), floor)
    return np.log10(fitted_mag) - np.log10(measured_mag)


def default_fit_range_weights() -> dict[str, float]:
    return {fit_range.key: fit_range.default_weight for fit_range in FIT_WEIGHT_RANGES}


def validate_fit_range_weights(weights: dict[str, float] | None) -> dict[str, float]:
    merged = default_fit_range_weights()
    if weights:
        for key, value in weights.items():
            if key not in merged:
                continue
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0.0:
                raise ShockleyFitError("Fit range weights must be finite numbers greater than or equal to 0.")
            merged[key] = parsed
    return merged


def range_log_magnitude_residuals(
    voltage: np.ndarray,
    fitted: np.ndarray,
    measured: np.ndarray,
    range_weights: dict[str, float] | None,
    *,
    floor: float,
) -> np.ndarray:
    voltage_arr = np.asarray(voltage, dtype=float)
    log_residual = log_magnitude_residual(fitted, measured, floor)
    weights = validate_fit_range_weights(range_weights)
    pieces: list[np.ndarray] = []
    for fit_range in FIT_WEIGHT_RANGES:
        weight = weights[fit_range.key]
        if weight <= 0.0:
            continue
        mask = _fit_range_mask(voltage_arr, fit_range)
        if np.any(mask):
            pieces.append(log_residual[mask] * weight)
    if not pieces:
        return np.array([], dtype=float)
    return np.concatenate(pieces)


def residual_objective_weights(
    voltage: np.ndarray,
    reverse_bias_weight: float = DEFAULT_REVERSE_BIAS_WEIGHT,
) -> np.ndarray:
    voltage_arr = np.asarray(voltage, dtype=float)
    weights = np.ones_like(voltage_arr, dtype=float)
    if reverse_bias_weight <= 0 or not math.isfinite(reverse_bias_weight):
        raise ShockleyFitError("Reverse-bias residual weight must be positive and finite.")
    weights[voltage_arr < 0.0] = float(reverse_bias_weight)
    return weights


def _fit_range_mask(voltage: np.ndarray, fit_range: FitWeightRange) -> np.ndarray:
    if fit_range.key == "pos_0p5_to_2":
        return (voltage >= fit_range.lower_v) & (voltage <= fit_range.upper_v)
    return (voltage >= fit_range.lower_v) & (voltage < fit_range.upper_v)


def _region_rmse(values: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(values, dtype=float)[mask]
    if region.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean(np.square(region))))


def _region_mean_abs(values: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(values, dtype=float)[mask]
    if region.size == 0:
        return float("nan")
    return float(np.mean(np.abs(region)))


def _region_max_abs(values: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(values, dtype=float)[mask]
    if region.size == 0:
        return float("nan")
    return float(np.max(np.abs(region)))


def _explicit_terms(
    voltage: np.ndarray,
    i0: float,
    n: float,
    temperature_k: float,
    rsh: float,
    a_tat: float,
    b_tat: float,
    vbi: float,
) -> np.ndarray:
    ideal = ideal_current_explicit(voltage, i0, n, temperature_k)
    shunt = 0.0 if not math.isfinite(rsh) else voltage / rsh
    tat = _tat_term(voltage, a_tat, b_tat, vbi)
    return ideal + shunt + tat


def _solve_implicit_current(
    voltage: float,
    *,
    i0: float,
    n: float,
    rs: float,
    temperature_k: float,
    rsh: float,
    a_tat: float,
    b_tat: float,
    vbi: float,
    previous: float,
) -> float:
    vt = thermal_voltage(temperature_k)
    beta = 1.0 / (n * vt)
    tat = float(_tat_term(np.array([voltage], dtype=float), a_tat, b_tat, vbi)[0])
    shunt_scale = 0.0 if not math.isfinite(rsh) else 1.0 / rsh

    def func(current_value: float) -> float:
        exponent = beta * (voltage - current_value * rs)
        ideal = i0 * (safe_exp(exponent) - 1.0)
        shunt = (voltage - current_value * rs) * shunt_scale
        return current_value - ideal - shunt - tat

    def derivative(current_value: float) -> float:
        exponent = beta * (voltage - current_value * rs)
        return 1.0 + i0 * beta * rs * safe_exp(exponent) + rs * shunt_scale

    guess = previous if math.isfinite(previous) else float(
        _explicit_terms(np.array([voltage]), i0, n, temperature_k, rsh, a_tat, b_tat, vbi)[0]
    )
    try:
        candidate = float(newton(func, guess, fprime=derivative, tol=1e-13, maxiter=30))
        if math.isfinite(candidate) and abs(func(candidate)) <= max(abs(candidate), 1.0) * 1e-8:
            return candidate
    except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
        pass

    scale = max(abs(guess), abs(previous) if math.isfinite(previous) else 0.0, abs(voltage) / max(rs, 1.0), 1e-12)
    low = min(guess, previous if math.isfinite(previous) else guess) - scale
    high = max(guess, previous if math.isfinite(previous) else guess) + scale
    f_low = func(low)
    f_high = func(high)
    for _ in range(80):
        if math.isfinite(f_low) and math.isfinite(f_high) and f_low <= 0.0 <= f_high:
            return float(brentq(func, low, high, xtol=1e-13, rtol=1e-12, maxiter=100))
        scale *= 2.0
        low -= scale
        high += scale
        f_low = func(low)
        f_high = func(high)
    raise ShockleyFitError("Could not solve implicit diode current.")


def _tat_term(voltage: np.ndarray, a_tat: float, b_tat: float, vbi: float) -> np.ndarray:
    voltage_arr = np.asarray(voltage, dtype=float)
    if a_tat == 0.0:
        return np.zeros_like(voltage_arr, dtype=float)
    denominator = voltage_arr - vbi
    exponent = np.divide(
        b_tat,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=np.abs(denominator) >= 1e-6,
    )
    exponent = np.clip(exponent, -TAT_EXP_LIMIT, TAT_EXP_LIMIT)
    term = a_tat * voltage_arr * np.exp(exponent)
    return np.where(voltage_arr < 0.0, term, 0.0)


def _make_row(name: str, initial: float, lower: float, upper: float) -> ParameterRow:
    label, unit = PARAMETER_INFO[name]
    clipped = min(max(float(initial), float(lower)), float(upper))
    return ParameterRow(name=name, label=label, unit=unit, initial=clipped, lower=float(lower), upper=float(upper))


def _estimate_i0(voltage: np.ndarray, current: np.ndarray) -> float:
    reverse = np.abs(current[voltage < 0])
    reverse = reverse[np.isfinite(reverse) & (reverse > 0)]
    if reverse.size:
        return float(np.clip(np.percentile(reverse, 10), 1e-18, 1e-6))
    positive = np.abs(current[np.isfinite(current) & (current != 0)])
    if positive.size:
        return float(np.clip(np.nanmin(positive) * 0.1, 1e-18, 1e-6))
    return 1e-12


def _estimate_n(voltage: np.ndarray, current: np.ndarray, temperature_k: float) -> float:
    positive = np.isfinite(voltage) & np.isfinite(current) & (current > 0)
    if np.count_nonzero(positive) < 3:
        return 2.0
    v_pos = voltage[positive]
    i_pos = current[positive]
    cutoff_low = np.percentile(i_pos, 20)
    cutoff_high = np.percentile(i_pos, 80)
    mask = (i_pos >= cutoff_low) & (i_pos <= cutoff_high)
    if np.count_nonzero(mask) < 3:
        mask = np.ones_like(i_pos, dtype=bool)
    slope, _ = np.polyfit(v_pos[mask], np.log(i_pos[mask]), 1)
    if slope <= 0 or not math.isfinite(slope):
        return 2.0
    return float(np.clip(1.0 / (thermal_voltage(temperature_k) * slope), 0.8, 6.0))


def _estimate_rs(voltage: np.ndarray, current: np.ndarray) -> float:
    positive = np.isfinite(voltage) & np.isfinite(current) & (current > 0)
    if np.count_nonzero(positive) < 4:
        return 0.0
    v_pos = voltage[positive]
    i_pos = current[positive]
    order = np.argsort(i_pos)
    v_sorted = v_pos[order]
    i_sorted = i_pos[order]
    start = max(int(0.75 * len(i_sorted)) - 1, 0)
    dv = np.diff(v_sorted[start:])
    di = np.diff(i_sorted[start:])
    slopes = dv[(di > 0)] / di[(di > 0)]
    slopes = slopes[np.isfinite(slopes) & (slopes >= 0)]
    if slopes.size == 0:
        return 0.0
    return float(np.clip(np.median(slopes), 0.0, 1e6))


def _estimate_rsh(voltage: np.ndarray, current: np.ndarray) -> float:
    low_current = np.isfinite(voltage) & np.isfinite(current) & (np.abs(current) > 0)
    if np.count_nonzero(low_current) < 3:
        return 1e9
    ratios = np.abs(voltage[low_current] / current[low_current])
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return 1e9
    return float(np.clip(np.percentile(ratios, 50), 1.0, 1e12))


def _positive_or_default(value: float, default: float) -> float:
    if math.isfinite(value) and value > 0:
        return float(value)
    return float(default)


def _validate_rows(model_key: str, rows: list[ParameterRow]) -> list[ParameterRow]:
    expected = list(MODEL_SPECS[model_key].parameter_names)
    by_name = {row.name: row for row in rows}
    missing = [name for name in expected if name not in by_name]
    if missing:
        raise ShockleyFitError(f"Missing parameter rows: {', '.join(missing)}")

    validated = []
    for name in expected:
        row = by_name[name]
        values = [row.initial, row.lower, row.upper]
        if not all(math.isfinite(value) for value in values):
            raise ShockleyFitError(f"Parameter {name} contains non-finite values.")
        if row.lower >= row.upper:
            raise ShockleyFitError(f"Parameter {name} lower bound must be less than upper bound.")
        initial = min(max(row.initial, row.lower), row.upper)
        validated.append(row.with_values(initial=initial))
    return validated


class _Transform:
    def __init__(self, encode: Callable[[float], float], decode: Callable[[float], float]):
        self.encode = encode
        self.decode = decode


def _transform_for(row: ParameterRow) -> _Transform:
    if row.name in {"I0", "Rsh"} and row.lower > 0 and row.upper > 0:
        return _Transform(lambda value: math.log10(value), lambda value: 10.0 ** value)
    return _Transform(lambda value: float(value), lambda value: float(value))


def _clip_initial(x0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    clipped = np.clip(x0, lower, upper)
    for index, value in enumerate(clipped):
        if value <= lower[index]:
            clipped[index] = np.nextafter(lower[index], upper[index])
        if value >= upper[index]:
            clipped[index] = np.nextafter(upper[index], lower[index])
    return clipped
