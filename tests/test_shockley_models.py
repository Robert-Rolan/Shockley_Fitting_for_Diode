import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shockley_models import (
    DEFAULT_FIXED_VALUES,
    ModelParameters,
    ShockleyFitError,
    component_currents,
    default_parameter_rows,
    default_fit_range_weights,
    export_fit_result,
    fit_iv_curve,
    ideal_current_explicit,
    model_current,
    prepare_log_preview,
    range_log_magnitude_residuals,
    read_iv_csv,
    residual_objective_weights,
)


class CsvLoadingTests(unittest.TestCase):
    def test_read_iv_csv_uses_first_and_last_columns_and_sorts_voltage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_text(
                "0.5,ignored,2e-6\n"
                "-1.0,ignored,-4e-7\n"
                "bad,row,ignored\n"
                "0.0,ignored,0\n",
                encoding="utf-8",
            )

            data = read_iv_csv(path)

        np.testing.assert_allclose(data.voltage, [-1.0, 0.0, 0.5])
        np.testing.assert_allclose(data.current, [-4e-7, 0.0, 2e-6])

    def test_read_iv_csv_rejects_files_without_two_numeric_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.csv"
            path.write_text("only-one-column\n", encoding="utf-8")

            with self.assertRaises(ShockleyFitError):
                read_iv_csv(path)


class PreviewTests(unittest.TestCase):
    def test_prepare_log_preview_uses_absolute_value_without_mutating_input(self):
        current = np.array([-1e-9, 0.0, 2e-6])
        original = current.copy()

        preview = prepare_log_preview(current)

        self.assertGreater(preview[1], 0.0)
        np.testing.assert_allclose(preview[[0, 2]], [1e-9, 2e-6])
        np.testing.assert_allclose(current, original)


class ModelEquationTests(unittest.TestCase):
    def test_ideal_model_with_zero_series_resistance_matches_explicit_equation(self):
        voltage = np.array([-0.1, 0.0, 0.1, 0.2])
        params = {"I0": 1e-12, "n": 1.6, "Rs": 0.0}

        implicit = model_current(
            "ideal",
            voltage,
            params,
            fixed={"T": 300.0},
        )
        explicit = ideal_current_explicit(voltage, 1e-12, 1.6, 300.0)

        np.testing.assert_allclose(implicit, explicit, rtol=1e-10, atol=1e-15)

    def test_tat_model_without_shunt_or_tat_terms_matches_ideal(self):
        voltage = np.array([-0.2, 0.0, 0.15, 0.25])
        ideal_params = {"I0": 2e-12, "n": 1.8, "Rs": 0.0}
        tat_params = {
            "I0": 2e-12,
            "n": 1.8,
            "Rs": 0.0,
            "Rsh": 1e30,
            "A_TAT": 0.0,
            "B": 0.2,
        }

        ideal = model_current("ideal", voltage, ideal_params, fixed={"T": 300.0})
        tat = model_current("tat", voltage, tat_params, fixed={"T": 300.0, "Vbi": 1.0})

        np.testing.assert_allclose(tat, ideal, rtol=1e-10, atol=1e-15)

    def test_tat_component_currents_sum_to_total_current(self):
        voltage = np.array([-0.4, 0.0, 0.4, 0.8])
        params = {
            "I0": 3e-12,
            "n": 1.9,
            "Rs": 8.0,
            "Rsh": 5e6,
            "A_TAT": 2e-13,
            "B": 0.25,
        }
        fixed = {"T": 300.0, "Vbi": 1.2}
        total = model_current("tat", voltage, params, fixed=fixed)

        components = component_currents("tat", voltage, total, params, fixed=fixed)

        self.assertEqual(
            ["ideal_diode_A", "ohmic_A", "tat_A"],
            list(components.keys()),
        )
        np.testing.assert_allclose(
            components["ideal_diode_A"] + components["ohmic_A"] + components["tat_A"],
            total,
            rtol=1e-8,
            atol=1e-14,
        )


class FittingTests(unittest.TestCase):
    def test_default_parameter_rows_are_finite_and_within_bounds(self):
        voltage = np.linspace(-1.0, 1.0, 21)
        current = ideal_current_explicit(voltage, 1e-11, 1.7, 300.0)

        rows = default_parameter_rows("ideal", voltage, current)

        self.assertEqual(["I0", "n", "Rs"], [row.name for row in rows])
        for row in rows:
            self.assertTrue(math.isfinite(row.initial), row)
            self.assertLessEqual(row.lower, row.initial, row)
            self.assertLessEqual(row.initial, row.upper, row)

    def test_residual_objective_weights_prioritize_reverse_bias(self):
        voltage = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        weights = residual_objective_weights(voltage)

        self.assertGreater(weights[0], weights[2])
        self.assertGreater(weights[1], weights[3])
        self.assertEqual(weights[2], weights[3])

    def test_range_log_residuals_use_configured_bias_range_weights(self):
        voltage = np.array([-1.5, -0.25, 0.25, 1.5, 2.4])
        measured = np.full_like(voltage, 1e-7, dtype=float)
        fitted = np.full_like(voltage, 1e-8, dtype=float)
        weights = {key: 0.0 for key in default_fit_range_weights()}
        weights["neg_2_to_neg_0p5"] = 3.0
        weights["pos_0p5_to_2"] = 7.0

        residuals = range_log_magnitude_residuals(voltage, fitted, measured, weights, floor=1e-18)

        np.testing.assert_allclose(residuals, [-3.0, -7.0])

    def test_fit_iv_curve_recovers_synthetic_ideal_parameters(self):
        voltage = np.linspace(-0.2, 0.55, 60)
        true_params = {"I0": 8e-12, "n": 1.7, "Rs": 4.0}
        current = model_current("ideal", voltage, true_params, fixed={"T": 300.0})
        rows = default_parameter_rows("ideal", voltage, current)
        rows_by_name = {row.name: row for row in rows}
        rows_by_name["I0"] = rows_by_name["I0"].with_values(initial=1e-11, lower=1e-14, upper=1e-8)
        rows_by_name["n"] = rows_by_name["n"].with_values(initial=1.5, lower=1.0, upper=3.0)
        rows_by_name["Rs"] = rows_by_name["Rs"].with_values(initial=2.0, lower=0.0, upper=20.0)
        params = ModelParameters(
            rows=[rows_by_name["I0"], rows_by_name["n"], rows_by_name["Rs"]],
            fixed=dict(DEFAULT_FIXED_VALUES),
        )

        result = fit_iv_curve("ideal", voltage, current, params)

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.parameters["I0"].value, true_params["I0"], delta=3e-12)
        self.assertAlmostEqual(result.parameters["n"].value, true_params["n"], delta=0.15)
        self.assertAlmostEqual(result.parameters["Rs"].value, true_params["Rs"], delta=2.0)

    def test_tat_fit_handles_voltage_equal_to_vbi_without_singular_blowup(self):
        voltage = np.linspace(-0.5, 1.5, 41)
        current = ideal_current_explicit(voltage, 2e-12, 1.8, 300.0)
        params = ModelParameters(
            rows=default_parameter_rows("tat", voltage, current),
            fixed={"T": 300.0, "Vbi": 1.0},
        )

        result = fit_iv_curve("tat", voltage, current, params, max_nfev=300)

        self.assertTrue(np.all(np.isfinite(result.fitted_current)))
        self.assertLess(result.metrics["rmse_A"], max(np.max(np.abs(current)) * 5.0, 1e-9))

    def test_fit_result_contains_component_currents(self):
        voltage = np.linspace(-0.1, 0.5, 24)
        current = ideal_current_explicit(voltage, 1e-12, 1.5, 300.0)
        params = ModelParameters(rows=default_parameter_rows("tat", voltage, current))

        result = fit_iv_curve("tat", voltage, current, params)

        self.assertEqual(
            ["ideal_diode_A", "ohmic_A", "tat_A"],
            list(result.component_currents.keys()),
        )
        np.testing.assert_allclose(
            result.component_currents["ideal_diode_A"]
            + result.component_currents["ohmic_A"]
            + result.component_currents["tat_A"],
            result.fitted_current,
            rtol=1e-8,
            atol=1e-14,
        )

    def test_fit_result_records_custom_bias_range_weights(self):
        voltage = np.linspace(-2.0, 2.0, 40)
        current = ideal_current_explicit(voltage, 1e-12, 1.6, 300.0)
        weights = default_fit_range_weights()
        weights["neg_2_to_neg_0p5"] = 12.0
        weights["pos_0p5_to_2"] = 8.0
        params = ModelParameters(
            rows=default_parameter_rows("tat", voltage, current),
            fit_range_weights=weights,
        )

        result = fit_iv_curve("tat", voltage, current, params)

        self.assertEqual(result.fit_range_weights["neg_2_to_neg_0p5"], 12.0)
        self.assertEqual(result.fit_range_weights["pos_0p5_to_2"], 8.0)

    def test_tat_fit_prioritizes_reverse_bias_on_log_scale_for_public_control_csv(self):
        path = Path(__file__).resolve().parents[1] / "control.csv"
        data = read_iv_csv(path)
        params = ModelParameters(rows=default_parameter_rows("tat", data.voltage, data.current))

        result = fit_iv_curve("tat", data.voltage, data.current, params, max_nfev=900)

        reverse = data.voltage < 0.0
        reverse_log_error = np.log10(prepare_log_preview(result.fitted_current[reverse])) - np.log10(
            prepare_log_preview(data.current[reverse])
        )
        self.assertLess(float(np.mean(np.abs(reverse_log_error))), 0.75)

    def test_export_fit_result_writes_curve_csv_and_parameter_json(self):
        voltage = np.linspace(-0.1, 0.4, 20)
        current = ideal_current_explicit(voltage, 1e-12, 1.5, 300.0)
        params = ModelParameters(rows=default_parameter_rows("ideal", voltage, current))
        result = fit_iv_curve("ideal", voltage, current, params, source_path="input.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            curve_path, parameter_path = export_fit_result(result, tmpdir)
            curve_text = Path(curve_path).read_text(encoding="utf-8")
            payload = json.loads(Path(parameter_path).read_text(encoding="utf-8"))

        self.assertIn(
            "V_V,I_measured_A,I_fitted_total_A,I_fitted_ideal_diode_A,"
            "I_fitted_ohmic_A,I_fitted_TAT_A,residual_A,residual_scaled",
            curve_text,
        )
        self.assertEqual(payload["source_path"], "input.csv")
        self.assertEqual(payload["model"], "ideal")
        self.assertIn("component_currents", payload)
        self.assertIn("I0", payload["parameters"])
        self.assertIn("rmse_A", payload["metrics"])


class GuiSmokeTests(unittest.TestCase):
    def test_gui_module_exposes_application_class_and_main(self):
        import shockley_gui

        self.assertTrue(callable(shockley_gui.ShockleyFittingApp))
        self.assertTrue(callable(shockley_gui.main))


if __name__ == "__main__":
    unittest.main()
