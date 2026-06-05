from __future__ import annotations

import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from shockley_models import (
    DEFAULT_FIXED_VALUES,
    FIT_WEIGHT_RANGES,
    MODEL_SPECS,
    IVData,
    FitResult,
    ModelParameters,
    ParameterRow,
    ShockleyFitError,
    default_parameter_rows,
    default_fit_range_weights,
    export_fit_result,
    fit_iv_curve,
    prepare_log_preview,
    read_iv_csv,
)


class ShockleyFittingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Shockley I-V 曲线拟合工具")
        self.root.geometry("1180x760")
        self.root.minsize(980, 620)

        self.data: IVData | None = None
        self.fit_result: FitResult | None = None
        self.parameter_widgets: dict[str, dict[str, tk.StringVar]] = {}
        self.parameter_rows: list[ParameterRow] = []
        if getattr(sys, "frozen", False):
            self.last_directory = Path(sys.executable).resolve().parent
        else:
            self.last_directory = Path(__file__).resolve().parent

        self.model_options = [(key, spec.label) for key, spec in MODEL_SPECS.items()]
        self.model_key_by_label = {label: key for key, label in self.model_options}
        self.model_label_by_key = {key: label for key, label in self.model_options}
        self.model_var = tk.StringVar(value=self.model_label_by_key["ideal"])
        self.temperature_var = tk.StringVar(value=_format_float(DEFAULT_FIXED_VALUES["T"]))
        self.vbi_var = tk.StringVar(value=_format_float(DEFAULT_FIXED_VALUES["Vbi"]))
        self.fit_weight_vars = {
            key: tk.StringVar(value=_format_float(value))
            for key, value in default_fit_range_weights().items()
        }
        self.status_var = tk.StringVar(value="请选择 CSV 数据文件。")

        self._build_layout()
        self._refresh_parameter_table()
        self._redraw_plot()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(0, weight=1)

        file_frame = ttk.LabelFrame(controls, text="数据与操作", padding=8)
        file_frame.grid(row=0, column=0, sticky="ew")
        file_frame.columnconfigure(0, weight=1)

        ttk.Button(file_frame, text="导入 I-V CSV 数据", command=self.load_data).grid(row=0, column=0, sticky="ew", pady=3)
        ttk.Button(file_frame, text="执行拟合", command=self.run_fit).grid(row=1, column=0, sticky="ew", pady=3)
        self.export_button = ttk.Button(file_frame, text="导出拟合数据", command=self.export_result, state=tk.DISABLED)
        self.export_button.grid(row=2, column=0, sticky="ew", pady=3)

        model_frame = ttk.LabelFrame(controls, text="拟合模型", padding=8)
        model_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        model_frame.columnconfigure(0, weight=1)
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[label for _, label in self.model_options],
            state="readonly",
            width=42,
        )
        model_combo.grid(row=0, column=0, sticky="ew")
        model_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_model_change())

        fixed_frame = ttk.LabelFrame(controls, text="固定输入", padding=8)
        fixed_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        fixed_frame.columnconfigure(1, weight=1)
        ttk.Label(fixed_frame, text="T (K)").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(fixed_frame, textvariable=self.temperature_var, width=16).grid(row=0, column=1, sticky="ew", pady=3)
        self.vbi_label = ttk.Label(fixed_frame, text="Vbi (V)")
        self.vbi_entry = ttk.Entry(fixed_frame, textvariable=self.vbi_var, width=16)
        self.vbi_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        self.vbi_entry.grid(row=1, column=1, sticky="ew", pady=3)

        weight_frame = ttk.LabelFrame(controls, text="偏压区间拟合权重（log坐标）", padding=8)
        weight_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        weight_frame.columnconfigure(1, weight=1)
        for row_index, fit_range in enumerate(FIT_WEIGHT_RANGES):
            ttk.Label(weight_frame, text=fit_range.label).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=3,
            )
            ttk.Entry(weight_frame, textvariable=self.fit_weight_vars[fit_range.key], width=12).grid(
                row=row_index,
                column=1,
                sticky="ew",
                pady=3,
            )

        parameter_frame = ttk.LabelFrame(controls, text="性能系数初值与上下限", padding=8)
        parameter_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        controls.rowconfigure(4, weight=1)
        self.parameter_table = ttk.Frame(parameter_frame)
        self.parameter_table.grid(row=0, column=0, sticky="nsew")

        plot_frame = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(7.0, 5.2), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.grid(row=1, column=0, sticky="ew")

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def current_model_key(self) -> str:
        return self.model_key_by_label[self.model_var.get()]

    def on_model_change(self) -> None:
        self.fit_result = None
        self.export_button.configure(state=tk.DISABLED)
        self._refresh_parameter_table()
        self._redraw_plot()
        self.status_var.set(f"已切换到 {self.model_var.get()}。")

    def load_data(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 I-V CSV 数据",
            initialdir=str(self.last_directory),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.data = read_iv_csv(path)
            self.fit_result = None
            self.export_button.configure(state=tk.DISABLED)
            self.last_directory = Path(path).resolve().parent
            self._refresh_parameter_table()
            self._redraw_plot()
            self.status_var.set(f"已导入 {Path(path).name}，共 {self.data.voltage.size} 个数据点。")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))
            self.status_var.set("导入失败。")

    def run_fit(self) -> None:
        if self.data is None:
            messagebox.showinfo("缺少数据", "请先导入 I-V CSV 数据。")
            return
        try:
            parameters = self._read_parameters_from_ui()
            self.status_var.set("正在拟合，请稍候...")
            self.root.update_idletasks()
            result = fit_iv_curve(
                self.current_model_key(),
                self.data.voltage,
                self.data.current,
                parameters,
                source_path=self.data.source_path,
            )
            self.fit_result = result
            self._write_fit_values(result)
            self.export_button.configure(state=tk.NORMAL)
            self._redraw_plot()
            self.status_var.set(
                "拟合完成：RMSE={:.4g} A，负向平均log偏差={:.3g} decade，迭代次数={}。".format(
                    result.metrics["rmse_A"],
                    result.metrics.get("reverse_log_mean_abs_decades", float("nan")),
                    result.nfev,
                )
            )
            if not result.success:
                messagebox.showwarning("拟合未完全收敛", result.message)
        except Exception as exc:
            messagebox.showerror("拟合失败", str(exc))
            self.status_var.set("拟合失败。")

    def export_result(self) -> None:
        if self.fit_result is None:
            messagebox.showinfo("缺少拟合结果", "请先执行拟合。")
            return
        directory = filedialog.askdirectory(
            title="选择导出文件夹",
            initialdir=str(self.last_directory),
        )
        if not directory:
            return
        try:
            curve_path, parameter_path = export_fit_result(self.fit_result, directory)
            self.status_var.set(f"已导出：{Path(curve_path).name} 和 {Path(parameter_path).name}。")
            messagebox.showinfo("导出完成", f"已导出：\n{curve_path}\n{parameter_path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            self.status_var.set("导出失败。")

    def _refresh_parameter_table(self) -> None:
        for child in self.parameter_table.winfo_children():
            child.destroy()
        self.parameter_widgets.clear()

        model_key = self.current_model_key()
        self._toggle_vbi(model_key == "tat")
        voltage, current = self._data_for_defaults()
        self.parameter_rows = default_parameter_rows(
            model_key,
            voltage,
            current,
            fixed=self._safe_fixed_values(),
        )

        headers = ["参数", "单位", "初始值", "下限", "上限", "拟合值"]
        for col, text in enumerate(headers):
            ttk.Label(self.parameter_table, text=text, font=("", 9, "bold")).grid(
                row=0, column=col, sticky="w", padx=4, pady=(0, 4)
            )
        for row_index, row in enumerate(self.parameter_rows, start=1):
            ttk.Label(self.parameter_table, text=row.label).grid(row=row_index, column=0, sticky="w", padx=4, pady=3)
            ttk.Label(self.parameter_table, text=row.unit or "-").grid(row=row_index, column=1, sticky="w", padx=4, pady=3)
            initial_var = tk.StringVar(value=_format_float(row.initial))
            lower_var = tk.StringVar(value=_format_float(row.lower))
            upper_var = tk.StringVar(value=_format_float(row.upper))
            value_var = tk.StringVar(value="")
            ttk.Entry(self.parameter_table, textvariable=initial_var, width=13).grid(
                row=row_index, column=2, sticky="ew", padx=4, pady=3
            )
            ttk.Entry(self.parameter_table, textvariable=lower_var, width=13).grid(
                row=row_index, column=3, sticky="ew", padx=4, pady=3
            )
            ttk.Entry(self.parameter_table, textvariable=upper_var, width=13).grid(
                row=row_index, column=4, sticky="ew", padx=4, pady=3
            )
            ttk.Label(self.parameter_table, textvariable=value_var, width=14).grid(
                row=row_index, column=5, sticky="w", padx=4, pady=3
            )
            self.parameter_widgets[row.name] = {
                "initial": initial_var,
                "lower": lower_var,
                "upper": upper_var,
                "value": value_var,
            }

    def _read_parameters_from_ui(self) -> ModelParameters:
        rows: list[ParameterRow] = []
        source_rows = {row.name: row for row in self.parameter_rows}
        for name in MODEL_SPECS[self.current_model_key()].parameter_names:
            source = source_rows[name]
            widgets = self.parameter_widgets[name]
            initial = _parse_float(widgets["initial"].get(), f"{name} 初始值")
            lower = _parse_float(widgets["lower"].get(), f"{name} 下限")
            upper = _parse_float(widgets["upper"].get(), f"{name} 上限")
            if lower >= upper:
                raise ShockleyFitError(f"{name} 的下限必须小于上限。")
            rows.append(source.with_values(initial=initial, lower=lower, upper=upper))
        return ModelParameters(
            rows=rows,
            fixed=self._read_fixed_values_from_ui(),
            fit_range_weights=self._read_fit_range_weights_from_ui(),
        )

    def _read_fixed_values_from_ui(self) -> dict[str, float]:
        temperature = _parse_float(self.temperature_var.get(), "T")
        if temperature <= 0:
            raise ShockleyFitError("T 必须大于 0 K。")
        fixed = {"T": temperature}
        if self.current_model_key() == "tat":
            fixed["Vbi"] = _parse_float(self.vbi_var.get(), "Vbi")
        else:
            fixed["Vbi"] = _parse_float(self.vbi_var.get(), "Vbi")
        return fixed

    def _safe_fixed_values(self) -> dict[str, float]:
        try:
            return self._read_fixed_values_from_ui()
        except Exception:
            return dict(DEFAULT_FIXED_VALUES)

    def _read_fit_range_weights_from_ui(self) -> dict[str, float]:
        weights = {}
        for fit_range in FIT_WEIGHT_RANGES:
            value = _parse_float(self.fit_weight_vars[fit_range.key].get(), f"{fit_range.label} 权重")
            if value < 0:
                raise ShockleyFitError(f"{fit_range.label} 权重必须大于或等于 0。")
            weights[fit_range.key] = value
        return weights

    def _write_fit_values(self, result: FitResult) -> None:
        for name, row in result.parameters.items():
            if name in self.parameter_widgets and row.value is not None:
                self.parameter_widgets[name]["value"].set(_format_float(row.value))

    def _data_for_defaults(self) -> tuple:
        if self.data is not None:
            return self.data.voltage, self.data.current
        return (
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 1e-10, 1e-8, 1e-6],
        )

    def _toggle_vbi(self, visible: bool) -> None:
        if visible:
            self.vbi_label.grid()
            self.vbi_entry.grid()
        else:
            self.vbi_label.grid_remove()
            self.vbi_entry.grid_remove()

    def _redraw_plot(self) -> None:
        self.axes.clear()
        if self.data is None:
            self.axes.text(0.5, 0.5, "请导入 I-V CSV 数据", ha="center", va="center", transform=self.axes.transAxes)
            self.axes.set_axis_off()
        else:
            self.axes.set_axis_on()
            self.axes.plot(
                self.data.voltage,
                prepare_log_preview(self.data.current),
                marker="o",
                linestyle="-",
                markersize=4,
                linewidth=1.2,
                label="原始数据 |I|",
            )
            if self.fit_result is not None:
                self.axes.plot(
                    self.fit_result.voltage,
                    prepare_log_preview(self.fit_result.fitted_current),
                    linestyle="-",
                    linewidth=2.0,
                    label="总拟合曲线 |I_fit|",
                )
                component_styles = [
                    ("ideal_diode_A", "ideal-diode current |I|", "--", 1.6),
                    ("ohmic_A", "ohmic current |I|", ":", 1.8),
                    ("tat_A", "TAT current |I|", "-.", 1.6),
                ]
                for component_key, label, linestyle, linewidth in component_styles:
                    if self.fit_result.model_key == "ideal" and component_key != "ideal_diode_A":
                        continue
                    self.axes.plot(
                        self.fit_result.voltage,
                        prepare_log_preview(self.fit_result.component_currents[component_key]),
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=label,
                    )
            self.axes.set_xlabel("Voltage V (V)")
            self.axes.set_ylabel("Current |I| (A)")
            self.axes.set_yscale("log")
            self.axes.grid(True, which="both", linestyle=":", linewidth=0.7)
            self.axes.legend(loc="best")
            self.axes.set_title("I-V 曲线预览（对数 y 轴，仅显示取绝对值）")
        self.figure.tight_layout()
        self.canvas.draw_idle()


def _parse_float(raw_value: str, label: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ShockleyFitError(f"{label} 必须是数字。") from exc
    if not math.isfinite(value):
        raise ShockleyFitError(f"{label} 必须是有限数字。")
    return value


def _format_float(value: float) -> str:
    return "{:.6g}".format(float(value))


def main() -> None:
    root = tk.Tk()
    ShockleyFittingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
