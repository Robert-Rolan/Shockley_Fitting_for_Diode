# Shockley Fitting for Diode

一个用于二极管 I-V 曲线 Shockley 方程拟合的 Windows/Python 小工具。项目包含图形界面、拟合模型、单元测试和一个公开标样数据文件 `control.csv`。

## 功能

- 从 CSV 读取 I-V 数据：第一列为电压，最后一列为电流。
- 支持理想 Shockley 二极管模型和带 TAT 项的非理想广义模型。
- 在图形界面中调整初值、上下限、温度、内建电势和偏压区间权重。
- 导出拟合曲线 `fitted_curve.csv` 和参数结果 `fit_parameters.json`。

## 从源码运行

建议使用 Python 3.11 或更新版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python shockley_gui.py
```

也可以双击 `start_shockley_fitting_tool.bat`，前提是当前环境已经安装了 `requirements.txt` 中的依赖。

## 运行测试

```powershell
python -B -m unittest discover -s tests -v
```

测试会使用仓库中的公开标样 `control.csv`，不会依赖个人实验数据。

## 可选打包

如果需要生成 Windows 可执行程序，可安装 PyInstaller 后运行：

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --noconsole --onedir --name "Shockley Equation Fitting" shockley_gui.py
```

生成的 `build/`、`dist/`、`.zip` 和 `.exe` 属于构建产物，不提交到仓库。
