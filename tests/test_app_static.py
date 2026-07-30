from pathlib import Path
import ast

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_file_compiles():
    source = APP.read_text(encoding="utf-8")
    compile(source, str(APP), "exec")


def test_expected_dashboard_functions_exist():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    expected = {
        "make_synthetic_forest_boxes",
        "forecast_tree_groups",
        "run_monte_carlo",
        "build_deterministic_forecast_chart",
        "build_uncertainty_chart",
        "main",
    }
    assert expected.issubset(function_names)


def test_dashboard_title_present():
    source = APP.read_text(encoding="utf-8")
    assert "Timber forecasting" in source
    assert "Turning a deterministic forest timber forecast into an uncertainty band" in source
