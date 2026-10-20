@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
echo ========================================================
echo Running Doom SNN Unit and Integration Tests via pytest
echo ========================================================
python -m pytest tests/ -q
if errorlevel 1 (
    echo [ERROR] pytest suite failed!
    exit /b 1
)

echo ========================================================
echo Running A2C Smoke Training on DummyEnv
echo ========================================================
python scripts/train.py --config configs/base.yaml --dummy-env --smoke --device cpu
if errorlevel 1 (
    echo [ERROR] Train smoke test failed!
    exit /b 1
)

echo ========================================================
echo Running Hardware Resource Accounting Test
echo ========================================================
python scripts/resource_report.py --neurons 96 --track c_rsnn_ac
if errorlevel 1 (
    echo [ERROR] Resource report failed!
    exit /b 1
)

echo ========================================================
echo ALL SMOKE TESTS PASSED SUCCESSFULLY!
echo ========================================================
