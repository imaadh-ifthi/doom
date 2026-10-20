#!/usr/bin/env bash
set -e
export KMP_DUPLICATE_LIB_OK=TRUE

echo "========================================================"
echo "Running Doom SNN Unit and Integration Tests via pytest"
echo "========================================================"
python -m pytest tests/ -q

echo "========================================================"
echo "Running A2C Smoke Training on DummyEnv"
echo "========================================================"
python scripts/train.py --config configs/base.yaml --dummy-env --smoke --device cpu

echo "========================================================"
echo "Running Hardware Resource Accounting Test"
echo "========================================================"
python scripts/resource_report.py --neurons 96 --track c_rsnn_ac

echo "========================================================"
echo "ALL SMOKE TESTS PASSED SUCCESSFULLY!"
echo "========================================================"
