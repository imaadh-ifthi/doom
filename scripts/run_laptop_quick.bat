@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
echo ========================================================
echo Running Quick Laptop SNN Training (Track C RSNN, beta=0.75)
echo ========================================================
python scripts/train.py --config configs/laptop_cpu.yaml --track c --beta 0.75 --episodes 50 --eval-every 25
if errorlevel 1 exit /b 1

echo ========================================================
echo Evaluating Baselines and Model
echo ========================================================
python scripts/eval.py --checkpoint runs/laptop_cpu/latest.pt --config configs/laptop_cpu.yaml --episodes 20 --seeds 6201 6202
echo Quick laptop run complete!
