@echo off
set KMP_DUPLICATE_LIB_OK=TRUE

echo ========================================================
echo Starting Full Server SNN Benchmark Training Battery
echo ========================================================

for %%S in (6201 6202 6203) do (
    echo --> Running Track C (beta=0.75, float) Seed %%S...
    python scripts/train.py --config configs/track_c_rsnn_ac_beta075_float.yaml --seed %%S --device auto
)

for %%S in (6201 6202 6203) do (
    echo --> Running Track C (beta=0.75, INT16 QAT) Seed %%S...
    python scripts/train.py --config configs/track_c_rsnn_ac_beta075_int16_qat.yaml --seed %%S --device auto
)

for %%S in (6201 6202 6203) do (
    echo --> Running Track D (FF SNN) Seed %%S...
    python scripts/train.py --config configs/track_d_ff_snn.yaml --seed %%S --device auto
)

echo --> Generating Unified Benchmark Summary...
python scripts/benchmark.py --config configs/benchmark.yaml --device auto

echo All server runs finished!
