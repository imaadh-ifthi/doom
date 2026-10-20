#!/usr/bin/env bash
set -e
export KMP_DUPLICATE_LIB_OK=TRUE

echo "========================================================"
echo "Starting Full Server SNN Benchmark Training Battery"
echo "Targeting ViZDoom BASIC under SNAP-V constraints"
echo "========================================================"

SEEDS=(6201 6202 6203)

# 1. Track C: RSNN Actor-Critic (beta=0.5, float)
for seed in "${SEEDS[@]}"; do
  echo "--> Running Track C (beta=0.5, float) Seed ${seed}..."
  python scripts/train.py --config configs/track_c_rsnn_ac_beta05_float.yaml --seed "$seed" --device auto
done

# 2. Track C: RSNN Actor-Critic (beta=0.75, float)
for seed in "${SEEDS[@]}"; do
  echo "--> Running Track C (beta=0.75, float) Seed ${seed}..."
  python scripts/train.py --config configs/track_c_rsnn_ac_beta075_float.yaml --seed "$seed" --device auto
done

# 3. Track C: RSNN Actor-Critic (beta=0.75, INT16 QAT)
for seed in "${SEEDS[@]}"; do
  echo "--> Running Track C (beta=0.75, INT16 QAT) Seed ${seed}..."
  python scripts/train.py --config configs/track_c_rsnn_ac_beta075_int16_qat.yaml --seed "$seed" --device auto
done

# 4. Track D: Feed-Forward SNN Ablation
for seed in "${SEEDS[@]}"; do
  echo "--> Running Track D (FF SNN Ablation) Seed ${seed}..."
  python scripts/train.py --config configs/track_d_ff_snn.yaml --seed "$seed" --device auto
done

# 5. Benchmark Comparison Summary Generation
echo "--> Generating Unified Benchmark Summary..."
python scripts/benchmark.py --config configs/benchmark.yaml --device auto

echo "All server training runs and benchmarks finished successfully!"
