# Running on Laptop and Server

## 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements-doom-snn.txt

# On Windows platforms, set the OpenMP flag:
set KMP_DUPLICATE_LIB_OK=TRUE  # CMD
$env:KMP_DUPLICATE_LIB_OK="TRUE"  # PowerShell
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/macOS
```

---

## 2. Quickstart Execution Commands

### 2.1 Smoke Test (Validates forward, backward, dummy env, and tests)
```bash
# Windows
scripts\run_smoke_tests.bat

# Linux / macOS
bash scripts/run_smoke_tests.sh
```

### 2.2 Laptop Quick Training (Track C RSNN, 50 episodes)
```bash
python scripts/train.py --config configs/laptop_cpu.yaml --track c --beta 0.75 --episodes 50
```

### 2.3 Server Full Training Battery (CUDA / 500 episodes)
Use `tmux` or `screen` for persistent background training:
```bash
tmux new -s doom_snn
bash scripts/run_server_full.sh
```

Or execute individual tracks:
```bash
# Track C (beta=0.75, float)
python scripts/train.py --config configs/track_c_rsnn_ac_beta075_float.yaml --device cuda --seed 6201

# Track C (beta=0.75, INT16 QAT)
python scripts/train.py --config configs/track_c_rsnn_ac_beta075_int16_qat.yaml --device cuda --seed 6201

# Track D (Feed-Forward SNN Ablation)
python scripts/train.py --config configs/track_d_ff_snn.yaml --device cuda --seed 6201
```

### 2.4 Resuming Training from Checkpoint
```bash
python scripts/train.py --config configs/track_c_rsnn_ac_beta075_float.yaml --resume runs/track_c_rsnn_ac_beta075_float/latest.pt
```

### 2.5 Evaluating Checkpoints
```bash
python scripts/eval.py --checkpoint runs/track_c_rsnn_ac_beta075_float/best.pt --episodes 100 --seeds 6201 6202 6203 --robustness
```

### 2.6 Running Benchmark Summary
```bash
python scripts/benchmark.py --config configs/benchmark.yaml --device auto
```

### 2.7 Hardware Resource Report
```bash
python scripts/resource_report.py --neurons 96 --track c_rsnn_ac
```
