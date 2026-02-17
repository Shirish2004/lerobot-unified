#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python examples/training/train_efficient_smolvla.py \
  --config examples/training/configs/efficient_smolvla_default.yaml \
  --out results/baseline_proposed.json

PYTHONPATH=src python examples/training/train_efficient_smolvla.py \
  --config examples/training/configs/efficient_smolvla_default.yaml \
  --out results/adapter_lora_only.json

PYTHONPATH=src python examples/training/train_efficient_smolvla.py \
  --config examples/training/configs/efficient_smolvla_move_flow_grpo.yaml \
  --out results/flow_grpo_move_finetune.json

cat <<'EOF'
Ablation entries produced:
- baseline_proposed.json
- adapter_lora_only.json
- flow_grpo_move_finetune.json
EOF
