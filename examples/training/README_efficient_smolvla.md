# Efficient SmolVLA + Fast Adaptation

## What changed in this revision

This revision updates the adaptation stage from GRPO to **Flow-GRPO** (group-relative policy optimization with a flow-matching objective inspired by flow-matching methods).

## Dataset integration: MOVE real-world format

`examples/training/train_efficient_smolvla.py` includes `MoveDataset`, which loads:
- `camera/color/Camera` (vision stream),
- `arm/jointStatePosition/joint_single` (robot action/state supervision).

Run with MOVE + Flow-GRPO:

```bash
PYTHONPATH=src python examples/training/train_efficient_smolvla.py \
  --config examples/training/configs/efficient_smolvla_move_flow_grpo.yaml \
  --out results/flow_grpo_move_finetune.json
```

Run local sanity training (no external dataset download):

```bash
PYTHONPATH=src python examples/training/train_efficient_smolvla.py \
  --config examples/training/configs/efficient_smolvla_default.yaml
```


## SmolVLA backbone + LoRA updates

The training script now supports using **SmolVLA/SmolVLM as the backbone** directly:
- set `use_smolvla_backbone: true`,
- choose `smolvla_model_name`,
- set LoRA rank via `smolvla_lora_rank`,
- optionally unfreeze the last text layers with `smolvla_unfreeze_last_n_layers`.

This keeps most backbone params frozen while enabling on-the-fly adaptation through:
1. LoRA updates on VLM projection/MLP layers,
2. compact adaptation heads (`EfficientSmolVLA`),
3. Flow-GRPO adaptation objective.

## Flow-GRPO objective

`policy_objective: flow_grpo` enables:
1. Group sampling from the MDN action distribution,
2. Relative advantage normalization across sampled candidates,
3. Flow-matching loss over straight-line interpolation from noise to policy endpoint,
4. A trust penalty (`flow_grpo_beta`) that keeps updates close to BC supervision,
5. A flow contribution scale (`flow_grpo_matching_weight`).

## Why the loss can still be negative

The BC term uses continuous MDN negative log-likelihood. Unlike discrete CE, continuous log-density can exceed 0 around narrow, high-density modes, so the resulting NLL can be negative. This is expected behavior for continuous density models.

Track these alongside total loss:
- `loss_bc`,
- `loss_flow_grpo`,
- `log_sigma` ranges,
- downstream task metrics (success rate / action error).

## Ablation scaffold

```bash
bash results/run_ablation.sh
```

Outputs:
- `results/baseline_proposed.json`
- `results/adapter_lora_only.json`
- `results/flow_grpo_move_finetune.json`
