import importlib.util
import sys
from pathlib import Path

import torch


def _load_train_module():
    module_path = Path("examples/training/train_efficient_smolvla.py")
    spec = importlib.util.spec_from_file_location("train_efficient_smolvla", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_loss_terms_with_flow_grpo():
    train_mod = _load_train_module()

    cfg = train_mod.TrainConfig(
        seed=0,
        steps=1,
        batch_size=2,
        hidden_size=64,
        action_dim=7,
        latent_dim=16,
        learning_rate=1e-4,
        adapter_learning_rate=1e-4,
        chunk_size=8,
        lora_rank=4,
        policy_objective="flow_grpo",
        flow_grpo_group_size=3,
        flow_grpo_beta=0.01,
        flow_grpo_matching_weight=1.0,
        synthetic_ratio=0.2,
        use_smolvla_backbone=False,
        smolvla_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        smolvla_load_weights=False,
        smolvla_unfreeze_last_n_layers=0,
        smolvla_lora_rank=4,
        dataset_name=None,
        dataset_config=None,
        dataset_split="train",
        max_dataset_samples=0,
        device="cpu",
        log_every=10,
    )

    model = train_mod.EfficientSmolVLA(hidden_size=64, action_dim=7, latent_dim=16, lora_rank=4, chunk_size=8)
    objective = train_mod.MultiObjectiveLoss(train_mod.LossWeights())
    batch = {
        "tokens": torch.randn(2, 8, 64),
        "actions": torch.randn(2, 7),
    }

    loss, logs = train_mod.collect_loss_terms(model, batch, objective, cfg)
    assert torch.is_tensor(loss)
    assert "loss_flow_grpo" in logs


def test_build_dataloader_toy():
    train_mod = _load_train_module()
    cfg = train_mod.TrainConfig(
        seed=0,
        steps=1,
        batch_size=2,
        hidden_size=32,
        action_dim=6,
        latent_dim=12,
        learning_rate=1e-4,
        adapter_learning_rate=1e-4,
        chunk_size=8,
        lora_rank=4,
        policy_objective="bc",
        flow_grpo_group_size=3,
        flow_grpo_beta=0.01,
        flow_grpo_matching_weight=0.5,
        synthetic_ratio=0.2,
        use_smolvla_backbone=False,
        smolvla_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        smolvla_load_weights=False,
        smolvla_unfreeze_last_n_layers=0,
        smolvla_lora_rank=4,
        dataset_name=None,
        dataset_config=None,
        dataset_split="train",
        max_dataset_samples=0,
        device="cpu",
        log_every=10,
    )

    loader = train_mod.build_dataloader(cfg)
    batch = next(iter(loader))
    assert batch["tokens"].shape[0] == 2
    assert batch["actions"].shape[-1] == 6
