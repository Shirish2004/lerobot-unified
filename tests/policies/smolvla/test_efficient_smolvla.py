import torch

from lerobot.policies.smolvla.efficient_smolvla import (
    BackboneLoRAConfig,
    EfficientSmolVLA,
    LossWeights,
    MultiObjectiveLoss,
    apply_lora_to_linear_layers,
)


def test_efficient_smolvla_shapes():
    model = EfficientSmolVLA(hidden_size=64, action_dim=7, latent_dim=16, discrete_vocab_size=32, chunk_size=4)
    tokens = torch.randn(3, 4, 64)
    out = model.predict_chunk(tokens)

    assert out["latent"].shape == (3, 16)
    assert out["mu"].shape == (3, 4, 7)
    assert out["log_sigma"].shape == (3, 4, 7)
    assert out["values"].shape == (3,)
    assert out["discrete_logits"].shape == (3, 32)


def test_multi_objective_loss():
    objective = MultiObjectiveLoss(LossWeights())
    terms = {
        "bc": torch.tensor(1.0),
        "align": torch.tensor(2.0),
        "temporal": torch.tensor(3.0),
        "distill": torch.tensor(4.0),
    }
    total, logs = objective(terms)

    assert total.item() > 0
    assert "loss_total" in logs
    assert "loss_bc" in logs


def test_rollout_chunks_runs():
    model = EfficientSmolVLA(hidden_size=48, action_dim=6, latent_dim=12, chunk_size=3)
    token_sequence = torch.randn(2, 9, 48)
    outputs = model.rollout_chunks(token_sequence, async_stride=3)

    assert len(outputs) == 3
    assert all("mu" in out for out in outputs)


def test_apply_lora_to_linear_layers():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )
    replaced = apply_lora_to_linear_layers(
        model,
        BackboneLoRAConfig(rank=2, alpha=4.0, target_modules=("0", "2")),
        freeze_non_lora=True,
    )

    assert len(replaced) == 2
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in name for name in trainable)
