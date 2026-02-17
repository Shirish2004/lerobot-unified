"""Train EfficientSmolVLA with staged BC/Flow-GRPO objectives and optional MOVE dataset loading."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lerobot.policies.smolvla.efficient_smolvla import (
    BackboneLoRAConfig,
    EfficientSmolVLA,
    LossWeights,
    MultiObjectiveLoss,
    SmolVLAAdaptiveModel,
    apply_lora_to_linear_layers,
    freeze_except_last_n_layers,
)


@dataclass
class TrainConfig:
    seed: int
    steps: int
    batch_size: int
    hidden_size: int
    action_dim: int
    latent_dim: int
    learning_rate: float
    adapter_learning_rate: float
    chunk_size: int
    lora_rank: int
    policy_objective: str
    flow_grpo_group_size: int
    flow_grpo_beta: float
    flow_grpo_matching_weight: float
    synthetic_ratio: float
    use_smolvla_backbone: bool
    smolvla_model_name: str
    smolvla_load_weights: bool
    smolvla_unfreeze_last_n_layers: int
    smolvla_lora_rank: int
    dataset_name: str | None
    dataset_config: str | None
    dataset_split: str
    max_dataset_samples: int
    device: str
    log_every: int


class ToyVLADataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, hidden_size: int, action_dim: int):
        self.tokens = torch.randn(n_samples, seq_len, hidden_size)
        self.actions = torch.tanh(self.tokens[:, :, :action_dim].mean(dim=1))

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"tokens": self.tokens[index], "actions": self.actions[index]}


class MoveDataset(Dataset):
    """Adapter for MOVE real-world pick-and-place dataset card structure."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        split: str,
        hidden_size: int,
        action_dim: int,
        seq_len: int,
        max_dataset_samples: int,
    ):
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, dataset_config, split=split)
        if max_dataset_samples > 0:
            dataset = dataset.select(range(min(max_dataset_samples, len(dataset))))
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.dataset)

    def _image_to_token(self, image: Any) -> Tensor:
        image_tensor = torch.as_tensor(image)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.float() / 255.0
            token = image_tensor.mean(dim=(0, 1))
        else:
            token = image_tensor.float().flatten()
        if token.shape[0] >= self.hidden_size:
            return token[: self.hidden_size]
        padded = torch.zeros(self.hidden_size, dtype=token.dtype)
        padded[: token.shape[0]] = token
        return padded

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.dataset[index]
        image = row["camera/color/Camera"]
        joint = torch.tensor(row["arm/jointStatePosition/joint_single"], dtype=torch.float32)

        token = self._image_to_token(image)
        tokens = token.unsqueeze(0).repeat(self.seq_len, 1)
        action = joint[: self.action_dim]
        if action.shape[0] < self.action_dim:
            action = F.pad(action, (0, self.action_dim - action.shape[0]))

        return {"tokens": tokens, "actions": action}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> TrainConfig:
    with open(path) as handle:
        data = yaml.safe_load(handle)
    return TrainConfig(**data)


def _sample_mdn_actions(params: dict[str, Tensor], group_size: int) -> tuple[list[Tensor], Tensor]:
    sigma = torch.exp(params["log_sigma"])
    mix = torch.distributions.Categorical(logits=params["logits"])
    sampled_actions: list[Tensor] = []
    rewards = []
    batch_size = params["mu"].shape[0]
    indices = torch.arange(batch_size, device=params["mu"].device)
    for _ in range(group_size):
        comp_idx = mix.sample()
        comp_mu = params["mu"][indices, comp_idx]
        comp_sigma = sigma[indices, comp_idx]
        sampled_action = comp_mu + torch.randn_like(comp_mu) * comp_sigma
        sampled_actions.append(sampled_action)
    return sampled_actions, torch.empty(0, device=params["mu"].device)


def _flow_grpo_loss(
    batch: dict[str, Tensor],
    params: dict[str, Tensor],
    group_size: int,
    beta: float,
    matching_weight: float,
) -> Tensor:
    """Flow-GRPO objective: group-relative weighting + flow-matching consistency."""
    sampled_actions, _ = _sample_mdn_actions(params, group_size)
    rewards = [-(sampled - batch["actions"]).pow(2).mean(dim=-1) for sampled in sampled_actions]
    reward_tensor = torch.stack(rewards, dim=1)
    advantages = (reward_tensor - reward_tensor.mean(dim=1, keepdim=True)) / (
        reward_tensor.std(dim=1, keepdim=True) + 1e-6
    )

    endpoint = params["mu"].mean(dim=1)
    flow_losses = []
    for idx, sampled_action in enumerate(sampled_actions):
        t = torch.rand(endpoint.shape[0], 1, device=endpoint.device)
        x0 = torch.randn_like(endpoint)
        # Straight-line interpolation between noise and model endpoint.
        _xt = (1 - t) * x0 + t * endpoint
        target_velocity = sampled_action - x0
        predicted_velocity = endpoint - x0
        fm_loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none").mean(dim=-1)
        weighted = fm_loss * advantages[:, idx]
        flow_losses.append(weighted.mean())

    flow_policy_loss = torch.stack(flow_losses).mean() * matching_weight
    trust_penalty = beta * F.mse_loss(endpoint, batch["actions"])
    return flow_policy_loss + trust_penalty


def collect_loss_terms(
    model: EfficientSmolVLA, batch: dict[str, Tensor], objective: MultiObjectiveLoss, cfg: TrainConfig
) -> tuple[Tensor, dict[str, float]]:
    out = _predict_chunk(model, batch["tokens"])
    params = {"logits": out["logits"], "mu": out["mu"], "log_sigma": out["log_sigma"]}

    bc = model.continuous_head.nll(params, batch["actions"])
    txt_anchor = batch["tokens"].mean(dim=1)[:, : out["latent"].shape[-1]]
    align = objective.info_nce(out["latent"], txt_anchor)
    temporal = (out["latent"][1:] - out["latent"][:-1]).pow(2).mean() if out["latent"].shape[0] > 1 else bc * 0
    distill = F.mse_loss(out["bayes_action"], batch["actions"].detach())
    synthetic = F.smooth_l1_loss(out["mu"].mean(dim=1), batch["actions"])

    flow_grpo = bc * 0
    if cfg.policy_objective.lower() == "flow_grpo":
        flow_grpo = _flow_grpo_loss(
            batch=batch,
            params=params,
            group_size=cfg.flow_grpo_group_size,
            beta=cfg.flow_grpo_beta,
            matching_weight=cfg.flow_grpo_matching_weight,
        )

    total, logs = objective(
        {
            "bc": bc,
            "align": align,
            "temporal": temporal,
            "distill": distill,
            "synthetic": synthetic,
            "ppo": flow_grpo,
        }
    )
    logs["loss_flow_grpo"] = flow_grpo.detach().item()
    return total, logs


def build_dataloader(cfg: TrainConfig) -> DataLoader:
    dataset: Dataset
    if cfg.dataset_name and cfg.dataset_config:
        try:
            dataset = MoveDataset(
                dataset_name=cfg.dataset_name,
                dataset_config=cfg.dataset_config,
                split=cfg.dataset_split,
                hidden_size=cfg.hidden_size,
                action_dim=cfg.action_dim,
                seq_len=cfg.chunk_size,
                max_dataset_samples=cfg.max_dataset_samples,
            )
            print(f"Loaded MOVE dataset: {cfg.dataset_name}/{cfg.dataset_config} ({cfg.dataset_split})")
        except Exception as error:
            print(f"Warning: failed to load MOVE dataset ({error}), falling back to ToyVLADataset.")
            dataset = ToyVLADataset(
                n_samples=512,
                seq_len=cfg.chunk_size,
                hidden_size=cfg.hidden_size,
                action_dim=cfg.action_dim,
            )
    else:
        dataset = ToyVLADataset(
            n_samples=512,
            seq_len=cfg.chunk_size,
            hidden_size=cfg.hidden_size,
            action_dim=cfg.action_dim,
        )
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)




def _predict_chunk(model: nn.Module, tokens: Tensor) -> dict[str, Tensor]:
    if hasattr(model, "predict_chunk"):
        return model.predict_chunk(tokens)
    if hasattr(model, "forward_from_embeddings"):
        return model.forward_from_embeddings(tokens)
    raise TypeError("Unsupported model type for predict_chunk")


def build_policy_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    if not cfg.use_smolvla_backbone:
        return EfficientSmolVLA(
            hidden_size=cfg.hidden_size,
            action_dim=cfg.action_dim,
            latent_dim=cfg.latent_dim,
            lora_rank=cfg.lora_rank,
            chunk_size=cfg.chunk_size,
        ).to(device)

    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    backbone = SmolVLMWithExpertModel(
        model_id=cfg.smolvla_model_name,
        load_vlm_weights=cfg.smolvla_load_weights,
        train_expert_only=False,
        freeze_vision_encoder=False,
    )
    vlm_model = backbone.get_vlm_model()
    lora_cfg = BackboneLoRAConfig(rank=cfg.smolvla_lora_rank)
    replaced = apply_lora_to_linear_layers(vlm_model, lora_cfg, freeze_non_lora=True)

    if cfg.smolvla_unfreeze_last_n_layers > 0:
        freeze_except_last_n_layers(vlm_model.text_model.layers, cfg.smolvla_unfreeze_last_n_layers)

    hidden_size = backbone.config.text_config.hidden_size
    model = SmolVLAAdaptiveModel(
        backbone=backbone,
        hidden_size=hidden_size,
        action_dim=cfg.action_dim,
        latent_dim=cfg.latent_dim,
        lora_rank=cfg.lora_rank,
    ).to(device)
    print(f"Initialized SmolVLA backbone with {len(replaced)} LoRA-injected layers.")
    return model


def train(cfg: TrainConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = build_policy_model(cfg, device)

    loader = build_dataloader(cfg)

    loss_weights = LossWeights()
    if cfg.policy_objective.lower() == "flow_grpo":
        loss_weights.ppo = 1.0
    objective = MultiObjectiveLoss(loss_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    model.train()
    logs: dict[str, float] = {}
    global_step = 0
    while global_step < cfg.steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total, step_logs = collect_loss_terms(model, batch, objective, cfg)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            logs = step_logs
            global_step += 1
            if global_step % cfg.log_every == 0:
                print(
                    f"step={global_step} loss={step_logs['loss_total']:.4f} "
                    f"flow_grpo={step_logs['loss_flow_grpo']:.4f}"
                )
            if global_step >= cfg.steps:
                break

    return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/efficient_smolvla_train_log.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logs = train(cfg)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"config": asdict(cfg), "metrics": logs}, indent=2))
    print(f"Wrote log to {output_path}")


if __name__ == "__main__":
    main()
