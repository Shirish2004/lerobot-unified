from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Lightweight LoRA wrapper for a linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        if rank > 0:
            self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.01)
            self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.rank <= 0:
            return out
        update = x @ self.lora_a @ self.lora_b
        return out + self.scale * update


class AdapterBlock(nn.Module):
    """Bottleneck adapter block for parameter-efficient adaptation."""

    def __init__(self, dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        update = self.up(F.silu(self.down(self.norm(x))))
        return x + update


class GatedFusion(nn.Module):
    """Shallow fusion block mapping VLM hidden states to compact latent."""

    def __init__(self, hidden_size: int, latent_dim: int, num_layers: int = 2, lora_rank: int = 8):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = hidden_size
        for _ in range(num_layers):
            layers.extend([LoRALinear(in_dim, hidden_size, rank=lora_rank), nn.SiLU(), AdapterBlock(hidden_size)])
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.proj = LoRALinear(hidden_size, latent_dim, rank=lora_rank)
        self.gate = nn.Linear(hidden_size, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        pooled = tokens.mean(dim=1)
        hidden = self.mlp(pooled)
        latent = self.proj(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return latent * gate


class ContinuousMDNHead(nn.Module):
    """Small MDN action head for continuous control."""

    def __init__(self, latent_dim: int, action_dim: int, num_components: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.num_components = num_components
        out_dim = num_components * action_dim
        self.logits = nn.Linear(latent_dim, num_components)
        self.mu = nn.Linear(latent_dim, out_dim)
        self.log_sigma = nn.Linear(latent_dim, out_dim)

    def forward(self, latent: Tensor) -> dict[str, Tensor]:
        bsz = latent.shape[0]
        mu = self.mu(latent).view(bsz, self.num_components, self.action_dim)
        log_sigma = self.log_sigma(latent).view(bsz, self.num_components, self.action_dim).clamp(-5.0, 2.0)
        logits = self.logits(latent)
        return {"logits": logits, "mu": mu, "log_sigma": log_sigma}

    def nll(self, params: dict[str, Tensor], target: Tensor) -> Tensor:
        target = target[:, None, :]
        sigma = torch.exp(params["log_sigma"])
        comp_log_prob = -0.5 * (((target - params["mu"]) / sigma) ** 2 + 2 * params["log_sigma"]).sum(dim=-1)
        log_mix = F.log_softmax(params["logits"], dim=-1)
        return -torch.logsumexp(comp_log_prob + log_mix, dim=-1).mean()


class DiscreteActionHead(nn.Module):
    def __init__(self, latent_dim: int, vocab_size: int):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, vocab_size)

    def forward(self, latent: Tensor) -> Tensor:
        return self.classifier(latent)


class ValueHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.value = nn.Linear(latent_dim, 1)

    def forward(self, latent: Tensor) -> Tensor:
        return self.value(latent).squeeze(-1)


@dataclass
class LossWeights:
    bc: float = 1.0
    align: float = 0.05
    temporal: float = 0.02
    distill: float = 0.1
    ppo: float = 0.5
    synthetic: float = 0.05


class MultiObjectiveLoss(nn.Module):
    """Configurable weighted loss for BC + SSL + distillation + PPO + synthetic consistency."""

    def __init__(self, weights: LossWeights, temperature: float = 0.07):
        super().__init__()
        self.weights = weights
        self.temperature = temperature

    def info_nce(self, img_proj: Tensor, txt_proj: Tensor) -> Tensor:
        img_proj = F.normalize(img_proj, dim=-1)
        txt_proj = F.normalize(txt_proj, dim=-1)
        logits = img_proj @ txt_proj.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)) * 0.5

    def forward(self, terms: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        total = torch.tensor(0.0, device=next(iter(terms.values())).device)
        logs: dict[str, float] = {}
        for key, weight in vars(self.weights).items():
            if key in terms:
                value = terms[key]
                total = total + weight * value
                logs[f"loss_{key}"] = value.detach().item()
        logs["loss_total"] = total.detach().item()
        return total, logs


class BayesianLinearHead(nn.Module):
    """Tiny Bayesian last layer for rapid posterior updates from few examples."""

    def __init__(self, in_dim: int, out_dim: int, prior_var: float = 1.0):
        super().__init__()
        self.weight_mean = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.register_buffer("precision", torch.eye(in_dim) / prior_var)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight_mean.transpose(0, 1) + self.bias

    @torch.no_grad()
    def update_posterior(self, x: Tensor, y: Tensor, noise_var: float = 1e-2) -> None:
        cov_update = x.transpose(0, 1) @ x / noise_var
        self.precision = self.precision + cov_update
        rhs = x.transpose(0, 1) @ y / noise_var
        mean = torch.linalg.solve(self.precision, rhs)
        self.weight_mean.copy_(mean.transpose(0, 1))


class EfficientSmolVLA(nn.Module):
    """Compact SmolVLM-to-action architecture with fast adaptation hooks."""

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        latent_dim: int = 32,
        discrete_vocab_size: int | None = None,
        lora_rank: int = 8,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.fusion = GatedFusion(hidden_size, latent_dim, num_layers=2, lora_rank=lora_rank)
        self.fast_adapter = AdapterBlock(latent_dim, bottleneck_dim=max(4, latent_dim // 2))
        self.continuous_head = ContinuousMDNHead(latent_dim, action_dim)
        self.discrete_head = DiscreteActionHead(latent_dim, discrete_vocab_size) if discrete_vocab_size else None
        self.value_head = ValueHead(latent_dim)
        self.bayesian_head = BayesianLinearHead(latent_dim, action_dim)

    def encode(self, vlm_tokens: Tensor) -> Tensor:
        latent = self.fusion(vlm_tokens)
        return self.fast_adapter(latent)

    def predict_chunk(self, vlm_tokens: Tensor) -> dict[str, Tensor]:
        latent = self.encode(vlm_tokens)
        cont = self.continuous_head(latent)
        values = self.value_head(latent)
        bayes = self.bayesian_head(latent)
        out = {"latent": latent, "values": values, "bayes_action": bayes, **cont}
        if self.discrete_head is not None:
            out["discrete_logits"] = self.discrete_head(latent)
        return out

    def rollout_chunks(self, token_sequence: Tensor, async_stride: int = 1) -> list[dict[str, Tensor]]:
        outputs: list[dict[str, Tensor]] = []
        for idx in range(0, token_sequence.shape[1], async_stride):
            chunk_tokens = token_sequence[:, idx : idx + self.chunk_size]
            outputs.append(self.predict_chunk(chunk_tokens))
        return outputs


def freeze_except_last_n_layers(modules: Iterable[nn.Module], last_n: int) -> None:
    stacked = list(modules)
    split = max(0, len(stacked) - last_n)
    for layer in stacked[:split]:
        for param in layer.parameters():
            param.requires_grad = False
    for layer in stacked[split:]:
        for param in layer.parameters():
            param.requires_grad = True


@dataclass
class BackboneLoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2")


def apply_lora_to_linear_layers(
    model: nn.Module,
    lora_config: BackboneLoRAConfig,
    freeze_non_lora: bool = True,
) -> list[str]:
    """Inject LoRA adapters into matching linear layers (in-place)."""
    replaced: list[str] = []

    def _replace_in_module(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(key in full_name for key in lora_config.target_modules):
                wrapped = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=lora_config.rank,
                    alpha=lora_config.alpha,
                )
                wrapped.base.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    wrapped.base.bias.data.copy_(child.bias.data)
                setattr(parent, name, wrapped)
                replaced.append(full_name)
            else:
                _replace_in_module(child, full_name)

    _replace_in_module(model)

    if freeze_non_lora:
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name

    return replaced


class SmolVLAAdaptiveModel(nn.Module):
    """SmolVLA backbone + lightweight adaptation heads for on-the-fly finetuning."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        action_dim: int,
        latent_dim: int = 32,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = EfficientSmolVLA(
            hidden_size=hidden_size,
            action_dim=action_dim,
            latent_dim=latent_dim,
            lora_rank=lora_rank,
        )

    def forward_from_embeddings(self, token_embeddings: Tensor) -> dict[str, Tensor]:
        return self.adapter.predict_chunk(token_embeddings)
