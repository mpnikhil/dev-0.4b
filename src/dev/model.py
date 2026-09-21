"""One bidirectional encoder, three parallel, non-generative heads."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class ThreeHeadEncoder(nn.Module):
    def __init__(self, encoder, pooling=None):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling or ("last" if encoder.config.model_type in {"qwen2", "qwen3"} else "first")
        hidden = encoder.config.hidden_size
        self.choice = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @property
    def score(self):
        return self.choice

    @property
    def noul(self):
        return self.choice

    @property
    def head(self):
        return self.choice

    @classmethod
    def pretrained(cls, name, revision=None, attn_implementation="sdpa"):
        # SDPA leverages hardware-fused kernels (FlashAttention on CUDA, MPSGraph on Apple Silicon) without materializing full attention matrices.
        return cls(AutoModel.from_pretrained(name, revision=revision, attn_implementation=attn_implementation))

    def forward(self, input_ids, attention_mask, candidate_mask, candidate_valid, **_):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "last":
            # Right-padded batches; the final token sees the complete objective and output.
            positions = attention_mask.sum(-1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        else:
            pooled = hidden[:, 0]
        mask = candidate_mask.to(hidden.dtype)
        candidates = torch.bmm(mask, hidden) / mask.sum(-1, keepdim=True).clamp_min(1)
        context = pooled[:, None].expand_as(candidates)
        logits = self.choice(torch.cat([context, candidates], -1)).squeeze(-1)
        logits = logits.masked_fill(~candidate_valid, -1e4)

        # In the unified architecture, Noul (Yes/No verification) uses candidate spans [No, Yes].
        # Softmax([s_no, s_yes])_1 == sigmoid(s_yes - s_no).
        # Returning s_yes - s_no allows BCEWithLogits to compute the exact 2-way cross-entropy contrast.
        if logits.shape[-1] >= 2:
            noul_logit = logits[:, 1] - logits[:, 0]
        else:
            noul_logit = logits[:, 0]

        return {
            "choice": logits,
            "score": logits,
            "logits": logits,
            "noul": noul_logit,
            "candidate_valid": candidate_valid,
        }


# SingleHeadEncoder alias for the unified universal architecture
SingleHeadEncoder = ThreeHeadEncoder


def multitask_loss(logits, labels, loss_mode="ce"):
    # -100 masks unavailable labels; all tasks route through the universal encoder and choice head.
    terms = []
    for key in ("choice", "score", "noul"):
        valid = labels[key] != -100
        if not valid.any():
            continue
        pred, target = logits[key][valid], labels[key][valid]
        if loss_mode == "brier":
            if key == "noul":
                # Binary Brier Score: (p_yes - target)^2
                p_yes = torch.sigmoid(pred)
                terms.append(F.mse_loss(p_yes, target.float()))
            elif key == "choice":
                # Multi-class Brier Score: sum_k (p_k - y_k)^2
                probs = F.softmax(pred, dim=-1)
                truth = F.one_hot(target.long(), pred.shape[-1]).float()
                mask = logits['candidate_valid'][valid].to(probs.dtype)
                probs = probs * mask
                brier = ((probs - truth * mask) ** 2).sum(-1)
                terms.append(brier.mean())
            elif key == "score":
                # Ordinal Ranked Probability Score (RPS) + multi-class Brier score
                probs = F.softmax(pred, dim=-1)
                truth = F.one_hot(target.long(), pred.shape[-1]).float()
                mask = logits['candidate_valid'][valid].to(probs.dtype)
                probs = probs * mask
                cdf_pred = probs.cumsum(-1)
                cdf_truth = (truth * mask).cumsum(-1)
                rps = (((cdf_pred - cdf_truth) ** 2) * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                brier = ((probs - truth * mask) ** 2).sum(-1)
                terms.append((brier + 0.5 * rps).mean())
        else:
            if key == "noul":
                # Binary cross-entropy on (s_yes - s_no) is mathematically identical to 2-way cross-entropy on [s_no, s_yes].
                terms.append(F.binary_cross_entropy_with_logits(pred, target.float()))
            else:
                loss = F.cross_entropy(pred, target.long())
                if key == "score":
                    # Penalize ordinal distance as well as categorical error.
                    truth = F.one_hot(target.long(), pred.shape[-1]).float()
                    distance = (pred.softmax(-1).cumsum(-1) - truth.cumsum(-1)).square()
                    mask = logits['candidate_valid'][valid].to(distance.dtype)
                    loss = loss + 0.25 * ((distance * mask).sum(-1) / mask.sum(-1)).mean()
                terms.append(loss)
    if not terms:
        raise ValueError("Batch contains no supervised labels")
    return sum(terms)
