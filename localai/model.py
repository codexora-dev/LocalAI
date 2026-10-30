
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def dialogue_training_loss(logits, inputs, targets, pad_id, prompt_weight, label_smoothing=0.0):
    """정답 응답 외 문맥에도 선택적으로 다음 토큰 학습을 적용합니다."""
    if prompt_weight == 0:
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100,
                               label_smoothing=label_smoothing)
    # 입력의 다음 위치를 이용하므로 아직 주어지지 않은 정답을 문맥에 넣지 않습니다.
    shifted = torch.cat([inputs[:, 1:], targets[:, -1:]], dim=1)
    context_targets = shifted.masked_fill(shifted.eq(pad_id), -100)
    labels = torch.where(targets.ne(-100), targets, context_targets)
    weights = torch.where(targets.ne(-100), 1.0, prompt_weight)
    weights = weights * labels.ne(-100)
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                            ignore_index=-100, reduction="none",
                            label_smoothing=label_smoothing).view_as(labels)
    return (losses * weights).sum() / weights.sum()


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )
        self.proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(
            torch.ones(
                config.block_size,
                config.block_size,
                dtype=torch.bool
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1, 1,
                config.block_size,
                config.block_size
            )
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                B, T, self.n_head, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~self.mask[:, :, :T, :T],
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)

        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                config.n_embd,
                4 * config.n_embd
            ),
            nn.GELU(),
            nn.Linear(
                4 * config.n_embd,
                config.n_embd
            ),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LocalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.final_norm = nn.LayerNorm(
            config.n_embd
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                "입력 길이가 block_size를 초과했습니다."
            )

        positions = torch.arange(
            T,
            device=idx.device
        )

        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    B * T,
                    self.config.vocab_size
                ),
                targets.reshape(B * T),
                # 사용자 입력과 오른쪽 패딩의 -100 위치는 손실에서 제외합니다.
                ignore_index=-100,
            )

        return logits, loss

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
        )
