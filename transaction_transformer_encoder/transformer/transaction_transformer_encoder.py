import math
from dataclasses import dataclass
from typing import Literal, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from transaction_transformer_encoder.transformer.basket_item_encoder import CategoricalFeatureConfig


@dataclass
class TransactionEncoderConfig:
    vocab_size: int
    day_size: int
    com_size: int
    embedding_dim: int
    num_layer: int
    num_heads: int

    # Features configs
    nan_feature: CategoricalFeatureConfig
    day_of_week_feature: CategoricalFeatureConfig
    com_group_feature: CategoricalFeatureConfig

    feedforward_dim: int | None = None  # default: 4 * embedding_dim
    dropout_p: float = 0.0
    activation: Literal["relu", "gelu"] = "gelu"

    # Whether to scale the basket item embeddings by sqrt(d_model)
    scale_input_embeddings: bool = False
    norm_first: bool = True

    bias: bool = False


class TransactionEncoder(pl.LightningModule):
    """Transformer Encoder only model to generate embeddings from transactional data."""

    def __init__(self, config: TransactionEncoderConfig) -> None:
        """Init the model"""
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wTokenEmbedding=nn.Embedding(
                    config.vocab_size, config.nan_feature.embedding_dim
                ),
                wDayEmbedding=nn.Embedding(
                    config.day_size, config.day_of_week_feature.embedding_dim
                ),
                wComGroupEmbedding=nn.Embedding(
                    config.com_size, config.com_group_feature.embedding_dim
                ),
            )
        )
        
    def forward(self, features: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batches, tokens = features["nan_indices"].size()
        print("batch b is %s and sequence length of tokens is %s", (batches, tokens))

        token_emb = self.transformer.wTokenEmbedding(features["nan_indices"])
        day_emb = self.transformer.wDayEmbedding(features["day_of_week_indices"])
        com_emb = self.transformer.wComGroupEmbedding(features["com_group_indices"])

        # what stategy to use to combine the embeddings?

        # transformer

        x = x

        mlm = x
        
        return mlm[:, 1:, :], x


class TransformerBlock(nn.Module):
    def __init__(self, config: TransactionEncoderConfig):
        super().__init__()
        # something to add herer
        self.s_attn = SelfAttention(config)
        # something to add here
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor):
        
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransactionEncoderConfig):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0
        # attention block architecture here

    def forward(self, x):
        B, T, C = x.size()
        # attention attention
        return 0


class FeedForward(nn.Module):
    def __init__(self, config: TransactionEncoderConfig):
        super().__init__()
        # FFN architecture here

    def forward(self, x):
        return x
