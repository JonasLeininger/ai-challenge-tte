from dataclasses import dataclass


@dataclass
class CategoricalFeatureConfig:
    embedding_dim: int
    cardinality: int


@dataclass
class ContinuousFeatureConfig:
    encoding_dim: int
