from collections import Counter
from itertools import chain
from types import MappingProxyType
from typing import Iterable
from typing import Tuple

import torch


class Vocab:
    """Vocab class.
    Special token indices are static for implementiation simplicity.
    torch.long is used by pytorch as an index tensor datatype
    """

    pad_token_idx = 0
    cls_token_idx = 1
    mask_token_idx = 2
    sep_token_idx = 3
    first_regular_token_idx = 4
    special_token_indices = (
        pad_token_idx,
        cls_token_idx,
        mask_token_idx,
        sep_token_idx,
    )

    _index_dtype = torch.long

    def __init__(self, token_list: list[str]):
        """Creates a vocabulary from an existing `token_list`.
        Please be aware that special tokens have reserved indices:
        - index 0: pad token
        - index 1: cls (out-of-vocabulary) token
        - index 2: mask token
        - index 3: sep token
        """

        # The string representations of special tokens are dynamic
        self.pad_token = token_list[self.pad_token_idx]
        self.cls_token = token_list[self.cls_token_idx]
        self.mask_token = token_list[self.mask_token_idx]
        self.sep_token = token_list[self.sep_token_idx]
        self.special_tokens = (
            self.pad_token,
            self.cls_token,
            self.mask_token,
            self.sep_token,
        )

        # Indices are mapped to tokens via simple list lookups
        self._idx_to_token = token_list

        # Create a _token_to_idx dict for efficient token -> idx lookups
        # Use MappingProxyType to create a read-only view on the underlying dict
        self._token_to_idx = MappingProxyType(
            {token: idx for idx, token in enumerate(token_list)}
        )

    @classmethod
    def fit(
        cls,
        token_candidates: Iterable[str],
        min_freq: int,
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        sep_token: str = "[SEP]",
    ) -> 'Vocab':
        """Fits a vocabulary based on an interable of `token_candidates`.
        Returns the vocab itself and the token frequencies as dict.
        """

        #token_freqs = cls._filter_tokens_by_freq(token_candidates, min_freq)

        special_tokens = (pad_token, cls_token, mask_token, sep_token)
        regular_tokens = (
            token for token in token_candidates if token not in special_tokens
        )
        token_list = list(chain(special_tokens, regular_tokens))

        vocab = cls(token_list)
        return vocab

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        indices = [
            self._token_to_idx.get(token, Vocab.cls_token_idx) for token in tokens
        ]
        return indices

    def lookup_tokens(self, indices: list[int]) -> list[str]:
        tokens = [self._idx_to_token[idx] for idx in indices]
        return tokens

    def size(self) -> int:
        return len(self._idx_to_token)

    def __len__(self) -> int:
        return self.size()

    def random_indices(
        self,
        size: torch.Size,
        exclude_special_tokens: bool,
    ) -> torch.Tensor:
        low = Vocab.first_regular_token_idx if exclude_special_tokens else 0
        random_indices = torch.randint(
            low=low, high=self.size(), size=size, dtype=Vocab._index_dtype
        )
        return random_indices

    @classmethod
    def padding_mask(cls, input: torch.Tensor) -> torch.Tensor:
        return input == cls.pad_token_idx

    @classmethod
    def special_token_mask(cls, input: torch.Tensor) -> torch.Tensor:
        return (
            (input == cls.pad_token_idx)
            | (input == cls.cls_token_idx)
            | (input == cls.mask_token_idx)
            | (input == cls.sep_token_idx)
        )

    @staticmethod
    def _filter_tokens_by_freq(
        token_candidates: Iterable[str],
        min_freq: int,
    ) -> dict[str, int]:
        """Filters tokens by minimum frequency and returns a sorted dictionary token -> frequency."""
        counter = Counter(token_candidates)
        filtered_token_freqs = (
            candidate_count for candidate_count in counter.items() if candidate_count[1] >= min_freq
        )
        sorted_token_freqs = dict(
            # Sort by frequency and then alphabetically by token (reversed -> DESC)
            sorted(
                filtered_token_freqs,
                key=lambda token_freq_tuple: (
                    -token_freq_tuple[1],
                    token_freq_tuple[0],
                ),
            )
        )
        return sorted_token_freqs
