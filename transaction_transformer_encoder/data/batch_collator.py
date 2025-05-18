import torch
from torch import Tensor

from transaction_transformer_encoder.data.vocab import Vocab


class BatchCollator:
    """Collates every feature within a list of SingleBasketFeatures into one batch."""

    def __init__(
        self,
        masking_select_probability=0.15,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self.masking_select_probability = masking_select_probability
        self.pad_to_multiple_of = pad_to_multiple_of

    def collate(
        self, single_features_list: list[dict]
    ) -> dict:
        ids_batch = [single_features['ids'] for single_features in single_features_list]
        nan_indices_batch, _ = self._collate_sequential_feature(
            single_features_list, "nan_indices"
        )
        day_of_week_indices_batch, context = self._collate_sequential_feature(
            single_features_list, "day_of_week_indices"
        )
        com_group_batch, _ = self._collate_sequential_feature(
            single_features_list, "com_group_indices"
        )
        features = {
            'ids': ids_batch,
            'nan_indices': nan_indices_batch,
            'com_group_indices': com_group_batch,
            'day_of_week_indices': day_of_week_indices_batch,
            'context_label': context
        }
        return features

    def _collate_sequential_feature(
        self,
        single_features_list: list[dict],
        feature_name: str,
    ) -> Tensor:
        feature_list: list[Tensor] = [
            single_features[feature_name] for single_features in single_features_list
        ]
        feature_batch = self._collate_batch(feature_list)
        return feature_batch

    def _collate_batch(self, tensor_list: list[Tensor]) -> Tensor:
        """Collates a list of tensors of different length (`shape[0]`) into one batch.

        Adds [SEP] token at the middle of each tensor.

        Supports padding to a length that is a multiple of an int defined by
        `pad_to_multiple_of`.
        """
        tmp_context_label = torch.zeros(len(tensor_list), 1)
        for i, tensor in enumerate(tensor_list):
            tensor_list[i] = torch.cat((torch.tensor([1]), tensor))

        pad_to_multiple_of = self.pad_to_multiple_of

        # Check if padding is necessary
        length_of_first = tensor_list[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in tensor_list)
        if are_tensors_same_length and (
            pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
        ):
            return torch.stack(tensor_list, dim=0)

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in tensor_list)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        result = tensor_list[0].new_full(
            size=[len(tensor_list), max_length], fill_value=Vocab.pad_token_idx
        )
        for i, example in enumerate(tensor_list):
            result[i, : example.shape[0]] = example

        context_label = torch.zeros(result.shape[0], result.shape[1], 1)
        context_label[:, 0, 0] = tmp_context_label[:, 0]
        return result, tmp_context_label