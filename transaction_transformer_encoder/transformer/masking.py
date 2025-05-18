
import torch

from transaction_transformer_encoder.data.vocab import Vocab


class FeatureMasking:
    """Masks/corrupts inputs and labels at random positions by applying a
    masked-language-modeling approach known from NLP models like BERT.
    """

    def __init__(
        self,
        nan_vocab: Vocab,
        day_of_week_vocab: Vocab,
        masking_select_probability=0.15,
    ) -> None:
        self.nan_vocab = nan_vocab
        self.day_of_week_vocab = day_of_week_vocab
        self.masking_select_probability = masking_select_probability

    def mask(self, features: dict) -> dict:
        """Prepare inputs and labels for input corruption (analogous to masked language
        modeling).
        Does not mutate the original features and returns a new BasketFeatures that
        contains `nan_labels`.
        We first select input tokens with a `masking_select_probability`, from which:
        - 80% are replaced by a special MASK token
        - 10% are replaced by another random token
        - 10% are left unchanged
        Mostly a copy of https://github.com/huggingface/transformers/blob/76924384af6081e58460231c3c637f9c83cabf97/src/transformers/data/data_collator.py#L750
        """
        # Clone the features before masking to avoid mutation of the originals
        nan_indices = features["nan_indices"].clone()
        day_of_week_indices = features["day_of_week_indices"].clone()
        context = features["context_label"].clone()
        com_group_indices = features["com_group_indices"].clone()

        # Select tokens for masking based on nan_indices
        probability_matrix = torch.full(
            nan_indices.shape, self.masking_select_probability
        )
        special_tokens_mask = Vocab.special_token_mask(nan_indices)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Create labels that reconstruct the original nans we are going to mask
        nan_labels = nan_indices.clone()
        mask = nan_labels.eq(0)
        nan_labels[mask] = 0
        #nan_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace selected input tokens with tokenizer.mask_token ([MASK])
        # replace_with_mask_token_mask = (
        #     torch.bernoulli(torch.full(nan_labels.shape, 1.0)).bool() & masked_indices
        # )
        # nan_indices[replace_with_mask_token_mask] = Vocab.mask_token_idx
        # day_of_week_indices[replace_with_mask_token_mask] = Vocab.mask_token_idx

        
        # # 10% of the time, we replace selected input tokens with a random token
        # replace_with_random_token_mask = (
        #     torch.bernoulli(torch.full(nan_labels.shape, 0.5)).bool()
        #     & masked_indices
        #     & ~replace_with_mask_token_mask
        # )
        # # Replace nan tokens with random ones
        # random_nan_indices = self.nan_vocab.random_indices(
        #     nan_labels.size(), exclude_special_tokens=True
        # )
        # nan_indices[replace_with_random_token_mask] = random_nan_indices[
        #     replace_with_random_token_mask
        # ]
        
        # Replace day_of_week tokens with random ones
        # random_day_of_week_indices = self.day_of_week_vocab.random_indices(
        #     features["day_of_week_indices"].size(), exclude_special_tokens=True
        # )
        # day_of_week_indices[
        #     replace_with_random_token_mask
        # ] = random_day_of_week_indices[replace_with_random_token_mask]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        # print(com_group_indices)
        # print(com_group_indices.shape)
        # print(nan_indices)
        # print(nan_indices.shape)
        masked_features_with_labels = {
            'ids': features["ids"],
            'nan_indices': nan_indices,
            'com_group_indices': com_group_indices,
            'day_of_week_indices': day_of_week_indices,
            'nan_labels': nan_labels[:, 1:],
            'context_label': context
        }
        return masked_features_with_labels
