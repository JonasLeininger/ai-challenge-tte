import logging
import os
import dill

import pytorch_lightning as pl
import torch
import torch.optim as optim

from transaction_transformer_encoder.transformer.transaction_transformer_encoder import TransactionEncoderConfig, TransactionEncoder


class TransformerPipeline(pl.LightningModule):
    """TransformerClusterPipeline to cluster transactional data."""

    def __init__(self, hparams, base_config, config_transformer: TransactionEncoderConfig):
        """Init TransformerClusterPipeline."""
        super().__init__()
        logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
        logging.info("Init TransformerClusterPipeline")
        self.hyperparams = hparams
        self.config = base_config
        self.config_transformer = config_transformer
        self.transformer = TransactionEncoder(self.config_transformer)
        self.save_hyperparameters()

        time_string = f"{self.config['gcp']['wwIdents']}_{self.config['gcp']['start_date']}-{self.config['gcp']['stop_date']}_"
        self.vocab_path_folder = f"{os.getcwd()}{self.config['artifact_path']}/"
        self.nan_vocab = dill.load(open(f"{self.vocab_path_folder}{time_string}{self.config['nan_vocab_path']}", 'rb'))
        self.com_group_vocab = dill.load(open(f"{self.vocab_path_folder}{time_string}{self.config['com_group_vocab_path']}", 'rb'))
        self.day_of_week_vocab = dill.load(open(f"{self.vocab_path_folder}{time_string}{self.config['dayofweek_vocab_path']}",'rb'))

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch: dict[str, torch.Tensor]):
        """Forward step through the pipeline."""
        token_em, encoding = self.transformer(batch)
        return token_em, encoding
    
    def training_step(self, bon_batch, batch_idx):
        #print(f"train step with batch size {len(bon_batch['nan_indices'])} and content {bon_batch}")
        return self._common_step(bon_batch)
    
    def _common_step(self, batch: dict) -> dict:
        """Common epoch start for all pipelines."""
        token_em, masked_encoding = self.transformer(batch)
        print(f"token_em shape: {token_em.shape}")
        print(f"masked_encoding shape: {masked_encoding.shape}")

        loss_dict = self._loss(batch, token_em, masked_encoding)
        return loss_dict
    
    def _loss(self, batch: dict, token_em: torch.Tensor, masked_encoding: torch.Tensor) -> dict:
        """Calculate the loss for the pipeline."""

        prediction = token_em.reshape(-1, self.config_transformer.nan_feature.cardinality)
        label = batch["nan_labels"].view(-1)
 
        transformer_loss = self.ce_loss(
            prediction,
            label,
        )
        loss = transformer_loss
        loss_dict = {
            "loss": loss,
        }
        return loss_dict

    def configure_optimizers(self):
        transformer = torch.nn.ParameterList([p for n, p in self.transformer.transformer.named_parameters()])
        optimizer = optim.AdamW(transformer, self.config["learning_rate"])
        return {"optimizer": optimizer}
