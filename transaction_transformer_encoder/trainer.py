import os
import json
import logging

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from transaction_transformer_encoder.config.base_config import base_config
from transaction_transformer_encoder.transaction_transformer_pipeline import TransformerPipeline
from transaction_transformer_encoder.data.basket_dataset import BasketDataset
from transaction_transformer_encoder.data.batch_collator import BatchCollator
from transaction_transformer_encoder.transformer.masking import FeatureMasking
from transaction_transformer_encoder.transformer.transaction_transformer_encoder import TransactionEncoderConfig
from transaction_transformer_encoder.transformer.basket_item_encoder import CategoricalFeatureConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Trainer:
    def __init__(self, predict=False):
        if predict:
            logging.info(f"Predicting...")
        else:
            logging.info(f"Training...")
        
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath = base_config['ckpt_path'],
                monitor="loss",
                mode="min",
                save_top_k=1, # feel free to save more than 1 checkpoint
                save_last=True,
                save_on_train_epoch_end=True,
                auto_insert_metric_name=True,
                filename="tte-{epoch}-{loss:.2f}",
            ),
        ]
        
        self.pl_trainer = pl.Trainer(
            max_epochs=base_config["epochs"],
            accelerator="auto",
            #strategy="ddp_find_unused_parameters_true",
            # devices=1,
            enable_checkpointing=True,
            callbacks=callbacks,
        )
    
    def fit(self):
        """Start training transaction transformer.
        
        pl.seed_everything is important in multi GPU training with DDP strategy
        """
        pl.seed_everything(1)
        
        train_dataloader, transformer_config = self._create_dataset()
        logging.info(f"Dataset created with {len(train_dataloader)} samples.")
        
        search_space = {
        "learning_rate": base_config["learning_rate"]
        }
        
        pipeline = TransformerPipeline(search_space, base_config, transformer_config)

        logging.info("Training started...")
        self.pl_trainer.fit(model=pipeline, train_dataloaders=train_dataloader)
    
    def _create_dataset(self, shuffle=True):
        train_dataset = BasketDataset(base_config)

        batch_collator = BatchCollator()
        masking = FeatureMasking(
            train_dataset.nan_vocab, train_dataset.day_of_week_vocab
        )
        collate_fn = lambda single_features_list: masking.mask(
            batch_collator.collate(single_features_list)
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=base_config["batch_size"],
            shuffle=shuffle,
            # num_workers=base_config["num_workers"], # num_workers can be a porblem with the dataloaders
            collate_fn=collate_fn,
        )

        nan_vocab_cardinality = train_dataset.nan_vocab.size()
        day_vocab_cardinality = train_dataset.day_of_week_vocab.size()
        comp_vocab_cardinality = train_dataset.com_group_vocab.size()
        transformer_config = TransactionEncoderConfig(
            vocab_size=nan_vocab_cardinality,
            day_size=day_vocab_cardinality,
            com_size=comp_vocab_cardinality,
            embedding_dim=base_config["embedding_encoder_dim"],
            num_layer=base_config["num_transformer_layer"],
            num_heads=base_config["num_transformer_heads"],
            nan_feature=CategoricalFeatureConfig(
                embedding_dim=base_config["nan_feature_embedding_dim"],
                cardinality=nan_vocab_cardinality,
            ),
            day_of_week_feature=CategoricalFeatureConfig(
                embedding_dim=base_config["day_of_week_embedding_dim"],
                cardinality=day_vocab_cardinality,
            ),
            com_group_feature=CategoricalFeatureConfig(
                base_config["com_group_feature_embedding_dim"],
                cardinality=comp_vocab_cardinality,
            ),
            scale_input_embeddings=True,
        )

        print(f"Vocab size {nan_vocab_cardinality}")
        return train_dataloader, transformer_config


def fit():
    trainer = Trainer()
    trainer.fit()


if __name__ == "__main__":
    fit()
