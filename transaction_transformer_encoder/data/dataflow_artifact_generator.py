import os
import uuid
import pandas as pd
import torch
import dill
import numpy as np
import typing
import json

from collections import Counter
from datetime import datetime, timezone
from google.cloud import secretmanager
import apache_beam as beam
from apache_beam.internal import pickler
from apache_beam.options.pipeline_options import PipelineOptions
from transaction_transformer_encoder.data.vocab import Vocab

pickler.set_library(pickler.USE_CLOUDPICKLE)


config = {
    'pipeline_options': {
    },
    'storage': {
    },
    'bigquery': {
        
    },
    'wandb': {
 
    }
}

class PrepareFilterExcludeComGroupsAndMinFreq(beam.DoFn):
    def process(self, transaction_as_json, *args_01, **kwargs):
        all_nans_flattened = transaction_as_json['artikelNr']
        counter = Counter(all_nans_flattened)
        nan_min_frequency = 1
        nans_to_delete = [candidate_count[0] for candidate_count in counter.items() if candidate_count[1] < nan_min_frequency]
        nans_to_delete.append(0) # some transactions only have one sku "0"
        com_groups_lvl2_to_delete = ['25','41','52','55','60','63','91','92','95','96','97','98','99']

        nans_filtered = []
        com_groups_filtered = []
        price_filtered = []
        product_count_filtered = []
        for idx, com_group in enumerate(transaction_as_json["warengruppe"]):
            com_lvl2 = int(str(com_group)[:2])
            if com_lvl2 not in com_groups_lvl2_to_delete:
                nan = transaction_as_json["artikelNr"][idx]
                #filter deposit with umsatzMenge > 0
                if nan not in nans_to_delete and transaction_as_json["umsatzMenge"][idx] > 0:
                    if nan not in nans_filtered:
                        nans_filtered.append(nan)
                        com_groups_filtered.append(transaction_as_json["warengruppe"][idx])
                        price_filtered.append(transaction_as_json["umsatzWertBrutto"][idx])
                        product_count_filtered.append(transaction_as_json["umsatzMenge"][idx])
        # clustering needs 2 skus, as we have cls-token (see README here and in clustering)
        #remove >1 after fix to >0
        #if len(nans_filtered) > 0:
        transaction_result = {'id': transaction_as_json['id'], 'bonTs': transaction_as_json['bonTs'], 'kassenNr': transaction_as_json['kassenNr']}
        if len(nans_filtered) > 1:
            transaction_result["artikelNr"] = nans_filtered
            transaction_result["warengruppe"] = com_groups_filtered
            transaction_result["umsatzWertBrutto"] = price_filtered
            transaction_result["umsatzMenge"] = product_count_filtered
        else:
            transaction_result["artikelNr"] = 'DATA_FILTERED'
            transaction_result["warengruppe"] = 'DATA_FILTERED'
            transaction_result["umsatzWertBrutto"] = 'DATA_FILTERED'
            transaction_result["umsatzMenge"] = 'DATA_FILTERED'
        yield transaction_result

class WaitUntilSideInputVocabIsCreated(beam.DoFn):
    def process(self, transaction, create_vocab_result):
        yield transaction


class ExtractComGroups(beam.DoFn):
    def process(self, transaction):
        for comgroup in transaction['warengruppe']:
            yield comgroup


class ExtractNans(beam.DoFn):
    def process(self, transaction):
        for nan in transaction['artikelNr']:
            yield nan


# class SkuToVocab(beam.DoFn):
#     def setup(self):
#         path_to_nan_vocab = f"{config['wandb']['artifact_path']}/{config['bigquery']['ww_ident']}_{config['wandb']['dataset_prefix']}_{config['wandb']['nan_vocab']}"
#         self.nan_vocab = dill.load(open(path_to_nan_vocab, 'rb'))


def create_vocab_fn(dummy_element, nans_skus, com_groups):
    nan_vocab = Vocab.fit(token_candidates=nans_skus, min_freq=1)
    print(f"Vocab len {len(nan_vocab)}")
    com_group_vocab = Vocab.fit(token_candidates=com_groups, min_freq=1)
    isoweekdays = [str(isoweekday) for isoweekday in range(1, 8)]
    dayofweek_vocab = Vocab.fit(token_candidates=isoweekdays, min_freq=1)
    hour_of_days = [str(hour_of_day) for hour_of_day in range(1,25)]
    hour_of_day_vocab = Vocab.fit(token_candidates=hour_of_days, min_freq=1)
    dill.dump(nan_vocab, open(f"{config['wandb']['artifact_path']}/{config['bigquery']['ww_ident']}_{config['wandb']['dataset_prefix']}_nan_vocab.dill", "wb" ))
    dill.dump(com_group_vocab, open(f"{config['wandb']['artifact_path']}/{config['bigquery']['ww_ident']}_{config['wandb']['dataset_prefix']}_com_group_vocab.dill", "wb" ))
    dill.dump(dayofweek_vocab, open(f"{config['wandb']['artifact_path']}/{config['bigquery']['ww_ident']}_{config['wandb']['dataset_prefix']}_dayofweek_vocab.dill", "wb" ))
    dill.dump(hour_of_day_vocab, open(f"{config['wandb']['artifact_path']}/{config['bigquery']['ww_ident']}_{config['wandb']['dataset_prefix']}_hourofday_vocab.dill", "wb" ))
    return True


class SortAndDistinctSkus(beam.DoFn):
    def process(self, transaction_as_json):
        tmp = list(set(transaction_as_json['artikelNr']))
        tmp.sort()
        transaction_as_json['artikelNr'] = tmp
        yield transaction_as_json


def upload_transactions_gcp(dummy_element, transactions):
    transactions_list = [transaction for transaction in transactions]
    path = f"{config['bigquery']['ww_ident']}_{config['wandb']['dataset_prefix']}{config['wandb']['dataset_filter_prefix']}{config['wandb']['dataset_filtered_suffix']}.json"
    local_path = f"{config['wandb']['artifact_path']}/{path}"
    with open(local_path, 'w') as f:
        json.dump(transactions_list, f, indent=4)


def run():
    data_query = "SELECT id, DATETIME(bonTs, 'Europe/Berlin') as bonTs, kassenNr, ARRAY_AGG(pl.artikelNr) as artikelNr," \
                " ARRAY_AGG(pl.warengruppe) as warengruppe, ARRAY_AGG(pl.umsatzMenge IGNORE NULLS) as umsatzMenge, ARRAY_AGG(pl.umsatzWertBrutto IGNORE NULLS) as umsatzWertBrutto" \
                " FROM `{database}.{input_table}`" \
                " CROSS JOIN UNNEST(`{database}.{input_table}`.payload) AS pl" \
                " WHERE bonTs BETWEEN \"{startDate}\" AND \"{endDate}\" AND wwIdent = {wwIdent}" \
                " GROUP BY id, bonTs, kassenNr;".format(database=config['bigquery']['database'],
                                            input_table=config['bigquery']['input_table'],
                                            startDate=config['bigquery']['start_date'],
                                            endDate=config['bigquery']['end_date'],
                                            wwIdent=config['bigquery']['ww_ident'])
    
    with beam.Pipeline(options=PipelineOptions(**config['pipeline_options'])) as p:
        
        transactions_filtered = (p
                        | 'Read transactions from BQ' >> beam.io.Read(beam.io.gcp.bigquery.ReadFromBigQuery(query=data_query, use_standard_sql=True))
                        | 'Prepare Filter exclude com groups and min skus frequency' >> beam.ParDo(PrepareFilterExcludeComGroupsAndMinFreq())
                        | 'Filter' >> beam.Filter(lambda transaction: transaction['artikelNr'] != 'DATA_FILTERED')
                        | 'Sort and distinct skus within one transaction' >> beam.ParDo(SortAndDistinctSkus())
                        )
        
        upload_transactions = (p
                        | 'Dummy step transactions' >> beam.Create([1]) #one dummy element to process pipeline
                        | 'Upload to gcp' >> beam.Map(upload_transactions_gcp, transactions=beam.pvalue.AsIter(transactions_filtered))
                        )

        
        com_groups = (transactions_filtered
                    | 'Extract com groups' >> beam.ParDo(ExtractComGroups())
                    | 'Distinct com groups' >> beam.Distinct()
                    | 'Convert com groups to int' >> beam.Map(lambda sku: int(sku))
                    )
        
        nan_skus = (transactions_filtered
                    | 'Extract nans' >> beam.ParDo(ExtractNans())
                    | 'Distinct nans' >> beam.Distinct()
                    | 'Convert nans to int' >> beam.Map(lambda nan: int(nan))
                    )

        create_vocab_result = (p
                        | 'Dummy step' >> beam.Create([1]) #one dummy element to process pipeline
                        | 'Create vocab' >> beam.Map(create_vocab_fn, nans_skus=beam.pvalue.AsIter(nan_skus), com_groups=beam.pvalue.AsIter(com_groups)))

        transactions_filtered_and_vocab_created = (transactions_filtered
                        | 'Wait until vocab files are generated' >> beam.ParDo(WaitUntilSideInputVocabIsCreated(), create_vocab_result=beam.pvalue.AsSingleton(create_vocab_result)))


if __name__ == '__main__':
    run()
