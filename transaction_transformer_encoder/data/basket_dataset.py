import os
import json
import dill
from datetime import datetime
from typing import Iterable
from typing import TypedDict
from collections import Counter

import torch
from torch.utils.data import Dataset


class ProductInputData(TypedDict):
    nan: str


class BasketInputData(TypedDict):
    cart: list[ProductInputData]
    bon_timestamp: str

class BasketDataset(Dataset):
    _index_dtype = torch.long

    def __init__(self, config: dict, baskets=None, is_predict: bool = False) -> None:
        nan_min_freq = config["nan_min_freq"]
        time_string = f"{config['gcp']['wwIdents']}_{config['gcp']['start_date']}-{config['gcp']['stop_date']}_"
        filtered_data_path = f"{os.getcwd()}/artifacts/{time_string}{config['dataset_filtered_suffix']}.json"
        
        # Extract pure baskets as list of lists containing NANs.
        if not is_predict:
            with open(filtered_data_path) as f:
                # baskets = [json.loads(line) for line in f.readlines()]
                baskets = json.load(f)
        else:
            baskets = baskets
        print(f"first basket: {baskets[0]}")
        print(f"last basket: {baskets[-1]}")
        print(f"Amount of transactions {len(baskets)}")
        self.basket_ids = self._extract_basket_ids(baskets)
        self.basket_nans = self._extract_basket_nans(baskets)
        self.basket_com_group = self._extract_basket_com(baskets)
        #todo filtering of nans is done above, delete here
        path_to_nan_vocab = f"{os.getcwd()}{config['artifact_path']}/{time_string}{config['nan_vocab_path']}"
        self.nan_vocab = self._load_vocab(path_to_nan_vocab)
        path_to_com_group_vocab = f"{os.getcwd()}{config['artifact_path']}/{time_string}{config['com_group_vocab_path']}"
        self.com_group_vocab = self._load_vocab(path_to_com_group_vocab)
        self.basket_nans, self.basket_com_group = self._sort_by_com_group(
            self.basket_nans, self.basket_com_group
        )
        path_to_dayofweek_vocab = f"{os.getcwd()}{config['artifact_path']}/{time_string}{config['dayofweek_vocab_path']}"
        self.day_of_week_vocab = self._load_vocab(path_to_dayofweek_vocab)
        self.day_of_week_indices = self._extract_day_of_week_indices(baskets)

        # path_to_hourofday_vocab = f"{config['artifact_path']}/{time_string}{config['hourofday_vocab_path']}"
        # self.hour_of_day_vocab = self._load_vocab(path_to_hourofday_vocab)
        # self.day_of_week_indices = self._extract_hour_of_day_indices(baskets)

    def _load_vocab(self, vocab_path):
        return dill.load(open(vocab_path, 'rb'))

    def __len__(self) -> int:
        return len(self.basket_nans)

    def __getitem__(self, idx):
        ids = self.basket_ids[idx]
        nans = self.basket_nans[idx]
        nan_indices = torch.tensor(
            self.nan_vocab.lookup_indices(nans), dtype=self._index_dtype
        )

        com_groups = self.basket_com_group[idx]
        com_group_indices = torch.tensor(
            self.com_group_vocab.lookup_indices(com_groups), dtype=self._index_dtype
        )

        # Make scalar basket features sequential by repeating them for each product/NAN
        day_of_week_indices = torch.full(
            nan_indices.shape, self.day_of_week_indices[idx], dtype=self._index_dtype
        )

        basket = {
            'ids': ids,
            'nan_indices': nan_indices,
            'com_group_indices': com_group_indices,
            'day_of_week_indices': day_of_week_indices,
            # 'hour_of_day_indices': hour_of_day_indices,
        }
        return basket

    @staticmethod
    def _sort_by_com_group(basket_nans, basket_com_group):
        sorted_basket_nans = []
        sorted_basket_com = []
        for i in range(len(basket_nans)):
            assert len(basket_nans[i]) == len(basket_com_group[i])
            sorted_basket_com.append(sorted(basket_com_group[i]))
            sorted_basket_nans.append(
                [x for (y, x) in sorted(zip(basket_com_group[i], basket_nans[i]))]
            )
        print(sorted_basket_com[0])
        print(basket_com_group[0])
        print(sorted_basket_nans[0])
        print(basket_nans[0])
        return basket_nans, basket_com_group

    @staticmethod
    def _extract_basket_ids(baskets: Iterable[BasketInputData]):
        ids = [basket['id'] for basket in baskets if len(basket["artikelNr"]) > 0]
        return ids

    @staticmethod
    def _extract_basket_nans(baskets: Iterable[BasketInputData]) -> list[list[str]]:
        nans = [[product for product in basket["artikelNr"]] for basket in baskets if len(basket["artikelNr"]) > 0]
        return nans

    @staticmethod
    def _extract_basket_com(baskets: Iterable[BasketInputData]) -> list[list[str]]:
        # carts = (basket["payload"] for basket in baskets if len(basket["payload"]) > 0)
        com_groups = [
            [wg for wg in basket["warengruppe"]] for basket in baskets if len(basket["warengruppe"]) > 0]
        return com_groups

    def _extract_day_of_week_indices(self, baskets: Iterable[BasketInputData]) -> list[int]:
        # old "2023-07-24 06:59:19"
        # new "2023-06-21T09:48:05"
        bon_timestamps = (
            datetime.strptime(basket["bonTs"], "%Y-%m-%dT%H:%M:%S")
            for basket in baskets
        )
        iso_weekday_strings = [str(bon_ts.isoweekday()) for bon_ts in bon_timestamps]
        print(f"bonTs days: {iso_weekday_strings[:3]}")
        day_of_week_indices = self.day_of_week_vocab.lookup_indices(iso_weekday_strings)
        return day_of_week_indices
    
    def _extract_hour_of_day_indices(self, baskets: Iterable[BasketInputData]) -> list[int]:
        # old "2023-07-24 06:59:19"
        # new "2023-06-21T09:48:05"
        bon_timestamps = (
            datetime.strptime(basket["bonTs"], "%Y-%m-%dT%H:%M:%S")
            for basket in baskets
        )
        hour_strings = [str(bon_ts.hour) for bon_ts in bon_timestamps]
        print(f"bonTs hours: {hour_strings[:3]}")
        hour_of_day_indices = self.hour_of_day_vocab.lookup_indices(hour_strings)
        return hour_of_day_indices

