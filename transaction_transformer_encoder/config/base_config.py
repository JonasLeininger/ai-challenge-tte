import math
import multiprocessing as mp
import os


# To avoid a pickling error by using MappingProxyType and 'spawn' as multiprocessing start method on mac os,
# we set the multiprocessing start method to 'fork' (default on linux)
# Credits to: https://medium.com/devopss-hole/python-multiprocessing-pickle-issue-e2d35ccf96a9#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6Ijc3NzBiMDg1YmY2NDliNzI2YjM1NzQ3NjQwMzBlMWJkZTlhMTBhZTYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2ODM1NTI3MDAsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExMzcxMzA4ODQzNjY2NjA5MzUyOSIsImhkIjoicmV3ZS1kaWdpdGFsLmNvbSIsImVtYWlsIjoidG9iaWFzLnN1bmRlcmRpZWtAcmV3ZS1kaWdpdGFsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJuYW1lIjoiVG9iaWFzIFN1bmRlcmRpZWsiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUdObXl4WkdpNDZOUnZSR1JRcDZWXzB4OE52LWtCcGRIcmtXRW9ROUNhbHM9czk2LWMiLCJnaXZlbl9uYW1lIjoiVG9iaWFzIiwiZmFtaWx5X25hbWUiOiJTdW5kZXJkaWVrIiwiaWF0IjoxNjgzNTUzMDAwLCJleHAiOjE2ODM1NTY2MDAsImp0aSI6ImQyYmVlYWMwN2YzOTcyMDU3NTViYWE1ZmFhNDAyMjM2NzFlMDEwY2YifQ.tD-jyEoZFTua3eeK3VEU0Px9WuQwVCi5vhg7xetCR8FNAihDKuSILWBThmP2Y2I9_jJNtbGKXukWX-einDB6xNJM4_huhUcBl1g721Yq-D7AjqXo-e8w2b0UhRWYPb_6TEID38liW1gqJtasjavn3Ny9BLuQYNECLRwwmg5lmYsdOTGE8dzlVEW_DH-KB8gU9aiTkf_TU_WmY4tJv_FEIlMb__IM4kFtfFxNx0X4_xYHz8DokQf0wa0XxSdbV964DpEmSqrqSSIa-neLJRNylpSxRB_rJpjJ6kS_9Je9RlTBqIJ4MD0iXeyjqD_QUQ05omJb81X0NNz0zU1fYzMg5g

#mp.set_start_method("fork")

base_config = {
    "gcp": {
    },
    "ckpt_path": './checkpoints',
    "debug_mode": True,
    "dataset_prefix": "cw30",
    "dataset_filter_prefix": "repeated0",
    "vocab_prefix": "cw30",
    "dataset_raw_suffix": "_raw",
    "dataset_filtered_suffix": "repeated0_filtered",
    "artifact_path": "/artifacts",
    "nan_vocab_path": "nan_vocab.dill",
    "com_group_vocab_path": "com_group_vocab.dill",
    "dayofweek_vocab_path": "dayofweek_vocab.dill",

    "is_tuning": False,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "num_workers": math.ceil(os.cpu_count() / 2)
    if os.cpu_count() > 1
    else os.cpu_count(),
    "epochs": 5,
    "encoding_total_size": 24,
    "embedding_encoder_dim": 24,
    "repeat_data": 0, # call filter_data.py to re-create dataset with this param
    "nan_min_freq": 1,
    "num_transformer_layer": 1,
    "num_transformer_heads": 1,
    "nan_feature_embedding_dim": 12,
    "day_of_week_embedding_dim": 4,
    "time_of_day_embedding_dim": 8,
    "com_group_feature_embedding_dim": 8,
    "alpha": 10.0,
    "ray": False,
}
