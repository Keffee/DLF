# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os
import logging
import logging.config
import yaml
import glob
import json
import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict
import fuxictr
from datetime import datetime


def load_config(config_dir, experiment_id, **kwargs):
    params = load_model_config(config_dir, experiment_id)
    if 'dataset_id' in kwargs:
        params['dataset_id'] = kwargs['dataset_id']
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params

def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params

def load_dataset_config(config_dir, dataset_id):
    params = {"dataset_id": dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, dataset_id + '_' + model_id + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    if params['verbose']:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                            handlers=[logging.FileHandler(log_file, mode='w'),
                                    logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                            handlers=[logging.FileHandler(log_file, mode='w')])
    logging.info("FuxiCTR version: " + fuxictr.__version__)

def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())


class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value

    def get_metrics(self):
        return list(self.kv_pairs.keys())


def load_pretrain_emb(pretrain_path, keys=["key", "value"]):
    if type(keys) != list:
        keys = [keys]
    if pretrain_path.endswith("h5"):
        with h5py.File(pretrain_path, 'r') as hf:
            values = [hf[k][:] for k in keys]
    elif pretrain_path.endswith("npz"):
        npz = np.load(pretrain_path)
        values = [npz[k] for k in keys]
    elif pretrain_path.endswith("parquet"):
        df = pd.read_parquet(pretrain_path)
        values = [df[k].values for k in keys]
    else:
        raise ValueError(f"Embedding format not supported: {pretrain_path}")
    return values[0] if len(values) == 1 else values


def not_in_whitelist(element, whitelist=[]):
    if not whitelist:
        return False
    elif type(whitelist) == list:
        return element not in whitelist
    else:
        return element != whitelist