import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import Monitor, not_in_whitelist
from tqdm import tqdm
from datetime import datetime
from fuxictr.pytorch.layers import MLP_Block, CrossNetV2, CrossNetMix, FactorizationMachine, SqueezeExcitation, BilinearInteractionV2, LogisticRegression, InnerProductInteraction, CompressedInteractionNet
from layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict
import torch.nn.functional as F
from DLF_layers import *
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, precision_recall_fscore_support, root_mean_squared_error, mean_squared_error
import time

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention

class BaseModel_all_feature(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 **kwargs):
        super(BaseModel_all_feature, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + datetime.now().strftime('_%Y%m%d-%H%M%S') + ".model"))
        self.validation_metrics = kwargs["metrics"]
        self.loss_stats = {feature: {'mean': 0, 'std': 1} for feature in self.feature_map.features.keys()}
        self.loss_stats['label'] = {'mean': 0, 'std': 1}
        self.token_classifiers_dict = {}

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self._embedding_regularizer:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self._net_regularizer:
                                    for net_p, net_lambda in net_reg:
                                        reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss += self.regularization_loss()
        return loss

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        
        batch_size = inputs['label'].size(0)
        num_head = 1  # 如果有多个head，请调整
        num_feature = len(inputs)  # 假设每个键都是一个特征
        half_batch_size = batch_size // 2

        half_mask_indices = []
        # 将输入数据转移到设备
        for feature in inputs.keys():
            if feature_source and not_in_whitelist(self.feature_map.features[feature]["source"], feature_source):
                continue
            X_dict[feature] = inputs[feature].to(self.device)
            half_mask_indice = torch.randperm(half_batch_size)[:int(half_batch_size*0.15)]
            half_mask_indices.append(half_mask_indice)
        # half_mask_indice在每个feature内采样了一个子序列
        half_mask_indices = torch.cat(half_mask_indices, dim=0).reshape(num_feature, -1)
        
        delta_tensor = torch.arange(num_feature) * int(half_batch_size*0.15)
        delta_tensor = delta_tensor.unsqueeze(1).repeat(1, int(half_batch_size*0.15))
        
        half_mask_indices_to_create_mask = half_mask_indices + delta_tensor
        half_mask_indices_to_create_mask = half_mask_indices_to_create_mask.reshape(-1)
        
        # 初始化mask
        # full_mask = torch.zeros(batch_size, num_head, num_feature, num_feature, device=self.device, dtype=torch.bool)

        full_mask = torch.ones(batch_size*num_feature, device=self.device, dtype=torch.bool)
        # 前半部分的mask
        mask_count = half_batch_size * int(num_feature * 0.15)
        full_mask[half_mask_indices_to_create_mask] = False
        full_mask = full_mask.reshape(num_feature, batch_size).T
        full_mask[half_batch_size:, -1] = False
        expanded_mask = full_mask.unsqueeze(-1).repeat(1,1,num_feature)
        transposed_mask = expanded_mask.transpose(-1, -2)
        # full_mask = expanded_mask & transposed_mask
        full_mask = transposed_mask

        return X_dict, full_mask, half_mask_indices

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        os.remove(self.checkpoint)  # delete self.checkpoint

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def update_loss_stats(self, feature, loss_value):
        # 更新损失的均值和标准差，使用滑动平均或其他方法
        # 这里假设使用简单的滑动平均
        alpha = 0.1  # 更新速率
        self.loss_stats[feature]['mean'] = (1 - alpha) * self.loss_stats[feature]['mean'] + alpha * loss_value
        self.loss_stats[feature]['std'] = (1 - alpha) * self.loss_stats[feature]['std'] + alpha * (loss_value - self.loss_stats[feature]['mean'])**2

    def normalize_loss(self, feature, loss_value):
        mean = self.loss_stats[feature]['mean']
        std = self.loss_stats[feature]['std']
        # 防止标准差为零的情况
        if std < 1e-6:
            std = 1.0
        return (loss_value - mean) / std

    def train_step(self, batch_index, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        full_loss = 0
        for i, feature in enumerate(return_dict['all_token_pred'].keys()):
            if feature == 'label':
                criterion = nn.BCELoss()
            elif self.feature_map.features[feature]['type'] == 'numeric':
                criterion = nn.MSELoss()
            elif self.feature_map.features[feature]['type'] == 'categorical':
                criterion = nn.CrossEntropyLoss()
            else:
                raise NotImplementedError("feature type={} not supported.".format(self.feature_map.features[feature]['type']))
            
            device_data = batch_data[feature].to(self.device)
            if feature == 'label' or self.feature_map.features[feature]['type'] == 'numeric':
                loss = criterion(return_dict['all_token_pred'][feature].view(-1).float(), device_data[return_dict['half_mask_indices'][i]].float())
            else:
                loss = criterion(return_dict['all_token_pred'][feature].float(), device_data[return_dict['half_mask_indices'][i]].long())
            
            # 归一化损失
            normalized_loss = self.normalize_loss(feature, loss.item())
            # full_loss += normalized_loss
            full_loss += loss

            # 更新损失统计
            self.update_loss_stats(feature, loss.item())

        criterion = nn.BCELoss()
        device_data = batch_data['label'].to(self.device)
        full_loss += criterion(return_dict['half_y_pred'].view(-1).float(), device_data[device_data.shape[0]//2:].float())
        full_loss += self.regularization_loss()
        # if batch_index % 10 == 0:
            # print(criterion(return_dict['half_y_pred'].view(-1).float(), device_data[device_data.shape[0]//2:].float()).item())
        # print('reg', self.regularization_loss())
        
        full_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return full_loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            t0 = time.time()
            loss = self.train_step(batch_index, batch_data)
            t1 = time.time()
            train_loss += loss.item()
            '''
            注意注意，这里必须要修改 or (self._total_steps+1) % 50 == 0
            '''
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
                t2 = time.time()
                # print(f"train time: {t1-t0}, eval time: {t2-t1}")
            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        val_logs = {}
        with torch.no_grad():
            t0 = time.time()
            batch_count = 0
            y_pred = []
            y_true = []
            all_true = {feature:[] for feature in self.feature_map.features.keys()}
            all_true['label'] = []
            all_pred = {feature:[] for feature in self.feature_map.features.keys()}
            all_pred['label'] = []
            all_f1_part = {feature:{'tp':0, 'fp':0, 'fn':0} for feature in self.feature_map.features.keys()}
            all_f1_part['label'] = {'tp':0, 'fp':0, 'fn':0}
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                t00 = time.time()
                return_dict = self.forward(batch_data)
                t01 = time.time()
                for i, feature in enumerate(return_dict['all_token_pred'].keys()):
                    if feature == 'label' or self.feature_map.features[feature]['type'] == 'numeric':
                        all_pred[feature].extend(return_dict['all_token_pred'][feature].detach().cpu().numpy().reshape(-1))
                        all_true[feature].extend(batch_data[feature][return_dict['half_mask_indices'][i]].cpu().numpy().reshape(-1))
                    else:
                        precision, recall, f1, support = precision_recall_fscore_support(
                            batch_data[feature][return_dict['half_mask_indices'][i]].cpu().numpy().reshape(-1), \
                            torch.argmax(return_dict['all_token_pred'][feature], dim=-1).detach().cpu().numpy().reshape(-1), \
                            average='weighted', \
                            zero_division=0
                        )
                        all_f1_part[feature]['tp'] += precision
                        all_f1_part[feature]['fp'] += (1 - precision)
                        all_f1_part[feature]['fn'] += (1 - recall)
                        
                t02 = time.time()
                y_pred.extend(return_dict["half_y_pred"].detach().cpu().numpy().reshape(-1))
                y_true.extend(batch_data['label'][batch_data['label'].shape[0]//2:].cpu().numpy().reshape(-1))
                batch_count += 1
                t03 = time.time()
                # print(f"Double circle: {t02-t01}, single circle: {(t02-t01)/len(return_dict['all_token_pred'].keys())}, store valid: {t03-t02}")
            t1 = time.time()
            all_rmse = 0
            rmse_count = 0
            all_f1 = 0
            f1_count = 0
            for feature in return_dict['all_token_pred'].keys():
                if feature == 'label' or self.feature_map.features[feature]['type'] == 'numeric':
                    all_rmse += root_mean_squared_error(y_true, y_pred)
                    rmse_count += 1
                elif self.feature_map.features[feature]['type'] == 'categorical':
                    # 这里先用f1，之后可以改
                    # all_f1 = f1_score(all_true[feature], all_pred[feature], average='weighted')
                    overall_precision = all_f1_part[feature]['tp'] / (all_f1_part[feature]['tp'] + all_f1_part[feature]['fp'])
                    overall_recall = all_f1_part[feature]['tp'] / (all_f1_part[feature]['tp'] + all_f1_part[feature]['fn'])
                    all_f1 += 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
                    f1_count += 1
            t2 = time.time()
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            val_logs['AUC'] = roc_auc_score(y_true, y_pred)
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            val_logs['logloss'] = log_loss(y_true, y_pred)
            val_logs['RMSE'] = all_rmse / rmse_count
            val_logs['F1'] = all_f1 / f1_count
            t3 = time.time()
            # print(f"Double circle: {t1-t0}, single circle: {(t1-t0)/batch_count}, cal all valid: {t2-t1}, store valid: {t3-t2}")
            
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def predict(self, data_generator):
        '''
        这个predict没改，因为我们似乎没用到？
        '''
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return_dict = OrderedDict()
        group_metrics = []
        for metric in metrics:
            if metric in ['logloss', 'binary_crossentropy']:
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return_dict[metric] = log_loss(y_true, y_pred)
            elif metric == 'AUC':
                return_dict[metric] = roc_auc_score(y_true, y_pred)
            elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
                return_dict[metric] = 0
                group_metrics.append(metric)
            else:
                raise ValueError("metric={} not supported.".format(metric))
        if len(group_metrics) > 0:
            assert group_id is not None, "group_index is required."
            metric_funcs = []
            for metric in group_metrics:
                try:
                    metric_funcs.append(eval(metric))
                except:
                    raise NotImplementedError('metrics={} not implemented.'.format(metric))
            score_df = pd.DataFrame({"group_index": group_id,
                                    "y_true": y_true,
                                    "y_pred": y_pred})
            results = []
            pool = mp.Pool(processes=mp.cpu_count() // 2)
            for idx, df in score_df.groupby("group_index"):
                results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
            pool.close()
            pool.join()
            results = [res.get() for res in results]
            sum_results = np.array(results).sum(0)
            average_result = list(sum_results[:, 0] / sum_results[:, 1])
            return_dict.update(dict(zip(group_metrics, average_result)))
        return return_dict

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))


class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + datetime.now().strftime('_%Y%m%d-%H%M%S') + ".model"))
        self.validation_metrics = kwargs["metrics"]

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self._embedding_regularizer:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self._net_regularizer:
                                    for net_p, net_lambda in net_reg:
                                        reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss += self.regularization_loss()
        return loss

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        for feature in inputs.keys():
            if feature in self.feature_map.labels:
                continue
            spec = self.feature_map.features[feature]
            if spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(spec["source"], feature_source):
                continue
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        os.remove(self.checkpoint)  # delete self.checkpoint

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return_dict = OrderedDict()
        group_metrics = []
        for metric in metrics:
            if metric in ['logloss', 'binary_crossentropy']:
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return_dict[metric] = log_loss(y_true, y_pred)
            elif metric == 'AUC':
                return_dict[metric] = roc_auc_score(y_true, y_pred)
            elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
                return_dict[metric] = 0
                group_metrics.append(metric)
            else:
                raise ValueError("metric={} not supported.".format(metric))
        if len(group_metrics) > 0:
            assert group_id is not None, "group_index is required."
            metric_funcs = []
            for metric in group_metrics:
                try:
                    metric_funcs.append(eval(metric))
                except:
                    raise NotImplementedError('metrics={} not implemented.'.format(metric))
            score_df = pd.DataFrame({"group_index": group_id,
                                    "y_true": y_true,
                                    "y_pred": y_pred})
            results = []
            pool = mp.Pool(processes=mp.cpu_count() // 2)
            for idx, df in score_df.groupby("group_index"):
                results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
            pool.close()
            pool.join()
            results = [res.get() for res in results]
            sum_results = np.array(results).sum(0)
            average_result = list(sum_results[:, 0] / sum_results[:, 1])
            return_dict.update(dict(zip(group_metrics, average_result)))
        return return_dict

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

class DLF(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DLF, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dim_emb = embedding_dim
        num_emb_lcb = kwargs.get("num_emb_lcb", 16)
        num_emb_fmb = kwargs.get("num_emb_fmb", 16)
        rank_fmb = kwargs.get("rank_fmb", 24)
        num_hidden_DLF = kwargs.get("num_hidden_DLF", 2)
        dim_hidden_DLF = kwargs.get("dim_hidden_DLF", 512)
        num_hidden_head = kwargs.get("num_hidden_head", 2)
        dim_hidden_head = kwargs.get("dim_hidden_head", 512)
        dim_output = kwargs.get("dim_output", 1)
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        num_emb_in = feature_map.num_fields
        dropout = net_dropout
        # 注意，这里我用attention_layer_num替代了DLF_layer_num，这是因为attention_layer的堆叠和DLF_layer的堆叠非常相似。
        num_layers = attention_layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.interaction_layers.append(
                DLFLayer(
                    feature_map.num_fields,
                    num_emb_in,
                    self.dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    rank_fmb,
                    num_hidden_DLF,
                    dim_hidden_DLF,
                    dropout,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb*2

        self.projection_head = DLF_MLP(
            (num_emb_lcb + num_emb_fmb*2) * self.dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout=dropout,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs) -> Tensor:
        X = self.get_inputs(inputs)
        inputs = self.embedding_layer(X)
        for i, layer in enumerate(self.interaction_layers):
            if i == 0:
                outputs = layer(inputs, inputs)
            else:
                outputs = layer(outputs, inputs)
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb*2) * self.dim_emb)
        outputs = self.projection_head(outputs)
        y_pred = self.output_activation(outputs)
        return_dict = {"y_pred": y_pred}

        return return_dict

class DLF_att_cross(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DLF_att_cross, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dim_emb = embedding_dim
        num_emb_lcb = kwargs.get("num_emb_lcb", 16)
        num_emb_fmb = kwargs.get("num_emb_fmb", 16)
        num_emb_att = kwargs.get("num_emb_att", 16)
        rank_fmb = kwargs.get("rank_fmb", 24)
        num_hidden_DLF = kwargs.get("num_hidden_DLF", 2)
        dim_hidden_DLF = kwargs.get("dim_hidden_DLF", 512)
        num_hidden_head = kwargs.get("num_hidden_head", 2)
        dim_hidden_head = kwargs.get("dim_hidden_head", 512)
        dim_output = kwargs.get("dim_output", 1)
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_att = num_emb_att
        num_emb_in = feature_map.num_fields
        dropout = net_dropout
        # 注意，这里我用attention_layer_num替代了DLF_layer_num，这是因为attention_layer的堆叠和DLF_layer的堆叠非常相似。
        num_layers = attention_layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.interaction_layers.append(
                DLFLayer_att_cross(
                    feature_map.num_fields,
                    num_emb_in,
                    self.dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    num_emb_att,
                    rank_fmb,
                    num_hidden_DLF,
                    dim_hidden_DLF,
                    dropout,
                    num_heads,
                    use_scale,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb*2 + num_emb_att

        self.projection_head = DLF_MLP(
            (num_emb_lcb + num_emb_fmb*2 + num_emb_att) * self.dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout=dropout,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs) -> Tensor:
        X = self.get_inputs(inputs)
        inputs = self.embedding_layer(X)
        for i, layer in enumerate(self.interaction_layers):
            if i == 0:
                outputs = layer(inputs, inputs)
            else:
                outputs = layer(outputs, inputs)
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb*2 + self.num_emb_att) * self.dim_emb)
        outputs = self.projection_head(outputs)
        y_pred = self.output_activation(outputs)
        return_dict = {"y_pred": y_pred}

        return return_dict
    
class DLF_att_self(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DLF_att_self, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dim_emb = embedding_dim
        num_emb_lcb = kwargs.get("num_emb_lcb", 16)
        num_emb_fmb = kwargs.get("num_emb_fmb", 16)
        # num_emb_att = kwargs.get("num_emb_att", 16)
        num_emb_att = feature_map.num_fields
        rank_fmb = kwargs.get("rank_fmb", 24)
        num_hidden_DLF = kwargs.get("num_hidden_DLF", 2)
        dim_hidden_DLF = kwargs.get("dim_hidden_DLF", 512)
        num_hidden_head = kwargs.get("num_hidden_head", 2)
        dim_hidden_head = kwargs.get("dim_hidden_head", 512)
        dim_output = kwargs.get("dim_output", 1)
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_att = num_emb_att
        num_emb_in = feature_map.num_fields
        dropout = net_dropout
        # 注意，这里我用attention_layer_num替代了DLF_layer_num，这是因为attention_layer的堆叠和DLF_layer的堆叠非常相似。
        num_layers = attention_layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.interaction_layers.append(
                DLFLayer_att_self(
                    feature_map.num_fields,
                    num_emb_in,
                    self.dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    num_emb_att,
                    rank_fmb,
                    num_hidden_DLF,
                    dim_hidden_DLF,
                    dropout,
                    num_heads,
                    use_scale,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb*2 + num_emb_att

        self.projection_head = DLF_MLP(
            (num_emb_lcb + num_emb_fmb*2 + num_emb_att) * self.dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout=dropout,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs) -> Tensor:
        X = self.get_inputs(inputs)
        inputs = self.embedding_layer(X)
        for i, layer in enumerate(self.interaction_layers):
            if i == 0:
                outputs = layer(inputs, inputs)
            else:
                outputs = layer(outputs, inputs)
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb*2 + self.num_emb_att) * self.dim_emb)
        outputs = self.projection_head(outputs)
        y_pred = self.output_activation(outputs)
        return_dict = {"y_pred": y_pred}

        return return_dict


class DLF_att_self_gate(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DLF_att_self_gate, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dim_emb = embedding_dim
        num_emb_lcb = kwargs.get("num_emb_lcb", 16)
        num_emb_fmb = kwargs.get("num_emb_fmb", 16)
        # num_emb_att = kwargs.get("num_emb_att", 16)
        num_emb_att = feature_map.num_fields
        rank_fmb = kwargs.get("rank_fmb", 24)
        num_hidden_DLF = kwargs.get("num_hidden_DLF", 2)
        dim_hidden_DLF = kwargs.get("dim_hidden_DLF", 512)
        num_hidden_head = kwargs.get("num_hidden_head", 2)
        dim_hidden_head = kwargs.get("dim_hidden_head", 512)
        dim_output = kwargs.get("dim_output", 1)
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_att = num_emb_att
        num_emb_in = feature_map.num_fields
        dropout = net_dropout
        # 注意，这里我用attention_layer_num替代了DLF_layer_num，这是因为attention_layer的堆叠和DLF_layer的堆叠非常相似。
        num_layers = attention_layers
        self.interaction_layers = nn.Sequential()
        for _ in range(num_layers):
            self.interaction_layers.append(
                DLFLayer_att_self_gate(
                    num_emb_in,
                    self.dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    num_emb_att,
                    rank_fmb,
                    num_hidden_DLF,
                    dim_hidden_DLF,
                    dropout,
                    num_heads,
                    use_scale,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb + num_emb_att

        self.projection_head = DLF_MLP(
            (num_emb_lcb + num_emb_fmb + num_emb_att) * self.dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout=dropout,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs) -> Tensor:
        X = self.get_inputs(inputs)
        outputs = self.embedding_layer(X)
        outputs = self.interaction_layers(outputs)
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb + self.num_emb_att) * self.dim_emb)
        outputs = self.projection_head(outputs)
        y_pred = self.output_activation(outputs)
        return_dict = {"y_pred": y_pred}

        return return_dict


class DLF_att_self_multi_gate(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DLF_att_self_multi_gate, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dim_emb = embedding_dim
        num_emb_lcb = kwargs.get("num_emb_lcb", 16)
        num_emb_fmb = kwargs.get("num_emb_fmb", 16)
        # num_emb_att = kwargs.get("num_emb_att", 16)
        num_emb_att = feature_map.num_fields
        rank_fmb = kwargs.get("rank_fmb", 24)
        num_hidden_DLF = kwargs.get("num_hidden_DLF", 2)
        dim_hidden_DLF = kwargs.get("dim_hidden_DLF", 512)
        num_hidden_head = kwargs.get("num_hidden_head", 2)
        dim_hidden_head = kwargs.get("dim_hidden_head", 512)
        dim_output = kwargs.get("dim_output", 1)
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_att = num_emb_att
        num_emb_in = feature_map.num_fields
        dropout = net_dropout
        # 注意，这里我用attention_layer_num替代了DLF_layer_num，这是因为attention_layer的堆叠和DLF_layer的堆叠非常相似。
        num_layers = attention_layers
        self.interaction_layers = nn.ModuleList()  # 改为 ModuleList
        for _ in range(num_layers):
            self.interaction_layers.append(
                DLFLayer_att_self(
                    num_emb_in,
                    self.dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    num_emb_att,
                    rank_fmb,
                    num_hidden_DLF,
                    dim_hidden_DLF,
                    dropout,
                    num_heads,
                    use_scale,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb + num_emb_att

        self.projection_head = DLF_MLP(
            (num_emb_lcb + num_emb_fmb + num_emb_att) * self.dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout=dropout,
        )
        self.gate_method = kwargs['gate_method']
        self.cat_mlp = DLF_MLP(
            num_layers * self.dim_emb,
            1,
            num_layers * self.dim_emb,
            self.dim_emb,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs) -> Tensor:
        X = self.get_inputs(inputs)
        outputs = self.embedding_layer(X)
        intermediate_outputs = []  # 用来保存每一层的中间向量
        for layer in self.interaction_layers:
            outputs = layer(outputs)  # 前向传播
            intermediate_outputs.append(outputs)
        intermediate_outputs = torch.stack(intermediate_outputs)
        if self.gate_method == 'add':
            outputs = torch.sum(intermediate_outputs, dim=0)
        elif self.gate_method == 'cat':
            outputs = intermediate_outputs.permute(1,2,0,3)
            outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2]*outputs.shape[3])
            outputs = self.cat_mlp(outputs)
        elif self.gate_method == 'gate':
            1
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb + self.num_emb_att) * self.dim_emb)
        outputs = self.projection_head(outputs)
        y_pred = self.output_activation(outputs)
        return_dict = {"y_pred": y_pred}

        return return_dict


class DLF_multi_gate(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DLF_multi_gate, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dim_emb = embedding_dim
        num_emb_lcb = kwargs.get("num_emb_lcb", 16)
        num_emb_fmb = kwargs.get("num_emb_fmb", 16)
        rank_fmb = kwargs.get("rank_fmb", 24)
        num_hidden_DLF = kwargs.get("num_hidden_DLF", 2)
        dim_hidden_DLF = kwargs.get("dim_hidden_DLF", 512)
        num_hidden_head = kwargs.get("num_hidden_head", 2)
        dim_hidden_head = kwargs.get("dim_hidden_head", 512)
        dim_output = kwargs.get("dim_output", 1)
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        num_emb_in = feature_map.num_fields
        dropout = net_dropout
        # 注意，这里我用attention_layer_num替代了DLF_layer_num，这是因为attention_layer的堆叠和DLF_layer的堆叠非常相似。
        num_layers = attention_layers
        self.interaction_layers = nn.ModuleList()  # 改为 ModuleList
        for _ in range(num_layers):
            self.interaction_layers.append(
                DLFLayer(
                    num_emb_in,
                    self.dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    rank_fmb,
                    num_hidden_DLF,
                    dim_hidden_DLF,
                    dropout,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb

        self.projection_head = DLF_MLP(
            (num_emb_lcb + num_emb_fmb) * self.dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout=dropout,
        )
        self.gate_method = kwargs['gate_method']
        self.cat_mlp = DLF_MLP(
            num_layers * self.dim_emb,
            1,
            num_layers * self.dim_emb,
            self.dim_emb,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs) -> Tensor:
        X = self.get_inputs(inputs)
        outputs = self.embedding_layer(X)
        intermediate_outputs = []  # 用来保存每一层的中间向量
        for layer in self.interaction_layers:
            outputs = layer(outputs)  # 前向传播
            intermediate_outputs.append(outputs)
        intermediate_outputs = torch.stack(intermediate_outputs)
        if self.gate_method == 'add':
            outputs = torch.sum(intermediate_outputs, dim=0)
        elif self.gate_method == 'cat':
            outputs = intermediate_outputs.permute(1,2,0,3)
            outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2]*outputs.shape[3])
            outputs = self.cat_mlp(outputs)
        elif self.gate_method == 'gate':
            1
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb)
        outputs = self.projection_head(outputs)
        y_pred = self.output_activation(outputs)
        return_dict = {"y_pred": y_pred}

        return return_dict

class AutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(AutoInt, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False) if use_wide else None
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=net_dropout, 
                                     use_residual=use_residual, 
                                     use_scale=use_scale,
                                     layer_norm=layer_norm) \
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        attention_out = self.self_attention(feature_emb)    # B, Feature_num, Embedding_dim
        attention_out = torch.flatten(attention_out, start_dim=1) # B, F*E
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X, mask=None):
        residual = X
        
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu() 
        return output

class HSTU_AutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(HSTU_AutoInt, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False) if use_wide else None
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[HSTU_MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=net_dropout, 
                                     use_residual=use_residual, 
                                     use_scale=use_scale,
                                     layer_norm=layer_norm,
                                     num_fields=feature_map.num_fields) \
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        attention_out = self.self_attention(feature_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class HSTU_MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False, num_fields=0):
        super(HSTU_MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self._embedding_dim = input_dim
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None
        
        # 下面是新增的
        # linear_hidden_dim = 160 # 暂时是直接写进来的，要是之后要修改，应该加载model_config.yaml中
        linear_hidden_dim = 4 * self.head_dim
        self._linear_dim = linear_hidden_dim
        self._embedding_dim = input_dim
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    input_dim,
                    linear_hidden_dim * 2 * num_heads
                    + self.head_dim * num_heads * 2,
                )
            ).normal_(mean=0, std=0.02),
        )
        self._o = torch.nn.Linear(
            in_features=self._linear_dim * num_heads * 1,
            out_features=input_dim,
        )
        self._linear_activation = 'silu'
        self.num_fields = num_fields
        self._dropout_ratio = dropout_rate

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim])
    
    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self.num_heads])

    def forward(self, X):
        '''
        input B, F, D
        '''
        # residual = X
        flattened_X = X.view(-1, self._embedding_dim)
        normed_x = self._norm_input(flattened_X)
        
        flattened_mm_output = torch.mm(normed_x, self._uvqk)
        batched_mm_output = flattened_mm_output.view(-1, self.num_fields, self._linear_dim * 2 * self.num_heads + self.head_dim * self.num_heads * 2)
        if self._linear_activation == 'silu':
            batched_mm_output = F.silu(batched_mm_output)
        elif self._linear_activation == 'none':
            batched_mm_output = batched_mm_output
        u, v, q, k = torch.split(
            batched_mm_output, 
            [
                self._linear_dim * self.num_heads,
                self._linear_dim * self.num_heads,
                self.head_dim * self.num_heads, 
                self.head_dim * self.num_heads, 
            ],
            dim=2,
        )
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q.view(-1, self.num_fields, self.num_heads, self.head_dim),
            k.view(-1, self.num_fields, self.num_heads, self.head_dim),
        )
        qk_attn = F.silu(qk_attn) / self.num_fields

        attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            # "bhnm,bhmd->bhnd",
            qk_attn,
            v.view(-1, self.num_fields, self.num_heads, self._linear_dim),
            ).reshape(-1, self.num_fields, self.num_heads * self._linear_dim)
        
        o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + X
        )
        
        return new_outputs
    

class FeedforwardNeuralNetwork(torch.nn.Module) :
    def __init__(self, input_size, hidden_size, output_size, activation: str, dropout: float, bias: bool = False) :
        super(FeedforwardNeuralNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.activation = activation
        # self.dropout = torch.nn.Dropout(dropout)
        self.lin2 = torch.nn.Linear(hidden_size, output_size, bias=bias)
        if activation == 'swiglu' :
            self.lin3 = torch.nn.Linear(input_size, hidden_size, bias=bias)
    
    def forward(self, X) :
        if self.activation == 'swiglu' :
            X = F.silu(self.lin1(X)) * self.lin3(X)
        else :
            X = activate(self.activation, self.lin1(X))
        X = self.lin2(X)
        return X
    
    def init(self) :
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)

class fuxi_alpha_AutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(fuxi_alpha_AutoInt, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False) if use_wide else None
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[fuxi_alpha_MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=net_dropout, 
                                     use_residual=use_residual, 
                                     use_scale=use_scale,
                                     layer_norm=layer_norm,
                                     num_fields=feature_map.num_fields) \
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        attention_out = self.self_attention(feature_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class fuxi_alpha_MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False, num_fields=0):
        super(fuxi_alpha_MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self._embedding_dim = input_dim
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None
        
        # 下面是新增的
        # linear_hidden_dim = 160 # 暂时是直接写进来的，要是之后要修改，应该加载model_config.yaml中
        linear_hidden_dim = 4 * self.head_dim
        self._linear_dim = linear_hidden_dim
        self._embedding_dim = input_dim
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    input_dim,
                    linear_hidden_dim * 2 * num_heads
                    + self.head_dim * num_heads * 2,
                )
            ).normal_(mean=0, std=0.02),
        )
        self._o = torch.nn.Linear(
            in_features=self._linear_dim * num_heads * 1,
            out_features=input_dim,
        )
        self._linear_activation = 'silu'
        self.num_fields = num_fields
        self._dropout_ratio = dropout_rate
        self._eps = 1e-6
        self._ffn = FeedforwardNeuralNetwork(
            # n_experts = n_experts,
            # n_active_experts = n_active_experts,
            # pre_softmax = True,
            input_size = self._embedding_dim,
            hidden_size = self._embedding_dim,
            output_size = self._embedding_dim,
            dropout = self._dropout_ratio,
            activation = 'swiglu'
        )
        self._ffn.init()

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)
    
    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, normalized_shape=[self._linear_dim * self.num_heads], eps=self._eps
        )

    def forward(self, X):
        '''
        input B, F, D
        '''
        # residual = X
        flattened_X = X.view(-1, self._embedding_dim)
        normed_x = self._norm_input(flattened_X)
        
        flattened_mm_output = torch.mm(normed_x, self._uvqk)
        batched_mm_output = flattened_mm_output.view(-1, self.num_fields, self._linear_dim * 2 * self.num_heads + self.head_dim * self.num_heads * 2)
        if self._linear_activation == 'silu':
            batched_mm_output = F.silu(batched_mm_output)
        elif self._linear_activation == 'none':
            batched_mm_output = batched_mm_output
        u, v, q, k = torch.split(
            batched_mm_output, 
            [
                self._linear_dim * self.num_heads,
                self._linear_dim * self.num_heads,
                self.head_dim * self.num_heads, 
                self.head_dim * self.num_heads, 
            ],
            dim=2,
        )
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q.view(-1, self.num_fields, self.num_heads, self.head_dim),
            k.view(-1, self.num_fields, self.num_heads, self.head_dim),
        )
        qk_attn = F.silu(qk_attn) / self.num_fields

        attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            # "bhnm,bhmd->bhnd",
            qk_attn,
            v.view(-1, self.num_fields, self.num_heads, self._linear_dim),
            ).reshape(-1, self.num_fields, self.num_heads * self._linear_dim)
        o_input = u * self._norm_attn_output(attn_output)

        block_output = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + X
        )
        normalized_ffn_input = self._norm_input(block_output)
        new_outputs = (
            self._ffn(
                F.dropout(
                    normalized_ffn_input, 
                    p = self._dropout_ratio,
                    training = self.training
                )
            ) + block_output
        )
        
        return new_outputs





class HSTU_pos_AutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(HSTU_pos_AutoInt, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False) if use_wide else None
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[HSTU_pos_MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=net_dropout, 
                                     use_residual=use_residual, 
                                     use_scale=use_scale,
                                     layer_norm=layer_norm,
                                     num_fields=feature_map.num_fields) \
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim, 1)
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            feature_map.num_fields,
            embedding_dim,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        feature_emb = feature_emb * (feature_emb.shape[-1]**0.5) + self._pos_emb(
            torch.arange(feature_emb.shape[1], device=feature_emb.device).unsqueeze(0).repeat(feature_emb.shape[0], 1)
        )
        attention_out = self.self_attention(feature_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class RelativePositionBasedBias(nn.Module):
    """
    Computes position-based relative attention bias.
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(self) -> torch.Tensor:
        """
        Returns:
            (N, N) position bias.
        """
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(N, 3 * N - 2)
        r = (2 * N - 1) // 2

        rel_pos_bias = t[:, r:-r]
        return rel_pos_bias

class HSTU_pos_MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False, num_fields=0):
        super(HSTU_pos_MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self._embedding_dim = input_dim
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None
        
        # 下面是新增的
        # linear_hidden_dim = 160 # 暂时是直接写进来的，要是之后要修改，应该加载model_config.yaml中
        linear_hidden_dim = 4 * self.head_dim
        self._linear_dim = linear_hidden_dim
        self._embedding_dim = input_dim
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    input_dim,
                    linear_hidden_dim * 2 * num_heads
                    + self.head_dim * num_heads * 2,
                )
            ).normal_(mean=0, std=0.02),
        )
        self._o = torch.nn.Linear(
            in_features=self._linear_dim * num_heads * 1,
            out_features=input_dim,
        )
        self._linear_activation = 'silu'
        self.num_fields = num_fields
        self._dropout_ratio = dropout_rate
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * num_fields - 1).normal_(mean=0, std=0.02),
        )
        self.relative_position_bias = RelativePositionBasedBias(max_seq_len=num_fields)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim])
    
    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self.num_heads])

    def forward(self, X):
        '''
        input B, F, D
        '''
        # residual = X
        flattened_X = X.view(-1, self._embedding_dim)
        normed_x = self._norm_input(flattened_X)
        
        flattened_mm_output = torch.mm(normed_x, self._uvqk)
        batched_mm_output = flattened_mm_output.view(-1, self.num_fields, self._linear_dim * 2 * self.num_heads + self.head_dim * self.num_heads * 2)
        if self._linear_activation == 'silu':
            batched_mm_output = F.silu(batched_mm_output)
        elif self._linear_activation == 'none':
            batched_mm_output = batched_mm_output
        u, v, q, k = torch.split(
            batched_mm_output, 
            [
                self._linear_dim * self.num_heads,
                self._linear_dim * self.num_heads,
                self.head_dim * self.num_heads, 
                self.head_dim * self.num_heads, 
            ],
            dim=2,
        )
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q.view(-1, self.num_fields, self.num_heads, self.head_dim),
            k.view(-1, self.num_fields, self.num_heads, self.head_dim),
        )
        
        rel_pos_bias = self.relative_position_bias()
        qk_attn += rel_pos_bias.unsqueeze(0).unsqueeze(0)
        
        qk_attn = F.silu(qk_attn) / self.num_fields

        attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            # "bhnm,bhmd->bhnd",
            qk_attn,
            v.view(-1, self.num_fields, self.num_heads, self._linear_dim),
            ).reshape(-1, self.num_fields, self.num_heads * self._linear_dim)
        
        o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + X
        )
        
        return new_outputs


class DCNv2(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2", 
                 gpu=-1,
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 pretrained_codes=None,
                 **kwargs):
        super(DCNv2, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        input_dim = feature_map.sum_emb_out_dim()
        if 'trans_embedding_dim' in kwargs and kwargs['trans_embedding_dim']:
            self.embedding_layer = FeatureEmbedding(feature_map, kwargs['pretrained_embedding_dim'], pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
            self.trans_embedding_dim_layer = MLP_Block(input_dim=kwargs['pretrained_embedding_dim']*feature_map.num_fields,
                                                       output_dim=input_dim,
                                                       hidden_activations=dnn_activations,
                                                       batch_norm=batch_norm)
        else:
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
        if 'use_id' in model_id:
            self.id_embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
            input_dim *= 2
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(input_dim, num_cross_layers, low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=dnn_activations,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = input_dim
        self.fc = nn.Linear(final_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        if hasattr(self, 'trans_embedding_dim_layer'):
            feature_emb = self.trans_embedding_dim_layer(feature_emb)
        if hasattr(self, 'id_embedding_layer'):
            feature_emb = torch.cat([feature_emb, self.id_embedding_layer(X, flatten_emb=True)], dim=-1)
        cross_out = self.crossnet(feature_emb)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_emb)], dim=-1)
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
class DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 pretrained_codes=None,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        input_dim = feature_map.sum_emb_out_dim()
        if 'trans_embedding_dim' in kwargs and kwargs['trans_embedding_dim']:
            self.embedding_layer = FeatureEmbedding(feature_map, kwargs['pretrained_embedding_dim'], pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
            self.trans_embedding_dim_layer = MLP_Block(input_dim=kwargs['pretrained_embedding_dim'],
                                                       output_dim=embedding_dim,
                                                       hidden_activations=hidden_activations,
                                                       batch_norm=batch_norm)
        else:
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
        if 'use_id' in model_id:
            self.id_embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
            input_dim *= 2
        self.fm = FactorizationMachine(feature_map)
        self.mlp = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        if hasattr(self, 'trans_embedding_dim_layer'):
            feature_emb = self.trans_embedding_dim_layer(feature_emb)
        if hasattr(self, 'id_embedding_layer'):
            feature_emb = torch.cat([feature_emb, self.id_embedding_layer(X)], dim=-1)
        y_pred = self.fm(X, feature_emb)
        y_pred += self.mlp(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
class FiBiNET(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FiBiNET", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 excitation_activation="ReLU",
                 reduction_ratio=3,
                 bilinear_type="field_interaction",
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 pretrained_codes=None,
                 **kwargs):
        super(FiBiNET, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        if 'trans_embedding_dim' in kwargs and kwargs['trans_embedding_dim']:
            self.embedding_layer = FeatureEmbedding(feature_map, kwargs['pretrained_embedding_dim'], pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
            self.trans_embedding_dim_layer = MLP_Block(input_dim=kwargs['pretrained_embedding_dim'],
                                                       output_dim=embedding_dim,
                                                       hidden_activations=hidden_activations,
                                                       batch_norm=batch_norm)
        else:
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
        if 'use_id' in model_id:
            self.id_embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
            embedding_dim *= 2
        num_fields = feature_map.num_fields
        self.senet_layer = SqueezeExcitation(num_fields, reduction_ratio, excitation_activation)
        self.bilinear_interaction1 = BilinearInteractionV2(num_fields, embedding_dim, bilinear_type)
        self.bilinear_interaction2 = BilinearInteractionV2(num_fields, embedding_dim, bilinear_type)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False)
        input_dim = num_fields * (num_fields - 1) * embedding_dim
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units, 
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X) # list of b x embedding_dim
        if hasattr(self, 'trans_embedding_dim_layer'):
            feature_emb = self.trans_embedding_dim_layer(feature_emb)
        if hasattr(self, 'id_embedding_layer'):
            feature_emb = torch.cat([feature_emb, self.id_embedding_layer(X)], dim=-1)
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction1(feature_emb)
        bilinear_q = self.bilinear_interaction2(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dnn_out = self.dnn(comb_out)
        y_pred = self.lr_layer(X) + dnn_out
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
class FM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 regularizer=None, 
                 pretrained_codes=None,
                 **kwargs):
        super(FM, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer, 
                                 net_regularizer=regularizer,
                                 **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pretrained_codes=pretrained_codes, vq_num_embeddings=kwargs['vq_num_embeddings'])
        self.fm = FactorizationMachine(feature_map)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.fm(X, feature_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
class PNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="PNN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 product_type="inner", 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(PNN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        if product_type != "inner":
            raise NotImplementedError("product_type={} has not been implemented.".format(product_type))
        self.inner_product_layer = InnerProductInteraction(feature_map.num_fields, output="inner_product")
        input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) \
                  + feature_map.num_fields * embedding_dim
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) 
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        inner_products = self.inner_product_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        y_pred = self.dnn(dense_input)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
class GDCNP(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="GDCNP",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(GDCNP, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.cross_net = GateCorssLayer(input_dim, num_cross_layers)
        self.fc = torch.nn.Linear(dnn_hidden_units[-1] + input_dim, 1)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_cn = self.cross_net(feature_emb)
        cross_mlp = self.dnn(feature_emb)
        y_pred = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class GDCNS(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="GDCNS",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(GDCNS, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.cross_net = GateCorssLayer(input_dim, num_cross_layers)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_cn = self.cross_net(feature_emb)
        y_pred = self.dnn(cross_cn)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class GateCorssLayer(nn.Module):
    #  The core structure： gated corss layer.
    def __init__(self, input_dim, cn_layers=3):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.wg = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])

        self.b = nn.ParameterList([nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

        for i in range(cn_layers):
            nn.init.uniform_(self.b[i].data)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x) # Feature Crossing
            xg = self.activation(self.wg[i](x)) # Information Gate
            x = x0 * (xw + self.b[i]) * xg + x
        return x
    
class LR(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="LR", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 regularizer=None, 
                 **kwargs):
        super(LR, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer, 
                                 net_regularizer=regularizer, 
                                 **kwargs)
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        y_pred = self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class xDeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="xDeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU",
                 cin_hidden_units=[16, 16, 16], 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(xDeepFM, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)     
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only CIN used
        self.lr_layer = LogisticRegression(feature_map, use_bias=False)
        self.cin = CompressedInteractionNet(feature_map.num_fields, cin_hidden_units, output_dim=1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X) # list of b x embedding_dim
        lr_logit = self.lr_layer(X)
        cin_logit = self.cin(feature_emb)
        y_pred = lr_logit + cin_logit # only LR + CIN
        if self.dnn is not None:
            dnn_logit = self.dnn(feature_emb.flatten(start_dim=1))
            y_pred += dnn_logit # LR + CIN + DNN
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class FinalMLP(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FinalMLP",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 mlp1_hidden_units=[64, 64, 64],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[64, 64, 64],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0,
                 mlp2_batch_norm=False,
                 use_fs=True,
                 fs_hidden_units=[64],
                 fs1_context=[],
                 fs2_context=[],
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FinalMLP, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        feature_dim = embedding_dim * feature_map.num_fields
        self.mlp1 = MLP_Block(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp1_dropout,
                              batch_norm=mlp1_batch_norm)
        self.mlp2 = MLP_Block(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp2_dropout, 
                              batch_norm=mlp2_batch_norm)
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(feature_map, 
                                              feature_dim, 
                                              embedding_dim, 
                                              fs_hidden_units, 
                                              fs1_context,
                                              fs2_context)
        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(X, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class FeatureSelection(nn.Module):
    def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], 
                 fs1_context=[], fs2_context=[]):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs1_context)
        self.fs2_context = fs2_context
        if len(fs2_context) == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs2_context)
        self.fs1_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)
        self.fs2_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs2_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)

    def forward(self, X, flat_emb):
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output
    
class FinalNet(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FinalNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 block_type="2B",
                 batch_norm=True,
                 use_feature_gating=False,
                 block1_hidden_units=[64, 64, 64],
                 block1_hidden_activations=None,
                 block1_dropout=0,
                 block2_hidden_units=[64, 64, 64],
                 block2_hidden_activations=None,
                 block2_dropout=0,
                 residual_type="concat",
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FinalNet, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        assert block_type in ["1B", "2B"], "block_type={} not supported.".format(block_type)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        self.use_feature_gating = use_feature_gating
        if use_feature_gating:
            self.feature_gating = FeatureGating(num_fields, gate_residual="concat")
            gate_out_dim = embedding_dim * num_fields * 2
        self.block_type = block_type
        self.block1 = FinalBlock(input_dim=gate_out_dim if use_feature_gating \
                                           else embedding_dim * num_fields,
                                 hidden_units=block1_hidden_units,
                                 hidden_activations=block1_hidden_activations,
                                 dropout_rates=block1_dropout,
                                 batch_norm=batch_norm,
                                 residual_type=residual_type)
        self.fc1 = nn.Linear(block1_hidden_units[-1], 1)
        if block_type == "2B":
            self.block2 = FinalBlock(input_dim=embedding_dim * num_fields,
                                     hidden_units=block2_hidden_units,
                                     hidden_activations=block2_hidden_activations,
                                     dropout_rates=block2_dropout,
                                     batch_norm=batch_norm,
                                     residual_type=residual_type)
            self.fc2 = nn.Linear(block2_hidden_units[-1], 1)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred, y1, y2 = None, None, None
        if self.block_type == "1B":
            y_pred = self.forward1(feature_emb)
        elif self.block_type == "2B":
            y1 = self.forward1(feature_emb)
            y2 = self.forward2(feature_emb)
            y_pred = 0.5 * (y1 + y2)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "y1": y1, "y2": y2}
        return return_dict

    def forward1(self, X):
        if self.use_feature_gating:
            X = self.feature_gating(X)
        block1_out = self.block1(X.flatten(start_dim=1))
        y_pred = self.fc1(block1_out)
        return y_pred

    def forward2(self, X):
        block2_out = self.block2(X.flatten(start_dim=1))
        y_pred = self.fc2(block2_out)
        return y_pred

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        if self.block_type == "2B":
            y1 = self.output_activation(return_dict["y1"])
            y2 = self.output_activation(return_dict["y2"])
            loss1 = self.loss_fn(y1, return_dict["y_pred"].detach(), reduction='mean')
            loss2 = self.loss_fn(y2, return_dict["y_pred"].detach(), reduction='mean')
            loss = loss + loss1 + loss2
        return loss


class FeatureGating(nn.Module):
    def __init__(self, num_fields, gate_residual="concat"):
        super(FeatureGating, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)
        assert gate_residual in ["concat", "sum"]
        self.gate_residual = gate_residual

    def reset_custom_params(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.ones_(self.linear.bias)

    def forward(self, feature_emb):
        gates = self.linear(feature_emb.transpose(1, 2)).transpose(1, 2)
        if self.gate_residual == "concat":
            out = torch.cat([feature_emb, feature_emb * gates], dim=1) # b x 2f x d
        else:
            out = feature_emb + feature_emb * gates
        return out


class FinalBlock(nn.Module):
    def __init__(self, input_dim, hidden_units=[], hidden_activations=None, 
                 dropout_rates=[], batch_norm=True, residual_type="sum"):
        # Factorized Interaction Block: Replacement of MLP block
        super(FinalBlock, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(FactorizedInteraction(hidden_units[idx],
                                                    hidden_units[idx + 1],
                                                    residual_type=residual_type))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i


class FactorizedInteraction(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, residual_type="sum"):
        """ FactorizedInteraction layer is an improvement of nn.Linear to capture quadratic 
            interactions between features.
            Setting `residual_type="concat"` keeps the same number of parameters as nn.Linear
            while `residual_type="sum"` doubles the number of parameters.
        """
        super(FactorizedInteraction, self).__init__()
        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        else:
            assert output_dim % 2 == 0, "output_dim should be divisible by 2."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        if self.residual_type == "concat":
            h = torch.cat([h2, h1 * h2], dim=-1)
        elif self.residual_type == "sum":
            h = h2 + h1 * h2
        return h
    

class SAM(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="SAM",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 interaction_type="SAM2E", # option in ["SAM2A", "SAM2E", "SAM3A", "SAM3E"]
                 aggregation="concat",
                 num_interaction_layers=3,
                 use_residual=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0,
                 **kwargs):
        super(SAM, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.block = SAMBlock(num_interaction_layers, feature_map.num_fields, embedding_dim, use_residual, 
                              interaction_type, aggregation, net_dropout)
        if aggregation == "concat":
            if interaction_type in ["SAM2A", "SAM2E"]:
                self.fc = nn.Linear(embedding_dim * (feature_map.num_fields ** 2), 1)
            else:
                self.fc = nn.Linear(feature_map.num_fields * embedding_dim, 1)
        else:
            self.fc = nn.Linear(embedding_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        interact_out = self.block(feature_emb)
        y_pred = self.fc(interact_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
        

class SAMBlock(nn.Module):
    def __init__(self, num_layers, num_fields, embedding_dim, use_residual=False, 
                 interaction_type="SAM2E", aggregation="concat", dropout=0):
        super(SAMBlock, self).__init__()
        assert aggregation in ["concat", "weighted_pooling", "mean_pooling", "sum_pooling"]
        self.aggregation = aggregation
        if self.aggregation == "weighted_pooling":
            self.weight = nn.Parameter(torch.ones(num_fields, 1))
        if interaction_type == "SAM2A":
            assert aggregation == "concat", "Only aggregation=concat is supported for SAM2A."
            self.layers = nn.ModuleList([SAM2A(num_fields, embedding_dim, dropout)])
        elif interaction_type == "SAM2E":
            assert aggregation == "concat", "Only aggregation=concat is supported for SAM2E."
            self.layers = nn.ModuleList([SAM2E(embedding_dim, dropout)])
        elif interaction_type == "SAM3A":
            self.layers = nn.ModuleList([SAM3A(num_fields, embedding_dim, use_residual, dropout) \
                                         for _ in range(num_layers)])
        elif interaction_type == "SAM3E":
            self.layers = nn.ModuleList([SAM3E(embedding_dim, use_residual, dropout) \
                                         for _ in range(num_layers)])
        else:
            raise ValueError("interaction_type={} not supported.".format(interaction_type))

    def forward(self, F):
        for layer in self.layers:
            F = layer(F)
        if self.aggregation == "concat":
            out = F.flatten(start_dim=1)
        elif self.aggregation == "weighted_pooling":
            out = (F * self.weight).sum(dim=1)
        elif self.aggregation == "mean_pooling":
            out = F.mean(dim=1)
        elif self.aggregation == "sum_pooling":
            out = F.sum(dim=1)
        return out


class SAM2A(nn.Module):
    def __init__(self, num_fields, embedding_dim, dropout=0):
        super(SAM2A, self).__init__()
        self.W = nn.Parameter(torch.ones(num_fields, num_fields, embedding_dim)) # f x f x d
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2)) # b x f x f
        out = S.unsqueeze(-1) * self.W # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM2E(nn.Module):
    def __init__(self, embedding_dim, dropout=0):
        super(SAM2E, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2)) # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F) # b x f x f x d
        out = S.unsqueeze(-1) * U # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3A(nn.Module):
    def __init__(self, num_fields, embedding_dim, use_residual=True, dropout=0):
        super(SAM3A, self).__init__()
        self.W = nn.Parameter(torch.ones(num_fields, num_fields, embedding_dim)) # f x f x d
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2)) # b x f x f
        out = (S.unsqueeze(-1) * self.W).sum(dim=2) # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3E(nn.Module):
    def __init__(self, embedding_dim, use_residual=True, dropout=0):
        super(SAM3E, self).__init__()
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2)) # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F) # b x f x f x d
        out = (S.unsqueeze(-1) * U).sum(dim=2) # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out
    
class AFN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AFN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 ensemble_dnn=True,
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU",
                 dnn_dropout=0,
                 afn_hidden_units=[64, 64, 64], 
                 afn_activations="ReLU",
                 afn_dropout=0,
                 logarithmic_neurons=5,
                 batch_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(AFN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.num_fields = feature_map.num_fields
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.coefficient_W = nn.Linear(self.num_fields, logarithmic_neurons, bias=False)
        self.dense_layer = MLP_Block(input_dim=embedding_dim * logarithmic_neurons,
                                     output_dim=1, 
                                     hidden_units=afn_hidden_units,
                                     hidden_activations=afn_activations,
                                     output_activation=None, 
                                     dropout_rates=afn_dropout, 
                                     batch_norm=batch_norm)
        self.log_batch_norm = nn.BatchNorm1d(self.num_fields)
        self.exp_batch_norm = nn.BatchNorm1d(logarithmic_neurons)
        self.ensemble_dnn = ensemble_dnn
        if ensemble_dnn:
            self.embedding_layer2 = FeatureEmbedding(feature_map, embedding_dim)
            self.dnn = MLP_Block(input_dim=embedding_dim * self.num_fields,
                                 output_dim=1, 
                                 hidden_units=dnn_hidden_units,
                                 hidden_activations=dnn_activations,
                                 output_activation=None, 
                                 dropout_rates=dnn_dropout, 
                                 batch_norm=batch_norm)
            self.fc = nn.Linear(2, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        dnn_input = self.logarithmic_net(feature_emb)
        afn_out = self.dense_layer(dnn_input)
        if self.ensemble_dnn:
            feature_emb2 = self.embedding_layer2(X)
            dnn_out = self.dnn(feature_emb2.flatten(start_dim=1))
            y_pred = self.fc(torch.cat([afn_out, dnn_out], dim=-1))
        else:
            y_pred = afn_out
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def logarithmic_net(self, feature_emb):
        feature_emb = torch.abs(feature_emb)
        feature_emb = torch.clamp(feature_emb, min=1e-5) # ReLU with min 1e-5 (better than 1e-7 suggested in paper)
        log_feature_emb = torch.log(feature_emb) # element-wise log 
        log_feature_emb = self.log_batch_norm(log_feature_emb) # batch_size * num_fields * embedding_dim 
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(logarithmic_out) # element-wise exp
        cross_out = self.exp_batch_norm(cross_out)  # batch_size * logarithmic_neurons * embedding_dim
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out
    
class AFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 attention_dropout=[0, 0],
                 attention_dim=10,
                 use_attention=True,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(AFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.use_attention = use_attention
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.product_layer = InnerProductInteraction(feature_map.num_fields, output="elementwise_product")
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)
        self.attention = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                       nn.ReLU(),
                                       nn.Linear(attention_dim, 1, bias=False),
                                       nn.Softmax(dim=1))
        self.weight_p = nn.Linear(embedding_dim, 1, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout[0])
        self.dropout2 = nn.Dropout(attention_dropout[1])
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        elementwise_product = self.product_layer(feature_emb) # bs x f(f-1)/2 x dim
        if self.use_attention:
            attention_weight = self.attention(elementwise_product)
            attention_weight = self.dropout1(attention_weight)
            attention_sum = torch.sum(attention_weight * elementwise_product, dim=1)
            attention_sum = self.dropout2(attention_sum)
            afm_out = self.weight_p(attention_sum)
        else:
            afm_out = torch.flatten(elementwise_product, start_dim=1).sum(dim=-1).unsqueeze(-1)
        y_pred = self.lr_layer(X) + afm_out
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict