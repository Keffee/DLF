import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
from fuxictr import datasets
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import models
import gc
import argparse
import os
from pathlib import Path
import torch
import numpy as np
import time


if __name__ == '__main__':
    for i in range(10):
        try:
            ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
            '''
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
            parser.add_argument('--expid', type=str, default='HSTU_AutoInt_default', help='The experiment id to run. AutoInt_default, HSTU_AutoInt_default or HSTU_pos_AutoInt_default.')
            parser.add_argument('--dataset_id', type=str, default='criteo_x1_default', help='The dataset id to run. criteo_x1_default or avazu_x1_default')
            parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
            parser.add_argument('--embedding', type=str, default='base', help='Embedding method, base or raw or vqemb.')
            parser.add_argument('--useid', type=str, default='False', help='Whether to use feature ID embeddings. 现在这里先只用False，不得更改。')
            parser.add_argument('--pre_emb_model', type=str, default='FM', help='Pretrained embedding model, FM, FFM, FFMv2 or DeepFM. Lower character is OK.')
            parser.add_argument('--verbose', type=int, default=1, help='Whether to print verbose information.')
            parser.add_argument('--layers', type=int, default=4, help='Number of layers in VQ-VAE.')
            parser.add_argument('--code_dim', type=int, default=256, help='The dim of codebook.')
            parser.add_argument('--cutdown', type=int, default=-1, help='When to use vqvae.')
            parser.add_argument('--use_freq', type=str, default='False', help='Whether to use feature frequency in VQ-VAE.')
            parser.add_argument('--attention_layers', type=int, default=-1, help='Number of attention layers.')
            parser.add_argument('--dnn_layers', type=int, default=2, help='How many layers of DNN will be used. 0, 1 or 2.')
            parser.add_argument('--emb_dim', type=int, default=64, help='The dim of embedding.')
            parser.add_argument('--batch_size', type=int, default=10000, help='The batch size.')
            parser.add_argument('--gate_method', type=str, default='add', help='The way to use gate, add or cat or gate.')
            args = vars(parser.parse_args())
            
            experiment_id = args['expid']
            num_embeddings = args['code_dim']
            layers = args['layers']
            cutdown = args['cutdown']
            params = load_config(args['config'], experiment_id, dataset_id=args['dataset_id'])
            params['gpu'] = args['gpu']
            params['verbose'] = args['verbose']
            params['vq_num_embeddings'] = num_embeddings
            vq_embedding_dim = int(int(params['embedding_dim']) / layers)
            if args['useid'] == 'True':
                params['model_id'] = experiment_id + '_' + args['embedding'] + '_' + 'use_id'
            else:
                params['model_id'] = experiment_id + '_' + args['embedding']
            if args['attention_layers'] != -1:
                params['attention_layers'] = args['attention_layers']
            if 'dnn_hidden_units' in params.keys() and len(params['dnn_hidden_units']) != args['dnn_layers']:
                params['dnn_hidden_units'] = [params['dnn_hidden_units'][0]] * args['dnn_layers']
            params['embedding_dim'] = args['emb_dim']
            params['attention_dim'] = args['emb_dim']
            params['batch_size'] = args['batch_size']
            params['gate_method'] = args['gate_method']
            
            params['num_emb_lcb'] = 32
            params['num_emb_fmb'] = 32
            params['num_emb_att'] = 32
            
            set_logger(params)
            logging.info("Params: " + print_to_json(params))
            # seed_everything(seed=params['seed'])

            print("[Running] python {}".format(' '.join(sys.argv), args['expid'].split('_')[0]))

            data_dir = os.path.join(params['data_root'], params['dataset_id'])
            feature_map_json = os.path.join(data_dir, "feature_map.json")
            
            # 判断feature_map_json是否存在
            if not os.path.exists(feature_map_json):
                # Build feature_map and transform data
                feature_encoder = FeatureProcessor(**params)
                params["train_data"], params["valid_data"], params["test_data"] = \
                    build_dataset(feature_encoder, **params)
            feature_map = FeatureMap(params['dataset_id'], data_dir)
            feature_map.load(feature_map_json, params)
            logging.info("Feature specs: " + print_to_json(feature_map.features))
            
            # params['num_emb_lcb'] = feature_map.num_fields
            # params['num_emb_fmb'] = feature_map.num_fields
            
            pretrained_codes = None
            if args['use_freq'] == 'True':
                code_book_name = '_feature_logfreq_cons_codebook/'
            else:
                code_book_name = '_feature_codebook/'
            # load pretrained embeds
            if args['embedding'] != 'base':
                if args['embedding'] == 'raw':
                    logging.info("Loading raw pretrained embeddings...")
                    files = []
                    directory = '/root/autodl-tmp/pretrain_result/' + params['dataset_id'] + '_' + args['pre_emb_model'].lower() + f'_embeddings'
                    for item in os.listdir(directory):
                        item_path = os.path.join(directory, item)
                        if os.path.isfile(item_path) and 'codes' not in item and 'embedding' in item:
                            files.append(item)
                    pretrained_embeddings = {}
                    for file in files:
                        feature_name = file.replace('_embedding', '').replace('.npy', '')
                        embedding_file = '/root/autodl-tmp/pretrain_result/' + params['dataset_id'] + '_' + args['pre_emb_model'].lower() + f'_embeddings/{feature_name}_embedding.npy'
                        pretrained_embeddings[feature_name] = np.load(embedding_file)
                else:
                    pretrained_codes = {}
                    logging.info("Loading vqvae embeddings...")
                    files = []
                    if cutdown > 0:
                        directory = '/root/autodl-tmp/pretrain_result/' + params['dataset_id'] + '_' + args['pre_emb_model'].lower() + f'_layers={layers}_codedim={num_embeddings}_embeddingdim={vq_embedding_dim}_{cutdown}{code_book_name}'
                    else:
                        directory = '/root/autodl-tmp/pretrain_result/' + params['dataset_id'] + '_' + args['pre_emb_model'].lower() + f'_layers={layers}_codedim={num_embeddings}_embeddingdim={vq_embedding_dim}{code_book_name}'
                    for feature_name in os.listdir(directory):
                        pretrained_codes[feature_name] = torch.tensor(np.load(directory + feature_name + f'/codes.npy'), device='cuda:%d'%args['gpu'], dtype=torch.long)
                    params['code_num'] = pretrained_codes[feature_name].shape[-1]
            
            model_class = getattr(models, params['model'])
            model = model_class(feature_map, pretrained_codes=pretrained_codes, **params)
            
            if args['embedding'] == 'vqemb':
                if cutdown>0:
                    directory = '/root/autodl-tmp/pretrain_result/' + params['dataset_id'] + '_' + args['pre_emb_model'].lower() + f'_layers={layers}_codedim={num_embeddings}_embeddingdim={vq_embedding_dim}_{cutdown}{code_book_name}'
                else:
                    directory = '/root/autodl-tmp/pretrain_result/' + params['dataset_id'] + '_' + args['pre_emb_model'].lower() + f'_layers={layers}_codedim={num_embeddings}_embeddingdim={vq_embedding_dim}{code_book_name}'
                    
                logging.info(f"Loading embeddings from codebook.")
                for dir_name in os.listdir(directory):
                    files = []
                    for i in range(params['code_num']):
                        files.append(directory + dir_name + f'/vq_layer_{i}_embeddings.npy')
                    model.embedding_layer.load_pretrained_embeddings(dir_name, files)
            # if args['embedding'] != 'base':
            #     # set pretrained embeddings
            #     for feature_name, weights in pretrained_embeddings.items():
            #         embedding_layer = model.embedding_layer.embedding_layer.embedding_layers[feature_name]
            #         weights_tensor = torch.from_numpy(weights).float()
            #         embedding_layer.weight.data.copy_(weights_tensor)
                    
            model.count_parameters() # print number of parameters used in model


            train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
            model.fit(train_gen, validation_data=valid_gen, **params)

            logging.info('****** Validation evaluation ******')
            valid_result = model.evaluate(valid_gen)
            del train_gen, valid_gen
            gc.collect()
            
            test_result = {}
            if params["test_data"]:
                logging.info('******** Test evaluation ********')
                test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
                test_result = model.evaluate(test_gen)
            
            dir_path = r'../results/' + params['dataset_id'][:params['dataset_id'].find('_default')]
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            result_filename = os.path.join(dir_path, Path(args['expid']).name.replace(".yaml", "") + '.csv')
            with open(result_filename, 'a+') as fw:
                fw.write(' {}, [command] python {}, [model] {}, [att_layers] {}, [dnn_layers] {}, [val] {}, [test] {}\n' \
                    .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                            ' '.join(sys.argv), args['expid'][:args['expid'].rfind('_')], params['attention_layers'], args['dnn_layers'], print_to_list(valid_result), print_to_list(test_result)))
            print("[Finished] python {}".format(' '.join(sys.argv), args['expid'][args['expid'].rfind('_')]))
            
            break
        except Exception as e:
            print(f"发生错误: {e}，将在1分钟后重启！")
            time.sleep(120)