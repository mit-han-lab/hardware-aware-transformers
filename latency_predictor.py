# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import random
import configargparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

class Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_layer_num):
        super(Net, self).__init__()

        self.first_layer = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x


class LatencyPredictor(object):
    def __init__(self, feature_norm, lat_norm, ckpt_path, lat_dataset_path='./latency_dataset/lat.tmp', feature_dim=10, hidden_dim=400, hidden_layer_num=3, train_steps=5000, bsz=128, lr=1e-5):
        self.dataset_path = lat_dataset_path
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.ckpt_path = ckpt_path

        self.dataset = None

        self.train_x = None
        self.train_y = None

        self.valid_x = None
        self.valid_y = None

        self.test_x = None
        self.test_y = None

        self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.train_steps = train_steps
        self.bsz = bsz

    def train(self):
        for i in range(self.train_steps):
            sample_ind = random.sample(range(len(self.train_x)), k=self.bsz)
            sample_x = [self.train_x[sample_ind[k]] for k in range(self.bsz)]
            sample_y = [self.train_y[sample_ind[k]] for k in range(self.bsz)]

            sample_x_tensor = torch.Tensor(sample_x)
            sample_y_tensor = torch.Tensor(sample_y)

            prediction = self.model(sample_x_tensor).squeeze()

            loss = self.criterion(prediction, sample_y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # validation
            if i % 100 == 0:
                with torch.no_grad():
                    sample_x_tensor = torch.Tensor(self.valid_x)
                    sample_y_tensor = torch.Tensor(self.valid_y)

                    prediction = self.model(sample_x_tensor).squeeze()
                    loss = self.criterion(prediction, sample_y_tensor)
                    print(f"Validation loss at {i} steps: {loss}")

        # test
        with torch.no_grad():
            sample_x_tensor = torch.Tensor(self.test_x)
            sample_y_tensor = torch.Tensor(self.test_y)
            prediction = self.model(sample_x_tensor).squeeze()
            loss = self.criterion(prediction, sample_y_tensor)
            print(f"Predicted latency: {prediction}")
            print(f"Real latency: {self.test_y}")
            print(f"Loss: {loss}")

            print(f"RMSE: {np.sqrt(self.criterion(self.lat_norm*sample_y_tensor, self.lat_norm*prediction))}")
            print(f"MAPD: {torch.mean(torch.abs((sample_y_tensor - prediction) / sample_y_tensor))}")

        torch.save(self.model.state_dict(), self.ckpt_path)

    def load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_path))

    def predict_lat(self, config):
        with torch.no_grad():
            features = utils.get_config_features(config)
            features_norm = np.array(features) / self.feature_norm

            prediction = self.model(torch.Tensor(features_norm)).item() * self.lat_norm

        return prediction

    def split(self):
        sample_num = len(self.dataset['x'])
        train_num = int(np.floor(0.8 * sample_num))
        valid_num = int(np.floor(0.1 * sample_num))

        self.train_x = self.dataset['x'][:train_num]
        self.train_y = self.dataset['y'][:train_num]

        self.valid_x = self.dataset['x'][train_num:(train_num+valid_num)]
        self.valid_y = self.dataset['y'][train_num:(train_num+valid_num)]

        self.test_x = self.dataset['x'][(train_num+valid_num):]
        self.test_y = self.dataset['y'][(train_num+valid_num):]

    def read_dataset(self):
        features_norm_all = []
        lats_all = []
        with open(self.dataset_path, 'r') as fid:
            next(fid) # skip first line of CSV
            for line in fid:
                features = line.split(',')[:self.feature_dim]
                features_eval = list(map(eval, features))
                features_norm = np.array(features_eval) / self.feature_norm
                features_norm_all.append(features_norm)

                lats = line.split(',')[self.feature_dim:]
                total_lat = eval(lats[0]) + eval(lats[1])
                lats_all.append(total_lat / self.lat_norm)
        tmp = list(zip(features_norm_all, lats_all))
        random.shuffle(tmp)
        features_norm_all, lats_all = zip(*tmp)
        self.dataset = {'x': features_norm_all, 'y': lats_all}


if __name__=='__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--configs', required=True, is_config_file=True)
    parser.add_argument('--dataset-path')

    parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/lat.tmp', help='the path to read latency dataset')
    parser.add_argument('--feature-norm', type=float, nargs='+', default=[640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2], help='normalizing factor for each feature')
    parser.add_argument('--lat-norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--feature-dim', type=int, default=10, help='dimension of feature vector')
    parser.add_argument('--hidden-dim', type=int, default=400, help='hidden dimension of FC layers in latency predictor')
    parser.add_argument('--hidden-layer-num', type=int, default=3, help='number of FC layers')
    parser.add_argument('--ckpt-path', type=str, default='latency_dataset/ckpts/tmp.pt', help='path to save latency predictor weights')
    parser.add_argument('--train-steps', type=int, default=5000, help='latency predictor training steps')
    parser.add_argument('--bsz', type=int, default=128, help='latency predictor training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='latency predictor training learning rate')

    args = parser.parse_args()
    print(args)

    predictor = LatencyPredictor(lat_dataset_path=args.lat_dataset_path,
                           feature_norm=args.feature_norm,
                           lat_norm=args.lat_norm,
                           feature_dim=args.feature_dim,
                           hidden_dim=args.hidden_dim,
                           hidden_layer_num=args.hidden_layer_num,
                           ckpt_path=args.ckpt_path,
                           train_steps=args.train_steps,
                           bsz=args.bsz,
                           lr=args.lr)

    predictor.read_dataset()
    predictor.split()
    predictor.train()
    print('Latency predictor training finished')

    predictor.load_ckpt()
    config_example = {
        'encoder': {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 6,
            'encoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 3072, 3072],
            'encoder_self_attention_heads': [8, 8, 8, 8, 8, 4],
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_layer_num': 5,
            'decoder_ffn_embed_dim': [2048, 3072, 3072, 3072, 1024],
            'decoder_self_attention_heads': [4, 8, 8, 4, 4],
            'decoder_ende_attention_heads': [4, 8, 8, 4, 4],
            'decoder_arbitrary_ende_attn':  [-1, 1, 1, 1, 1]
        }
    }

    predict = predictor.predict_lat(config_example)
    print(f'Example config: {config_example}')
    print(f'Example latency: {predict}')
