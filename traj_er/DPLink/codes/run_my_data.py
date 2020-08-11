# -*- coding=utf-8 -*
from __future__ import print_function, division

import os
import time
import torch
import mlflow
import argparse
import setproctitle
import numpy as np

from match import run_experiments, run_single_dataset_experiments


class CustomSettings(object):
    def __init__(self, seed=None, data='weibo', epoch=20, threshold=2000, lr_pretrain=0.001, lr_match=0.0005,
                 layers=1, rnn_mod="GRU", attn_mod='dot', lr_step=5, lr_decay=0.1, batch_size=32, l2=0.0,
                 poi=None, loss_mode="BCELoss", neg=32, intersect=1, topk=5, noise=0, pretrain=1, dropout_p=0.5,
                 hidden_size=200, loc_emb_size=200, tim_emb_size=10, poi_type=0, poi_size=21, poi_emb_size=10,
                 data_path="/data1/input/", save_path="/data1/output/", vid_size=None, traj_length=1):

        if seed is None:
            np.random.seed(1)
            torch.manual_seed(1)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        vid_size_dict = {
            "geolife": {24.0: 57754},
            "singapore" : {24.0: 13613}
        }

        self.data_path = os.path.join(os.getcwd().strip('DPLink/codes') + 'dataset/dplink_dataset' + data + '_mcan')
        self.save_path = os.path.join(os.getcwd() + '/save', data, 'output')
        self.data_name = data
        self.epoch = epoch
        self.threshold = threshold
        self.poi = poi
        self.lr_pretrain = lr_pretrain
        self.intersect = intersect
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.topk = topk

        self.rnn_mod = rnn_mod
        self.attn_mod = attn_mod
        self.layers = layers

        self.neg = neg
        self.loss_mode = loss_mode
        self.noise = noise
        self.pretrain = pretrain

        self.lr_match = lr_match
        self.batch_size = batch_size
        self.l2 = l2
        self.dropout_p = dropout_p

        self.hidden_size = hidden_size
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size

        self.poi_type = poi_type
        self.poi_size = poi_size
        self.poi_emb_size = poi_emb_size
        self.vid_size = vid_size_dict[data][traj_length]


if __name__ == '__main__':
    settings = {"weibo": {"hidden_size": 200,
                          "loc_emb_size": 200,
                          "tim_emb_size": 10,
                          "dropout_p": 0.3,
                          "lr_match": 0.0005,
                          "loss_mode": "BCELoss",
                          "l2": 1e-6},
                "porto": {"hidden_size": 200,  # origin 50
                               "loc_emb_size": 200,  # origin 50
                               "tim_emb_size": 10,
                               "dropout_p": 0.5,
                               "lr_match": 0.0004,
                               "loss_mode": "BCELoss",
                               "l2": 1e-5,
                               "vid_size": 2109
                          },
                "geolife": {"hidden_size": 300, # origin 50
                               "loc_emb_size": 300, # origin 50
                               "tim_emb_size": 60, # origin 10
                               "dropout_p": 0.5,
                               "lr_match": 0.0004,
                               "loss_mode": "BCELoss",
                               "l2": 1e-5,
                               "vid_size": 57754
                          },
                "singapore": {"hidden_size": 300,  # origin 50
                          "loc_emb_size": 300,  # origin 50
                          "tim_emb_size": 60,
                          "dropout_p": 0.5,
                          "lr_match": 0.0004,
                          "loss_mode": "BCELoss",
                          "l2": 1e-5,
                          "vid_size": 13612}
                }

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="porto", choices=['porto', "geolife", "singapore", 'gowalla', 'foursquare'])
    parser.add_argument("--gpu", type=str, default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--model", type=str, default="ERC", choices=["ERC", "ERPC"])
    parser.add_argument("--rnn", type=str, default="GRU")
    parser.add_argument("--noise_level", type=int, default=0)
    parser.add_argument("--poi_type", type=int, default=0)
    parser.add_argument("--use_poi", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=3200)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--pretrain_unit", type=str, default="ERCF", choices=["N", "E", "R", "C", "F", "ERCF"])
    parser.add_argument("--length", type=float)
    args = parser.parse_args()

    USE_POI = (args.use_poi == 1)
    device = torch.device("cuda:" + args.gpu)

    thre = args.threshold
    rnn_unit = 'GRU'
    attn_unit = 'dot'
    test_pretrain = False  # test the effect of different pretrain degree, working with run_pretrain
    pre_path, rank_pre2, hit_pre2 = None, None, None

    hidden_size = settings[args.data]["hidden_size"]
    loc_emb_size = settings[args.data]["loc_emb_size"]
    tim_emb_size = settings[args.data]["tim_emb_size"]
    dropout_p = settings[args.data]["dropout_p"]
    l2 = settings[args.data]["l2"]
    lr_match = settings[args.data]["lr_match"]
    # if run_id == 0:
    #     loss_mode = "BCELoss"
    # else:
    #     loss_mode = settings[args.data]["loss_mode"]
    loss_mode = settings[args.data]["loss_mode"]
    run_settings = CustomSettings(data=args.data, neg=32, seed=int(time.time()), pretrain=args.pretrain,
                                    loss_mode=loss_mode, lr_match=lr_match, l2=l2, dropout_p=dropout_p,
                                    tim_emb_size=tim_emb_size, loc_emb_size=loc_emb_size,
                                    hidden_size=hidden_size, epoch=args.epoch, threshold=args.threshold,
                                    rnn_mod=rnn_unit, attn_mod=attn_unit, save_path=None,
                                    noise=0 if USE_POI else args.noise_level, poi_type=args.poi_type,
                                    vid_size=settings[args.data]["vid_size"], traj_length=args.length)
    model, rank, hit, rank_pre, hit_pre = run_single_dataset_experiments(run_settings, model_type=args.model,
                                                            device=device, USE_POI=USE_POI,
                                                            unit=args.pretrain_unit)
