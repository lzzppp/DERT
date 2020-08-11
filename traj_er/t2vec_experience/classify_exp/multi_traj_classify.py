""" 测试每次输入一个司机的多条轨迹进行分类的提升效果

    （因为是简答测试，把所有模块写在一个文件了）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import pandas as pd 
import pickle
import h5py
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class TrajPairData(Dataset):

    def __init__(self, idxs, labels):
        self.idxs = idxs
        self.labels = labels

    def __getitem__(self, index):
        return self.idxs[index], self.labels[index]
    
    def __len__(self):
        return len(self.idxs)

def custom_collate_fn(batch):
    """array到tensor，同时根据长度sort"""
    idxs, labels = zip(*batch)

    lens = torch.tensor(list(map(len, idxs)))
    lens, indices = torch.sort(lens, descending=True)

    labels = torch.tensor(labels).cuda()[indices]
    idxs = [torch.tensor(seq).cuda() for seq in idxs]  
    idxs = pad_sequence(idxs, batch_first=True)[indices]  # B x T x *
    return idxs, lens, labels
    
class MultiTrajClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, traj_embeddings=None):
        super(MultiTrajClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if traj_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(traj_embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch, lengths):
        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True)
        outputs, (ht, ct) = self.lstm(packed_input)  # 测试用默认的hidden state（应该是0）

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output

def get_emb(path, layer):
    print(path)
    with h5py.File(path, 'r') as f:
        vec = np.array(f['layer%d' % layer])
    return vec

def load_data(data_pkl_path, h5_embedding_paths):
    """载入pickle存储的数据和h5存储的trajectory emebdding，得到训练数据
    
        train部分分割为 train, val，train和test h5的embedding合并，
        所以每个test部分index要加上len(train_emb)
        Args:
        - data_pkl_path: str
        - h5_embedding_paths: tuple(train_path_str, test_path_str)
    """
    with open(data_pkl_path, 'rb') as f:
        info = pickle.load(f)
    train_data, train_label = info['train_traj_idx'], info['train_label']
    test_data, test_label = info['test_traj_idx'], info['test_label']
    assert len(train_data) == len(train_label) and len(test_data) == len(test_label), "Lens of train data and label are not matched!"
    
    train_data, train_label = np.array(train_data), np.array(train_label)
    np.random.seed(7)
    random_idx = np.random.choice(len(train_data), len(train_data), replace=False)
    train_data, train_label = list(train_data[random_idx]), list(train_label[random_idx])

    # 分割train到 train, val
    new_len_train = int(len(train_data) * 0.875)
    train_data, val_data = train_data[:new_len_train], train_data[new_len_train:]
    train_label, val_label = train_label[:new_len_train], train_label[new_len_train:]

    # load embedding
    train_emb = get_emb(h5_embedding_paths[0], 3)
    test_emb = get_emb(h5_embedding_paths[1], 3)

    len_train_emb = len(train_emb)

    test_data = [[idx + len_train_emb for idx in seq] for seq in test_data]
    return train_data, train_label, val_data, val_label, test_data, test_label, np.concatenate((train_emb, test_emb), axis=0)

def train(data_pkl_path, h5_embedding_paths, model_args):
    train_data, train_label, val_data, val_label, test_data, test_label, emb = load_data(data_pkl_path, h5_embedding_paths)

    train_dataset = TrajPairData(train_data, train_label)
    train_data_loader = DataLoader(train_dataset, batch_size=model_args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataset = TrajPairData(val_data, val_label)
    val_data_loader = DataLoader(val_dataset, batch_size=model_args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_dataset = TrajPairData(test_data, test_label)
    test_data_loader = DataLoader(test_dataset, batch_size=model_args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    num_class = len(set(train_label + test_label))
    print('driver number:', num_class)
    model = MultiTrajClassifier(emb.shape[0], emb.shape[1], model_args.lstm_hidden_dim, num_class, traj_embeddings=emb)
    device = torch.device("cuda:" + model_args.gpu)
    # device = torch.device("cpu")
    model.to(device)
    Loss = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=model_args.lr)
    print("Start training process...")
    for epoch in range(model_args.epochs):
        print("Start epoch %d..." % epoch)

        num_correct = 0
        total_loss = 0
        total_size = 0
        for batch_seq, lens, labels in train_data_loader:
            model.train()
            batch_seq = batch_seq.to(device)
            lens = lens.to(device)
            labels = labels.to(device)

            pred = model(batch_seq, lens)

            loss = Loss(pred, labels)
            optimizer.zero_grad()
            # print(loss.size())
            loss.backward()
            optimizer.step()

            _, pred_label = pred.max(1)

            # print(pred_label)
            num_correct += (pred_label == labels).sum()
            total_loss += loss.item()
            total_size += batch_seq.size(0)

        val_loss, val_acc = evaluate(model, val_data_loader, Loss, device)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss/len(train_data), float(num_correct) / total_size,
                                                                                val_loss, val_acc))

    test_loss, test_acc = evaluate(model, test_data_loader, Loss, device)

    print("Accuracy on test", test_acc)


def evaluate(model, data_loader, Loss, device):
    total_size = 0
    num_correct = 0
    total_loss = 0
    model.eval()
    for batch_seq, lens, labels in data_loader:
        batch_seq = batch_seq.to(device)
        lens = lens.to(device)
        labels = labels.to(device)
        # model.zero_grad()
        pred = model(batch_seq, lens)
        loss = Loss(pred, labels)
        # loss.backward()

        total_size += batch_seq.size(0)

        _, pred_label = pred.max(1)
        num_correct += (pred_label == labels).sum()

        total_loss += loss.item()

    return total_loss / total_size, float(num_correct) / total_size
