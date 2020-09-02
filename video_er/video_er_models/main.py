
import os
import sys
import time
import torch
import pyprind
import argparse
import torch.nn as nn
from tqdm import tqdm
from set_models import set_model
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim import lr_scheduler
from data_process import data_processer


def print_final_stats(epoch, runtime, datatime, stats):
    print (('Finished Epoch {epoch} || Run Time: {runtime:6.1f} | '
            'Load Time: {datatime:6.1f} || F1: {f1:6.2f} | Prec: {prec:6.2f} | '
            'Rec: {rec:6.2f} || Ex/s: {eps:6.2f}\n').format (
        epoch=epoch,
        runtime=runtime,
        datatime=datatime,
        f1=stats.f1 (),
        prec=stats.precision (),
        rec=stats.recall (),
        eps=stats.examples_per_sec ()))

def compute_scores(output, target):
    predictions = output.max (1)[1].data
    correct = (predictions == target.data).float ()
    incorrect = (1 - correct).float ()
    positives = (target.data == 1).float ()
    negatives = (target.data == 0).float ()

    tp = torch.dot (correct, positives)
    tn = torch.dot (correct, negatives)
    fp = torch.dot (incorrect, negatives)
    fn = torch.dot (incorrect, positives)

    return tp, tn, fp, fn

class Statistics(object):
    def __init__(self):
        self.loss_sum = 0
        self.examples = 0
        self.tps = 0
        self.tns = 0
        self.fps = 0
        self.fns = 0
        self.start_time = time.time()

    def update(self, loss=0, tps=0, tns=0, fps=0, fns=0):
        examples = tps + tns + fps + fns
        self.loss_sum += loss * examples
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        self.examples += examples

    def loss(self):
        return self.loss_sum / self.examples

    def f1(self):
        prec = self.precision()
        recall = self.recall()
        return 2 * prec * recall / max(prec + recall, 1)

    def precision(self):
        return 100 * self.tps / max(self.tps + self.fps, 1)

    def recall(self):
        return 100 * self.tps / max(self.tps + self.fns, 1)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples

    def examples_per_sec(self):
        return self.examples / (time.time() - self.start_time + 1)

def tally_parameters(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])

def main():
    parser = argparse.ArgumentParser (formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument ('-shallow_cam', default=1, help="root path to data directory")
    parser.add_argument ('-compatibility', default=1, help="root path to data directory")
    parser.add_argument ('--branches', nargs='+', type=str, default=['global', 'abd'])
    parser.add_argument ('--global-dim', type=int, default=1024)
    parser.add_argument ('--model_type', type=str, default='MCAN', choices=['MCAN', 'ABD-Net', 'RAG-Net', 'RAG-Net-PA'])
    parser.add_argument ('--abd-dim', type=int, default=1024)
    parser.add_argument ('--abd-np', type=int, default=2)
    parser.add_argument ('--abd-dan', nargs='+', type=str, default=['cam', 'pam'])
    parser.add_argument ('--abd-dan-no-head', action='store_true')
    parser.add_argument ('--shallow-cam', action='store_true')
    parser.add_argument ('--dropout', type=float, default=0.5)
    parser.add_argument ('--global-max-pooling', action='store_true')
    parser.add_argument ('--Warm-up-epoch', default=5, help="Warm-up-epochs")
    parser.add_argument ('--epoch', default=40, help="epochs")
    parser.add_argument ('--pos_neg', default=True, help="POS_NEG")
    parser.add_argument('--dataset', help="dataset", choices=['market1501', 'occluded_dukeMTMC'])
    args = parser.parse_args()
    best_save_path = "model_params_" + args.dataset + '_' + args.model_type + '.pth'
    epochs = args.epoch
    if args.dataset == 'market1501':
        path = os.getcwd().strip('video_er_models') + 'market1501/'
    elif args.dataset == 'occluded_dukeMTMC':
        path = os.getcwd().strip('video_er_models') + 'occluded_dukeMTMC/'
    train_dataloader, test_dataloader, valid_dataloader, train_dataset, test_dataset = data_processer(path)
    
    if args.pos_neg:    
        test_labels = list(test_dataset.labels)
        pos_neg_ratio = int((len (test_labels) - sum (test_labels)) / sum (test_labels))
        pos_neg_ratio = max(1, pos_neg_ratio) + 1 # Since the positive sample ratio of the data cannot be exactly equal to 0.25, 1 can be added flexibly here.
        print ("[Info] pos_neg_ratio: ", pos_neg_ratio)
    
        pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
        neg_weight = 2 - pos_weight
    
    model = set_model(args)
    model.init_models (train_dataset)
    optimizer = Adam (filter (lambda p: p.requires_grad, model.parameters ()), lr=0.001)
    if args.pos_neg:
        criterion = nn.NLLLoss(weight=torch.Tensor ([neg_weight, pos_weight])).cuda ()
    else:
        criterion = nn.NLLLoss().cuda ()
    model = model.cuda ()
    
    best_score = 0.0
    for epoch in tqdm (range (5), desc="Training first"):
        train (model, optimizer, criterion, train_dataloader, epoch, train=True)
    
    for param in model.parameters ():  # nn.Module有成员函数parameters()
        param.requires_grad = True
    optimizer = Adam (model.parameters (), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.StepLR (optimizer, step_size=3, gamma=0.9)
    
    for epoch in tqdm (range (epochs), desc="Training second"):
        train (model, optimizer, criterion, train_dataloader, epoch, train=True)
        scheduler.step ()
        scores = test (model, criterion, valid_dataloader, epoch, train=False)
        
        if scores > best_score:
            print ('* Best F1:', scores)
            best_score = scores
            new_best_found = True
            if best_save_path and new_best_found:
                print ('Saving best model...')
                torch.save (model.state_dict (), best_save_path)
                print ('Done.')
    
    model.load_state_dict(torch.load(best_save_path))
    scores = test (model, criterion, test_dataloader, epoch, train=False)
    print ("The test scores is : .4f" % scores.cpu ().numpy ())

def train(model, optimizer, criterion, dataloader, epoch, train=True, log_freq=5):
    if train:
        model.train ()
        run_type = "TRAIN"
    else:
        model.eval ()
        run_type = "TEST"

    datatime = 0
    runtime = 0
    cum_stats = Statistics ()
    stats = Statistics ()

    if train and epoch == 0:
        print ('* Number of trainable parameters:', tally_parameters (model))
    epoch_str = 'Epoch {0:d}'.format (epoch + 1)
    print ('===> ', run_type, epoch_str)
    batch_end = time.time ()
    pbar = pyprind.ProgBar (len (dataloader) // log_freq, bar_char='*', width=30)
    for batch_idx, batch in enumerate (dataloader):
        batch = tuple (t.cuda() for t in batch)
        batch_start = time.time()
        datatime += batch_start - batch_end

        output = model (batch[0], batch[1], 'gpu')

        loss = float ('NaN')
        if criterion:
            loss = criterion (output, batch[2])

        if batch[2] is not None:
            scores = compute_scores (output, batch[2])
        else:
            scores = [0] * 4

        cum_stats.update (float (loss), *scores)
        stats.update (float (loss), *scores)

        if (batch_idx + 1) % log_freq == 0:
            pbar.update ()
            stats = Statistics ()

        if train:
            model.zero_grad ()
            loss.backward ()
            optimizer.step ()

        batch_end = time.time ()
        runtime += batch_end - batch_start
    sys.stderr.flush ()
    print_final_stats (epoch + 1, runtime, datatime, cum_stats)

def test(model, optimizer, criterion, dataloader, epoch, train=False, log_freq=5):
    if train:
        model.train ()
        run_type = "TRAIN"
    else:
        model.eval ()
        run_type = "TEST"

    datatime = 0
    runtime = 0
    cum_stats = Statistics ()
    stats = Statistics ()

    if train and epoch == 0:
        print ('* Number of trainable parameters:', tally_parameters (model))
    epoch_str = 'Epoch {0:d}'.format (epoch + 1)
    print ('===> ', run_type, epoch_str)
    batch_end = time.time ()
    pbar = pyprind.ProgBar (len (dataloader) // log_freq, bar_char='*', width=30)
    for batch_idx, batch in enumerate (dataloader):
        batch = tuple (t.cuda () for t in batch)
        batch_start = time.time ()
        datatime += batch_start - batch_end

        output = model (batch[0], batch[1], 'gpu')

        loss = float ('NaN')
        if criterion:
            loss = criterion (output, batch[2])

        if batch[2] is not None:
            scores = compute_scores (output, batch[2])
        else:
            scores = [0] * 4

        cum_stats.update (float (loss), *scores)
        stats.update (float (loss), *scores)

        if (batch_idx + 1) % log_freq == 0:
            pbar.update ()
            stats = Statistics ()

        batch_end = time.time ()
        runtime += batch_end - batch_start
    sys.stderr.flush ()
    print_final_stats (epoch + 1, runtime, datatime, cum_stats)
    return cum_stats.f1 ()

if __name__ == "__main__":
    main()
