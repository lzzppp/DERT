import os
import argparse
import mcan as dm

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', required=True, type=str)
parser.add_argument('-type', required=True, choices=('StructuralWithValue', 'Structural',
                                                     'Textual', 'Dirty', 'Dirty1', 'Dirty2'))
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-epoch', default=23, type=int)
parser.add_argument('-pos_neg', action='store_true')
parser.add_argument('-ablation_part', default='none', type=str)
parser.add_argument('-length', type=float)
opt = parser.parse_args()

dataset = opt.dataset
data_type = opt.type
epoch = opt.epoch
is_pos_neg = opt.pos_neg
batch_size = opt.batch_size

# data_dir = os.path.join("dataset", data_type, dataset)
data_dir = os.path.join(os.getcwd().strip('MCAN') + 'dataset/mcan_dataset', dataset)

train, validation, test = dm.data.process(
    path=data_dir,
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    embeddings_cache_path=os.getcwd()[:-13] + '/text_er/MCAN/dataset',
    ignore_columns=['ltable_id', 'rtable_id']
)

pos_neg_ratio = None
if is_pos_neg:
    test_labels = list(test.label)
    pos_neg_ratio = int((len(test_labels) - sum(test_labels)) / sum(test_labels))
    pos_neg_ratio = max(1, pos_neg_ratio) # Since the positive sample ratio of the data cannot be exactly equal to 0.25, 1 can be added flexibly here.
    print("[Info] pos_neg_ratio: ", pos_neg_ratio)

model = dm.MCANModel()

model.run_train(
    train,
    validation,
    epochs=epoch,
    batch_size=batch_size,
    best_save_path='mcan_model_%s.pth'%opt.dataset,
    pos_neg_ratio=pos_neg_ratio,
    ablation_part = opt.ablation_part
)

model.run_eval(test)
