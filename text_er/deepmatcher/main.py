import os
import argparse
import deepmatcher_textual as dm


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', required=True)
parser.add_argument('-type', required=True, choices=('StructuralWithValue', 'Structural', 'Textual', 'Dirty', 'Dirty1', 'Dirty2'))
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-epoch', default=10, type=int)
parser.add_argument('-pos_neg', action='store_true')
opt = parser.parse_args()

dataset = opt.dataset
data_type = opt.type
epoch = opt.epoch
is_pos_neg = opt.pos_neg
batch_size = opt.batch_size

data_dir = os.path.join("dataset", data_type, dataset)

train, validation, test = dm.data.process(
    path=data_dir,
    cache='train_cache.pth',
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    embeddings_cache_path='./dataset',
    use_magellan_convention=True,
    ignore_columns=['ltable_id', 'rtable_id']
)

pos_neg_ratio = None
if is_pos_neg:
    test_labels = list(test.label)
    pos_neg_ratio = int((len(test_labels) - sum(test_labels)) / sum(test_labels))
    pos_neg_ratio = max(1, pos_neg_ratio)
    print("[Info] pos_neg_ratio: ", pos_neg_ratio)

model = dm.MCANModel(attr_comparator='concat-mul-diff', classifier='2-layer-highway')

model.run_train(
    train,
    validation,
    epochs=epoch,
    batch_size=batch_size,
    best_save_path='mcan_model.pth',
    pos_neg_ratio=pos_neg_ratio
)

model.run_eval(test)
