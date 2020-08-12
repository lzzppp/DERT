
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from abd_net.abd_modules import init_model
from abd_net.modules import _merge_module, _transform_module, Transform

class BinaryClassifier(nn.Sequential):
    def __init__(self, transform_network, hidden_size=None):
        super(BinaryClassifier, self).__init__()
        if transform_network:
            self.add_module('transform',
                            _transform_module(transform_network, hidden_size))
        self.add_module('softmax_transform',
                        Transform(
                            '1-layer', non_linearity=None, output_size=2))
        self.add_module('softmax', nn.LogSoftmax(dim=1))

class mcan_reid(nn.Module):
    def __init__(self, args, attr_comparator='concat-mul-diff', classifier='2-layer-highway'):
        super (mcan_reid, self).__init__ ()
        self.abd_net = init_model("resnet50", num_classes=751, loss={'xent'}, use_gpu=True, args=vars(args))
        for p in self.parameters():
            p.requires_grad = False
        self.attr_comparator = attr_comparator
        self.classifier = classifier
        self.hidden_size = 3072
        self.attr_comparator = _merge_module (self.attr_comparator)
        self.classifier = BinaryClassifier(self.classifier, hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def init_models(self, train_dataset):
        run_iter = DataLoader(train_dataset, batch_size=4, shuffle=False, pin_memory=False)
        init_batch = next(run_iter.__iter__ ())
        self.forward(init_batch[0], init_batch[1], torch.device('cpu'))

    def forward(self, left, right, device):

        left_final = self.abd_net(left)
        right_final = self.abd_net(right)

        entity_comparison = self.attr_comparator(left_final, right_final)
        entity_comparison = self.dropout(entity_comparison)
        return self.classifier(entity_comparison)
