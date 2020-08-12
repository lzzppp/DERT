
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rga_classifier.model_example import ResNetCommonBranch, init_pretrained_weights, model_urls, ResNet, Bottleneck
from rga_classifier.modules import _merge_module, _transform_module, Transform

def resnet50_backbone(args):

    network = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,  # Always remove down-sampling
    )
    init_pretrained_weights(network, model_urls['resnet50'])

    return ResNetCommonBranch('resnet50', network, args)

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
        self.resnet = resnet50_backbone(vars(args))
        for p in self.parameters():
            p.requires_grad = False
        self.attr_comparator = attr_comparator
        self.classifier = classifier
        self.hidden_size = 512
        self.attr_comparator = _merge_module (self.attr_comparator)
        self.classifier = BinaryClassifier(self.classifier, hidden_size=self.hidden_size)
    
    def init_models(self, train_dataset):
        run_iter = DataLoader(train_dataset, batch_size=4, shuffle=False, pin_memory=False)
        init_batch = next(run_iter.__iter__ ())
        self.forward(init_batch[0], init_batch[1], torch.device('cpu'))
    
    def forward(self, left, right, device):
        
        left_input_, right_input_ = self.resnet(left, right)

        entity_comparison = self.attr_comparator(left_input_, right_input_)
        
        return self.classifier(entity_comparison)
