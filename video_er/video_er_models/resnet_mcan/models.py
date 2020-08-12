
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from resnet_mcan.model_example import ResNetCommonBranch, init_pretrained_weights, model_urls, ResNet, Bottleneck
from resnet_mcan.modules import _merge_module, _transform_module, Transform, RNN, SelfAttention, PairAttention, Fusion, \
    GatingMechanism, GlobalAttention, AttrTensor

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
        # self.resnet = ResNetCommonBranch("None", backbone, vars(args))
        # self.rga_model = ResNet50_RGA_Model(num_feat=512, height=64, width=128, dropout=0.1, branch_name='rgasc')
        self.resnet = resnet50_backbone(vars(args))
        for p in self.parameters():
            p.requires_grad = False
        self.attr_comparator = attr_comparator
        self.classifier = classifier
        self.hidden_size = 512
        self.attr_comparator = _merge_module (self.attr_comparator)
        self.classifier = BinaryClassifier(self.classifier, hidden_size=self.hidden_size)
        self.gru = RNN('gru', hidden_size=self.hidden_size)
        self.self_attention = SelfAttention(hidden_size=self.hidden_size, alignment_network="dot")
        self.pair_attention = PairAttention(alignment_network='bilinear')
        self.word_fusion = Fusion(hidden_size=self.hidden_size, is_meta=True)
        self.gate_mechanism = GatingMechanism(hidden_size=self.hidden_size, dropout=0.1)
        self.global_attention = GlobalAttention(hidden_size=self.hidden_size, style="dot", dropout=0.5)

    def init_models(self, train_dataset):
        run_iter = DataLoader(train_dataset, batch_size=4, shuffle=False, pin_memory=False)
        init_batch = next(run_iter.__iter__ ())
        self.forward(init_batch[0], init_batch[1], torch.device('cpu'))

    def forward(self, left, right, device):

        left_input_, right_input_ = self.resnet(left, right)

        left_input_ = left_input_.view (left.shape[0], 512, -1).transpose (2, 1)
        right_input_ = right_input_.view (left.shape[0], 512, -1).transpose (2, 1)

        left_input_data = {"attr": (left_input_, torch.tensor ([128] * left_input_.shape[0], dtype=torch.long))}
        right_input_data = {"attr": (right_input_, torch.tensor ([128] * right_input_.shape[0], dtype=torch.long))}

        left_input = AttrTensor (**left_input_data)
        right_input = AttrTensor (**right_input_data)

        left_contextualized = self.gru (left_input)
        right_contextualized = self.gru (right_input)

        left_contextualized = self.self_attention (left_contextualized, device)
        right_contextualized = self.self_attention (right_contextualized, device)

        left_fused = self.pair_attention (
            left_contextualized, right_contextualized, left_input, right_input, device)
        right_fused = self.pair_attention (
            right_contextualized, left_contextualized, right_input, left_input, device)

        left_fused = self.word_fusion (left_contextualized, left_fused)
        right_fused = self.word_fusion (right_contextualized, right_fused)
        # left_gated, right_gated = left_fused, right_fused
        left_gated = self.gate_mechanism (left_input, left_fused)
        right_gated = self.gate_mechanism (right_input, right_fused)

        left_final = self.global_attention (left_gated, device)
        right_final = self.global_attention (right_gated, device)

        entity_comparison = self.attr_comparator (left_final.data, right_final.data)

        return self.classifier(entity_comparison)
