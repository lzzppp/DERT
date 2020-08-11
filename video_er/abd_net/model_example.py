
import torch
import logging
import argparse
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.utils.model_zoo as model_zoo
from copy import deepcopy
#from rga_modules import RGA_Module

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))

# class CAM_Module(nn.Module):
#     """ Channel attention module"""
#
#     def __init__(self, in_dim):
#         super().__init__()
#         self.channel_in = in_dim
#
#         self.gamma = Parameter(torch.zeros(1))
#         self.softmax = Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = x.view(m_batchsize, C, -1)
#         proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
#         energy_new = max_energy_0 - energy
#         attention = self.softmax(energy_new)
#         proj_value = x.view(m_batchsize, C, -1)
#
#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)
#
#         logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
#         gamma = self.gamma.to(out.device)
#         out = gamma * out + x
#         return out
#
# class ShallowCAM(nn.Module):
#
#     def __init__(self, args, feature_dim: int):
#
#         super().__init__()
#         self.input_feature_dim = feature_dim
#
#         use = args['shallow_cam']
#
#         if use:
#             self._cam_module = cam_module = CAM_Module(self.input_feature_dim)
#
#             if args['compatibility']:
#                 self._cam_module_abc = cam_module  # Forward Compatibility
#         else:
#             self._cam_module = None
#
#     def forward(self, x):
#
#         if self._cam_module is not None:
#             x = self._cam_module(x)
#
#         return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers, last_stride=2):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

# class PairAttention(LazyModule):
#     def _init(self,
#               hidden_size=None,
#               input_dropout=0,
#               alignment_network='bilinear',
#               score_dropout=0,
#               value_merge='concat',
#               transform_dropout=0,
#               comparison_merge='concat-mul-diff',
#               comparison_network='2-layer-highway',
#               input_size=None):
#         hidden_size = hidden_size if hidden_size is not None else input_size[0]
#         self.hidden_size = hidden_size
#         self.alignment_network = _alignment_module(alignment_network, hidden_size)
#
#         self.input_dropout = nn.Dropout(input_dropout)
#         self.transform_dropout = nn.Dropout(transform_dropout)
#         self.score_dropout = nn.Dropout(score_dropout)
#         self.softmax = nn.Softmax(dim=2)
#
#         self.raw_alignment = True
#
#     def _forward(self,
#                  input_with_meta,
#                  context_with_meta,
#                  raw_input_with_meta,
#                  raw_context_with_meta, device):
#         input = self.input_dropout(input_with_meta.data)
#         context = self.input_dropout(context_with_meta.data)
#         raw_input = self.input_dropout(raw_input_with_meta.data)
#         raw_context = self.input_dropout(raw_context_with_meta.data)
#
#         queries = input
#         keys = context
#         values = context
#         if self.raw_alignment:
#             queries = raw_input
#             keys = raw_context
#
#         # Dims: batch x len1 x len2
#         alignment_scores = self.score_dropout(self.alignment_network(queries, keys))
#
#         if context_with_meta.lengths is not None:
#             mask = sequence_mask(context_with_meta.lengths)
#             if device == torch.device('cpu'):
#                 mask = mask.unsqueeze(1)
#             else:
#                 mask = mask.unsqueeze(1).cuda()  # Make it broadcastable.
#             alignment_scores.data.masked_fill_(~mask, -float('inf'))
#
#         # Make values along dim 2 sum to 1.
#         normalized_scores = self.softmax(alignment_scores)
#
#         # Dims: batch x len1 x channels
#         values_aligned = torch.bmm(normalized_scores, values)
#
#         return AttrTensor.from_old_metadata(values_aligned, input_with_meta)

class pair_attention(nn.Module):
    def __init__(self):
        super(pair_attention, self).__init__()
        self.hidden_size = 512

    def forward(self, x, y):

        x_r = x.reshape (x.shape[0], x.shape[1], -1)
        y_r = y.reshape (y.shape[0], x.shape[1], -1)

        x_w = F.softmax (torch.bmm (x_r, y_r.transpose(1, 2)), dim=2)
        y_w = F.softmax (torch.bmm (y_r, x_r.transpose(1, 2)), dim=2)

        x_c_p = torch.bmm(x_w, y_r).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) + x
        y_c_p = torch.bmm(y_w, x_r).reshape(y.shape[0], y.shape[1], y.shape[2], x.shape[3]) + y

        x_w_m = F.softmax(torch.bmm(x_r.transpose(1, 2), y_r), dim=2)
        y_w_m = F.softmax(torch.bmm(y_r.transpose(1, 2), x_r), dim=2)

        x_p_p = (torch.bmm(x_w_m, y_r.transpose(1, 2)).transpose(1, 2)).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) + x
        y_p_p = (torch.bmm(y_w_m, x_r.transpose(1, 2)).transpose(1, 2)).reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3]) + y

        x_new = x_c_p + x_p_p
        y_new = y_c_p + y_p_p

        return x_new, y_new

class ResNetCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()
        self.height = 128
        self.width = 64
        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            RGA_Module(256, (self.height // 4) * (self.width // 4), use_spatial=True, use_channel=True,
                    cha_ratio=8, spa_ratio=8, down_ratio=8)
        )
        # self.rga_att1 = RGA_Module(256, (self.height // 4) * (self.width // 4), use_spatial=True, use_channel=True,
        #                             cha_ratio=8, spa_ratio=8, down_ratio=8)
        # self.rga_att2 = RGA_Module(512, (self.height // 8) * (self.width // 8), use_spatial=True, use_channel=True,
        #                             cha_ratio=8, spa_ratio=8, down_ratio=8)
        # self.shallow_cam = ShallowCAM(args, 256)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            # backbone.layer3,
            RGA_Module(512, (self.height // 8) * (self.width // 8), use_spatial=True, use_channel=True,
                        cha_ratio=8, spa_ratio=8, down_ratio=8)
        )
        # self.pair = pair_attention()
        self.weight = nn.Parameter(torch.Tensor(256, 256))
        self.dropout = 0.1
        self.feat_bn = nn.BatchNorm1d(512)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        self.feat_bn.apply(weights_init_kaiming)

    def forward(self, x, y):

        x = self.backbone1(x)
        y = self.backbone1(y)

        # intermediate = x = self.shallow_cam(x)
        feat_x = self.backbone2(x)
        feat_y = self.backbone2(y)

        # feat_x_, feat_y_ = self.pair(feat_x, feat_y)

        # feat_x = F.avg_pool2d(feat_x, feat_x.size()[2:]).view(feat_x.size(0), -1)
        # feat_y = F.avg_pool2d(feat_y, feat_y.size()[2:]).view(feat_y.size(0), -1)

        # featx = self.feat_bn(feat_x)
        # featy = self.feat_bn(feat_y)
        #
        # if self.dropout > 0:
        #     featx = self.drop(featx)
        #     featy = self.drop(featy)

        return feat_x, feat_y
