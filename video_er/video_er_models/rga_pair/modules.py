from __future__ import division

from functools import reduce
import abc
import math

import six

# import mcan_structured as mcan
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable

# from . import _utils
# from ..batch import AttrTensor

AttrTensor_ = namedtuple('AttrTensor', ['data', 'lengths'])

class AttrTensor(AttrTensor_):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        if len(kwargs) == 0:
            return super(AttrTensor, cls).__new__(cls, *args)
        else:
            attr = kwargs['attr']
            if isinstance(attr, tuple):
                data = attr[0]
                lengths = attr[1]
            else:
                data = attr
                lengths = None
            # word_probs = None
            # if 'word_probs' in train_info.metadata:
            #     raw_word_probs = train_info.metadata['word_probs'][name]
            #     word_probs = torch.Tensor(
            #         [[raw_word_probs[w] for w in b] for b in data.data])
            #     if data.is_cuda:
            #         word_probs = word_probs.cuda()
            # pc = None
            # if 'pc' in train_info.metadata:
            #     pc = torch.Tensor(train_info.metadata['pc'][name])
            #     if data.is_cuda:
            #         pc = pc.cuda()
            return AttrTensor(data, lengths)

    @staticmethod
    def from_old_metadata(data, old_attrtensor):
        return AttrTensor(data, *old_attrtensor[1:])

def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(
        lengths.unsqueeze(1)))

def get_module(cls, op, required=False, op_kwarg=None, **kwargs):
    if op is None and not required or isinstance(op, cls):
        return op
    elif required:
        return cls(**kwargs)
    elif isinstance(op, six.string_types):
        if op_kwarg is not None:
            kwargs[op_kwarg] = op
            return cls(**kwargs)
        else:
            return cls(op, **kwargs)
    elif six.callable(op) and not isinstance(op, torch.nn.Module):
        return LazyModuleFn(op)
    else:
        raise ValueError(
            str(cls) + ' arg must be a valid string, a ' + str(cls) + ' object, or a '
            'callable.')

def check_nan(*values):
    for value in values:
        if isinstance(value, AttrTensor):
            value = value.data
        if isinstance(value, torch.Tensor) and (value != value).any():
            print('NaN detected!!!')
            pdb.set_trace()

@six.add_metaclass(abc.ABCMeta)
class LazyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LazyModule, self).__init__()
        self._init_args = args
        self._init_kwargs = kwargs
        self._initialized = False
        self._fns = []
        self.signature = None

    def forward(self, input, *args, **kwargs):
        if not self._initialized:
            try:
                self._init(
                    *self._init_args,
                    input_size=self._get_input_size(input, *args, **kwargs),
                    **self._init_kwargs)
            except TypeError as e:
                self._init(*self._init_args, **self._init_kwargs)
            for fn in self._fns:
                super(LazyModule, self)._apply(fn)

            if self.signature is not None:
                self._verify_signature(input, *args)
            _check_nan = True
            if _check_nan:
                self.register_forward_hook(LazyModule._check_nan_hook)
                self.register_backward_hook(LazyModule._check_nan_hook)

            self._initialized = True

        return self._forward(input, *args, **kwargs)

    def expect_signature(self, signature):
        self.signature = signature

    def _verify_signature(self, *args):
        return True

    def _get_input_size(self, *args, **kwargs):
        if len(args) > 1:
            return [self._get_input_size(input) for input in args]
        elif isinstance(args[0], (AttrTensor, Variable)):
            return args[0].data.size(-1)
        else:
            return None

    def _apply(self, fn):
        if not self._initialized:
            self._fns.append(fn)
        else:
            super(LazyModule, self)._apply(fn)

    @staticmethod
    def _check_nan_hook(m, *tensors):
        check_nan(*tensors)

    def _init(self):
        pass

    @abc.abstractmethod
    def _forward(self):
        pass


class NoMeta(nn.Module):
    def __init__(self, module):
        super(NoMeta, self).__init__()
        self.module = module

    def forward(self, *args):
        module_args = []
        for arg in args:
            module_args.append(arg.data if isinstance(arg, AttrTensor) else arg)

        results = self.module(*module_args)

        if not isinstance(args[0], AttrTensor):
            return results
        else:
            if not isinstance(results, tuple):
                results = (results,)

            if len(results) != len(args) and len(results) != 1 and len(args) != 1:
                raise ValueError(
                    'Number of inputs must equal number of outputs, or '
                    'number of inputs must be 1 or number of outputs must be 1.')

            results_with_meta = []
            for i in range(len(results)):
                arg_i = min(i, len(args) - 1)
                results_with_meta.append(
                    AttrTensor.from_old_metadata(results[i], args[arg_i]))

            if len(results_with_meta) == 1:
                return results_with_meta[0]

            return tuple(results_with_meta)


class ModuleMap(nn.Module):
    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, module):
        setattr(self, name, module)

    def __delitem__(self, name):
        delattr(self, name)


class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        modules = list(self._modules.values())
        inputs = modules[0](*inputs)
        for module in modules[1:]:
            if isinstance(inputs, tuple) and not isinstance(inputs, AttrTensor):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class LazyModuleFn(LazyModule):
    def _init(self, fn, *args, **kwargs):
        self.module = fn(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class RNN(LazyModule):
    _supported_styles = ['rnn', 'gru', 'lstm']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self,
              unit_type='gru',
              hidden_size=None,
              layers=1,
              bidirectional=True,
              dropout=0,
              input_dropout=0,
              last_layer_dropout=0,
              bypass_network=None,
              connect_num_layers=1,
              input_size=None,
              **kwargs):
        hidden_size = input_size if hidden_size is None else hidden_size
        last_layer_dropout = dropout if last_layer_dropout is None else last_layer_dropout

        if bidirectional:
            hidden_size //= 2

        if bypass_network is not None:
            assert layers % connect_num_layers == 0
            rnn_groups = layers // connect_num_layers
            layers_per_group = connect_num_layers
        else:
            rnn_groups = 1
            layers_per_group = layers

        bad_args = [
            'input_size', 'input_size', 'num_layers', 'batch_first', 'dropout',
            'bidirectional'
        ]
        assert not any([a in kwargs for a in bad_args])

        self.rnn_groups = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.bypass_networks = nn.ModuleList()
        self.input_dropout = NoMeta(nn.Dropout(input_dropout))

        rnn_in_size = input_size
        # for g in range(rnn_groups):
        #     self.rnn_groups.append(
        #         self._get_rnn_module(
        #             unit_type,
        #             input_size=rnn_in_size,
        #             hidden_size=hidden_size,
        #             num_layers=layers_per_group,
        #             batch_first=True,
        #             dropout=dropout,
        #             bidirectional=bidirectional,
        #             **kwargs))
        #
        #     if g != rnn_groups:
        #         self.dropouts.append(nn.Dropout(dropout))
        #     else:
        #         self.dropouts.append(nn.Dropout(last_layer_dropout))
        #     self.bypass_networks.append(_bypass_module(bypass_network))
        #
        #     if bidirectional:
        #         rnn_in_size = hidden_size * 2
        #     else:
        #         rnn_in_size = hidden_size
        self.rnn = nn.GRU(input_size=rnn_in_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def _forward(self, input_with_meta):
        output = self.input_dropout(input_with_meta.data)

        # for rnn, dropout, bypass in zip(self.rnn_groups, self.dropouts,
        #                                 self.bypass_networks):
        #     new_output = dropout(rnn(output)[0])
        #     if bypass:
        #         new_output = bypass(new_output, output)
        #     output = new_output
        # self.rnn.flatten_parameters()
        # self.rnn.flatten_parameters()
        new_output = self.rnn(output)[0]
        
        return AttrTensor.from_old_metadata(new_output, input_with_meta)

    def _get_rnn_module(self, unit_type, *args, **kwargs):
        return getattr(nn, unit_type.upper())(*args, **kwargs)


class GatingMechanism(LazyModule):
    def _init(self,
              hidden_size=None,
              input_size=None,
              dropout=0.1):
        hidden_size = input_size if hidden_size is None else hidden_size
        input_size = hidden_size
        self.in1_features = input_size
        self.in2_features = input_size
        self.out_features = hidden_size
        self.w1 = nn.Parameter(torch.FloatTensor(self.out_features, self.in1_features))
        self.w2 = nn.Parameter(torch.FloatTensor(self.out_features, self.in2_features))
        self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def _forward(self, input_with_meta, hidden_with_meta):
        E = input_with_meta.data
        H = hidden_with_meta.data
        G = E.matmul(self.w1.t()) + H.matmul(self.w2.t()) + self.bias
        G = torch.sigmoid(G)
        output = E.mul(G) + H.mul(1 - G)
        output = self.dropout(output)
        return AttrTensor.from_old_metadata(output, input_with_meta)


class Fusion(LazyModule):
    def _init(self, merge="concat-mul-diff", transfrom="2-layer-highway",
              hidden_size=None, input_size=None, is_meta=True):
        hidden_size = hidden_size if hidden_size is not None else input_size[0]
        self.merge_network = _merge_module(merge)
        self.transform_network = _transform_module(transfrom, hidden_size)
        self.is_meta = is_meta

    def _forward(self, raw_input_meta, contextualized_meta):
        if self.is_meta:
            raw_input = raw_input_meta.data
            contextualized = contextualized_meta.data
        else:
            raw_input = raw_input_meta
            contextualized = contextualized_meta
        merged = self.merge_network(raw_input, contextualized)
        transformed = self.transform_network(merged)
        return AttrTensor.from_old_metadata(transformed, raw_input_meta) if self.is_meta else transformed

class GlobalAttention(LazyModule):
    def _init(self,
              hidden_size=None,
              input_size=None,
              dropout=0.1,
              style="dot",
              use_meta=True,
              transform_network="2-layer-highway"):
        self.hidden_size = input_size if hidden_size is None else hidden_size
        self.input_size = self.hidden_size
        self.alignment_network = nn.Linear(hidden_size, 1)
        self.input_dropout = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.output_dropout = nn.Dropout(p=dropout)
        self.style = style
        self.use_meta = use_meta
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)        

    def _forward(self, input_with_meta, device):
        if self.use_meta:
            input = self.input_dropout(input_with_meta.data)
        else:
            input = input_with_meta
        k = v = input
        # Get Attention Score
        # calculate attention score
        attn_score = self.alignment_network(k)
        attn_score = attn_score.squeeze(2)
        # Mask Padding
        if self.use_meta:
            if input_with_meta.lengths is not None:
                if device == torch.device('cpu'):
                    mask = sequence_mask (input_with_meta.lengths)
                else:
                    mask = sequence_mask(input_with_meta.lengths).cuda()
                attn_score.data.masked_fill_(~mask, -float('inf'))
        # if meta_data is not None:
        #     mask = meta_data
        #     if isinstance(attn_score.data, torch.FloatTensor):
        #         attn_score.data.masked_fill_(1 - mask, -float('inf'))
        #     else:
        #         attn_score.data.masked_fill_(1 - mask.cuda(), -float('inf'))

        normalized_score = F.softmax(attn_score, dim=1)
        normalized_score = self.attn_dropout(normalized_score)

        output = torch.bmm(normalized_score.unsqueeze(1), v)
        output = output.squeeze(1)

        output = self.output_dropout(output)
        if self.use_meta:
            return AttrTensor.from_old_metadata(output, input_with_meta)
        else:
            return output

class SelfAttention(LazyModule):
    def _init(self,
              heads=1,
              hidden_size=None,
              input_dropout=0,
              alignment_network='dot',
              scale=False,
              score_dropout=0,
              value_transform_network=None,
              value_merge='concat',
              transform_dropout=0,
              output_transform_network=None,
              output_dropout=0,
              bypass_network='highway',
              input_size=None,
              raw_input=False):
        hidden_size = hidden_size if hidden_size is not None else input_size

        self.alignment_networks = nn.ModuleList()
        for head in range(heads):
            self.alignment_networks.append(
                _alignment_module(alignment_network, hidden_size))

        if value_transform_network is None and heads > 1:
            value_transform_network = Transform(
                '1-layer-highway', non_linearity=None, hidden_size=hidden_size // heads)
        self.value_transform_network = _transform_module(
            value_transform_network, hidden_size // heads)

        self.value_merge = _merge_module(value_merge)

        self.softmax = nn.Softmax(dim=2)

        if output_transform_network is None and heads > 1:
            output_transform_network = Transform(
                '1-layer-highway', non_linearity=None, hidden_size=hidden_size)
        self.output_transform_network = _transform_module(
            output_transform_network, hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(output_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.bypass_network = _bypass_module(bypass_network)

        self.heads = heads
        self.scale = scale
        self.hidden_size = hidden_size
        self.raw_input = raw_input

    def _forward(self, input_with_meta, device):
        input = self.input_dropout(input_with_meta.data)
        # if raw_input_with_meta is not None:
        #     raw_input = self.input_dropout(raw_input_with_meta.data)

        values_aligned = []
        for head in range(self.heads):
            # Dims: batch x len1 x len2
            if self.raw_input:
                alignment_scores = self.score_dropout(self.alignment_networks[head](raw_input, raw_input))
            else:
                alignment_scores = self.score_dropout(self.alignment_networks[head](input, input))

            if self.scale:
                alignment_scores = alignment_scores / math.sqrt(self.hidden_size)

            if input_with_meta.lengths is not None:
                mask = sequence_mask(input_with_meta.lengths)
                if device == torch.device('cpu'):
                    mask = mask.unsqueeze(1)
                else:
                    mask = mask.unsqueeze(1).cuda()  # Make it broadcastable.
                alignment_scores.data.masked_fill_(~mask, -float('inf'))
                # alignment_scores.data.masked_fill_(1 - mask, 0)
            normalized_scores = self.softmax(alignment_scores)
            if self.value_transform_network is not None:
                values_transformed = self.transform_dropout(
                    self.value_transform_network(input))
            else:
                values_transformed = input

            # Dims: batch x len1 x channels
            values_aligned.append(torch.bmm(normalized_scores, values_transformed))

        values_merged = self.value_merge(*values_aligned)

        output = values_merged
        if self.output_transform_network:
            output = self.output_transform_network(output)
        output = self.output_dropout(output)

        final_output = self.bypass_network(output, input)

        return AttrTensor.from_old_metadata(final_output, input_with_meta)


class AlignmentNetwork(LazyModule):
    _supported_styles = ['dot', 'general', 'decomposable', 'bilinear']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self,
              style='decomposable',
              hidden_size=None,
              transform_network='2-layer-highway',
              input_size=None):
        if style in ['general', 'decomposable']:
            if style == 'general':
                assert hidden_size is None or hidden_size == input_size
            self.transform = _transform_module(transform_network, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.style = style
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def _forward(self, input, context):
        if self.style == 'dot':
            return torch.bmm(
                input,  # batch x len1 x input_size
                context.transpose(1, 2))  # batch x ch x input_size
        elif self.style == 'general':
            return torch.bmm(
                input,  # batch x len1 x input_size
                self.transform(context).transpose(1, 2))  # batch x input_size x len2
        elif self.style == 'decomposable':
            return torch.bmm(
                self.transform(input),  # batch x hidden_size x len2
                self.transform(context).transpose(1, 2))  # batch x hidden_size x len2
        elif self.style == "biliear":
            return torch.bmm(input.matmul(self.weight), context.transpose(1, 2))


class PairAttention(LazyModule):
    def _init(self,
              hidden_size=None,
              input_dropout=0,
              alignment_network='bilinear',
              score_dropout=0,
              value_merge='concat',
              transform_dropout=0,
              comparison_merge='concat-mul-diff',
              comparison_network='2-layer-highway',
              input_size=None):
        hidden_size = hidden_size if hidden_size is not None else input_size[0]
        self.hidden_size = hidden_size
        self.alignment_network = _alignment_module(alignment_network, hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(score_dropout)
        self.softmax = nn.Softmax(dim=2)

        self.raw_alignment = True

    def _forward(self,
                 input_with_meta,
                 context_with_meta,
                 raw_input_with_meta,
                 raw_context_with_meta, device):
        input = self.input_dropout(input_with_meta.data)
        context = self.input_dropout(context_with_meta.data)
        raw_input = self.input_dropout(raw_input_with_meta.data)
        raw_context = self.input_dropout(raw_context_with_meta.data)

        queries = input
        keys = context
        values = context
        if self.raw_alignment:
            queries = raw_input
            keys = raw_context

        # Dims: batch x len1 x len2
        alignment_scores = self.score_dropout(self.alignment_network(queries, keys))

        if context_with_meta.lengths is not None:
            mask = sequence_mask(context_with_meta.lengths)
            if device == torch.device('cpu'):
                mask = mask.unsqueeze(1)
            else:
                mask = mask.unsqueeze(1).cuda()  # Make it broadcastable.
            alignment_scores.data.masked_fill_(~mask, -float('inf'))

        # Make values along dim 2 sum to 1.
        normalized_scores = self.softmax(alignment_scores)

        # Dims: batch x len1 x channels
        values_aligned = torch.bmm(normalized_scores, values)

        return AttrTensor.from_old_metadata(values_aligned, input_with_meta)


class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)


class Pool(LazyModule):
    _supported_styles = [
        'avg', 'divsqrt', 'inv-freq-avg', 'sif', 'max', 'last', 'last-simple',
        'birnn-last', 'birnn-last-simple'
    ]

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self, style, alpha=0.001):
        assert self.supports_style(style)
        self.style = style.lower()
        self.register_buffer('alpha', torch.Tensor([alpha]))

    def _forward(self, input_with_meta):
        input = input_with_meta.data

        if self.style == 'last':
            lengths = input_with_meta.lengths
            lasts = Variable(lengths.view(-1, 1, 1).repeat(1, 1, input.size(2))) - 1
            output = torch.gather(input, 1, lasts).squeeze(1).float()
        elif self.style == 'last-simple':
            output = input[:, input.size(1), :]
        elif self.style == 'birnn-last':
            hsize = input.size(2) // 2
            lengths = input_with_meta.lengths
            lasts = Variable(lengths.view(-1, 1, 1).repeat(1, 1, hsize)) - 1

            forward_outputs = input.narrow(2, 0, input.size(2) // 2)
            forward_last = forward_outputs.gather(1, lasts).squeeze(1)

            backward_last = input[:, 0, hsize:]
            output = torch.cat((forward_last, backward_last), 1)
        elif self.style == 'birnn-last-simple':
            forward_last = input[:, input.size(1), :hsize]
            backward_last = input[:, 0, hsize:]
            output = torch.cat((forward_last, backward_last), 1)
        elif self.style == 'max':
            if input_with_meta.lengths is not None:
                mask = sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(2)  # Make it broadcastable.
                input.data.masked_fill_(1 - mask, -float('inf'))
            output = input.max(dim=1)[0]
        else:
            if input_with_meta.lengths is not None:
                mask = sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(2)  # Make it broadcastable.
                input.data.masked_fill_(1 - mask, 0)

            lengths = Variable(input_with_meta.lengths.clamp(min=1).unsqueeze(1).float())
            if self.style == 'avg':
                output = input.sum(1) / lengths
            elif self.style == 'divsqrt':
                output = input.sum(1) / lengths.sqrt()
            elif self.style == 'inv-freq-avg':
                inv_probs = self.alpha / (input_with_meta.word_probs + self.alpha)
                weighted = input * Variable(inv_probs.unsqueeze(2))
                output = weighted.sum(1) / lengths.sqrt()
            elif self.style == 'sif':
                inv_probs = self.alpha / (input_with_meta.word_probs + self.alpha)
                weighted = input * Variable(inv_probs.unsqueeze(2))
                v = (weighted.sum(1) / lengths.sqrt())
                pc = Variable(input_with_meta.pc).unsqueeze(0).repeat(v.shape[0], 1)
                proj_v_on_pc = torch.bmm(v.unsqueeze(1), pc.unsqueeze(2)).squeeze(2) * pc
                output = v - proj_v_on_pc
            else:
                raise NotImplementedError(self.style + ' is not implemented.')

        return AttrTensor.from_old_metadata(output, input_with_meta)


def add(x, y):
    return x+y


class Merge(LazyModule):
    _style_map = {
        'sum': lambda *args: reduce(add, args),
        'concat': lambda *args: torch.cat(args, args[0].dim() - 1),
        'diff': lambda x, y: x - y,
        'abs-diff': lambda x, y: torch.abs(x - y),
        'concat-diff': lambda x, y: torch.cat((x, y, x - y), x.dim() - 1),
        'concat-abs-diff': lambda x, y: torch.cat((x, y, torch.abs(x - y)), x.dim() - 1),
        'mul': lambda x, y: torch.mul(x, y),
        'concat-mul-diff': lambda x, y: torch.cat((x, y, torch.mul(x, y), torch.abs(x - y)), x.dim() - 1)
    }

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._style_map

    def _init(self, style):
        assert self.supports_style(style)
        self.op = Merge._style_map[style.lower()]

    def _forward(self, *args):
        return self.op(*args)


class Bypass(LazyModule):
    _supported_styles = ['residual', 'highway']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self, style, residual_scale=True, highway_bias=-2, input_size=None):
        assert self.supports_style(style)
        self.style = style.lower()
        self.residual_scale = residual_scale
        self.highway_bias = highway_bias
        self.highway_gate = nn.Linear(input_size[1], input_size[0])

    def _forward(self, transformed, raw):
        assert transformed.shape[:-1] == raw.shape[:-1]

        tsize = transformed.shape[-1]
        rsize = raw.shape[-1]
        adjusted_raw = raw
        if tsize < rsize:
            assert rsize / tsize <= 50
            if rsize % tsize != 0:
                padded = F.pad(raw, (0, tsize - rsize % tsize))
            else:
                padded = raw
            adjusted_raw = padded.view(*raw.shape[:-1], -1, tsize).sum(-2) * math.sqrt(
                tsize / rsize)
        elif tsize > rsize:
            multiples = math.ceil(tsize / rsize)
            adjusted_raw = raw.repeat(*([1] * (raw.dim() - 1)), multiples).narrow(
                -1, 0, tsize)

        if self.style == 'residual':
            res = transformed + adjusted_raw
            if self.residual_scale:
                res *= math.sqrt(0.5)
            return res
        elif self.style == 'highway':
            transform_gate = torch.sigmoid(self.highway_gate(raw) + self.highway_bias)
            carry_gate = 1 - transform_gate
            return transform_gate * transformed + carry_gate * adjusted_raw


class Transform(LazyModule):
    _supported_nonlinearities = [
        'sigmoid', 'tanh', 'relu', 'elu', 'selu', 'glu', 'leaky_relu'
    ]

    @classmethod
    def supports_nonlinearity(cls, nonlin):
        return nonlin.lower() in cls._supported_nonlinearities

    def _init(self,
              style,
              layers=1,
              bypass_network=None,
              non_linearity='leaky_relu',
              hidden_size=None,
              output_size=None,
              input_size=None):
        hidden_size = hidden_size or input_size
        output_size = output_size or hidden_size

        parts = style.split('-')

        if 'layer' in parts:
            layers = int(parts[parts.index('layer') - 1])

        for part in parts:
            if Bypass.supports_style(part):
                bypass_network = part
            if Transform.supports_nonlinearity(part):
                non_linearity = part

        self.transforms = nn.ModuleList()
        self.bypass_networks = nn.ModuleList()

        assert (non_linearity is None or self.supports_nonlinearity(non_linearity))
        self.non_linearity = non_linearity.lower() if non_linearity else None

        transform_in_size = input_size
        transform_out_size = hidden_size
        for layer in range(layers):
            if layer == layers - 1:
                transform_out_size = output_size
            self.transforms.append(nn.Linear(transform_in_size, transform_out_size))
            self.bypass_networks.append(_bypass_module(bypass_network))
            transform_in_size = transform_out_size

    def _forward(self, input):
        output = input

        for transform, bypass in zip(self.transforms, self.bypass_networks):
            new_output = transform(output)
            if self.non_linearity:
                new_output = getattr(F, self.non_linearity)(new_output)
            if bypass:
                new_output = bypass(new_output, output)
            output = new_output

        return output


def _merge_module(op):
    module = get_module(Merge, op)
    if module:
        module.expect_signature('[AxB, AxB] -> [AxC]')
    return module


def _bypass_module(op):
    module = get_module(Bypass, op)
    if module:
        module.expect_signature('[AxB, AxC] -> [AxB]')
    return module


def _transform_module(op, hidden_size, output_size=None):
    output_size = output_size or hidden_size
    module = get_module(
        Transform, op, hidden_size=hidden_size, output_size=output_size)
    if module:
        module.expect_signature('[AxB] -> [AxC]')
        module.expect_signature('[AxBxC] -> [AxBxD]')
    return module


def _alignment_module(op, hidden_size):
    module = get_module(
        AlignmentNetwork, op, hidden_size=hidden_size, required=True)
    module.expect_signature('[AxBxC, AxDxC] -> [AxBxD]')
    return module
