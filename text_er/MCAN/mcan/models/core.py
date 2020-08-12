import copy
from collections import Mapping

import six

import mcan
import torch
import torch.nn as nn

from .modules import RNN, GatingMechanism, SelfAttention, PairAttention, GlobalAttention, Fusion
from ..data import MatchingDataset, MatchingIterator
from ..runner import Runner
from ..utils import Bunch


class MCANModel(nn.Module):
    def __init__(self):
        super(MCANModel, self).__init__()
        self.attr_merge = 'concat'
        self.hidden_size = 300
        self._train_buffers = set()
        self._initialized = False

    def run_train(self, *args, **kwargs):
        return Runner.train(self, *args, **kwargs)

    def run_eval(self, *args, **kwargs):
        return Runner.eval(self, *args, **kwargs)

    def run_prediction(self, *args, **kwargs):
        return Runner.predict(self, *args, **kwargs)

    def initialize(self, train_dataset, init_batch=None, ablation_part=None):
        if self._initialized:
            return

        self.meta = Bunch(**train_dataset.__dict__)
        if hasattr(self.meta, 'fields'):
            del self.meta.fields
            del self.meta.examples

        self._register_train_buffer('state_meta', Bunch(**self.meta.__dict__))
        del self.state_meta.metadata

        self.ablation_part = ablation_part  # For ablation experiment 

        self.gru = RNN('gru', hidden_size=self.hidden_size)
        if 'SA' not in self.ablation_part:
            self.self_attention = SelfAttention(hidden_size=self.hidden_size, alignment_network="dot")

        if 'PA' not in self.ablation_part:
            self.pair_attention = PairAttention(alignment_network='bilinear')
            self.word_fusion = Fusion(hidden_size=self.hidden_size, is_meta=True)
        
        if 'GM' not in self.ablation_part:
            self.gate_mechanism = GatingMechanism(hidden_size=self.hidden_size, dropout=0.1)
        if 'GA' not in self.ablation_part:
            self.global_attention = GlobalAttention(hidden_size=self.hidden_size, style="dot", dropout=0.1)
        self.text_fusion = Fusion(hidden_size=self.hidden_size, is_meta=False)
        self.classifier = BinaryClassifier()

        self._reset_embeddings(train_dataset.vocabs)

        if not init_batch:
            run_iter = MatchingIterator(
                train_dataset,
                train_dataset,
                train=False,
                batch_size=4,
                device=-1,
                sort_in_buckets=False)
            init_batch = next(run_iter.__iter__())
        self.forward(init_batch)
        self.state_meta.init_batch = init_batch
        self._initialized = True

    def _reset_embeddings(self, vocabs):
        self.embed = mcan.modules.ModuleMap()
        field_vectors = {}
        for name in self.meta.all_text_fields:
            vectors = vocabs[name].vectors
            if vectors not in field_vectors:
                vectors_size = vectors.shape
                embed = nn.Embedding(vectors_size[0], self.hidden_size)
                embed.weight.data.copy_(vectors)
                embed.weight.requires_grad = False # change
                field_vectors[vectors] = mcan.modules.NoMeta(embed)
            self.embed[name] = field_vectors[vectors]

    def forward(self, input):
        left_input = self.embed["ltable_value"](getattr(input, "ltable_value"))
        right_input = self.embed["rtable_value"](getattr(input, "rtable_value"))

        left_contextualized = self.gru(left_input)
        right_contextualized = self.gru(right_input)

        # left_contextualized, right_contextualized = left_input, right_input

        if 'SA' not in self.ablation_part:
            left_contextualized = self.self_attention(left_contextualized)
            right_contextualized = self.self_attention(right_contextualized)

        if 'PA' not in self.ablation_part:
            left_fused = self.pair_attention(
                left_contextualized, right_contextualized, left_input, right_input)
            right_fused = self.pair_attention(
                right_contextualized, left_contextualized, right_input, left_input)

            left_fused = self.word_fusion(left_contextualized, left_fused)
            right_fused = self.word_fusion(right_contextualized, right_fused)
        else:
            left_fused = left_contextualized
            right_fused = right_contextualized

        if 'GM' not in self.ablation_part:
            left_gated = self.gate_mechanism(left_input, left_fused)
            right_gated = self.gate_mechanism(right_input, right_fused)
        else:
            left_gated = left_fused
            right_gated = right_fused

        if 'GA' not in self.ablation_part:
            left_summary = self.global_attention(left_gated)
            right_summary = self.global_attention(right_gated)
        else:
            left_summary = left_gated
            right_summary = right_gated

        left_summary, right_summary = left_summary.data, right_summary.data

        entity_comparison = self.text_fusion(left_summary, right_summary)
        return self.classifier(entity_comparison)

    def _register_train_buffer(self, name, value):
        self._train_buffers.add(name)
        setattr(self, name, value)

    def save_state(self, path, include_meta=True):
        state = {'model': self.state_dict()}
        for k in self._train_buffers:
            if include_meta or k != 'state_meta':
                state[k] = getattr(self, k)
        torch.save(state, path)

    def load_state(self, path):
        state = torch.load(path)
        for k, v in six.iteritems(state):
            if k != 'model':
                self._train_buffers.add(k)
                setattr(self, k, v)

        if hasattr(self, 'state_meta'):
            train_info = copy.copy(self.state_meta)

            train_info.metadata = train_info.orig_metadata
            MatchingDataset.finalize_metadata(train_info)

            self.initialize(train_info, self.state_meta.init_batch)

        self.load_state_dict(state['model'])


class BinaryClassifier(nn.Sequential):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.add_module('softmax_transform',
                        mcan.modules.Transform(
                            '1-layer', non_linearity=None, output_size=2))
        self.add_module('softmax', nn.LogSoftmax(dim=1))
