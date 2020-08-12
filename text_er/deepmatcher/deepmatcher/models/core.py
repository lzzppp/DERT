import copy
from collections import Mapping

import six

import deepmatcher as dm
import torch
import torch.nn as nn

from .modules import RNN, GatingMechanism, SelfAttention, PairAttention, GlobalAttention, Fusion
from ..data import MatchingDataset, MatchingIterator
from ..runner import Runner
from ..utils import Bunch


is_cuda = torch.cuda.is_available()


class MCANModel(nn.Module):
    def __init__(self, attr_comparator='concat-mul-diff', classifier='2-layer-highway'):
        super(MCANModel, self).__init__()
        self.attr_merge = 'sum'    # TODO: sum or concat
        self.attr_condense_factor = 'auto'
        self.attr_comparator = attr_comparator
        self.classifier = classifier
        self.hidden_size = 300
        self._train_buffers = set()
        self._initialized = False

    def run_train(self, *args, **kwargs):
        return Runner.train(self, *args, **kwargs)

    def run_eval(self, *args, **kwargs):
        return Runner.eval(self, *args, **kwargs)

    def run_prediction(self, *args, **kwargs):
        return Runner.predict(self, *args, **kwargs)

    def initialize(self, train_dataset, init_batch=None):
        if self._initialized:
            return

        self.meta = Bunch(**train_dataset.__dict__)
        if hasattr(self.meta, 'fields'):
            del self.meta.fields
            del self.meta.examples

        self._register_train_buffer('state_meta', Bunch(**self.meta.__dict__))
        del self.state_meta.metadata  # we only need `self.meta.orig_metadata` for state.

        self.attr_summarizers = dm.modules.ModuleMap()
        self.attr_summarizer = TextSummarizer(hidden_size=self.hidden_size)

        self.meta.canonical_text_fields.sort()

        for name in self.meta.canonical_text_fields:
            self.attr_summarizers[name] = copy.deepcopy(self.attr_summarizer)
        if self.attr_condense_factor == 'auto':
            self.attr_condense_factor = min(len(self.meta.canonical_text_fields), 6)
            if self.attr_condense_factor == 1:
                self.attr_condense_factor = None
        if not self.attr_condense_factor:
            self.attr_condensors = None
        else:
            self.attr_condensors = dm.modules.ModuleMap()
            for name in self.meta.canonical_text_fields:
                self.attr_condensors[name] = dm.modules.Transform(
                    '1-layer-highway',
                    non_linearity=None,
                    output_size=self.hidden_size // self.attr_condense_factor)

        # self.text_fusions = dm.modules.ModuleMap()
        # self.text_fusion = Fusion(hidden_size=self.hidden_size, is_meta=False)
        self.attr_comparators = dm.modules.ModuleMap()
        self.attr_comparator = dm.modules._merge_module(self.attr_comparator)
        for name in self.meta.canonical_text_fields:
            self.attr_comparators[name] = copy.deepcopy(self.attr_comparator)

        # for name in self.meta.canonical_text_fields:
        #     self.text_fusions[name] = copy.deepcopy(self.text_fusion)

        # self.attr_merge = dm.modules._merge_module(self.attr_merge)
        self.attr_merge = GlobalAttention(hidden_size=self.hidden_size // self.attr_condense_factor * 4, style="dot", use_meta=False)
        self.classifier = Classifier(self.classifier, hidden_size=self.hidden_size)

        self._reset_embeddings(train_dataset.vocabs)

        # Instantiate all components using a small batch from training set.
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
        self.embed = dm.modules.ModuleMap()
        field_vectors = {}
        for name in self.meta.all_text_fields:
            vectors = vocabs[name].vectors
            if vectors not in field_vectors:
                vectors_size = vectors.shape
                embed = nn.Embedding(vectors_size[0], vectors_size[1])
                embed.weight.data.copy_(vectors)
                embed.weight.requires_grad = False
                field_vectors[vectors] = dm.modules.NoMeta(embed)
            self.embed[name] = field_vectors[vectors]

    def forward(self, input):
        embeddings = {}
        for name in self.meta.all_text_fields:
            attr_input = getattr(input, name)
            embeddings[name] = self.embed[name](attr_input)

        attr_comparisons = []

        # new_add
        meta_data = []

        for name in self.meta.canonical_text_fields:
            left, right = self.meta.text_fields[name]

            # new_add
            len1 = embeddings[left].lengths
            len2 = embeddings[right].lengths
            temp = []
            for i, j in zip(len1, len2):
                if i == 2 or j == 2:
                    temp.append(0)
                else:
                    temp.append(1)
            meta_data.append(temp)

            left_summary, right_summary = self.attr_summarizers[name](embeddings[left],
                                                                      embeddings[right])
            left_summary, right_summary = left_summary.data, right_summary.data

            if self.attr_condensors:
                left_summary = self.attr_condensors[name](left_summary)
                right_summary = self.attr_condensors[name](right_summary)
            attr_comparisons.append(self.attr_comparators[name](left_summary,
                                                                right_summary).unsqueeze(1))

        # new_add
        meta_data = torch.ByteTensor(meta_data)
        meta_data = meta_data.transpose(0, 1)

        attr_comparisons = torch.cat(attr_comparisons, dim=1)
        entity_comparison = self.attr_merge(attr_comparisons, meta_data)
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


class TextSummarizer(dm.modules.LazyModule):
    def _init(self,
              hidden_size=None):
        self.gru = RNN('gru', hidden_size=hidden_size)
        self.self_attention = SelfAttention(hidden_size=hidden_size, alignment_network="dot")
        self.pair_attention = PairAttention(alignment_network='bilinear')
        self.word_fusion = Fusion(hidden_size=hidden_size, is_meta=True)
        self.gate_mechanism = GatingMechanism(hidden_size=hidden_size, dropout=0.1)
        self.global_attention = GlobalAttention(hidden_size=hidden_size, style="dot", dropout=0.1)

    def _forward(self, left_input, right_input):
        left_contextualized = self.gru(left_input)
        right_contextualized = self.gru(right_input)

        left_contextualized = self.self_attention(left_contextualized)
        right_contextualized = self.self_attention(right_contextualized)

        left_fused = self.pair_attention(
            left_contextualized, right_contextualized, left_input, right_input)
        right_fused = self.pair_attention(
            right_contextualized, left_contextualized, right_input, left_input)

        left_fused = self.word_fusion(left_contextualized, left_fused)
        right_fused = self.word_fusion(right_contextualized, right_fused)
        # left_gated, right_gated = left_fused, right_fused
        left_gated = self.gate_mechanism(left_input, left_fused)
        right_gated = self.gate_mechanism(right_input, right_fused)

        left_final = self.global_attention(left_gated)
        right_final = self.global_attention(right_gated)

        return left_final, right_final


class Classifier(nn.Sequential):
    def __init__(self, transform_network, hidden_size=None):
        super(Classifier, self).__init__()
        if transform_network:
            self.add_module('transform',
                            dm.modules._transform_module(transform_network, hidden_size))
        self.add_module('softmax_transform',
                        dm.modules.Transform(
                            '1-layer', non_linearity=None, output_size=2))
        self.add_module('softmax', nn.LogSoftmax(dim=1))
