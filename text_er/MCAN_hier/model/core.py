import copy
from collections import Mapping

import six

import mcan_structured as mcan
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules import RNN, GatingMechanism, SelfAttention, PairAttention, GlobalAttention, Fusion, _transform_module
from ..data import MatchingDataset, MatchingIterator
from ..runner import Runner
from ..utils import Bunch


is_cuda = torch.cuda.is_available()


class MCANModel(nn.Module):
    def __init__(self, attr_comparator='concat-mul-diff', classifier='2-layer-highway'):
        super(MCANModel, self).__init__()
        self.attr_merge = 'sum'
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

        self.attr_summarizers = mcan.modules.ModuleMap()
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
            self.attr_condensors = mcan.modules.ModuleMap()
            for name in self.meta.canonical_text_fields:
                self.attr_condensors[name] = mcan.modules.Transform(
                    '1-layer-highway',
                    non_linearity=None,
                    output_size=self.hidden_size // self.attr_condense_factor)

        # self.text_fusions = mcan.modules.ModuleMap()
        # self.text_fusion = Fusion(hidden_size=self.hidden_size, is_meta=False)
        self.attr_comparators = mcan.modules.ModuleMap()
        self.attr_comparator = mcan.modules._merge_module(self.attr_comparator)
        for name in self.meta.canonical_text_fields:
            self.attr_comparators[name] = copy.deepcopy(self.attr_comparator)

        # for name in self.meta.canonical_text_fields:
        #     self.text_fusions[name] = copy.deepcopy(self.text_fusion)

        # self.attr_merge = mcan.modules._merge_module(self.attr_merge)
        self.transform = mcan.modules._transform_module(self.classifier, self.hidden_size)
        self.highway_layer = mcan.modules._transform_module(self.classifier, self.hidden_size)
        self.highway_layer_alignment = mcan.modules._transform_module(self.classifier, self.hidden_size)
        self.linear_token_compare = nn.Linear(self.hidden_size, 1)
        self.label = nn.Linear(self.hidden_size, 2)
        self.attr_merge = GlobalAttention(hidden_size=self.hidden_size // self.attr_condense_factor * 4, style="dot", use_meta=False)
        self.classifier = BinaryClassifier(self.classifier, hidden_size=self.hidden_size)

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
        self.embed = mcan.modules.ModuleMap()
        field_vectors = {}
        for name in self.meta.all_text_fields:
            vectors = vocabs[name].vectors
            if vectors not in field_vectors:
                vectors_size = vectors.shape
                embed = nn.Embedding(vectors_size[0], vectors_size[1])
                embed.weight.data.copy_(vectors)
                embed.weight.requires_grad = False
                field_vectors[vectors] = mcan.modules.NoMeta(embed)
            self.embed[name] = field_vectors[vectors]
    
    def element_wise_compare(self, tensor_1, tensor_2):
        compare_result = torch.abs(tensor_1 - tensor_2)
        return compare_result
    
    def token_level_attention(self, left, right):
        left_expand = left.clone()
        right_expand = right.clone()
        size_left = left.size()
        size_right = right.size()

        left_expand = left_expand.view(size_left[0], size_left[1], 1, size_left[2])
        left_expand = left_expand.repeat(1, 1, size_right[1], 1)

        right_expand = right_expand.view(size_right[0], 1, size_right[1], size_right[2])
        right_expand = right_expand.repeat(1, size_left[1], 1, 1)

        compare_result = self.element_wise_compare(left_expand, right_expand)
        compare_result = self.highway_layer_alignment(compare_result)
        sim_values = self.linear_token_compare(compare_result)
        sim_values = sim_values.view(sim_values.size()[0], sim_values.size()[1], -1)
        sim_values = F.softmax(sim_values, dim=2)
        compare_matrix = sim_values
        return compare_matrix
        #
        # return torch.bmm(
        #         self.transform(left),  # batch x hidden_size x len2
        #         self.transform(right).transpose(1, 2))  # batch x hidden_size x len2
    
    def process_pad_token(self, attention_matrix, token_mask_list_cat):
        '''
            Set attention values of pad tokens to 0
        '''
        # token_mask_list_cat = torch.cat(token_mask_list, 0)
        token_mask_list_cat = token_mask_list_cat.permute(1, 0).contiguous()
        token_mask_list_cat = token_mask_list_cat.view(token_mask_list_cat.size()[0], 1, token_mask_list_cat.size()[1])
        token_mask_list_cat = token_mask_list_cat.repeat(1, attention_matrix.size()[1], 1)
        if self._initialized:
            attention_matrix = attention_matrix * token_mask_list_cat.float().cuda()
        else:
            attention_matrix = attention_matrix * token_mask_list_cat.float()
        return attention_matrix
    
    def to_one_hot(self, token_compare_res):
        # max_index = torch.max(token_compare_res, 2)[1]
        max_value = torch.max(token_compare_res, 2)[0]
        max_value = max_value.view(max_value.size()[0], max_value.size()[1], 1)
        max_value_expand = max_value.expand(max_value.size()[0], max_value.size()[1], token_compare_res.size()[2])
        # mask = torch.ones(token_compare_res.size()[0], token_compare_res.size()[1], token_compare_res.size()[2])
        mask = (token_compare_res == max_value_expand).float()
        token_compare_res = torch.mul(token_compare_res, mask)
        mask2 = token_compare_res + 0.0000001
        token_compare_res = torch.div(token_compare_res, mask2)
        return token_compare_res
    
    def forward(self, input):
        embeddings = {}
        for name in self.meta.all_text_fields:
            attr_input = getattr(input, name)
            embeddings[name] = self.embed[name](attr_input)

        attr_comparisons = []

        # new_add
        meta_data, l_meta_data, r_meta_data = [], [], []
        
        left_summary_all, right_summary_all = [], []
        for name in self.meta.canonical_text_fields:
            left, right = self.meta.text_fields[name]

            # new_add
            len1 = embeddings[left].lengths
            len2 = embeddings[right].lengths
            temp, l_temp, r_temp = [], [], []
            for i, j in zip(len1, len2):
                if i == 2 and j == 2:
                    temp.append(0)
                    l_temp.append(0)
                    r_temp.append(0)
                elif i == 2 and j != 2:
                    temp.append(0)
                    l_temp.append(0)
                    r_temp.append(1)
                elif i != 2 and j == 2:
                    temp.append(0)
                    l_temp.append(1)
                    r_temp.append(0)
                elif i != 2 and j != 2:
                    temp.append(1)
                    l_temp.append(1)
                    r_temp.append(1)
            meta_data.append(temp)
            l_meta_data.append(l_temp)
            r_meta_data.append(r_temp)
            # temp = []
            # for i, j in zip(len1, len2):
            #     if i == 2 or j == 2:
            #         temp.append(0)
            #     else:
            #         temp.append(1)
            # meta_data.append(temp)

            left_summary, right_summary = self.attr_summarizers[name](embeddings[left],
                                                                      embeddings[right])
            left_summary, right_summary = left_summary.data, right_summary.data
            
            left_summary_all.append(left_summary)
            right_summary_all.append(right_summary)
            # if self.attr_condensors:
            #     left_summary = self.attr_condensors[name](left_summary)
            #     right_summary = self.attr_condensors[name](right_summary)
            # attr_comparisons.append(self.attr_comparators[name](left_summary,
            #                                                     right_summary).unsqueeze(1))
        meta_data = torch.ByteTensor(meta_data)
        meta_data = meta_data.transpose(0, 1)
        l_meta_data = torch.ByteTensor(l_meta_data)
        # l_meta_data = l_meta_data.transpose(0, 1)
        r_meta_data = torch.ByteTensor(r_meta_data)
        # r_meta_data = r_meta_data.transpose(0, 1)

        left_summary_all, right_summary_all = torch.stack(left_summary_all, dim=1), torch.stack(right_summary_all, dim=1)

        attention_matrix = self.token_level_attention(left_summary_all, right_summary_all)
        attention_matrix = self.process_pad_token(attention_matrix, r_meta_data)
        attention_matrix = self.to_one_hot(attention_matrix)
        left_aligned_representation = torch.bmm(attention_matrix, right_summary_all)
        left_token_compare_result = self.element_wise_compare(left_summary_all, left_aligned_representation).view(left_aligned_representation.shape[0], -1)

        attention_matrix = self.token_level_attention(right_summary_all, left_summary_all)
        attention_matrix = self.process_pad_token(attention_matrix, l_meta_data)
        attention_matrix = self.to_one_hot(attention_matrix)
        right_aligned_representation = torch.bmm(attention_matrix, left_summary_all)
        right_token_compare_result = self.element_wise_compare(right_summary_all, right_aligned_representation).view(right_aligned_representation.shape[0], -1)

        compare_concat = torch.cat((left_token_compare_result, right_token_compare_result), dim=-1)
        entity_comparison = self.highway_layer(compare_concat)
        entity_comparison = self.label(entity_comparison)
        output = F.log_softmax(entity_comparison, dim=-1)
        
        # # new_add
        # meta_data = torch.ByteTensor(meta_data)
        # meta_data = meta_data.transpose(0, 1)
        #
        # attr_comparisons = torch.cat(attr_comparisons, dim=1)
        # entity_comparison = self.attr_merge(attr_comparisons, meta_data)
        return output # self.classifier(entity_comparison)

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


class TextSummarizer(mcan.modules.LazyModule):
    def _init(self,
              hidden_size=None):
        self.gru = RNN('gru', hidden_size=hidden_size)
        self.self_attention = SelfAttention(hidden_size=hidden_size, alignment_network="dot")
        self.pair_attention = PairAttention(alignment_network='biliear')
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


class BinaryClassifier(nn.Sequential):
    def __init__(self, transform_network, hidden_size=None):
        super(BinaryClassifier, self).__init__()
        if transform_network:
            self.add_module('transform',
                            mcan.modules._transform_module(transform_network, hidden_size))
        self.add_module('softmax_transform',
                        mcan.modules.Transform(
                            '1-layer', non_linearity=None, output_size=2))
        self.add_module('softmax', nn.LogSoftmax(dim=1))
