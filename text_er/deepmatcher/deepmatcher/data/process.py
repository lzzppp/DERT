import io
import os

from torchtext.utils import unicode_csv_reader

from .dataset import MatchingDataset
from .field import MatchingField


def _check_header(header, id_attr, left_prefix, right_prefix, label_attr, ignore_columns):
    if label_attr:
        assert label_attr in header

    for attr in header:
        if attr not in (id_attr, label_attr) and attr not in ignore_columns:
            if not attr.startswith(left_prefix) and not attr.startswith(right_prefix):
                raise ValueError('Attribute ' + attr + ' is not a left or a right table '
                                 'column, not a label or id and is not ignored. Not sure '
                                 'what it is...')

    num_left = sum(attr.startswith(left_prefix) for attr in header)
    num_right = sum(attr.startswith(right_prefix) for attr in header)
    assert num_left == num_right


def _make_fields(header, id_attr, label_attr, ignore_columns, lower, tokenize,
                 include_lengths):
    text_field = MatchingField(
        lower=lower,
        tokenize=tokenize,
        init_token='<BOS>',
        eos_token='<EOS>',
        batch_first=True,
        include_lengths=include_lengths)
    numeric_field = MatchingField(
        sequential=False, preprocessing=lambda x: int(x), use_vocab=False)
    id_field = MatchingField(sequential=False, use_vocab=False, id=True)

    fields = []
    for attr in header:
        if attr == id_attr:
            fields.append((attr, id_field))
        elif attr == label_attr:
            fields.append((attr, numeric_field))
        elif attr in ignore_columns:
            fields.append((attr, None))
        else:
            fields.append((attr, text_field))
    return fields


def _maybe_download_nltk_data():
    import nltk
    nltk.download('perluniprops', quiet=True)
    nltk.download('nonbreaking_prefixes', quiet=True)
    nltk.download('punkt', quiet=True)


def process(path,
            train=None,
            validation=None,
            test=None,
            cache='cache.pth',
            check_cached_data=True,
            auto_rebuild_cache=True,
            tokenize='nltk',
            lowercase=True,
            embeddings='fasttext.en.bin',
            embeddings_cache_path='~/.vector_cache',
            ignore_columns=(),
            include_lengths=True,
            id_attr='id',
            label_attr='label',
            left_prefix='left_',
            right_prefix='right_',
            use_magellan_convention=True,
            pca=True):

    if use_magellan_convention:
        id_attr = '_id'
        left_prefix = 'ltable_'
        right_prefix = 'rtable_'

    a_dataset = train or validation or test
    with io.open(os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8") as f:
        header = next(unicode_csv_reader(f))

    # _maybe_download_nltk_data()
    _check_header(header, id_attr, left_prefix, right_prefix, label_attr, ignore_columns)
    fields = _make_fields(header, id_attr, label_attr, ignore_columns, lowercase,
                          tokenize, include_lengths)

    column_naming = {
        'id': id_attr,
        'left': left_prefix,
        'right': right_prefix,
        'label': label_attr
    }

    datasets = MatchingDataset.splits(
        path,
        train,
        validation,
        test,
        fields,
        embeddings,
        embeddings_cache_path,
        column_naming,
        cache,
        check_cached_data,
        auto_rebuild_cache,
        train_pca=pca)

    datasets[0].ignore_columns = ignore_columns
    datasets[0].tokenize = tokenize
    datasets[0].lowercase = lowercase
    datasets[0].include_lengths = include_lengths

    return datasets
