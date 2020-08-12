from __future__ import division

from torchtext import data

from ..batch import MatchingBatch


class MatchingIterator(data.BucketIterator):
    def __init__(self,
                 dataset,
                 train_info,
                 train,
                 batch_size,
                 sort_in_buckets=None,
                 **kwargs):
        if sort_in_buckets is None:
            sort_in_buckets = train
        self.sort_in_buckets = sort_in_buckets
        self.train_info = train_info
        super(MatchingIterator, self).__init__(
            dataset, batch_size, train=train, repeat=False, sort=False, **kwargs)

    @classmethod
    def splits(cls, datasets, batch_sizes=None, **kwargs):
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            ret.append(
                cls(datasets[i],
                    train_info=datasets[0],
                    train=i==0,
                    batch_size=batch_sizes[i],
                    **kwargs))
        return tuple(ret)

    def __iter__(self):
        for batch in super(MatchingIterator, self).__iter__():
            yield MatchingBatch(batch, self.train_info)

    def create_batches(self):
        if self.sort_in_buckets:
            return data.BucketIterator.create_batches(self)
        else:
            return data.Iterator.create_batches(self)
