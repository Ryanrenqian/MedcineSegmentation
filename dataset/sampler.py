import torch
from torch._six import int_classes as _int_classes
import random
from torch.utils.data import  Sampler

class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source,  num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples
        self._sampled=set()
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        tumor_size,normal_size=self.data_source.shape
        sample = self.num_samples//2  # 各抽取一半样本
        tumor_list=torch.randint(high=tumor_size, size=(sample,), dtype=torch.int64).tolist()
        normal_list=torch.randint(low=tumor_size,high=n, size=(sample,), dtype=torch.int64).tolist()
        samples=tumor_list+normal_list
        self._sampled= set.union(self._sampled,set(samples))
        random.shuffle(samples) # 重新排列
        return iter(samples)

    def __len__(self):
        return self.num_samples
