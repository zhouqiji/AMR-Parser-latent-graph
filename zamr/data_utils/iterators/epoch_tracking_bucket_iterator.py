# -*- coding: utf-8 -*-

import logging
from typing import List, Tuple
import warnings

from zamr.data_utils.iterators.data_iterator import DataIterator
from zamr.data_utils.iterators.bucket_iterator import BucketIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("epoch_tracking_bucket")
class EpochTrackingBucketIterator(BucketIterator):

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False) -> None:
        super().__init__(sorting_keys=sorting_keys,
                         padding_noise=padding_noise,
                         biggest_batch_first=biggest_batch_first,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         track_epoch=True,
                         cache_instances=cache_instances)
        warnings.warn("EpochTrackingBucketIterator is deprecated, "
                      "please just use BucketIterator with track_epoch=True",
                      DeprecationWarning)
