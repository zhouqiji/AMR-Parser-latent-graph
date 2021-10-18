# -*- coding: utf-8 -*-

from typing import Dict, MutableMapping
from zamr.data_utils.fields.field import DataArray, Field
from zamr.data_utils.vocabulary import Vocabulary


class Instance:
    """
    An ``Instance`` is a collection of :class:`~zamr.data_utils.fields.field.Field` objects,
    specifying the inputs and outputs to some model.  We don't make a distinction between inputs and outputs here,
    though - all operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.
    The ``Fields`` in an ``Instance`` can start out either indexed or un-indexed.  During the data
    processing pipeline, all fields will be indexed, after which multiple instances can be combined
    into a ``Batch`` and then converted into padded arrays.
    Parameters
    ----------
    """

    def __init__(self, fields: MutableMapping[str, Field]) -> None:
        self.fields = fields
        self.indexed = False

    def add_filed(self, filed_name: str, field: Field, vocab: Vocabulary = None) -> None:
        """
        Add the field to the existing fields mapping.
        If we have already indexed the Instance, then we also index `field`, so
        it is necessary to supply the vocab.
        """
        self.fields[filed_name] = field
        if self.indexed:
            field.index(vocab)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        Increments counts in the given ``counter`` for all of the vocabulary items in all of the
        ``Fields`` in this ``Instance``.
        """
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary) -> None:
        """
        Indexes all fields in this ``Instance`` using the provided ``Vocabulary```
        This `mutate` the current object, it does not return a new ``Instance``,
        A ``DataIterator`` will call this on each pass through a dataset; we use the ``indexed``
        flag to make sure that indexing only happens ones.
        """
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a dictionary of padding lengths, keyed by field name .
        Each ``Field`` returns a mapping from padding keys to actual lengths, and we
        just key that dictionary field name.
        """
        lengths = {}
        for field_name, filed in self.fields.items():
            lengths[field_name] = filed.get_padding_lengths()
        return lengths

    def as_tensor_dict(self,
                       padding_lengths: Dict[str, Dict[str, int]] = None) -> Dict[str, DataArray]:
        """
        Pads each ``Field`` in this instance to the lengths given in ``padding_lengths`` (which is
        keyed by field name, then by padding key, the same as the return value in
        :func:`get_padding_lengths`), returning a list of torch tensors for each field.
        If ``padding_lengths`` is omitted, we will call ``self.get_padding_lengths()`` to get the
        sizes of the tensors to create.
        """
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors

    def __str__(self) -> str:
        base_string = f"Instance with fields:\n"
        return " ".join([base_string] + [f"\t {name}: {field} \n"
                                         for name, field in self.fields.items()])
