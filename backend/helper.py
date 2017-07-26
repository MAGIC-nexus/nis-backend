# -*- coding: utf-8 -*-

import collections
from multidict import MultiDict, CIMultiDict

# #################################################################
# CASE SeNsItIvE or INSENSITIVE names (flows, funds, processors, ...)
#


class CaseInsensitiveDict(collections.MutableMapping):
    """
    A dictionary with case insensitive Keys.
    Prepared also to support TUPLES as keys, required because compound keys are required
    """
    def __init__(self, data=None, **kwargs):
        from collections import OrderedDict
        self._store = OrderedDict()
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        # Use the lowercased key for lookups, but store the actual
        # key alongside the value.
        if not isinstance(key, tuple):
            self._store[key.lower()] = (key, value)
        else:
            self._store[tuple([k.lower() for k in key])] = (key, value)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self._store[key.lower()][1]
        else:
            return self._store[tuple([k.lower() for k in key])][1]

    def __delitem__(self, key):
        if not isinstance(key, tuple):
            del self._store[key.lower()]
        else:
            del self._store[tuple([k.lower() for k in key])]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in self._store.values())

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return (
            (lowerkey, keyval[1])
            for (lowerkey, keyval)
            in self._store.items()
        )

    def __contains__(self, key):  # "in" operator to check if the key is present in the dictionary
        if not isinstance(key, tuple):
            return key.lower() in self._store
        else:
            return tuple([k.lower() for k in key]) in self._store

    def __eq__(self, other):
        if isinstance(other, collections.Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    # Copy is required
    def copy(self):
        return CaseInsensitiveDict(self._store.values())

    def __repr__(self):
        return str(dict(self.items()))


def create_dictionary(case_sensitive=False, multi_dict=False):
    """
    Factory to create dictionaries

    :param case_sensitive: True to create a case sensitive dictionary, False to create a case insensitive one
    :param multi_dict: True to create a "MultiDict", capable of storing several values
    :return:
    """

    if not multi_dict:
        if case_sensitive:
            return {}
        else:
            return CaseInsensitiveDict()
    else:
        if case_sensitive:
            return MultiDict()
        else:
            return CIMultiDict()
