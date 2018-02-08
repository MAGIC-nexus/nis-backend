# -*- coding: utf-8 -*-

import collections
import ast
import json
import pandas as pd
from multidict import MultiDict, CIMultiDict
from functools import partial

import backend
from backend import case_sensitive, \
                    SDMXConcept

# #####################################################################################################################
# >>>> CASE SeNsItIvE or INSENSITIVE names (flows, funds, processors, ...) <<<<
# #####################################################################################################################


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


def create_dictionary(case_sens=case_sensitive, multi_dict=False, data={}):
    """
    Factory to create dictionaries

    :param case_sens: True to create a case sensitive dictionary, False to create a case insensitive one
    :param multi_dict: True to create a "MultiDict", capable of storing several values
    :return:
    """

    if not multi_dict:
        if case_sens:
            return {}.update(data)  # Normal, "native" dictionary
        else:
            return CaseInsensitiveDict(data)
    else:
        if case_sens:
            return MultiDict(data)
        else:
            return CIMultiDict(data)


def strcmp(s1, s2):
    """
    Compare two strings for equality or not, considering a flag for case sensitiveness or not

    It also removes leading and trailing whitespace from both strings, so it is not sensitive to this possible
    difference, which can be a source of problems

    :param s1:
    :param s2:
    :return:
    """
    if case_sensitive:
        return s1.strip() == s2.strip()
    else:
        return s1.strip().lower() == s2.strip().lower()

# #####################################################################################################################
# >>>> DYNAMIC IMPORT <<<<
# #####################################################################################################################


def import_names(package, names):
    """
    Dynamic import of a list of names from a module

    :param package: String with the name of the package
    :param names: Name or list of names, string or list of strings with the objects inside the package
    :return: The list (or not) of objects under those names
    """
    if not isinstance(names, list):
        names = [names]
        not_list = True
    else:
        not_list = False

    try:
        tmp = __import__(package, fromlist=names)
    except:
        tmp = None

    if tmp:
        tmp2 = [getattr(tmp, name) for name in names]
        if not_list:
            tmp2 = tmp2[0]
        return tmp2
    else:
        return None

# #####################################################################################################################
# >>>> JSON FUNCTIONS <<<<
# #####################################################################################################################


def _json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    from datetime import datetime
    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial
    raise TypeError("Type not serializable")


JSON_INDENT = 4
ENSURE_ASCII = False


def generate_json(o):
    return json.dumps(o,
                      default=_json_serial,
                      sort_keys=True,
                      indent=JSON_INDENT,
                      ensure_ascii=ENSURE_ASCII,
                      separators=(',', ': ')
                      ) if o else None

# #####################################################################################################################
# >>>> KEY -> VALUE STORE, WITH PARTIAL KEY INDEXATION <<<<
# #####################################################################################################################


class PartialRetrievalDictionary:
    """ The key is a dictionary, the value an object. Allows partial search.

    >> IF "case_sensitive==False" -> VALUES are CASE INSENSITIVE <<<<<<<<<<<<<<<<<

        It is prepared to store different key compositions, with a pd.DataFrame per key
        When retrieving (get), it can match several of these, so it can return results from different pd.DataFrames

        pd.DataFrame with MultiIndex:

import pandas as pd
df = pd.DataFrame(columns=["a", "b", "c"])  # Empty DataFrame
df.set_index(["a", "b"], inplace=True)  # First columns are the MultiIndex
df.loc[("h", "i"), "c"] = "hi"  # Insert values in two cells
df.loc[("h", "j"), "c"] = "hj"
df.loc[(slice(None), "j"), "c"]  # Retrieve using Partial Key
df.loc[("h", slice(None)), "c"]
    """
    def __init__(self):
        self._dfs = {}  # Dict from sorted key-dictionary-keys (the keys in the key dictionary) to DataFrame
        self._df_sorted = {}  # A Dict telling if each pd.DataFrame is sorted
        self._key_lst = []  # List of sets of keys, used in "get" when matching partial keys

    def put(self, key, value):
        """
        Store a value using a dictionary key which can have None values

        :param key:
        :param value:
        :return:
        """
        # Sort keys
        keys = [k for k in key]
        s_tuple = tuple(sorted(keys))
        if case_sensitive:
            df_key = tuple([key[k] if key[k] is not None else slice(None) for k in s_tuple])
        else:
            df_key = tuple([(key[k] if k.startswith("__") else key[k].lower()) if key[k] is not None else slice(None) for k in s_tuple])

        if s_tuple not in self._dfs:
            # Append to list of sets of keys
            self._key_lst.append(set(keys))
            # Add New DataFrame to the dictionary of pd.DataFrame
            cols = [s for s in s_tuple]
            cols.append("value")
            df = pd.DataFrame(columns=cols)
            self._dfs[s_tuple] = df
            df.set_index([s for s in s_tuple], inplace=True)
        else:
            # Use existing pd.DataFrame
            df = self._dfs[s_tuple]

        # Do the insertion into the pd.DataFrame
        df.loc[df_key, "value"] = value

        # Flag the pd.DataFrame as unsorted
        self._df_sorted[s_tuple] = False

    def get(self, key, key_and_value: bool=False):
        """
        Return elements of different kinds, matching the totally or partially specified key.

        :param key: A dictionary with all or part of the key of elements to be retrieved
        :param key_and_value: If True, return a list of tuple (key, value). If not, return a list of values
        :return: A list of elements which can be (key, value) or "value", depending on the parameter "key_and_value"
        """
        keys = [k for k in key]
        s_tuple = tuple(sorted(keys))
        if s_tuple not in self._dfs:
            # Try partial match
            s = set(keys)
            df = []
            for s2 in self._key_lst:
                if s.issubset(s2):
                    s_tuple = tuple(sorted(s2))
                    df.append((self._dfs[s_tuple], s_tuple))
        else:
            df = [(self._dfs[s_tuple], s_tuple)]

        if df:
            res = []
            for df_, s_tuple in df:
                try:
                    if case_sensitive:
                        df_key = tuple([key[k] if k in key else slice(None) for k in s_tuple])
                    else:
                        df_key = tuple([(key[k] if k.startswith("__") else key[k].lower()) if k in key else slice(None) for k in s_tuple])

                    if s_tuple not in self._df_sorted or not self._df_sorted[s_tuple]:
                        df_.sort_index(ascending=True, inplace=True)
                        self._df_sorted[s_tuple] = True

                    tmp = df_.loc[df_key, "value"]
                    if isinstance(tmp, pd.Series):
                        for i, v in enumerate(tmp):
                            if key_and_value:
                                k = {rk: rv for rk, rv in zip(s_tuple, tmp.index[i])}
                                res.append((k, v))
                            else:
                                res.append(v)
                    else:  # Single result, standardize to always (k, v)
                        if key_and_value:
                            k = {rk: rv for rk, rv in zip(s_tuple, df_key)}
                            res.append((k, tmp))
                        else:
                            res.append(tmp)
                except (IndexError, KeyError):
                    pass
            return res
        else:
            return []

    def delete(self, key):
        """
        Remove elements matching each of the keys, total or partial, passed to the method

        :param key:
        :return:
        """

        def delete_single(key):
            """
            Remove elements matching the total or partial key

            :param key:
            :return:
            """
            """
            Return elements of different kinds, matching the totally or partially specified key.
    
            :param key: A dictionary with all or part of the key of elements to be retrieved
            :param key_and_value: If True, return a list of tuple (key, value). If not, return a list of values
            :return: A list of elements which can be (key, value) or "value", depending on the parameter "key_and_value"
            """
            keys = [k for k in key]
            s_tuple = tuple(sorted(keys))
            if s_tuple not in self._dfs:
                # Try partial match
                s = set(keys)
                df = []
                for s2 in self._key_lst:
                    if s.issubset(s2):
                        s_tuple = tuple(sorted(s2))
                        df.append((self._dfs[s_tuple], s_tuple))
            else:
                df = [(self._dfs[s_tuple], s_tuple)]

            if df:
                res = 0
                for df_, s_tuple in df:
                    if case_sensitive:
                        df_key = tuple([key[k] if k in key else slice(None) for k in s_tuple])
                    else:
                        df_key = tuple([(key[k] if k.startswith("__") else key[k].lower()) if k in key else slice(None) for k in s_tuple])
                    try:
                        tmp = df_.loc[df_key, "value"]
                        if isinstance(tmp, pd.Series):
                            df_.drop(tmp.index, inplace=True)
                            res += len(tmp)
                        else:  # Single result, standardize to always (k, v)
                            df_.drop(df_key, inplace=True)
                            res += 1
                    except (IndexError, KeyError):
                        pass
                return res
            else:
                return 0

        if isinstance(key, list):
            res_ = 0
            for k in key:
                res_ += delete_single(k)
            return res_
        else:
            return delete_single(key)

    def to_pickable(self):
        # Convert to a jsonpickable structure
        out = {}
        for k in self._dfs:
            df = self._dfs[k]
            out[k] = df.to_dict()
        return out

    def from_pickable(self, inp):
        # Convert from the jsonpickable structure (obtained by "to_pickable") to the internal structure
        self._dfs = {}
        self._key_lst = []
        for t in inp:
            t_ = ast.literal_eval(t)
            self._key_lst.append(set(t_))
            inp[t]["value"] = {ast.literal_eval(k): v for k, v in inp[t]["value"].items()}
            df = pd.DataFrame(inp[t])
            df.index.names = t_
            self._dfs[t_] = df

        return self  # Allows the following: prd = PartialRetrievalDictionary().from_pickable(inp)

# #####################################################################################################################
# >>>> EXTERNAL DATASETS <<<<
# #####################################################################################################################


def get_statistical_dataset_structure(source, dset_name):
    # Obtain DATASET: Datasource -> Database -> DATASET -> Dimension(s) -> CodeList (no need for "Concept")
    dset = backend.data_source_manager.get_dataset_structure(source, dset_name)

    # TODO Generate "dims", "attrs" and "meas" from "dset"
    dims = create_dictionary()  # Each dimension has a name, a description and a code list
    attrs = create_dictionary()
    meas = create_dictionary()
    for dim in dset.dimensions:
        if dim.is_measure:
            meas[dim.code] = None
        else:
            # Convert the code list to a dictionary
            if dim.code_list:
                cl = dim.code_list.to_dict()
            else:
                cl = None
            dims[dim.code] = SDMXConcept("dimension", dim.code, dim.is_time, "", cl)

    time_dim = False

    lst_dim = []

    for l in dims:
        lst_dim_codes = []
        if dims[l].istime:
            time_dim = True
        else:
            lst_dim.append((dims[l].name, lst_dim_codes))

        if dims[l].code_list:
            for c in dims[l].code_list:
                lst_dim_codes.append((c, dims[l].code_list[c]))

    if time_dim:
        lst_dim.append(("startPeriod", None))
        lst_dim.append(("endPeriod", None))

    return lst_dim, (dims, attrs, meas)


def obtain_dataset_source(dset_name):
    # from backend.model.rdb_persistence.persistent import DBSession
    # if not backend.data_source_manager:
    #     backend.data_source_manager = register_external_datasources(app.config, DBSession)

    lst = backend.data_source_manager.get_datasets()  # ALL Datasets, (source, dataset)
    ds = create_dictionary(data={d[0]: t[0] for t in lst for d in t[1]})  # Dataset to Source (to obtain the source given the dataset name)

    if dset_name in ds:
        source = ds[dset_name]
    else:
        source = None
    # else:  # Last resource, try using how the dataset starts
    #     if dset_name.startswith("ssp_"):
    #         # Obtain SSP
    #         source = "SSP"
    #     else:
    #         source = "Eurostat"
    return source


def check_dataset_exists(dset_name):
    if len(dset_name.split(".")) == 2:
        source, d_set = dset_name.split(".")
    else:
        d_set = dset_name
    res = obtain_dataset_source(d_set)
    return res is not None


def obtain_dataset_metadata(dset_name, source=None):
    d_set = dset_name
    if not source:
        if len(dset_name.split(".")) == 2:
            source, d_set = dset_name.split(".")
        else:
            source = obtain_dataset_source(d_set)

    _, metadata = get_statistical_dataset_structure(source, d_set)

    return metadata

# #####################################################################################################################
# >>>> DECORATORS <<<<
# #####################################################################################################################


class Memoize:
    """
    Cache of function calls (non-persistent, non-refreshable)
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res