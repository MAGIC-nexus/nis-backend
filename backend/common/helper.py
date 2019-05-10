# -*- coding: utf-8 -*-
import ast
import base64
import collections
import functools
import gzip
import io
import itertools
import json
import mimetypes
import tempfile
import urllib
import urllib.request
import uuid
from functools import partial
from io import BytesIO
from typing import IO, List, Tuple, Dict, Any, Optional, Iterable, Callable, TypeVar, Type, Union
from urllib.parse import urlparse
from uuid import UUID

import jsonpickle
import numpy as np
import pandas as pd
from flask import after_this_request, request
from multidict import MultiDict, CIMultiDict
from pandas import DataFrame

import backend
from backend import case_sensitive, SDMXConcept, get_global_configuration_variable
from backend.command_generators import Issue
from backend.models import ureg
import webdav.client as wc


# #####################################################################################################################
# >>>> JSON FUNCTIONS <<<<
# #####################################################################################################################

def _json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    from datetime import datetime
    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial
    elif isinstance(obj, CaseInsensitiveDict):
        return str(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, Issue):
        return obj.__repr__()
    raise TypeError(f"Type {type(obj)} not serializable")


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


class Encodable:
    """
    Abstract class with the method encode() that should be implemented by a subclass to be encoded into JSON
    using the json.dumps() method together with the option cls=CustomEncoder.
    """
    def encode(self) -> Dict[str, Any]:
        raise NotImplementedError("users must define encode() to use this base class")

    @staticmethod
    def parents_encode(obj: "Encodable", cls: type) -> Dict[str, Any]:
        """
        Get the state of all "cls" parent classes for the selected instance "obj"
        :param obj: The instance. Use "self".
        :param cls: The base class which parents we want to get. Use "__class__".
        :return: A dictionary with the state of the instance "obj" for all inherited classes.

        """
        d = {}
        for parent in cls.__bases__:
            if issubclass(parent, Encodable) and parent is not Encodable:
                d.update(parent.encode(obj))
        return d


class CustomEncoder(json.JSONEncoder):
    """
    Encoding class used by json.dumps(). It should be passed as the "cls" argument.
    Example: print(json.dumps({'A': 2, 'b': 4}), cls=CustomEncoder)
    """
    def default(self, obj):
        # Does the object implement its own encoder?
        if isinstance(obj, Encodable):
            return obj.encode()

        # Use the default encoder for handled types
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return json.JSONEncoder.default(self, obj)

        # For other unhandled types, like set, use universal json encoder "jsonpickle"
        return jsonpickle.encode(obj, unpicklable=False)


# #####################################################################################################################
# >>>> CASE SeNsItIvE or INSENSITIVE names (flows, funds, processors, ...) <<<<
# #####################################################################################################################
# from backend.models.musiasem_concepts import Taxon  IMPORT LOOP !!!!! AVOID !!!!


class CaseInsensitiveDict(collections.MutableMapping, Encodable):
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

    def encode(self):
        return self.get_data()

    def get_original_data(self):
        return {casedkey: mappedvalue for casedkey, mappedvalue in self._store.values()}

    def get_data(self):
        return {key: self._store[key][1] for key in self._store}

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


def create_dictionary(case_sens=case_sensitive, multi_dict=False, data=dict()):
    """
    Factory to create dictionaries

    :param case_sens: True to create a case sensitive dictionary, False to create a case insensitive one
    :param multi_dict: True to create a "MultiDict", capable of storing several values
    :param data: Dictionary with which the new dictionary is initialized
    :return:
    """

    if not multi_dict:
        if case_sens:
            tmp = {}
            tmp.update(data)
            return tmp  # Normal, "native" dictionary
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


def istr(s1: str) -> str:
    """ Return a lowercase version of a string if program works ignoring case sensitiveness """
    if case_sensitive:
        return s1
    else:
        return s1.lower()

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
# >>>> KEY -> VALUE STORE, WITH PARTIAL KEY INDEXATION <<<<
# #####################################################################################################################


class PartialRetrievalDictionary2:  # DEPRECATED!!!: Use PartialRetrievalDictionary
    """
    Implementation using pd.DataFrame, very slow!!!

    The key is a dictionary, the value an object. Allows partial search.

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
                except Exception as e:
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

        def delete_single(key_):
            """
            Remove elements matching the total or partial key

            :param key_:
            :return:
            """
            """
            Return elements of different kinds, matching the totally or partially specified key.
    
            :param key: A dictionary with all or part of the key of elements to be retrieved
            :param key_and_value: If True, return a list of tuple (key, value). If not, return a list of values
            :return: A list of elements which can be (key, value) or "value", depending on the parameter "key_and_value"
            """
            keys = [k_ for k_ in key_]
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
                        df_key = tuple([key_[k_] if k_ in key_ else slice(None) for k_ in s_tuple])
                    else:
                        df_key = tuple([(key_[k_] if k_.startswith("__") else key_[k_].lower()) if k_ in key_ else slice(None) for k_ in s_tuple])
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


class PartialRetrievalDictionary:
    def __init__(self):
        # A dictionary of key-name to dictionaries, where the dictionaries are each of the values of the key and the
        # value is a set of IDs having that value
        # dict(key-name, dict(key-value, set(obj-IDs with that key-value))
        self._keys = {}
        # Dictionary from ID to the tuple (composite-key-elements dict, object)
        self._objs = {}
        self._rev_objs = {}  # From object to ID
        # Counter
        self._id_counter = 0

    def get(self, key, key_and_value=False, full_key=False, just_oid=False):
        """
        Retrieve one or more objects matching "key"
        If "key_and_value" is True, return not only the value, also matching key (useful for multiple matching keys)
        If "full_key" is True, zero or one objects should be the result
        :param key:
        :param full_key:
        :return: A list of matching elements
        """
        if True:
            # Lower case values
            # Keys can be all lower case, because they will be internal Key components, not specified by users
            if case_sensitive:
                key2 = {k.lower(): v for k, v in key.items()}
            else:
                key2 = {k.lower(): v if k.startswith("__") else v.lower() for k, v in key.items()}
        else:
            key2 = key

        sets = [self._keys.get(k, {}).get(v, set()) for k, v in key2.items()]

        # Find shorter set and Remove it from the list
        min_len = 1e30
        min_len_set_idx = None
        for i, s in enumerate(sets):
            if len(s) < min_len:
                min_len = len(s)
                min_len_set_idx = i
        min_len_set = sets[min_len_set_idx]
        del sets[min_len_set_idx]
        # Compute intersections
        result = min_len_set.intersection(*sets)
        if just_oid:
            return result

        # Obtain list of results
        if full_key and len(result) > 1:
            raise Exception("Zero or one results were expected. "+str(len(result)+" obtained."))
        if not key_and_value:
            return [self._objs[oid][1] for oid in result]
        else:
            return [self._objs[oid] for oid in result]

    def put(self, key, value):
        """
        Insert implies the key does not exist
        Update implies the key exists
        Upsert does not care

        :param key:
        :param value:
        :return:
        """
        ptype = 'i'  # 'i', 'u', 'ups' (Insert, Update, Upsert)
        if True:
            # Lower case values
            # Keys can be all lower case, because they will be internal Key components, not specified by users
            if case_sensitive:
                key2 = {k.lower(): v for k, v in key.items()}
            else:
                key2 = {k.lower(): v if k.startswith("__") else v.lower() if isinstance(v, str) else v for k, v in key.items()}
        else:
            key2 = key
        # Arrays containing key: values "not-present" and "present"
        not_present = []  # List of tuples (dictionary of key-values, value to be stored)
        present = []  # List of sets storing IDs having same key-value
        for k, v in key2.items():
            d = self._keys.get(k, {})
            if len(d) == 0:
                self._keys[k] = d
            if v not in d:
                not_present.append((d, v))
            else:
                present.append(d.get(v))

        if len(not_present) > 0:
            is_new = True
        else:
            if len(present) > 1:
                is_new = len(present[0].intersection(*present[1:])) == 0
            elif len(present) == 1:
                is_new = len(present) == 0
            else:
                is_new = False

        # Insert, Update or Upsert
        if is_new:  # It seems to be an insert
            # Check
            if ptype == 'u':
                raise Exception("Key does not exist")
            # Insert
            if value in self._rev_objs:
                oid = self._rev_objs[value]
            else:
                self._id_counter += 1
                oid = self._id_counter
                self._objs[oid] = (key, value)
                self._rev_objs[value] = oid

            # Insert
            for d, v in not_present:
                s = set()
                d[v] = s
                s.add(oid)
            for s in present:
                s.add(oid)
        else:
            if ptype == 'i':
                raise Exception("Key '+"+str(key2)+"' already exists")
            # Update
            # Find the ID for the key
            res = self.get(key, just_oid=True)
            if len(res) != 1:
                raise Exception("Only one result expected")
            # Update value (key is the same, ID is the same)
            self._objs[res[0]] = value

    def delete(self, key):
        def delete_single(key):
            if True:
                # Lower case values
                # Keys can be all lower case, because they will be internal Key components, not specified by users
                if case_sensitive:
                    key2 = {k.lower(): v for k, v in key.items()}
                else:
                    key2 = {k.lower(): v if k.startswith("__") else v.lower() for k, v in key.items()}
            else:
                key2 = key

            # Get IDs
            oids = self.get(key, just_oid=True)
            if len(oids) > 0:
                # From key_i: value_i remove IDs (set difference)
                for k, v in key2.items():
                    d = self._keys.get(k, None)
                    if d:
                        s = d.get(v, None)
                        if s:
                            s2 = s.difference(oids)
                            d[v] = s2
                            if not s2:
                                del d[v]  # Remove the value for the key

                # Delete oids
                for oid in oids:
                    del self._objs[oid]

                return len(oids)
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
        return dict(keys=self._keys, objs=self._objs, cont=self._id_counter)

    def from_pickable(self, inp):
        self._keys = inp["keys"]
        self._objs = {int(k): v for k, v in inp["objs"].items()}
        self._rev_objs = {v[1]: k for k, v in self._objs.items()}
        self._id_counter = inp["cont"]

        return self  # Allows the following: prd = PartialRetrievalDictionary().from_pickable(inp)


# #####################################################################################################################
# >>>> EXTERNAL DATASETS <<<<
# #####################################################################################################################


def get_statistical_dataset_structure(source, dset_name, local_datasets=None):
    from backend.ie_imports.data_sources.ad_hoc_dataset import AdHocDatasets
    # Register AdHocDatasets
    if local_datasets:
        # Register AdHocSource, which needs the current state
        adhoc = AdHocDatasets(local_datasets)
        backend.data_source_manager.register_datasource_manager(adhoc)

    # Obtain DATASET: Datasource -> Database -> DATASET -> Dimension(s) -> CodeList (no need for "Concept")
    dset = backend.data_source_manager.get_dataset_structure(source, dset_name)

    # Generate "dims", "attrs" and "meas" from "dset"
    dims = create_dictionary()  # Each dimension has a name, a description and a code list
    attrs = create_dictionary()
    meas = create_dictionary()
    for dim in dset.dimensions:
        if dim.is_measure:
            meas[dim.code] = None
        else:
            # Convert the code list to a dictionary
            if dim.get_hierarchy():
                cl = dim.get_hierarchy().to_dict()
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
            for c, description in dims[l].code_list.items():
                lst_dim_codes.append((c, description))

    if time_dim:
        lst_dim.append(("StartPeriod", None))
        lst_dim.append(("EndPeriod", None))

    # Unregister AdHocDatasets
    if local_datasets:
        backend.data_source_manager.unregister_datasource_manager(adhoc)

    return lst_dim, (dims, attrs, meas)


def obtain_dataset_source(dset_name, local_datasets=None):
    from backend.ie_imports.data_sources.ad_hoc_dataset import AdHocDatasets
    # Register AdHocDatasets
    if local_datasets:
        # Register AdHocSource, which needs the current state
        adhoc = AdHocDatasets(local_datasets)
        backend.data_source_manager.register_datasource_manager(adhoc)

    # Obtain the list of ALL datasets, and find the desired one, then find the source of the dataset
    lst = backend.data_source_manager.get_datasets()  # ALL Datasets, (source, dataset)
    ds = create_dictionary(data={d[0]: t[0] for t in lst for d in t[1]})  # Dataset to Source (to obtain the source given the dataset name)

    if dset_name in ds:
        source = ds[dset_name]
    else:
        source = None

    # Unregister AdHocDatasets
    if local_datasets:
        backend.data_source_manager.unregister_datasource_manager(adhoc)

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


class Memoize2(object):
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


def gzipped(f):
    """
    Decorator to ZIP the response
    """
    @functools.wraps(f)
    def view_func(*args, **kwargs):
        @after_this_request
        def zipper(response):
            accept_encoding = request.headers.get('Accept-Encoding', '')

            if 'gzip' not in accept_encoding.lower():
                return response

            response.direct_passthrough = False

            if (response.status_code < 200 or
                response.status_code >= 300 or
                'Content-Encoding' in response.headers):
                return response
            gzip_buffer = IO()
            gzip_file = gzip.GzipFile(mode='wb',
                                      fileobj=gzip_buffer)
            gzip_file.write(response.data)
            gzip_file.close()

            response.data = gzip_buffer.getvalue()
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Vary'] = 'Accept-Encoding'
            response.headers['Content-Length'] = len(response.data)

            return response

        return f(*args, **kwargs)

    return view_func

# #####################################################################################################################
# >>>> MAPPING <<<<
# #####################################################################################################################


# Cython version is in module "helper_accel.pyx"
def augment_dataframe_with_mapped_columns(
        df: DataFrame,
        dict_of_maps: Dict[str, Tuple[str, List[Dict]]],
        measure_columns: List[str]) -> DataFrame:
    """
    Elaborate a pd.DataFrame from the input DataFrame "df" and
    "dict_of_maps" which is a dictionary of "source_column" to a tuple ("destination_column", map)
    where map is of the form:
        [ {origin category: [{d: destination category, w: weight assigned to destination category}] } ]

    Support not only "Many to One" (ManyToOne) but also "Many to Many" (ManyToMany)

    :param df: pd.DataFrame to process
    :param dict_of_maps: dictionary from "source" to a tuple of ("destination", "map"), see previous introduction
    :param measure_columns: list of measure column names in "df"
    :return: The pd.DataFrame resulting from the mapping
    """
    mapped_cols = {}  # A dict from mapped column names to column index in "m"
    measure_cols = {}  # A dict from measure columns to column index in "m". These will be the columns affected by the mapping weights
    non_mapped_cols = {}  # A dict of non-mapped columns (the rest)
    for i, c in enumerate(df.columns):
        if c in dict_of_maps:
            mapped_cols[c] = i
        elif c in measure_columns:
            measure_cols[c] = i
        else:
            non_mapped_cols[c] = i

    # "np.ndarray" from "pd.DataFrame" (no index, no column labels, only values)
    m = df.values

    # First pass is just to obtain the size of the target ndarray
    ncols = 2*len(mapped_cols) + len(non_mapped_cols) + len(measure_cols)
    nrows = 0
    for r in range(m.shape[0]):
        # Obtain each code combination
        n_subrows = 1
        for c_name, c in mapped_cols.items():
            map_ = dict_of_maps[c_name][1]
            code = m[r, c]
            n_subrows *= len(map_[code])
        nrows += n_subrows

    # Second pass, to elaborate the elements of the destination array
    new_cols_base = len(mapped_cols)
    non_mapped_cols_base = 2*len(mapped_cols)
    measure_cols_base = non_mapped_cols_base + len(non_mapped_cols)

    # Output matrix column names
    col_names = [col for col in mapped_cols]
    col_names.extend([dict_of_maps[col][0] for col in mapped_cols])
    col_names.extend([col for col in non_mapped_cols])
    col_names.extend([col for col in measure_cols])
    assert len(col_names) == ncols

    # Output matrix
    mm = np.empty((nrows, ncols), dtype=object)

    # For each ROW of ORIGIN matrix
    row = 0  # Current row in output matrix
    for r in range(m.shape[0]):
        # Obtain combinations from current codes
        lst = []
        n_subrows = 1
        for c_name, c in mapped_cols.items():
            map_ = dict_of_maps[c_name][1]
            code = m[r, c]
            n_subrows *= len(map_[code])
            lst.append(map_[code])

        combinations = list(itertools.product(*lst))

        for icomb in range(n_subrows):
            combination = combinations[icomb]
            # Mapped columns
            # At the same time, compute the weight for the measures
            w = 1.0
            for i, col in enumerate(mapped_cols.keys()):
                mm[row+icomb, i] = m[r, mapped_cols[col]]
                mm[row+icomb, new_cols_base+i] = ifnull(combination[i]["d"], '')
                if combination[i]["w"]:
                    w *= float(combination[i]["w"])

            # Fill the measures
            for i, col in enumerate(measure_cols.keys()):
                mm[row+icomb, measure_cols_base+i] = w * m[r, measure_cols[col]]

            # Non-mapped columns
            for i, col in enumerate(non_mapped_cols.keys()):
                mm[row+icomb, non_mapped_cols_base+i] = m[r, non_mapped_cols[col]]

        row += n_subrows

    # Now elaborate a DataFrame back
    tmp = pd.DataFrame(data=mm, columns=col_names)

    return tmp


def is_boolean(v):
    return v.lower() in ["true", "false"]


def to_boolean(v):
    return v.lower() == "true"


def is_integer(v):
    try:
        int(v)
        return True
    except ValueError:
        return False


def to_integer(v):
    return int(v)


def is_float(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def to_float(v):
    return float(v)


def is_datetime(v):
    try:
        from dateutil.parser import parse
        parse(v)
        return True
    except ValueError:
        return False


def to_datetime(v):
    from dateutil.parser import parse
    return parse(v)


def is_url(v):
    """
    From https://stackoverflow.com/a/36283503
    """
    min_attributes = ('scheme', 'netloc')
    qualifying = min_attributes
    token = urllib.parse.urlparse(v)
    return all([getattr(token, qualifying_attr) for qualifying_attr in qualifying])


def to_url(v):
    return urllib.parse.urlparse(v)


def is_uuid(v):
    try:
        UUID(v, version=4)
        return True
    except ValueError:
        return False


def to_uuid(v):
    return UUID(v, version=4)


def is_category(v):
    # TODO Get all hierarchies, get all categories from all hierarchies, find if "v" is one of them
    return False


def to_category(v):
    # TODO
    return None #  Taxon  # Return some Taxon


def is_geo(v):
    # TODO Check if "v" is a GeoJSON, or a reference to a GeoJSON
    return None


def to_geo(v):
    return None


def is_str(v):
    return True


def to_str(v):
    return str(v)

# #####################################################################################################################
# >>>> LOAD DATASET FROM URL INTO PD.DATAFRAME <<<<
# #####################################################################################################################


def load_dataset(location: str=None):
    """
    Loads a dataset into a DataFrame
    If the dataset is present, it decompresses it in memory to obtain one of the four datasets per file
    If the dataset is not downloaded, downloads it and decompresses into the corresponding version directory
    :param location: URL of the dataset data
    :return: pd.DataFrame
    """

    if not location:
        df = None
    else:
        pr = urlparse(location)
        if pr.scheme != "":
            # Load from remote site
            if pr.netloc.lower() == "nextcloud.data.magic-nexus.eu":
                # WebDAV
                parts = location.split("/")
                for i, p in enumerate(parts):
                    if p == "nextcloud.data.magic-nexus.eu":
                        url = "/".join(parts[:i+1]) + "/"
                        fname = "/" + "/".join(parts[i+1:])
                        break

                options = {
                    "webdav_hostname": url,
                    "webdav_login": get_global_configuration_variable("FS_USER"),
                    "webdav_password": get_global_configuration_variable("FS_PASSWORD")
                }
                client = wc.Client(options)
                with tempfile.NamedTemporaryFile(delete=True) as temp:
                    client.download_sync(remote_path=fname, local_path=temp.name)
                    f = open(temp.name, "rb")
                    data = io.BytesIO(f.read())
                    f.close()
            else:
                data = urllib.request.urlopen(location).read()
                data = io.BytesIO(data)
        else:
            data = urllib.request.urlopen(location).read()
            data = io.BytesIO(data)

        # Then, try to read it
        t = mimetypes.guess_type(location, strict=True)
        if t[0] == "text/csv":
            df = pd.read_csv(data)
        elif t[0] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(data)

    return df


def prepare_dataframe_after_external_read(ds, df):
    issues = []
    dims = set()  # Set of dimensions, to index Dataframe on them
    cols = []  # Same columns, with exact case (matching Dataset object)
    for c in df.columns:
        for d in ds.dimensions:
            if strcmp(c, d.code):
                cols.append(d.code)  # Exact case
                if not d.is_measure:
                    dims.add(d.code)
                break
        else:
            issues.append("Column '" + c + "' not found in the definition of Dataset '" + ds.code + "'")
    if len(issues) == 0:
        df.columns = cols
        df.set_index(list(dims), inplace=True)

    return issues


# #####################################################################################################################
# >>>> DATAFRAME <<<<
# #####################################################################################################################

def get_dataframe_copy_with_lowercase_multiindex(dataframe: DataFrame) -> DataFrame:
    """
    Create a copy of an input MultiIndex dataframe where all the index values have been lowercased.
    :param dataframe: a MultiIndex dataframe
    :return: A copy of the input dataframe with lowercased index values.
    """
    df = dataframe.copy()
    levels = [df.index.get_level_values(n).str.lower() for n in range(df.index.nlevels)]
    df.index = pd.MultiIndex.from_arrays(levels)
    return df

# #####################################################################################################################
# >>>> OTHER STUFF <<<<
# #####################################################################################################################


def str2bool(v: str):
    return str(v).lower() in ("yes", "true", "t", "1")


def ascii2uuid(s: str) -> str:
    """
    Convert an ASCII string to an UUID hexadecimal string
    :param s: an ASCII string
    :return: an UUID hexadecimal string
    """
    return str(uuid.UUID(bytes=base64.a85decode(s)))


def ifnull(var, val):
    """ Returns first value if not None, otherwise returns second value """
    if var is None:
        return val
    return var


T = TypeVar('T')


def head(l: List[T]) -> Optional[T]:
    """
    Returns the head element of the list or None if the list is empty.
    :param l: The input list
    :return: The head element of the list or None
    """
    if l:
        return l[0]
    else:
        return None


def first(iterable: Iterable[T],
          condition: Callable[[T], bool] = lambda x: True,
          default: Optional[T] = None) -> Optional[T]:
    """
    Returns the first item in the `iterable` that satisfies the `condition`.
    If the condition is not given, returns the first item of the iterable.

    Returns the `default` value if no item satisfying the condition is found.

    >>> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first( () )
    None
    >>> first( (), default="Some" )
    Some
    """
    return next((x for x in iterable if condition(x)), default)


def translate_case(current_names: List[str], new_names: List[str]) -> List[str]:
    """
    Translate the names in the current_names list according the existing names in the new_names list that
    can have a different case.
    :param current_names: a list of names to translate
    :param new_names: a list of new names to use
    :return: the current_names list where some or all of the names are translated
    """
    new_names_dict = {name.lower(): name for name in new_names}
    translated_names = [new_names_dict.get(name.lower(), name) for name in current_names]
    return translated_names


def values_of_nested_dictionary(d: Dict)-> List:
    for v in d.values():
        if not isinstance(v, Dict):
            yield v
        else:
            yield from values_of_nested_dictionary(v)


def name_and_id_dict(obj: object) -> Optional[Dict]:
    if obj:
        return {"name": obj.name, "id": obj.uuid}
    else:
        return None


def get_value_or_list(current_value, additional_value):
    """
    Add a new value to another existing value/s. If a value doesn't exist it returns the new value otherwise
    returns a list with the existing value/s and the new value.
    :param current_value: the current value
    :param additional_value: the new value
    :return: a single value or a list
    """
    if current_value:
        if isinstance(current_value, list):
            return current_value + [additional_value]
        else:
            return [current_value, additional_value]
    else:
        return additional_value


def class_full_name(c: Type) -> str:
    """ Get the full name of a class """
    module = c.__module__
    if module is None:
        return c.__name__
    else:
        return module + '.' + c.__name__


def object_full_name(o: object) -> str:
    """ Get the full class name of an object """
    return class_full_name(o.__class__)


def split_and_strip(s: str, sep=",") -> List[str]:
    """Split a string representing a comma separated list of strings into a list of strings
    where each element has been stripped. If the string has no elements an empty list is returned."""
    string_list: List[str] = []
    if s is not None and isinstance(s, str):
        string_list = [s.strip() for s in s.split(sep)]

    return string_list


FloatOrStringT = Union[str, float]


class FloatOrString:
    @staticmethod
    def to_float(value: Optional[FloatOrStringT]) -> Optional[FloatOrStringT]:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def multiply(a: Optional[FloatOrStringT], b: Optional[FloatOrStringT]) -> Optional[FloatOrStringT]:
        value_a = FloatOrString.to_float(a)
        value_b = FloatOrString.to_float(b)

        if value_a is None or value_b is None:
            return ifnull(value_a, value_b)

        if isinstance(value_a, float) and isinstance(value_b, float):
            return value_a * value_b
        else:
            return f"({value_a})*({value_b})"

    @staticmethod
    def multiply_with_float(a: FloatOrStringT, b: float) -> FloatOrStringT:
        value_a = FloatOrString.to_float(a)

        if isinstance(value_a, float):
            return value_a * b
        else:
            return f"({value_a})*{b}"


class UnitConversion:
    @staticmethod
    def ratio(from_unit: str, to_unit: str) -> float:
        return ureg(from_unit).to(ureg(to_unit)).magnitude

    @staticmethod
    def get_scaled_weight(weight: FloatOrStringT,
                          source_from_unit: str, source_to_unit: Optional[str],
                          target_from_unit: Optional[str], target_to_unit: str) -> FloatOrStringT:
        ratio = 1.0
        if source_to_unit:
            ratio *= UnitConversion.ratio(source_from_unit, source_to_unit)

        if target_from_unit:
            ratio *= UnitConversion.ratio(target_from_unit, target_to_unit)

        return FloatOrString.multiply_with_float(weight, ratio)


def add_label_columns_to_dataframe(ds_name, df, prd):
    """
    Add columns containing labels describing codes in the input Dataframe
    The labels must be in the CodeHierarchies or CodeLists

    :param ds_name: Dataset name
    :param df: pd.Dataframe to enhance
    :param prd: PartialRetrievalDictionary
    :return: Enhanced pd.Dataframe
    """
    from backend.models.musiasem_concepts import Hierarchy
    # Merge with Taxonomy LABELS, IF available
    for col in df.columns:
        hs = prd.get(Hierarchy.partial_key(ds_name + "_" + col))
        if len(hs) == 1:
            h = hs[0]
            nodes = h.get_all_nodes()
            tmp = []
            for nn in nodes:
                t = nodes[nn]
                tmp.append([t[0].lower(), t[1]])  # CSens
            if not backend.case_sensitive and df[col].dtype == 'O':
                df[col + "_l"] = df[col].str.lower()
                col = col + "_l"

            # Dataframe of codes and descriptions
            df_dst = pd.DataFrame(tmp, columns=['sou_rce', col + "_desc"])
            df = pd.merge(df, df_dst, how='left', left_on=col, right_on='sou_rce')
            del df['sou_rce']
            if not backend.case_sensitive:
                del df[col]

    return df


if __name__ == '__main__':
    import random
    import string
    from timeit import default_timer as timer

    class Dummy:
        def __init__(self, a):
            self._a = a

    def rndstr(n):
        return random.choices(string.ascii_uppercase + string.digits, k=n)

    prd = PartialRetrievalDictionary2()
    ktypes = [("a", "b", "c"), ("a", "b"), ("a", "d"), ("a", "f", "g")]
    # Generate a set of keys and empty objects
    vals = []
    print("Generating sample")
    for i in range(30000):
        # Choose random key
        ktype = ktypes[random.randrange(len(ktypes))]
        # Generate the element
        vals.append(({k: ''.join(rndstr(6)) for k in ktype}, Dummy(rndstr(12))))

    print("Insertion started")
    df = pd.DataFrame()
    start = timer()
    # Insert each element
    for v in vals:
        prd.put(v[0], v[1])
    stop = timer()
    print(stop-start)

    print("Reading started")

    # Select all elements
    start = timer()
    # Insert each element
    for v in vals:
        r = prd.get(v[0], False)
        if len(r) == 0:
            raise Exception("Unexpected!")
    stop = timer()
    print(stop-start)

    print("Deleting started")

    # Select all elements
    start = timer()
    # Insert each element
    for v in vals:
        r = prd.delete(v[0])
        if r == 0:
            raise Exception("Unexpected!")
    stop = timer()
    print(stop-start)

    print("Finished!!")
