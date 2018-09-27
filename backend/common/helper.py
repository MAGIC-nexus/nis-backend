# -*- coding: utf-8 -*-

import collections
import functools
import gzip
import itertools
# from numba import jit
import ast
import json
import urllib
from typing import IO
from uuid import UUID

import pandas as pd
import numpy as np
from flask import after_this_request, request
from multidict import MultiDict, CIMultiDict
from functools import partial

import backend
from backend import case_sensitive, \
                    SDMXConcept


# #####################################################################################################################
# >>>> CASE SeNsItIvE or INSENSITIVE names (flows, funds, processors, ...) <<<<
# #####################################################################################################################
# from backend.models.musiasem_concepts import Taxon  IMPORT LOOP !!!!! AVOID !!!!


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
    elif isinstance(obj, CaseInsensitiveDict):
        return str(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
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
                key2 = {k.lower(): v if k.startswith("__") else v.lower() for k, v in key.items()}
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
        lst_dim.append(("startPeriod", None))
        lst_dim.append(("endPeriod", None))

    return lst_dim, (dims, attrs, meas)


def obtain_dataset_source(dset_name):
    # from backend.model.persistent_db.persistent import DBSession
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


# TODO Consider optimization using numba or others
def augment_dataframe_with_mapped_columns(df, maps, measure_columns):
    """
    Elaborate a pd.DataFrame from the input DataFrame "df" and
    "maps" which is a list of tuples ("source_column", "destination_column", map)
    where map is of the form:
        [{"o": "", "to": [{"d": "", "w": ""}]}]
        [ {o: origin category, to: [{d: destination category, w: weight assigned to destination category}] } ]

    :param df: pd.DataFrame to process
    :param maps: list of tuples (source, destination, map), see previous introduction
    :param measure_columns: list of measure column names in "df"
    :return: The pd.DataFrame resulting from the mapping
    """
    dest_col = {t[0]: t[1] for t in maps}  # Destination column from origin column
    mapped_cols = {}  # A dict from mapped column names to column index in "m"
    measure_cols = {}  # A dict from measure columns to column index in "m". These will be the columns affected by the mapping weights
    non_mapped_cols = {}  # A dict of non-mapped columns (the rest)
    for i, c in enumerate(df.columns):
        if c in dest_col.keys():
            mapped_cols[c] = i
        elif c in measure_columns:
            measure_cols[c] = i
        else:
            non_mapped_cols[c] = i

    # A map relating origin column to a tuple formed by the destination column and the full map
    dict_of_maps = {}
    for t in maps:
        dict_of_maps[t[0]] = (t[1], {d["o"]: d["to"] for d in t[2]})  # Destination column and a new mapping repeating the origin

    # "np.ndarray" from "pd.DataFrame" (no index, no column labels, only values)
    m = df.values

    # First pass is just to obtain the size of the target ndarray
    ncols = 2*len(mapped_cols) + len(non_mapped_cols) + len(measure_cols)
    nrows = 0
    for r in range(m.shape[0]):
        # Obtain each code combination
        n_subrows = 1
        for c_name, c in mapped_cols.items():
            # dest_col = map_of_maps[c_name][0]
            map_ = dict_of_maps[c_name][1]
            code = m[r, c]
            n_subrows *= len(map_[code])
        nrows += n_subrows

    # Second pass, to elaborate the elements of the destination array
    new_cols_base = len(mapped_cols)
    non_mapped_cols_base = 2*len(mapped_cols)
    measure_cols_base = non_mapped_cols_base + len(non_mapped_cols)

    # Output matrix column names
    col_names = [col for col in mapped_cols.keys()]
    col_names.extend([dest_col[col] for col in mapped_cols.keys()])
    col_names.extend([col for col in non_mapped_cols.keys()])
    col_names.extend([col for col in measure_cols.keys()])
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
            # dest_col = dict_of_maps[c_name][0]
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
                mm[row+icomb, new_cols_base+i] = combination[i]["d"]
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


def str2bool(v: str):
    return str(v).lower() in ("yes", "true", "t", "1")

