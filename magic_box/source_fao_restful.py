""" Download FAO data using RESTful API """

import re

import requests
import numpy as np
import pandas as pd

__version__ = '0.1.0'
__fao_url__ = 'http://data.fao.org/developers/api/resources'  # 'api/v2/en/resources'

# Elaborated from technical reference at: http://api.data.fao.org/1.0/esb-rest/resources/resourcesAPI.html

#
# Address for BULK Download of FAOSTAT database (820 MB, zipped)
# http://fenixservices.fao.org/faostat/static/bulkdownloads/FAOSTAT.zip
#


def __getitems(jsoncode):
    """ Convert json code to dataframe """
    try:
        items = jsoncode['result']['list']['items']

        # Convert to list if only one items
        if isinstance(items, dict):
            items = [items]

        # Convert to list
        datajs = [it for it in items]

        # Retrieve number of items
        total = jsoncode['result']['list']['total']
        returned = jsoncode['result']['list']['page']
        returned *= jsoncode['result']['list']['pageSize']

    except (KeyError, TypeError):
        return None, 0, 0

    # Convert to dataframes
    data = pd.DataFrame(datajs)

    return data, total, returned


def get_databases():
    """ List of FAO databases

    Returns
    -----------
    db : pandas.core.frame.DataFrame
        List of FAO databases. Columns of the data frame are
        * label: FAO label
        * uri: weblink the FAO website
        * uuid: Unique identifier
        * mnemonic: Short version of the FAO label

    Example
    -----------
    >>> from faodata import faodata
    >>> db = faodata.get_databases()
    >>> db.columns()
    Index([u'label', u'uri', u'urn', u'uuid', u'mnemonic'], dtype='object')
    """
    page = 1
    done = False
    lst = []
    while not done:
        url = '%s/database' % __fao_url__
        params = {'page': page, 'version': '1.0'}
        req = requests.get(url, params=params)
        jsoncode = req.json()
        databases, total, ret = __getitems(jsoncode)
        databases['mnemonic'] = databases['database']
        lst.append(databases)
        done = total <= ret
        page += 1

    return pd.concat(lst)


def get_resource_types(database):
    """ List of the dataset in a given FAO database

    Parameters
    -----------
    database : str
        Database mnemonic (e.g. 'faostat')

    Returns
    -----------
    ds : pandas.core.frame.DataFrame
        List of FAO databasets. Columns of the data frame are
        * _version_: Version of the dataset
        * database: Database name
        * description: Plain text description of the dataset
        * label: Short name of the dataset
        * mnemonic: FAO code for the dataset
        * uri: Weblink in the FAO website

    """

    page = 1
    done = False
    lst = []
    while not done:
        url = __fao_url__
        params = {'page': page, 'version': '1.0', 'database': database}
        req = requests.get(url, params=params)
        jsoncode = req.json()
        resource_types, total, ret = __getitems(jsoncode)
        done = total <= ret
        page += 1

        cols = ['type', 'label', 'URI']
        try:
            resource_types = resource_types.loc[:, cols]
            lst.append(resource_types)
        except (KeyError, AttributeError):
            pass

    if len(lst) > 0:
        return pd.concat(lst)
    else:
        return None


def get_datasets(database):
    """ List of the dataset in a given FAO database

    Parameters
    -----------
    database : str
        Database mnemonic (e.g. 'faostat')

    Returns
    -----------
    ds : pandas.core.frame.DataFrame
        List of FAO databasets. Columns of the data frame are
        * _version_: Version of the dataset
        * database: Database name
        * description: Plain text description of the dataset
        * label: Short name of the dataset
        * mnemonic: FAO code for the dataset
        * uri: Weblink in the FAO website

    Example
    -----------
    >>> from faodata import faodata
    >>> ds = faodata.get_datasets('faostat')

    """

    page = 1
    done = False
    lst = []
    while not done:
        url = '%s/%s/datasets.json' % (__fao_url__, database)
        params = {'page': page, 'fields': 'mnemonic,label@en,description@en, uri'}
        req = requests.get(url, params=params)
        jsoncode = req.json()
        datasets, total, ret = __getitems(jsoncode)
        done = total <= ret
        page += 1

        cols = ['description', 'label', 'mnemonic', 'uri']
        try:
            datasets = datasets.loc[:, cols]
            lst.append(datasets)
        except (KeyError, AttributeError):
            pass

    if len(lst) > 0:
        return pd.concat(lst)
    else:
        return None


def get_fields(database, dataset):
    """ Get info related to a particular dataset

    Parameters
    -----------
    database : str
        Database mnemonic (e.g. 'faostat')
    dataset : str
        Dataset mnemonic (e.g. 'crop-prod')

    Returns
    -----------
    fields : pandas.core.frame.DataFrame
        List of fields in dataset

    Example
    -----------
    >>> from faodata import faodata
    >>> database = 'faostat'
    >>> dataset = 'crop-prod'
    >>> fields = faodata.get_fields(database, dataset)

    """

    url = '%s/%s/%s' % (__fao_url__, database, dataset)

    params = {'fields': 'mnemonic, label@en, unitMeasure, uri'}

    # Get measures
    req = requests.get('%s/measures.json?' % url, params=params)
    jsoncode = req.json()
    fields, total, ret = __getitems(jsoncode)

    try:
        fields = fields.loc[:, ['mnemonic', 'label', 'unitMeasure', 'uri']]
    except (KeyError, AttributeError):
        return None

    return fields


def get_data(database, dataset, field, country=None, year=None):
    """ Get data from specific a field in a dataset

    Parameters
    -----------
    database : str
        Database mnemonic (e.g. 'faostat')
    dataset : str
        Dataset mnemonic (e.g. 'crop-prod')
    field : str
        Field mnemonic (e.g. 'm5510')
    country : str
        ISO3 country code (optional, if none returns data for all countries)
    year : int
        Year (optional, if none returns data for all years)

    Returns
    -----------
    fields : pandas.core.frame.DataFrame
        List of fields in dataset

    Example
    -----------
    >>> from faodata import faodata
    >>> database = 'faostat'
    >>> dataset = 'crop-prod'
    >>> field = 'm5511'
    >>> df = faodata.get_data(database, dataset, field)
    """

    url = '%s/%s/%s' % (__fao_url__, database, dataset)

    params = {
        'fields':('year,cnt.iso3 ' + \
            'as country,item as item, %s as value') % field,
        'page': 1,
        'pageSize':50
    }

    if not country is None:
        params.update({'filter':'cnt.iso3 eq %s' % country})

    if not year is None:
        if not 'filter' in params:
            params.update({'filter':'year eq %d' % year})

        else:
            params['filter'] = '%s and year eq %d' % ( \
                    params['filter'], year)

    # Get data - first pass
    req = requests.get('%s/facts.json?' % url, params=params)
    jsoncode = req.json()
    data, total, ret = __getitems(jsoncode)

    # Get data - second pass
    # with updates on the number of pages
    if total > ret:
        params['pageSize'] = total

        req = requests.get('%s/facts.json?' % url, params=params)
        jsoncode = req.json()
        data, total, ret = __getitems(jsoncode)

    if data is None:
        return data

    # Convert value to float
    try:
        data['value'] = data['value'].astype(float)
    except KeyError:
        return None

    # Remove data with no country
    idx = data['country'] != 'null'
    idx = idx & (data['value'] >= 0)
    if np.sum(idx) == 0:
        return None

    data = data[idx]

    return data

if __name__ == '__main__':
    db = get_databases()
    for r in ["faostat"]:
        get_resource_types(r)

    for r in db.iterrows():
        get_resource_types(r[1].mnemonic)
        ds = get_datasets(r[1].mnemonic)
        if ds is not None:
            for r2 in ds.iterrows():
                print(r[1].mnemonic+" : "+r2[1].mnemonic+"("+str(r2[1].label)+", "+str(r2[1].description)+")")
