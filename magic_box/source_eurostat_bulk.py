import os
import requests
import gzip
import numpy as np
import pandas as pd
import tempfile
from io import StringIO


def download_file(url, local_filename, update=False):
    """
    Download a file (general purpose, not only for Eurostat datasets)

    :param url:
    :param local_filename:
    :param update:
    :return:
    """
    if os.path.isfile(local_filename):
        if not update:
            return
        else:
            os.remove(local_filename)

    r = requests.get(url, stream=True)
    # http://stackoverflow.com/questions/15352668/download-and-decompress-gzipped-file-in-memory
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_eurostat_dataset_into_dataframe(dataset, update=False):

    def multi_replace(text, rep):
        import re
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    dataframe_fn = "/tmp/" + dataset + ".bin"
    df = None
    if os.path.isfile(dataframe_fn):
        df = pd.read_msgpack(dataframe_fn)

    if df is None:
        import requests_cache
        d = {"backend": "sqlite", "include_get_headers": True, "cache_name": "/tmp/eurostat_bulk_datasets"}
        requests_cache.install_cache(**d)
        url = "http://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?downfile=data%2F" + dataset + ".tsv.gz"
        zip_name = tempfile.gettempdir() + "/" + dataset + '.tsv.gz'
        download_file(url, zip_name, update)
        with gzip.open(zip_name, "rb") as gz:
            # Read file
            # Remove flags (documented at http://ec.europa.eu/eurostat/data/database/information)
            # b = break in time series
            # c = confidential
            # d = definition differs, see metadata. The relevant explanations are usually provided in the annex of the metadata.
            # e = estimated
            # f = forecast
            # i = see metadata (phased out) x
            # n = not significant
            # p = provisional
            # r = revised (phased out) x
            # s = Eurostat estimate (phased out) x
            # u = low reliability
            # z = not applicable
            st = multi_replace(gz.read().decode("utf-8"),
                               {":": "NaN", " p": "", " e": "", " f": "", " n": "", " c": "", " u": "", " z": "", " r": "", " b": "", " d": ""})
            fc = StringIO(st)
            #fc = StringIO(gz.read().decode("utf-8").replace(" p\t", "\t").replace(":", "NaN"))
        os.remove(zip_name)
        # Remove ":" -> NaN
        # Remove " p" -> ""
        df = pd.read_csv(fc, sep="\t")

        def split_codes(all_codes):  # Split, strip and lower
            return [s.strip().lower() for s in all_codes.split(",")]

        original_column = df.columns[0]
        new_cols = [s.strip() for s in original_column.split(",")]
        new_cols[-1] = new_cols[-1][:new_cols[-1].find("\\")]
        temp = list(zip(*df[original_column].map(split_codes)))
        del df[original_column]
        df.columns = [c.strip() for c in df.columns]
        # Convert to numeric
        for cn in df.columns:
            df[cn] = df[cn].astype(np.float)
            # df[cn] = df[cn].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        # Add index columns
        for i, c in enumerate(new_cols):
            df[c] = temp[i]
        # set index on the dimension columns
        df.set_index(new_cols, inplace=True)
        # Save df
        df.to_msgpack(dataframe_fn)

    return df


def get_eurostat_filtered_dataset_into_dataframe(dataset, update=False):
    """
    Main function of this module, it allows obtaining a Eurostat dataset

    :param dataset:
    :param update:
    :return:
    """

    df1 = get_eurostat_dataset_into_dataframe(dataset, update)

    return df1


if __name__ == "__main__":
    df = get_eurostat_filtered_dataset_into_dataframe("nrg_110a", {'startDate': '2014', 'endDate': '2015', 'unit': 'ktoe', 'geo': ['es', 'pt']})
    data_frame = get_eurostat_dataset_into_dataframe("nrg_110a")
