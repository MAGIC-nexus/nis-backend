import mimetypes
from io import BytesIO
import pandas as pd
from typing import List, Tuple
import urllib.request

from backend.common.helper import create_dictionary
from backend.ie_imports.data_source_manager import IDataSourceManager, filter_dataset_into_dataframe
from backend.models.statistical_datasets import Dataset, DataSource, Database


def load_dataset(ds):
    """
    Loads a dataset into a DataFrame
    If the dataset is present, it decompresses it in memory to obtain one of the four datasets per file
    If the dataset is not downloaded, downloads it and decompresses into the corresponding version directory
    :param code:
    :param date:
    :param ds_lst: list of FADN datasets
    :param directory:
    :param base_url:
    :return:
    """

    location = ds.attributes["_location"]
    if not location:
        df = None
    else:
        # Try to load the Dataset from the specified location
        data = urllib.request.urlopen(location).read()
        data = BytesIO(data)
        # Then, try to read it
        t = mimetypes.guess_type(location, strict=True)
        if t[0] == "text/csv":
            df = pd.read_csv(data)
        elif t[0] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(data)

    return df


class AdHocDatasets(IDataSourceManager):
    """
    Datasets in a file (external, or the file currently being analyzed) with format defined by Magic project
    """
    def __init__(self, datasets_list: List[Dataset]):
        self._registry = create_dictionary()
        for ds in datasets_list:
            self.register_dataset(ds.code, ds)

    def initialize_datasets_registry(self, datasets_list: List[Dataset]):
        """
        Receive a list of the datasets and make a copy

        :param datasets_list:
        :return:
        """
        self._registry = create_dictionary()
        for ds in datasets_list:
            self.register_dataset(ds.code, ds)

    def register_dataset(self, name, ds):
        self._registry[name] = ds

    def get_name(self) -> str:
        """ Source name """
        return self.get_datasource().name

    def get_datasource(self) -> DataSource:
        """ Data source """
        src = DataSource()
        src.name = "AdHoc"
        src.description = "A special, ad-hoc, data source, providing datasets elaborated inside an execution. Datasets are local to the execution."
        return src

    def get_databases(self) -> List[Database]:
        """ List of databases in the data source """
        db = Database()
        db.code = ""
        db.description = "AdHoc is just a single database"
        return [db]

    def get_datasets(self, database=None) -> list:
        """ List of datasets in a database, or in all the datasource (if database==None)
            Return a list of tuples (database, dataset)
        """

        lst = []
        for d in self._registry:
            lst.append((d, self._registry[d].description))  # [(name, description)]

        return lst

    def get_datasets(self, database=None) -> list:
        """ List of datasets in a database, or in all the datasource (if database==None)
            Return a list of tuples (database, dataset)
        """
        return [k for k in self._registry]

    def get_dataset_structure(self, dataset) -> Dataset:
        """ Obtain the structure of a dataset: concepts, dimensions, attributes and measures """
        return self.etl_dataset(dataset, update=False)

    def etl_full_database(self, database=None, update=False):
        pass

    def etl_dataset(self, dataset, update=False) -> str:
        """
        Read dataset data and metadata into NIS databases

        :param url:
        :param local_filename:
        :param update:
        :return: String with full file name
        """
        pass

    def get_dataset_filtered(self, dataset, dataset_params: List[Tuple]) -> Dataset:
        """ This method has to consider the last dataset download, to re"""
        # Read dataset structure
        ds = self.get_dataset_structure(None, dataset)

        if not ds.data:
            df = load_dataset(ds)

        # Obtain dataset dictionary
        d = None
        for dd, ds in self._registry.items:
            if ds.code == dataset:
                d = dd
                break

        # Filter it using generic Pandas filtering capabilities
        if dataset_params["StartPeriod"] and dataset_params["EndPeriod"]:
            years = [str(y) for y in range(int(dataset_params["StartPeriod"][0]), int(dataset_params["EndPeriod"][0])+1)]
            dataset_params["year"] = years
            del dataset_params["StartPeriod"]
            del dataset_params["EndPeriod"]
        ds.data = filter_dataset_into_dataframe(df, dataset_params)

        return ds

    def get_refresh_policy(self):  # Refresh frequency for list of databases, list of datasets, and dataset
        pass


