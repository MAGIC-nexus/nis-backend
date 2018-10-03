from backend.common.helper import create_dictionary
from backend.ie_imports.data_source_manager import IDataSourceManager
from backend.models.statistical_datasets import Dataset


class AdHocDatasets(IDataSourceManager):
    """
    Datasets in a file (external, or the file currently being analyzed) with format defined by Magic project
    """
    def __init__(self):
        self._registry = create_dictionary()

    def initialize_datasets_registry(self, datasets_list):
        """
        Receive a list of the datasts and make a copy

        :param datasets_list:
        :return:
        """
        pass

    def register_dataset(self, name, location):
        self._registry[name] = location

    def get_name(self) -> str:
        """ Source name """
        return "AdHoc"

    def get_databases(self) -> list[str]:
        """ List of databases in the data source """
        return []

    def get_datasets(self, database=None) -> list:
        """ List of datasets in a database, or in all the datasource (if database==None)
            Return a list of tuples (database, dataset)
        """
        return [k for k in self._registry]

    def get_dataset_structure(self, dataset) -> Dataset:
        """ Obtain the structure of a dataset: concepts, dimensions, attributes and measures """
        return Dataset.construct()

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

    def get_dataset_filtered(self, dataset, dataset_params: list[tuple]) -> Dataset:
        """ This method has to consider the last dataset download, to re"""
        pass

    def get_refresh_policy(self):  # Refresh frequency for list of databases, list of datasets, and dataset
        pass


