from backend.external_data.data_source_manager import IDataSourceManager
from backend.external_data.rdb_model import Dataset


class AdHocDataset(IDataSourceManager):
    """
    A dataset in a file (external, or the file currently being analyzed)
    """
    def __init__(self, location):
        self._location = location

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
        return []

    def get_dataset_structure(self, dataset) -> Dataset:
        """ Obtain the structure of a dataset: concepts, dimensions, attributes and measures """
        return Dataset.construct()

    def etl_full_database(self, database=None, update=False):
        """ If bulk download is supported, refresh full database """
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


