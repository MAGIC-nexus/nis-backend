"""
QQQQ
Buenas tardes,

Me gustaría saber si existe la posibilidad de descargar por completo (bulk) desde alguna URL los datsets PRODCOM,
además sin requerir interacción humana.

Ahora mismo más de 6000 de los datasets de Eurostat se pueden descargar perfectamente de esta manera.
Sin embargo no he encontrado los de PRODCOM, y era simplemente saber si es que no es posible o si la URL
de descarga es distinta.

Muchas gracias.
---------------------------------------------------------------------------------------------------------------------
AAAA
Gracias por su interés en la página web de Eurostat, www.ec.europa.eu/eurostat, y por plantearnos su consulta telefónica.

He encontrado la opción de descarga para PRODCOM. Le adjunto a continuación el enlace a esos archivos y la ruta seguida.
El único problema es que los datos son de finales de 2017. Hemos preguntado al equipo responsable y, como le acabo de comentar por teléfono, nos han comunicado que están trabajando en las actualizaciones y que esperan subir nuevos datos para finales de la próxima semana, en principio.

Aquí tiene el enlace a los datos de PRODCOM:

http://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?sort=1&dir=comext%2FCOMEXT_OTHER_DATA%2FEUROPROMS

Ruta: Bulk Download Listing > comext > COMEXT OTHER DATA > Europroms > seleccione a continuación el archivo o archivos
que más le interesen.

Esperamos que esta información le sea de utilidad.
---------------------------------------------------------------------------------------------------------------------
QQQQ
La información que me pasas ya me sirve bastante.

Para completar el desarrollo que tengo que hacer, necesitaría saber también cómo consultar los metadatos. Con los demás
datasets de Eurostat existe una URL donde se obtiene una información en formato SDMX. Me pasa algo parecido a lo
anterior, y es que no sé qué URL habría para consultar los datasets de COMEXT. Por ejemplo para los demás datasets de
Eurostat esta URL es:

http://ec.europa.eu/eurostat/SDMX/diss-web/rest/dataflow/ESTAT/all/latest

¿Hay alguna equivalente para COMEXT?
---------------------------------------------------------------------------------------------------------------------
AAAA
Buenos días de nuevo:

Me alegro mucho de que le haya servido. Me comunican que para COMEXT y PRODCOM no disponemos de un archivo en formato
SDMX como los otros que ha encontrado. Los metadatos sí que los tenemos en la página de Eurostat para las tablas de
PRODCOM:

http://ec.europa.eu/eurostat/web/prodcom/data/database

En el símbolo con la M al lado del nombre de la carpeta encontrará la información. Le paso el enlace también:

http://ec.europa.eu/eurostat/cache/metadata/en/prom_esms.htm

Espero que, teniendo la información, pueda manejarla usted mismo para su proyecto. En cualquier caso, recuerde mirar
los otros enlaces la semana que viene, por si hubiera una actualización en este sentido.

"""

from abc import abstractmethod

from typing import List

from backend.ie_imports.data_source_manager import IDataSourceManager
from backend.models.statistical_datasets import DataSource, Database, Dataset


class COMEXT(IDataSourceManager):
    @abstractmethod
    def get_name(self) -> str:
        """ Source name """
        pass

    @abstractmethod
    def get_datasource(self) -> DataSource:
        """ Data source """
        pass

    @abstractmethod
    def get_databases(self) -> List[Database]:
        """ List of databases in the data source """
        pass

    @abstractmethod
    def get_datasets(self, database=None) -> list:
        """ List of datasets in a database, or in all the datasource (if database==None)
            Return a list of tuples (database, dataset)
        """
        pass

    @abstractmethod
    def get_dataset_structure(self, database, dataset) -> Dataset:
        """ Obtain the structure of a dataset: concepts, dimensions, attributes and measures """
        pass

    @abstractmethod
    def etl_full_database(self, database=None, update=False):
        """ If bulk download is supported, refresh full database """
        pass

    @abstractmethod
    def etl_dataset(self, dataset, update=False):
        """ If bulk download is supported, refresh full dataset """
        pass

    @abstractmethod
    def get_dataset_filtered(self, dataset, dataset_params: list) -> Dataset:
        """ Obtains the dataset with its structure plus the filtered values
            The values can be in a pd.DataFrame or in JSONStat compact format
            After this, new dimensions can be joined, aggregations need to be performed
        """
        pass

    @abstractmethod
    def get_refresh_policy(self):  # Refresh frequency for list of databases, list of datasets, and dataset
        pass
