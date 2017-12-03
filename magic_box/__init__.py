from flask import Flask
import collections

# SDMX Concept can be: dimension, attribute or measure. Stored in "metadatasets" associated to a dataset by its name
SDMXConcept = collections.namedtuple('Concept', 'type name istime description code_list')


class Registry:
    def __init__(self):
        self.processors = None  # Layers of processors. Current "registry"
        self.maps = None  # Maps of names
        self.connections = None  # Maps of links
        self.datasets = None  # Current "dfs"
        self.metadatasets = None  # Metadata about datasets

the_registry = Registry()

app = Flask(__name__)
app.debug = True
UPLOAD_FOLDER = '/tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Initialize configuration
try:
    app.config.from_envvar('MAGIC_SERVICE_CONFIG_FILE')
except Exception:
    print("MAGIC_SERVICE_CONFIG_FILE environment variable not defined!")
