# Configuration

If no configuration file is provided (environment variable MAGIC_NIS_SERVICE_CONFIG_FILE not specified), a default configuration file is generated which can be later modified with a text editor.

To specify a custom configuration, a text file with the typical syntax of a **Variable** and its **Value** per line must be created:

`VAR1="VALUE"`

`VAR2="VALUE"`

Variable name | Value | Example
--- | --- | --- |
DB_CONNECTION_STRING | Metadata database, SQLAlchemy compliant connection string | "sqlite:///nis_metadata.db" |
DATA_CONNECTION_STRING | Dataset cache database, SQLAlchemy compliant connection string | "sqlite:///nis_cached_data.db" |
CASE_STUDIES_DIR | Directory where case studies would be stored | "/srv/nis_data/cs/" |
FAO_DATASETS_DIR | Directory where FAO datasets are downloaded and cached | "/srv/faostat/" |
FADN_FILES_LOCATION | Directory where FADN datasets are downloaded and cached | "/srv/fadn" |
CACHE_FILE_LOCATION | Directory where SDMX datasets are downloaded and cached | "/srv/sdmx_datasets_cache" |
REDIS_HOST_FILESYSTEM_DIR | If REDIS_HOST='filesystem:local_session', directory where sessions are stored | "/srv/sessions" |
SSP_FILES_DIR | Not used | "" |
NIS_FILES_LIST | A comma-separated list of URLs to CSV files where NIS case studies or parts of them are enumerated. Each CSV file starts with a header, with four columns: name, url, description and example (True if it is an example) | "" |  
REDIS_HOST | "localhost" expects a REDIS server available at localhost:6379; "redis-local" creates a local REDIS instance; "filesystem:local_session" uses filesystem to store sessions (a good option for execution in PC/laptop) | "" |
TESTING | "True"| "True" |
SELF_SCHEMA | Name of the host where Backend RESTful service responds, preceded by the protocol (http or https) | "https://one.nis.magic-nexus.eu/" |
FS_TYPE | "Webdav" | "Webdav" |
FS_SERVER | Host name of the WebDAV server | "nextcloud.data.magic-nexus.eu" |
FS_USER | User name used. Files and folders must be readable and writable by this user | "<webdav user>" |
FS_PASSWORD | Password for the previous user | "<password in clear>" |
GAPI_CREDENTIALS_FILE | Path to a file obtained from Google API management web, to directly access a NIS workbook file in Google Sheets | "/srv/credentials.json" |
GAPI_TOKEN_FILE | Path to a file used to stored authorization token | "/srv/tocken.pickle" |
