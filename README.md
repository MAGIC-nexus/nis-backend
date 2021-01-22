<a href="http://magic-nexus.eu"><img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/nexinfosys/frontend/assets/images/logo_favicon.png" width=125 height=125 title="MAGIC project logo" alt="MAGIC project logo"></a>


# nis-backend
> A <a href="https://en.wikipedia.org/wiki/MuSIASEM">MuSIASEM</a> instance

NIS (Nexus Information System) is a software system being developed inside MAGIC project (<a href="https://cordis.europa.eu/project/id/689669">H2020 grant #689669</a>). 

NIS enables accounting the biophysical flows in complex bioeconomic systems according to MuSIASEM concepts and methodology, as a way to assess the sustainability of current -or future- scenarios in socio-ecologic systems.

NIS is made of four components: **nis-backend**, **nis-frontend**, [nis-client](https://github.com/MAGIC-nexus/nis-python-client) and [nis-eda](https://github.com/MAGIC-nexus/nis-eda). **nis-backend** is the backend component of NIS, which is deployed embedding another NIS component, **nis-frontend** (an Angular web application) and it is (programmatically) actionable through a third, <a href="https://github.com/MAGIC-nexus/nis-python-client">nis-client</a>, using Python or R. **nis-eda** is an exploratory data analysis tool written in R and Shiny which connects to a **nis-backend**.

**nis-backend** "runs" MuSIASEM models by **interpreting** (lexically, syntactically and semantically) and **resolving** (quantitatively) a representation of a MuSIASEM grammar (an open model), using a **tabular syntax** ("dataframe" compatible), and producing a set of outputs **exportable** for further use by external, complementary tools.

It is an open system as it can be used to integrate MuSIASEM as a formalism in analytic/scientific workflows thanks to the RESTful API (and the easier to use [nis-client](https://github.com/MAGIC-nexus/nis-python-client), for Python and R scripts).

**Disclaimer**: this README is still under elaboration, details may be missing or innacurate.

<!-- Badges
 [![Build Status](http://img.shields.io/travis/badges/badgerbadgerbadger.svg?style=flat-square)](https://travis-ci.org/badges/badgerbadgerbadger) [![Dependency Status](http://img.shields.io/gemnasium/badges/badgerbadgerbadger.svg?style=flat-square)](https://gemnasium.com/badges/badgerbadgerbadger) [![Coverage Status](http://img.shields.io/coveralls/badges/badgerbadgerbadger.svg?style=flat-square)](https://coveralls.io/r/badges/badgerbadgerbadger) [![Code Climate](http://img.shields.io/codeclimate/github/badges/badgerbadgerbadger.svg?style=flat-square)](https://codeclimate.com/github/badges/badgerbadgerbadger) [![Github Issues](http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/issues.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger/issues) [![Pending Pull-Requests](http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/pulls.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger/pulls) [![Gem Version](http://img.shields.io/gem/v/badgerbadgerbadger.svg?style=flat-square)](https://rubygems.org/gems/badgerbadgerbadger) [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org) [![Badges](http://img.shields.io/:badges-9/9-ff6799.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger)

- For more on these wonderful ~~badgers~~ badges, refer to <a href="http://badges.github.io/badgerbadgerbadger/" target="_blank">`badgerbadgerbadger`</a>.

 -->
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

<!-- Insert a representative Screenshot or an animated GIF (use "Recordit"?)-->

## Table of Contents

- [Documentation](#documentation)
- [Getting started](#getting-started)
  - [Features](#features)
  - [Installing and executing **nis-backend**](#installing-and-executing-nis-backend)
    - [Pip package](#pip-package)
    - [Docker image](#docker-image)
    - [Source code](#source-code)
  - [Models with Commands](#models-with-commands)
  - [**nis-frontend** quick intro](#quick-intro-to-nis-frontend)
- [Accessing **nis-backend** in Python and R scripts with **nis-client**](#accessing-nis-backend-in-python-and-r-scripts-with-nis-client)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [People](#people)
- [Contact](#contact)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Keywords](#keywords)
  
## Documentation

Full documentation:

- How-to  (Annex 4 - "NIS frontend how-to manual").
- Technical reference (Annex 2 - "Report on NIsys toolset development").
- Format of input files (Annex 3 - "Format for specification of NIS case studies").

is available in the annexes of the deliverable of the project regarding "nis-backend", in the official web page, at http://www.magic-nexus.eu/documents/deliverable-33-report-datasets-and-software-tools.

NOTE: the "Format of input files" document is updated outside of the main document.

## Getting started

### Features

- **MuSIASEM adaptation**. MuSIASEM concepts have been adapted to an object model using a DSL (Domain Specific Language) with tabular syntax (spreadsheet, dataframe).
- **Formalization, standardization**. The former point has as consequence a formal and standard definition of MuSIASEM grammars/models, for repeatability, reproducibility, replicability, findability of case studies and of MuSIASEM itself, and also for automatic processing of specifications.
- **Uses**. Apt for both learning and teaching; for research, economic analysis, decision making, policy making, diagnosis and prognosis, etc. 
- **Integrability**. Easy to integrate into analytical workflows (GIS, statistical, engineering, ...).
- **Inputs - External**. Integrated use of statistical data sources: Eurostat, OECD, FAO, FADN.
- **Inputs - Internal**. Integration of data from custom external spreadsheets.
- **Reusability of model parts**. Similar to libraries/packages in programming languages.
- **Integrated solver**. Capable of exploiting available quantifications and relationships, in multiple:
  - Systems, e.g. interconnected countries.
  - Scales, e.g. food and macronutrients.
  - Scenarios, which are named sets of values for parameters, e.g. "optimistic", "expected", "pessimistic".
  - Sources, so several observations can be made about the same fact.
  - Times, in years or in months.
- **Outputs**. Exportable as datasets (statistical cubes), graphs (networks), matrices, models (transformations of the input model), geolayers, scripts (Python or R).
- **Deployment**. It can be deployed in multiple ways: personal computer (pip, source, Docker), or server (source or Docker).
- **Configurable**. Deployment for server can be configured to use different database servers and REDIS for session management.
- **Open**. All functionality can be accessed through a RESTful service. Behind the scenes, because HTTP protocol is stateless, a complex serialization/deserialization function combined with a key/value store and a browser session enable saving the state in memory after a service call ends, just to recover it when another invocation is done. 
- **Two expertise levels**. Two components wrap the RESTful interface, **nis-frontend** and **nis-client**. These components match two expertise levels: **nis-frontend** does not require programming knowledge, while **nis-client** can be used in Python/R scripts.

### Installing and executing "nis-backend"

**nis-backend** is a Python package. There are several options to have it installed.

#### pip package

The pip version is obviously for execution as package, and does not include the frontend, the other two deployment options below include the frontend.

* Set a Python environment
* Install the package with

`pip install nexinfosys`

* Use class **NIS**, with methods similar to **NISClient** class in [nexinfosys-client](https://github.com/MAGIC-nexus/nis-python-client) package.

#### Docker image

For a lab setup, this option allows to configure and run nis-backend inside a Docker environment (Docker must be installed in the target computer).
 
The instructions are at "Docker Hub" site, please follow the link:

https://hub.docker.com/r/magicnexush2020/magic-nis-backend

#### Source code

**NOTE**: Python3 and git required.

Clone this repository and execute "service_main.py":

```
git clone https://github.com/MAGIC-nexus/nis-backend
cd nis-backend
git checkout develop
pip install -r requirements.txt
PYTHONPATH=. python3 nexinfosys/restful_service/service_main.py
```

Alternately to the last line, for a more robust server (**gunicorn** assumed to be installed):

```bash
gunicorn --workers=3 --log-level=2000 --bind 0.0.0.0:5000 nexinfosys.restful_service.service_main:app
```

Change the port to the one desired

<!--
#### Windows executable

TO-DO - (installer_cmd_maker.py: script using PyInstaller to create executable, in single file form)
-->

### Models with Commands

To specify MuSIASEM grammars (models), NIS proposes a syntactically simple "block oriented DSL" (DSL - Domain Specific Language), structured in a spreadsheet, where each worksheet is considered a batch of Commands of the same kind, each row is an instance of the command and each column is a field for the command.

In order to take advantage of **nis-backend**, analysts write a workbook considering that commands (worksheets) are read and executed from left to right, and inside a worksheet from the first row to the last.

The reference of a command appears when a command from the "Commands" tab is selected in **nis-frontend**. A document with a complete reference is available at ¿ZENODO?. An HTML called "Commands reference" can be obtained from "Exportable datasets" tab, containing an up-to-date reference.

The next section introduces **nis-frontend** which is a tool embedded in **nis-backend** whose goal is to help in writing MuSIASEM grammars.  
 
### Quick intro to **nis-frontend** 
 
**nis-backend** embeds the compiled version of a full Angular application called **nis-frontend** which is the indicated interface to **nis-backend** for most users.
 
Once **nis-backend** is executing in your computer, open a browser a go to the address:
 
<a href="http://localhost:5000/nis_client/">http://localhost:5000/nis_client </a>
 
A screen similar to the following should open:

<img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/docs/initial_screen.png" style="border:1px solid black" title="NIS frontend initial screen" alt="NIS frontend initial screen">
 
The screen has a top bar (blue colored) where a menu, the Play button and a field allowing the specification of a URL are present.
 
The rest of the screen is divided in two zones, at left and right.

The right zone is where the process of representation of a MuSIASEM grammar (model) can be edited (textually or graphically), issues are showed for correction, and outputs can be explored or exported for further elaboration.

The left zone contains helpers allowing the insertion of empty Commands (worksheets) and the selection of external Datasets for the elaboration of internal ones.

The URL field allows editing a workbook using Google Sheets (the application must be authorized first). **nis-backend** reads directly from the specified worksheet, so it needs access enabled to the file.

For extensive documentation see the [deliverable](http://www.magic-nexus.eu/documents/deliverable-33-report-datasets-and-software-tools). Online help for commands appears below the list of commands:

<img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/docs/detail_of_help_window.png" style="border:1px solid black" title="Help in NIS frontend" alt="Help in NIS frontend" height=500>

and help for each field in a command appears when clicking on the name of a field in a worksheet:

<img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/docs/detail_of_field_help_popup.png" style="border:1px solid black" title="Popup help for a field, in NIS frontend" alt="Popup help for a field, in NIS frontend" height=500>

When information is typed for a field in the embedded worksheet, syntactic validity is checked. 

**IMPORTANT**: **nis-frontend** is closed source due to a restriction in the license of a commercial component.

## Accessing **nis-backend** in Python and R scripts with nis-client

**nis-backend** is, from a software engineering point-of-view, a RESTful service. All its functionality is accessed through this interface.

**nis-frontend** makes use of it behind the scenes, it is not necessary for users to know the technical details in this regards. The other alternative designed to access this service is a Python component called "nexinfosys-client". It can be installed using "pip":

`pip install nexinfosys-client`

With this package it possible to integrate the MuSIASEM related capabilities of **nis-backend** in a processing workflow. See <a href="https://github.com/MAGIC-nexus/nis-python-client">nis-client</a> README for basic usage example.

Thanks to R capabilities to use Python packages, the component can be used from R, see also the former README. 

## Configuration

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

## Contributing

Contributions are welcome. Follows a list of possible ways in which you could do it:

* Testing as end user. Install "nis-backend" (this repository) and execute it, launching the user interface to specify and submit case studies, maybe with the help of an external spreadsheet software. Use Github issues when you find incorrect behavior. Please follow [Mastering Issues](https://guides.github.com/features/issues/) guidelines.
* Testing as scripts developer. Install "nis-backend" and use it in your scripts using either NIS class or the subproject [nis-client](https://github.com/MAGIC-nexus/nis-python-client) connecting it to a running "nis-backend".
* Prepare reusable MuSIASEM library files, containing model parts, like structures of high level functions in different domains (water, energy, food), interface type hierarchies, technical data about structurals, mappings from specific official statistical datasets to MuSIASEM interface types, calculation of scalar indicators, benchmarks, etc. All of them may be compiled in separate files, as long as the composition of them is consistent.
* If you are a developer willing to contribute to "nis-backend", please mail one of the contact persons (see Contact below) to discuss some of the features and improvements which could be interesting to have implemented. Once you are ready to contribute, please do it through Github Pull Requests, following the [standard PR workflow](https://guides.github.com/introduction/flow/).

The standard [contributor covenant](CODE_OF_CONDUCT.md) code of conduct is earnestly applied in interactions with the community.

## People

An enumeration of people who have contributed in different manners to the elaboration of NIS during MAGIC project lifetime:

* Rafael Nebot. ITC-DCCT (Instituto Tecnológico de Canarias - Departamento de Computación).
* Marco Galluzzi.
* Michele Staiano. UniNa (Università degli Studi di Napoli Federico II). NIS supervisor, mentor, catalyzer.
* Paula Moreno. ITC-DCCT.
* Mario Giampietro, Ansel Renner, Violeta Cabello, Cristina Madrid, Maddalena Rippa, Juan Cadillo, Raúl Velasco, Louisa di Felice, Sandra Bukkens. ICTA-UAB (Institut de Ciència i Tecnologia Ambientals - Universitat Autònoma de Barcelona). MuSIASEM creators and mentors, analysts.
* Ignacio López.
* Alberto Sosa, Francisco Socorro, María Artiles, Carlos Caraballo, Ivet Cabrera. Internship students at ITC-DCCT.

## Contact

Please send any question regarding this repository to [rnebot@itccanarias.org](mailto:rnebot@itccanarias.org).

<!--
## FAQ
## Support
-->

## License
This project is licensed under the BSD-3 License - see the [LICENSE](LICENSE) file for details

## Acknowledgements
The development of this software was supported by the European Union’s Horizon 2020 research and innovation programme under Grant Agreement No. 689669 (MAGIC). This work reflects the authors' view only; the funding agencies are not responsible for any use that may be made of the information it contains.

## Keywords

    Sustainability - Bioeconomy - Socio-Ecological Systems - Complex Adaptive Systems - Water-Energy-Food Nexus
