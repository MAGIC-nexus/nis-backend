# Getting started

## Features

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

## Installing and executing "nis-backend"

**nis-backend** is a Python package. There are several options to have it installed.

### pip package

The pip version is obviously for execution as package, and does not include the frontend, the other two deployment options below include the frontend.

* Set a Python environment
* Install the package with

`pip install nexinfosys`

* Use class **NIS**, with methods similar to **NISClient** class in [nexinfosys-client](https://github.com/MAGIC-nexus/nis-python-client) package.

### Docker image

For a lab setup, this option allows to configure and run nis-backend inside a Docker environment (Docker must be installed in the target computer).
 
The instructions are at "Docker Hub" site, please follow the link:

https://hub.docker.com/r/magicnexush2020/magic-nis-backend

### Source code

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

## Models with Commands

To specify MuSIASEM grammars (models), NIS proposes a syntactically simple "block oriented DSL" (DSL - Domain Specific Language), structured in a spreadsheet, where each worksheet is considered a batch of Commands of the same kind, each row is an instance of the command and each column is a field for the command.

In order to take advantage of **nis-backend**, analysts write a workbook considering that commands (worksheets) are read and executed from left to right, and inside a worksheet from the first row to the last.

The reference of a command appears when a command from the "Commands" tab is selected in **nis-frontend**. A document with a complete reference is available at ¿ZENODO?. An HTML called "Commands reference" can be obtained from "Exportable datasets" tab, containing an up-to-date reference.

The next section introduces **nis-frontend** which is a tool embedded in **nis-backend** whose goal is to help in writing MuSIASEM grammars.  
 
## Quick intro to **nis-frontend** 
 
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
