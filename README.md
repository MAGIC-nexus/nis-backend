<a href="http://magic-nexus.eu"><img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/nexinfosys/frontend/assets/images/logo_favicon.png" width=125 height=125 title="MAGIC project logo" alt="MAGIC project logo"></a>


# nis-backend
> A <a href="https://en.wikipedia.org/wiki/MuSIASEM">MuSIASEM</a> instance

NIS (Nexus Information System) is a software system being developed inside MAGIC project (<a href="https://cordis.europa.eu/project/id/689669">H2020 grant #689669</a>). 

NIS enables accounting the biophysical flows in complex bioeconomic systems according to MuSIASEM concepts and methodology, as a way to assess the sustainability of current -or future- scenarios in socio-ecologic systems.

It does so by **interpreting** and **resolving -quantitatively-** a representation of a MuSIASEM grammar (kind of a model), using a **tabular syntax** ("dataframe" compatible), and producing a set of outputs **exportable** for further use by external, complementary tools.

**nis-backend** is the backend component of NIS, which is deployed embedding another NIS component, **nis-frontend** and actionable (programmatically) through a third, <a href="https://github.com/MAGIC-nexus/nis-python-client">nis-client</a>, using Python or R. 

<!-- Badges
 [![Build Status](http://img.shields.io/travis/badges/badgerbadgerbadger.svg?style=flat-square)](https://travis-ci.org/badges/badgerbadgerbadger) [![Dependency Status](http://img.shields.io/gemnasium/badges/badgerbadgerbadger.svg?style=flat-square)](https://gemnasium.com/badges/badgerbadgerbadger) [![Coverage Status](http://img.shields.io/coveralls/badges/badgerbadgerbadger.svg?style=flat-square)](https://coveralls.io/r/badges/badgerbadgerbadger) [![Code Climate](http://img.shields.io/codeclimate/github/badges/badgerbadgerbadger.svg?style=flat-square)](https://codeclimate.com/github/badges/badgerbadgerbadger) [![Github Issues](http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/issues.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger/issues) [![Pending Pull-Requests](http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/pulls.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger/pulls) [![Gem Version](http://img.shields.io/gem/v/badgerbadgerbadger.svg?style=flat-square)](https://rubygems.org/gems/badgerbadgerbadger) [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org) [![Badges](http://img.shields.io/:badges-9/9-ff6799.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger)

- For more on these wonderful ~~badgers~~ badges, refer to <a href="http://badges.github.io/badgerbadgerbadger/" target="_blank">`badgerbadgerbadger`</a>.

 -->
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

<!-- Insert a representative Screenshot or an animated GIF (use "Recordit"?)-->

## Table of Contents

- [Getting started](#getting-started)
  - [Features](#features)
  - [Installing and executing **NIS backend**](#installing-and-executing-nis-backend)
    - [Windows and Linux installers](#windows-and-linux-installers)
    - [pip](#pip)
    - [Docker image](#docker-image-lab-setup)
    - [Source code](#source-code)
  - [Models](#models)
  - [Basic usage - **NIS frontend**](#basic-usage-nis-frontend)
- [Advanced - **NIS client**](#advanced-nis-client)
- [Configuration](#configuration)
- [People](#people)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Keywords](#keywords)
  
## Getting started

### Features

- **MuSIASEM adaptation**. Adapts MuSIASEM concepts to an object model using a DSL (Domain Specific Language) with tabular syntax (spreadsheet, dataframe).
- **Formalization**. This results in formal definition of MuSIASEM grammars/models, for repeatibility, reproducibility, replicability, findability of case studies and MuSIASEM itself.
  - Standardization, ambiguity prevention
- **Uses**. Apt for both learning and teaching; for research, analysis, decision making, policy making, diagnosis and prognosis, etc. 
- **Integrability**. Easy to integrate into analytical workflows (GIS, statistical, engineering, ...).
- **Inputs - External**. Integrated use of statistical data sources: Eurostat, OECD, FAO, FADN.
- **Inputs - Internal**. Integration of data from external spreadsheets.
- **Reusability of model parts**. Similar to libraries/packages in programming languages.
- **Integrated solver**. Capable of exploiting available quantifications and relationships, in multiple:
  - Systems (e.g. interconnected countries)
  - Scales (e.g. food and macronutrients)
  - Scenarios (e.g. )
  - Sources (observations for the same fact)
  - Times
- **Outputs**. Exportable as datasets (statistical cubes), graphs (networks), matrices, models, geolayers, scripts.
- **Deployment**. It can be deployed in multiple ways: single computer (installer or pip), common server (Docker).
- **Configurable**. Deployment for server can be configured to different database servers and REDIS.
- **Open**. because all functionality can be accessed through a RESTful service. Behind the scenes, because HTTP protocol is stateless, a complex serialization/deserialization function combined with a key/value store and a browser session enable saving the state in memory after a service call ends, just to recover it when another invocation is done. 
- **Two expertise levels**. On top of this interface, two components, NIS frontend and NIS Python client, hide the RESTful service in two expertise levels: non-initiated to Python/R programming (most users) and capable of writing Python/R scripts

### Installing and executing "NIS backend"

**nis-backend** is a Python package. There are several options to have it installed.

#### Windows, Mac OS X and Linux installers

TO-DO - (PyInstaller output for each platform in both directory and single file forms, giving six possibilities)

#### pip

<!-- Prepare package to upload it to pypi -->

* Set a Python environment
* Install the package with

TO-DO `pip install nexinfosys`

* Start the server

`python3 nexinfosys.restful_service.service_main.py`
 
#### Docker image

For a lab setup this option allows to configure and run nis-backend inside a Docker environment.
 
<!-- Upload image to docker hub -->

#### Source code

Just clone this repository and execute "service_main.py"

### Models

To specify MuSIASEM grammars (models), NIS proposes a syntactically simple "block oriented DSL" (DSL - Domain Specific Language), structured in a spreadsheet, where each worksheet is considered a batch of Commands of the same kind, each row is an instance of the command and each column is a field for the command.

In order to take advantage of NIS backend, analysts write a workbook considering that commands (worksheets) are read and executed from left to right, and inside a worksheet from the first row to the last.

The reference of commands can be found at ¿ZENODO?

The next section introduces NIS frontend which is a tool embedded in NIS backend whose goal is to help in writing MuSIASEM grammars.  
 
### Basic usage (NIS frontend)
 
NIS backend embeds the compiled version of a full Angular application called "NIS frontend" which is the indicated interface to NIS backend for most users.
 
Once "NIS backend" is executing, open a browser a go to the address:
 
`http://localhost:5000/nis_client`
 
A screen similar to the following will open:

<img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/docs/initial_screen.png" title="NIS frontend initial screen" alt="NIS frontend initial screen">
 
The screen has a top bar (blue colored) where a menu, the Play button and a field allowing the specification of a URL are present.
 
The rest of the screen is divided in two: a left and a right zone.

The right zone is where the process of representation of a MuSIASEM grammar (model) can be edited (textually or graphically) and outputs can be explored or exported for further elaboration.

The left zone contains helpers allowing the insertion of empty commands (worksheets) and the selection of external datasets for the elaboration of internal ones.

The URL field allows editing a workbook using Google Sheets. NIS backend reads directly from the specified worksheet, so it needs access enabled to the file.

For extensive documentation see the **User's Manual**. 

## Advanced (NIS Client)

NIS backend is, from a software engineering point-of-view, a RESTful service. All its functionality is accessed through this interface.

NIS frontend makes use of it behind the scenes, it is not necessary for users to know the technical details in this regards. The other alternative designed to access this service is a Python component called "nexinfosys-client". It can be installed using "pip":

`pip install nexinfosys-client`

With this package it possible to integrate the MuSIASEM related capabilities of NIS backend in a processing workflow. **See github repository README...** for basic usage with Python.

Similarly, thanking to R capabilities to use Python packages, the component can be used from R **(see ...)**. 

## Configuration

TO-DO explain the configuration variables in a configuration file and how to specify the name of the configuration file before execution of the server

## People

* Rafael Nebot. ITC (Instituto Tecnológico de Canarias, SA). Departamento de Computación.
* Marco Galluzzi
* Paula Moreno. ITC. Departamento de Computación.
* Michele Staiano. UniNa.
* Ansel Renner. UAB ICTA.
* MuSIASEM makers and analysts: Mario Giampietro, Violeta Cabello, Cristina Madrid, Maddalena Rippa, Juan Cadillo, Raúl Velasco, Louisa di Felice.

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
