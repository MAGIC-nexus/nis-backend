<a href="http://magic-nexus.eu"><img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/nexinfosys/frontend/assets/images/logo_favicon.png" width=125 height=125 title="MAGIC project logo" alt="MAGIC project logo"></a>


# nis-backend
> A <a href="https://en.wikipedia.org/wiki/MuSIASEM">MuSIASEM</a> instance

NIS (Nexus Information System) is a software system being developed inside MAGIC project (<a href="https://cordis.europa.eu/project/id/689669">H2020 grant #689669</a>). 

NIS enables accounting the biophysical flows in complex bioeconomic systems according to MuSIASEM concepts and methodology, as a way to assess the sustainability of current -or future- scenarios in socio-ecologic systems.

NIS is made of three components: **nis-backend**, **nis-frontend** and **nis-client**. **nis-backend** is the backend component of NIS, which is deployed embedding another NIS component, **nis-frontend** and it is (programmatically) actionable through a third, <a href="https://github.com/MAGIC-nexus/nis-python-client">nis-client</a>, using Python or R.

**nis-backend** "runs" MuSIASEM models by **interpreting** (lexically, syntactically and semantically) and **resolving** (quantitatively) a representation of a MuSIASEM grammar (an open model), using a **tabular syntax** ("dataframe" compatible), and producing a set of outputs **exportable** for further use by external, complementary tools.

**Disclaimer**: this README is still under elaboration, details may be missing or innacurate.

<!-- Badges
 [![Build Status](http://img.shields.io/travis/badges/badgerbadgerbadger.svg?style=flat-square)](https://travis-ci.org/badges/badgerbadgerbadger) [![Dependency Status](http://img.shields.io/gemnasium/badges/badgerbadgerbadger.svg?style=flat-square)](https://gemnasium.com/badges/badgerbadgerbadger) [![Coverage Status](http://img.shields.io/coveralls/badges/badgerbadgerbadger.svg?style=flat-square)](https://coveralls.io/r/badges/badgerbadgerbadger) [![Code Climate](http://img.shields.io/codeclimate/github/badges/badgerbadgerbadger.svg?style=flat-square)](https://codeclimate.com/github/badges/badgerbadgerbadger) [![Github Issues](http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/issues.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger/issues) [![Pending Pull-Requests](http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/pulls.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger/pulls) [![Gem Version](http://img.shields.io/gem/v/badgerbadgerbadger.svg?style=flat-square)](https://rubygems.org/gems/badgerbadgerbadger) [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org) [![Badges](http://img.shields.io/:badges-9/9-ff6799.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger)

- For more on these wonderful ~~badgers~~ badges, refer to <a href="http://badges.github.io/badgerbadgerbadger/" target="_blank">`badgerbadgerbadger`</a>.

 -->
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

<!-- Insert a representative Screenshot or an animated GIF (use "Recordit"?)-->

## Table of Contents

- [Getting started](#getting-started)
  - [Features](#features)
  - [Installing and executing **nis-backend**](#installing-and-executing-nis-backend)
    - [Windows and Linux installers](#windows-and-linux-installers)
    - [pip](#pip)
    - [Docker image](#docker-image-lab-setup)
    - [Source code](#source-code)
  - [Models with Commands](#models-with-commands)
  - [Basic usage - **nis-frontend**](#basic-usage-nis-frontend)
- [Advanced - **nis-client**](#advanced-nis-client)
- [Configuration](#configuration)
- [People](#people)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Keywords](#keywords)
  
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
- **Deployment**. It can be deployed in multiple ways: personal computer (installer or pip), or server (Docker or pip).
- **Configurable**. Deployment for server can be configured to use different database servers and REDIS for session management.
- **Open**. All functionality can be accessed through a RESTful service. Behind the scenes, because HTTP protocol is stateless, a complex serialization/deserialization function combined with a key/value store and a browser session enable saving the state in memory after a service call ends, just to recover it when another invocation is done. 
- **Two expertise levels**. Two components wrap the RESTful interface, **nis-frontend** and **nis-client**. These components match two expertise levels: **nis-frontend** does not require programming knowledge, while **nis-client** can be used in Python/R scripts.

### Installing and executing "nis-backend"

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

For a lab setup, this option allows to configure and run nis-backend inside a Docker environment.
 
<!-- Upload image to docker hub -->

#### Source code

Just clone this repository and execute "service_main.py"

### Models with Commands

To specify MuSIASEM grammars (models), NIS proposes a syntactically simple "block oriented DSL" (DSL - Domain Specific Language), structured in a spreadsheet, where each worksheet is considered a batch of Commands of the same kind, each row is an instance of the command and each column is a field for the command.

In order to take advantage of **nis-backend**, analysts write a workbook considering that commands (worksheets) are read and executed from left to right, and inside a worksheet from the first row to the last.

The reference of commands can be found at ¿ZENODO?

The next section introduces **nis-frontend** which is a tool embedded in **nis-backend** whose goal is to help in writing MuSIASEM grammars.  
 
### Basic usage (**nis-frontend**)
 
**nis-backend** embeds the compiled version of a full Angular application called **nis-frontend** which is the indicated interface to **nis-backend** for most users.
 
Once **nis-backend** is executing, open a browser a go to the address:
 
<a href="http://localhost:5000/nis_client/">http://localhost:5000/nis_client</a>
 
A screen similar to the following will open:

<img src="https://github.com/MAGIC-nexus/nis-backend/raw/develop/docs/initial_screen.png" style="border:1px solid black" title="NIS frontend initial screen" alt="NIS frontend initial screen">
 
The screen has a top bar (blue colored) where a menu, the Play button and a field allowing the specification of a URL are present.
 
The rest of the screen is divided in two zones, at left and right.

The right zone is where the process of representation of a MuSIASEM grammar (model) can be edited (textually or graphically), issues are showed for correction, and outputs can be explored or exported for further elaboration.

The left zone contains helpers allowing the insertion of empty Commands (worksheets) and the selection of external Datasets for the elaboration of internal ones.

The URL field allows editing a workbook using Google Sheets (the application must be authorized first). **nis-backend** reads directly from the specified worksheet, so it needs access enabled to the file.

For extensive documentation see the **User's Manual**. 

**IMPORTANT**: **nis-frontend** is closed source due to a restriction in the license of a commercial component.

## Advanced (**nis-client**)

**nis-backend** is, from a software engineering point-of-view, a RESTful service. All its functionality is accessed through this interface.

**nis-frontend** makes use of it behind the scenes, it is not necessary for users to know the technical details in this regards. The other alternative designed to access this service is a Python component called "nexinfosys-client". It can be installed using "pip":

`pip install nexinfosys-client`

With this package it possible to integrate the MuSIASEM related capabilities of **nis-backend** in a processing workflow. See <a href="https://github.com/MAGIC-nexus/nis-python-client">nis-client</a> README for basic usage example.

Thanks to R capabilities to use Python packages, the component can be used from R, see also the former README. 

## Configuration

TO-DO explain the configuration variables in a configuration file and how to specify the name of the configuration file before execution of the server

## People

The following is an enumeration of people who have contributed in different manners to the elaboration of NIS.

* [Rafael Nebot](mailto:rnebot@itccanarias.org). ITC-DCCT (Instituto Tecnológico de Canarias, SA - Departamento de Computación).
* Marco Galluzzi.
* Michele Staiano. UniNa (Università degli Studi di Napoli Federico II).
* Paula Moreno. ITC-DCCT.
* MuSIASEM creators and analysts at ICTA-UAB (Institut de Ciència i Tecnologia Ambientals - Universitat Autònoma de Barcelona): Mario Giampietro, Ansel Renner, Violeta Cabello, Cristina Madrid, Maddalena Rippa, Juan Cadillo, Raúl Velasco, Louisa di Felice.
* Ignacio López.
* Internship students at ITC-DCCT. Alberto Sosa, Francisco Socorro, María Artiles, Carlos Caraballo, Ivet Cabrera.

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
