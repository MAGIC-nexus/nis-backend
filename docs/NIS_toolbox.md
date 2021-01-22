# NIS Toolbox

NIS (Nexus Information System) Toolbox is a software ecosystem developed inside MAGIC project (<a href="https://cordis.europa.eu/project/id/689669">H2020 grant #689669</a>). 

It was designed to enable "accounting the biophysical flows in complex bioeconomic systems" according to MuSIASEM concepts and methodology, as a way to assess the sustainability of current -or future- scenarios in socio-ecologic systems.

NIS Toolbox is made of four main components:

* **nis-backend** is the module in charge of processing MAGIC models (a format created during the project) via a suitable interface: it is programmatically actionable via a [RESTful API](restful_api.md), ([live](https://one.nis.magic-nexus.eu/nis_api/sources/-/databases/-) example -list of all available external datasets-), or the **nis-client**; or through the experimental graphical user interface **nis-frontend**).
* **nis-frontend** ([live](https://one.nis.magic-nexus.eu/nis_client/)) is a web application encompassing the different steps of the analysis, from the specification of the model to the management of results.  
* **nis-client**, can be used from scripts written in Python ([usage example](https://github.com/MAGIC-nexus/nis-python-client/blob/develop/nexinfosys/basic_usage2.py)) or R ([example usage](https://github.com/MAGIC-nexus/nis-python-client/blob/develop/nexinfosys/basic_usage.R)) languages.  
* **nis-eda** ([live](https://aware.nis.magic-nexus.eu/nis-eda/)) is an exploratory data analysis tool written in R and Shiny which processes analysis outputs (or connects to an instance of **nis-backend** to directly obtain the outputs to be explored).

Source code is open and published in Github, under "[MAGIC-nexus H2020](https://github.com/MAGIC-nexus)" organization. Non-publishable code (due to integration of third party commercial components) can be requested to the contact person (see [Social](social.md)).

NIS Toolbox is **open** also in the sense that it can be used to integrate MuSIASEM in reproducible analytic/scientific workflows.

## Documentation

Full documentation:

- How-to  (Annex 4 - "NIS frontend how-to manual").
- Technical reference (Annex 2 - "Report on NIsys toolset development").
- Format of input files (Annex 3 - "Format for specification of NIS case studies").

is available in the annexes of the deliverable of the project regarding "nis-backend", in the official web page, at http://www.magic-nexus.eu/documents/deliverable-33-report-datasets-and-software-tools.

NOTE: the "Format of input files" document is updated outside of the main document.

## Getting started

see [Getting started](getting_started.md)

## Configuration

see [Configuration](configuration.md)

## Social

see [Social](social.md)

## License
This project is licensed under the BSD-3 License - see the [LICENSE](../LICENSE) file for details

## Acknowledgements
The development of this software was supported by the European Union’s Horizon 2020 research and innovation programme under Grant Agreement No. 689669 (MAGIC). This work reflects the authors' view only; the funding agencies are not responsible for any use that may be made of the information it contains.

## Keywords

    Sustainability - Bioeconomy - Socio-Ecological Systems - Complex Adaptive Systems - Water-Energy-Food Nexus


