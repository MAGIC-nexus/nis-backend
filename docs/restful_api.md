# RESTful API
[nis-backend](https://github.com/MAGIC-nexus/nis-backend) is made accessible to external components through its RESTful API. It is implemented using Flask, which is very lightweight but powerful enough for what it is intended for (asynchronous systems are much more efficient but they are intended for thousands of simultaneous lightweight calls, while “nis-backend” is for a few heavy calls). All API calls are implemented in the module [service_main.py](https://github.com/MAGIC-nexus/nis-backend/blob/develop/nexinfosys/restful_service/service_main.py), together with other helper functions. An enumeration of the available calls follows, accompanied by a brief description of their purpose and parameters. Many are wrapped by [nis-client](https://github.com/MAGIC-nexus/nis-python-client).

POST **/isession**. Start an interactive session. Should be the first call.

GET **/isession**. Check if an interactive session is open.

DELETE **/isession/state**. Reset state in open interactive session.

PUT **/isession/identity**. Login as user:
* PUT **/isession/identity?user=...&password=...** The only user is “test_user”
* Add headers “token” and “service” for OAuth0 authentication.

GET **/isession/identity**. Get who is logged in.

DELETE **/isession/identity**. Logout.

POST **/isession/generator.json**. Convert input file to JSON.

POST **/isession/generator.to_dc.xml**. Obtain a simple Dublin Core XML record of the metadata.

POST **/isession/rsession**. Start a reproducible session. Query parameters:
* uuid. Of previously existing case study or case study version.
* read_version_state. True or False. Retrieve the state (or recalculate).
* create_new: “case_study”, “version”, “no”.
* allow_saving. If true, allow saving when closing the reproducible session.

GET **/isession/rsession**. Check if there is an open reproducible session.

POST **/isession/rsession/command**. Submit a single command.

POST **/isession/rsession/generator**. Submit and execute (parse and solve) a generator file (a NIS workbook) with multiple commands. The main call.

GET **/isession/rsession/command_generators/<order>**. Obtain one of the command generators registered in the reproducible session.

PUT **/isession/rsession/state**. Save the state directly to a file at “CASE_STUDIES_DIR” (configuration file).
* code. Name to assign to the state file. Should be unique (if repeated, it is overwritten).

GET **/isession/rsession/state**. Load the state directly from a file at “CASE_STUDIES_DIR” (configuration file).
* code. Name of the assumed existing state file.

GET **/isession/rsession/state.pickled**. Obtain a string with the serialized state.

GET **/isession/rsession/state_query/issues**. List of issues after execution.

GET **/isession/rsession/state_query/everything_executed**. True if all commands of current submission have been executed (useful to check the status of execution).

GET **/isession/rsession/state_query/outputs**. List of outputs available after an execution.

GET **/isession/rsession/state_query/parameters**. List of parameters, available after an execution to change them dynamically.

PUT **/isession/rsession/state_query/parameters**. Modify parameters and solve a dynamic scenario, generating special outputs.

GET **/isession/rsession/state_query/geolayer.<format>**. If execution was successful, generate the geolayer with location and attributes for Processors annotated with geographic information.

GET **/isession/rsession/state_query/ontology.<format>**. If execution was successful, generate an “.owl” file with an ontology representation of model and results.

GET **/isession/rsession/state_query/python_script.<format>**. If execution was successful, generate a Python or Jupyter notebook script.

GET **/isession/rsession/state_query/r_script.<format>**. If execution was successful, generate an R or Jupyter notebook script.

GET **/isession/rsession/state_query/model.<format>**. If execution was successful, generate the model in JSON or idempotent (NIS workbook) format.

GET **/isession/rsession/state_query/flow_graph.<format>**. If execution was successful, generate a flow graph in VisJS or GML format.

GET **/isession/rsession/state_query/processors_graph.visjs**. If execution was successful, generate a processors graph in VisJS format.

GET **/isession/rsession/state_query/sankey_graph.json**. If execution was successful, generate a JSON file which can be used to create a Sankey graph.

GET **/isession/rsession/state_query/datasets/<name>.<format>**. If execution was successful, prepare the specified dataset (name) in CSV or XLSX format.

DELETE **/isession/rsession**. Save and close a reproducible session.
* save_before_close. If true and “allow_saving” was true when opening the reproducible session, save.
* cs_uuid. If defined, save inside an existing case study.
* cs_name. If defined, assign it as the name of the case study version.

GET **/sources/**. External datasets. A list of sources (Eurostat, FAO, FADN, …)

GET **/sources/<source>**. External datasets. A list of databases of a source.

GET **/sources/<source>/databases/<database>**. External datasets. A list of datasets of a database in a source. If no database specified, all datasets of a source.

GET **/sources/<source>/databases/<database>/datasets/<dataset>**. External datasets. A JSON with metadata of a dataset in a database of a source.

PUT **/isession/external_xslx**. Retrieve, rewrite and download an Excel workbook from a URL.

POST **/isession/regenerate_xlsx**. Rewrite and download an Excel workbook passed as binary parameter.

GET **/commands_and_fields**. Help. Obtain a list of available commands and their fields. 

GET **/commands_reference.json**. Help. Obtain all the commands for the list . 

POST **/command_reference.json**. Help. Obtain help for one or more commands. 

POST **/command_fields_reference.json**. Help. Obtain help for one or more fields in commands. 

POST **/validate_command_record**. Lexically and syntactically validate one or more fields of a record in a command. Syntax validation.