from abc import ABCMeta, abstractmethod  # Abstract Base Class


class IExecutableCommand(metaclass=ABCMeta):
    """ A command prepared for its execution. Commands have direct access to the current STATE """

    @abstractmethod
    def execute(self, state: "State"):
        """ Execute """
        return None, None  # Issues, Output

    @abstractmethod
    def estimate_execution_time(self):
        pass

    @abstractmethod
    def json_serialize(self):
        pass

    @abstractmethod
    def json_deserialize(self, json_input):
        """
        Read command parameters from a JSON string
        After deserialize, the command can be executed
        
        :param json_input: JSON in a Unicode String 
        :return: --- (the object state is updated, ready for execution)
        """
        pass

"""

API
* Open/close interactive session
* Identify
* Open a reproducible session (optionally load existing commands/state). Close it, optionally save.
* CRUD case studies, versions and variables (objects)

* Browse datasets
* Browse case study objects: mappings, hierarchies, grammars

* Submit Worksheet
  - Interactive session
  - Open work session
  - Submit file
	- the file produces a sequence of commands
	- execute
	  - elaborate output file and compile issues
  - Close Interactive session (save, new case study version)
  - Close user session
* Submit R script
  - Interactive session
  - Open work session
  - Submit file
	- the file produces commands
	- execute
	  - elaborate output file ¿?, compile issues
  - Close work session (save, new case study version)
  - Close Interactive session
* Execute sparse commands (from R client, Python client, Google Sheets...)
  - Interactive session
  - parse then execute the command

* Export/import
  - Internal export. No need to be identified
  - Packaged case study export into Zenodo
  - Import case study (reverse of "internal export"). No need to be identified, the import needs a close work session with save or not
* Manage users
  - Link authenticators with identities
* Manage permissions
  - Share/unshare, READ, CONTRIBUTE, SHARE, EXPORT

RESTFul

* Open/close the interactive session
  - Open interactive, persisted session. Creation date registered. Register from where does the connection come
    (last call is also registered so after some time the session can be closed)
    POST /isession  -> return the ID, generate a COOKIE for the interactive session
  - Close interactive session
    DELETE /isession/ -> (the cookie has to be passed)
* Identify
  - Method 1. Through a OAuth2 token

USE CASES
* Start a case study from scratch
* Continue working on a case study. Implies saving the case study. A status flag for the case study (in elaboration, ready, publishable, ...)
* Play with an existing case study. CRUD elements, modify parameters, solve, analyze (read), ....
* Start an anonymous case study from scratch. It will not be saved
* Create case study from an existing one, as new version ("branch") or totally new case study
  *
* Analyze case study (open a case study, play with it)

"""

"""
Module containing high level function calls, controlling the different use cases
These function are called by the RESTful interface, which is the gate for the different other clients: R client, Web,...

The function allow a user to start a Work Session on a Case Study. All work is in memory. Specific functions allow
storing in database or in file.

A Work Session has a memory state related to one or more case studies, and a user. To shorten commands, there will be a
default case study.

* Queries (to database) encompassing several case studies? That would break the assumption of a case study per work
  session ¿do not assume that then? It implies having a default case study -context- to simplify commands
  * It would be like multi-database queries. The result is a valid structure, the retrieval process has to deal with opening
* Spin-off objects (objects born inside a case study which can be used in other case studies). Consider clone or
  reference options.


Use cases:
* Start a case study from scratch
* Continue working on a case study. Implies saving the case study. A status flag for the case study (in elaboration, ready, publishable, ...)
* Play with an existing case study. CRUD elements, modify parameters, solve, analyze (read), ....
* Start an anonymous case study from scratch. It will not be saved
* Create case study from an existing one, as new version ("branch") or totally new case study
  *
* Export case study for publication
* Export objects
* Analyze case study
* Import case study
* Import objects


Work Session level command types:
* Open a session
* Create a case study
* Load case study (from DB, from file)
* Clone case study. Specify if it is a new version or a new case study
  * If it is a new case study, reset metadata
* Save case study. Only for case studies with metadata.
  * The case study
  *
* Case study commands
  * Execute
  * Add to the case study


Case study level command types:
* Case study Metadata ("librarian/archivist")
* Data import ("traders"). Prepare external data for its use into the case study
* Specification ("writers"). Modify the structures and values composing the case study, including metadata of elements of the case study
* Solve ("info-reactors"). Deduce new information from existing structures/constraints
* Read/Query ("readers")
* Visualization ("story tellers")

(In-memory) Object types:
* Work session
  * Work session variables
* Case study
* Heterarchy, which includes hierarchy
  * Categories
  * Flows
  * Funds
* Transformation from a hierarchy to another
* Indicator + Benchmark
  * Metabolic rate
  * Many others
* Grammar. A grammar is specialized in a case study. It also serves as a set of constraints.
  * Changes should be allowed before "instantiation"
* Sequences connection
* Upscale transform
* Dataset
* Data process
* Dataset source
* NUSAP matrix or profile
* Analysis matrices: end use, environmental impact, externalization
* DPSIR
* Clustering


The main objects are:
* Work session
* Case study
* Command

A work session can hold a registry of several commands.
A case study contains objects generated by commands. ¿References
A command can generate commands
A persisted case study must register sessions. It must register either a sequence of commands or a generator of commands (an Excel file, an R script). The three possibilities

"""


"""
The higher level object is a case study
It can be embedded in a file or registered
Each case study can have several IDs: OID, UUID, name
A sequence of commands
Modes of work
* exploratory elaboration of a version of a case study
* review of a version. READ ONLY commands, not added to the case study itself
* continue elaborating version of a case study (is a new version?)
* 

Commands. A case study is evolved through commands which produce MuSIASEM primitives and provoke moving them, by ETL, by solving, output, etc.
Commands need to be adapted to the different interfaces, like Excel or R
Commands are executed in sequence, in transactions (the commands are grouped, then submitted in a pack)
Special commands to signal START of SUBMISSION and END of SUBMISSION of case study
Commands may be stored under a registered case study or simply executed and forgotten

Command
Several types. Compatibility with tool, ETL/specification/solving/visualization/export, process control (new case study, new submission, new transaction, login) or not
Metadata
Hierarchy

Pre-steps (in no particular order)
* Categories, taxonomies
* Mappings
* Dataset
*

Steps
* Identify compartments
* Identify relations
* Values
* Reasoning-Solving

"""

"""
* angular
* pyparsing - antlr
* python avanzado
* selenium

* estructuras, api, comandos, formato excel, persistencia

* estructura del proyecto en backend, incluyendo Python y R
* jupyterhub
* interacción (análisis), empaquetado para reproducibilidad y para demostración

* interface: restful. authentication (OAuth, JWS, OAuth2, other?)
* interface: navigation diagram
* excel: 
* commands: specification
* model: enumerate
* patterns: command, data structures, trace from command to generated entities, duck typing, jobs execution queue
* requirements: list of commands
* project structure. multiple languages, multiple modules, tests, README, setup, other?
* taiga 

"""

"""
almacenes (server URL)
casos de estudio (UUID)
envíos (ID, relative to the case study)
transacciones (ID, relative to the transaction)
comandos (ID, relative to the transaction)
primitivas (ID, relative to the command)

proc
fact
pf
valor. expresión

jerarq o lista
procs en jerarquía
fact en jerarquía


declare hierarchy1
add p
p is h1.asdf
p is h2.wer
p [p1 p2]
p3 [p1 p4]
(p1 € p y p3)

f [f1 f2]

add q

p.f1=3 m³*.F2
p.F2=5 m²

p1 > p2
p1.f1 > p2.f1

p1.f1 = (QQ)
p1.

swagger r
list commands, find an Excel expression

"""


