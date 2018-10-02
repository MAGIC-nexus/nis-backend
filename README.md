# nis-backend

## Technological Features of NIS tool

**DONE**
* Dockerized: a Dockerfile for Apache2, receiving the name of the configuration file on image creation
  * One of the configuration is devised to be almost autonomous: it needs only a REDIS instance. It is based on
  using SQLite. And because the Dockerfile exposes "/srv" for volumes, the SQLite databases persist between container restarts.
  * To solve: all configurations ARE inside the image, use a VOLUME to store the configuration and look
    for the configuration file there
* RESTful web service, allowing multiple language/system bindings
  * Interactive session, with different identification (authentication) systems
  * ACL for case studies
  * Case study management: creation, versioning and access
* Frontend published through the Web Service, in a separate Path: /nis_client/index.html
  * The Angular2 frontend has to be built (ng build --prod --aot --base-href /nis_client/) then the result
  has to be copied to the "frontend" directory. After this, the Dockerfile can pack things properly. 
* Serialization/deserialization of complex web session (to support web services), using:
  * jsonpickle
  * Custom serialization for SQLAlchemy objects representing Persistent Sessions
  * Flask-Session
* Separate Metadata and Data databases. The first contains all MuSIASEM things, plus datasets metadata. The second aims
  to be an OLAP repository.
* Access to REDIS to persist temporarily open interactive sessions (needed because HTTP is stateless)
* Web service for use from different domain (like official web site). Using CORS
* Script automating the deployment of related Docker containers: Databases, Redis, Nextcloud
  - Not published for now. It is called "run-data-management-tools.py", can be requested.

**TO-DO**
* Possibility to execute R scripts in a specialized Docker container, which provides a sandbox
  * Access to resources should be through URLs, not file system PATHs. Or standardized file system PATHs
* MonetDB Docker used as data repository
* Dockerized: a Dockerfile for NGinx, allowing the execution of Celery workers for long executions and background jobs
* OAuth2 tokens validation using third parties (Google)
* WebDAV access to NextCloud to store files and prepare case studies in DMP format
* Configuration file divided in sections, .INI file style.

## Features useful for NIS construction

**DONE**
* Automated Test Units enabled, and prepared
  * API (directly)
  * In Process Web Service (API through RESTful, needs REDIS)
  * Separate Process Web Service (API through RESTful, needs REDIS)

* Locally executable, to ease debugging

**TO-DO**
* Web service specified through Swagger, allowing quick creation/maintenance of clients
