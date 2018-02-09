import io
import os
import sys
import redis
import logging
from pathlib import Path
import sqlalchemy.schema
from sqlalchemy.pool import StaticPool
# from flask import (jsonify, abort, redirect, url_for,
#
#                    )
from flask import (Response, request, session as flask_session, send_from_directory
                   )
from flask.helpers import get_root_path
from flask_session import Session as FlaskSessionServerSide
from flask_cors import CORS
from werkzeug.exceptions import NotFound
import json


# >>>>>>>>>> IMPORTANT <<<<<<<<<
# For debugging in local mode, prepare an environment variable "MAGIC_NIS_SERVICE_CONFIG_FILE", with value "./nis_local.conf"
# >>>>>>>>>> IMPORTANT <<<<<<<<<
if __name__ == '__main__':
    print("Executing locally!")
    os.environ["MAGIC_NIS_SERVICE_CONFIG_FILE"] = "./nis_local.conf"

from backend.common.helper import generate_json, strcmp, obtain_dataset_source
from backend.model.persistent_db.persistent import *
from backend.common.create_database import create_pg_database_engine, create_monet_database_engine
from backend.restful_service import app
import backend
from backend.command_generators.json import create_command
from backend.command_executors.specification.metadata_command import generate_dublin_core_xml
from backend.model_services import State
from backend.model_services.workspace import InteractiveSession, CreateNew, ReproducibleSession, \
    execute_command_container, convert_generator_to_native
from backend.restful_service import nis_api_base, nis_client_base, nis_external_client_base, log_level, \
    tm_default_users, \
    tm_authenticators, \
    tm_object_types, \
    tm_permissions, \
    tm_case_study_version_statuses
from backend.restful_service.serialization import serialize, deserialize
from backend.external_data.data_source_manager import DataSourceManager
from backend.external_data.data_sources.eurostat_bulk import Eurostat
from backend.external_data.data_sources.faostat import FAOSTAT


# #####################################################################################################################
# >>>> BOOT TIME. FUNCTIONS AND CODE <<<<
# #####################################################################################################################


def initialize_database_data():
    # Load base tables
    load_table(DBSession, User, tm_default_users)
    load_table(DBSession, Authenticator, tm_authenticators)
    load_table(DBSession, CaseStudyStatus, tm_case_study_version_statuses)
    load_table(DBSession, ObjectType, tm_object_types)
    load_table(DBSession, PermissionType, tm_permissions)
    # Create and insert a user
    session = DBSession()
    # Create test User, if it does not exist
    u = session.query(User).filter(User.name == 'test_user').first()
    if not u:
        u = User()
        u.name = "test_user"
        u.uuid = "27c6a285-dd80-44d3-9493-3e390092d301"
        session.add(u)
        session.commit()
    DBSession.remove()

# Initialize DATABASEs


def initialize_databases():
    recreate_db = False
    if 'DB_CONNECTION_STRING' in app.config:
        db_connection_string = app.config['DB_CONNECTION_STRING']
        print("Connecting to metadata server")
        print(db_connection_string)
        print("-----------------------------")
        if db_connection_string.startswith("sqlite://"):
            backend.engine = sqlalchemy.create_engine(db_connection_string,
                                                      echo=True,
                                                      connect_args={'check_same_thread': False},
                                                      poolclass=StaticPool)
        else:
            backend.engine = create_pg_database_engine(db_connection_string, "magic_nis", recreate_db=recreate_db)

        # global DBSession # global DBSession registry to get the scoped_session
        DBSession.configure(bind=backend.engine)  # reconfigure the sessionmaker used by this scoped_session
        tables = ORMBase.metadata.tables
        connection = backend.engine.connect()
        table_existence = [backend.engine.dialect.has_table(connection, tables[t].name) for t in tables]
        connection.close()
        if False in table_existence:
            ORMBase.metadata.bind = backend.engine
            ORMBase.metadata.create_all()
        # connection = backend.engine.connect()
        # table_existence = [backend.engine.dialect.has_table(connection, tables[t].name) for t in tables]
        # connection.close()
        # Load base tables
        initialize_database_data()
    else:
        print("No database connection defined (DB_CONNECTION_STRING), exiting now!")
        sys.exit(1)

    if 'DATA_CONNECTION_STRING' in app.config:
        data_connection_string = app.config['DB_CONNECTION_STRING']
        print("Connecting to data server")
        if data_connection_string.startswith("monetdb"):
            backend.data_engine = create_monet_database_engine(data_connection_string, "magic_data")
        elif data_connection_string.startswith("sqlite://"):
            backend.data_engine = sqlalchemy.create_engine(data_connection_string,
                                                           echo=True,
                                                           connect_args={'check_same_thread': False},
                                                           poolclass=StaticPool)
        else:
            backend.data_engine = create_pg_database_engine(data_connection_string, "magic_data", recreate_db=recreate_db)
    else:
        print("No data connection defined (DATA_CONNECTION_STRING), exiting now!")
        sys.exit(1)


def register_external_datasources(cfg):
    dsm2 = DataSourceManager(session_factory=DBSession)
    dsm2.register_datasource_manager(Eurostat())
    if 'FAO_DATASETS_DIR' in cfg:
        fao_dir = cfg['FAO_DATASETS_DIR']
    else:
        fao_dir = "/home/rnebot/DATOS/FAOSTAT/"
    dsm2.register_datasource_manager(FAOSTAT(datasets_directory=fao_dir,
                                           metadata_session_factory=DBSession,
                                           data_engine=backend.data_engine))

    sources = dsm2.get_supported_sources()
    return dsm2


def connect_redis():
    # A REDIS instance needs to be available. Check it
    # A local REDIS could be as simple as:
    #
    # docker run --rm -p 6379:6379 redis:alpine
    #
    if 'REDIS_HOST' in app.config:
        r_host = app.config['REDIS_HOST']
        if r_host == "redis_lite":
            import redislite
            rs2 = redislite.Redis(serverconfig={'port': '6379'})
        else:
            rs2 = redis.Redis(r_host)
        try:
            print("Trying connection to REDIS '"+r_host+"'")
            rs2.ping()
            print("Connected to REDIS instance '"+r_host+"'")
        except:
            print("REDIS instance '"+r_host+"' not reachable, exiting now!")
            sys.exit(1)
    else:
        print("No REDIS instance configured, exiting now!")
        sys.exit(1)
    return rs2

# #####################################################################################################################
# >>>> THE INITIALIZATION CODE <<<<
#


initialize_databases()
backend.data_source_manager = register_external_datasources(app.config)
backend.redis = connect_redis()

# Now initialize Flask-Session, using the REDIS instance
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_KEY_PREFIX"] = "nis:"
app.config["SESSION_PERMANENT"] = False
#app.config["PERMANENT_SESSION_LIFETIME"] = 3600
app.config["SESSION_REDIS"] = backend.redis

FlaskSessionServerSide(app)
CORS(app,
     resources={r"/nis_api/*": {"origins": "http://localhost:4200"}},
     supports_credentials=True
     )

logger = logging.getLogger(__name__)
logging.getLogger('flask_cors').level = logging.DEBUG
app.logger.setLevel(log_level)
logger.setLevel(log_level)

# #####################################################################################################################
# >>>> UTILITY FUNCTIONS <<<<
# #####################################################################################################################


def reset_database():
    """
    Empty ALL data in the database !!!!

    Used in testing web services

    :return:
    """
    if is_testing_enabled():
        connection2 = backend.engine.connect()
        tables = ORMBase.metadata.tables
        table_existence = [backend.engine.dialect.has_table(connection2, tables[t].name) for t in tables]
        connection2.close()
        if False in table_existence:
            ORMBase.metadata.bind = backend.engine
            ORMBase.metadata.create_all()

    for tbl in reversed(ORMBase.metadata.sorted_tables):
        backend.engine.execute(tbl.delete())


def build_json_response(obj, status=200):
    return Response(generate_json(obj),
                    mimetype="text/json",
                    status=status)


def serialize_isession_and_close_db_session(sess: InteractiveSession):
    # Serialize WorkSession apart, if it exists
    if sess._reproducible_session:
        csvs = sess._reproducible_session._session
        o_list = [csvs.version.case_study, csvs.version, csvs]
        o_list.extend(csvs.commands)
        d_list = serialize(o_list)
        s = jsonpickle.encode({"allow_saving": sess._reproducible_session._allow_saving, "pers": d_list})
        flask_session["rsession"] = s
        sess._reproducible_session = None
    else:
        if "rsession" in flask_session:
            del flask_session["rsession"]

    tmp = sess.get_sf()
    sess.set_sf(None)
    sess._reproducible_session = None
    # Serialize sess._state and sess._identity
    s = jsonpickle.encode(sess)
    flask_session["isession"] = s
    sess.set_sf(tmp)
    sess.close_db_session()


def deserialize_isession_and_prepare_db_session(return_error_response_if_none=True) -> InteractiveSession:
    if "isession" in flask_session:
        s = flask_session["isession"]
        sess = jsonpickle.decode(s)
        sess.set_sf(DBSession)
        if "rsession" in flask_session:
            rs = ReproducibleSession(sess)
            rs.set_sf(sess.get_sf())
            d = jsonpickle.decode(flask_session["rsession"])
            rs._allow_saving = d["allow_saving"]
            o_list = deserialize(d["pers"])
            rs._session = o_list[2]  # type: CaseStudyVersionSession
            sess._reproducible_session = rs
    else:
        sess = None

    if not sess and return_error_response_if_none:
        return NO_ISESS_RESPONSE
    else:
        return sess


def is_testing_enabled():
    if "TESTING" in app.config:
        if isinstance(app.config["TESTING"], bool):
            testing = app.config["TESTING"]
        else:
            testing = app.config["TESTING"].lower() in ["true", "1"]
    else:
        testing = False
    return testing


NO_ISESS_RESPONSE = build_json_response({"error": "No interactive session active. Please, open one first ('POST /isession')"}, 400)

# >>>> SPECIAL FUNCTIONS <<<<


# @app.before_request
# def print_headers():
#     print("HEADER Authorization")
#     found = False
#     for h in request.headers:
#         if h[0] in ["Authorization", "Autorizacion"]:
#             print(h[0] + ": " + str(h[1]))
#             found = True
#     if not found:
#         print("-- not sent --")


@app.after_request
def after_a_request(response):
    for i in request.cookies.items():
        response.set_cookie(i[0], i[1])

    if "__invalidate__" in flask_session:
        response.delete_cookie(app.session_cookie_name)

    return response

# #####################################################################################################################
# >>>> SERVE ANGULAR2 CLIENT FILES <<<<
# #####################################################################################################################


@app.route(nis_client_base + "/<path:path>", methods=["GET"])
@app.route(nis_external_client_base + "/<path:path>", methods=["GET"])
def send_web_client_file(path):
    """
    Serve files from the Angular2 client
    To generate these files (ON EACH UPDATE TO THE CLIENT:
    * CD to the Angular2 project directory
    * ng build --prod --aot --base-href /nis_client/
    * CP * <FRONTEND directory>

    :param path:
    :return:

    """
    base = Path(get_root_path("backend.restful_service"))
    base = str(base.parent.parent)+"/frontend"
    # logger.debug("BASE DIRECTORY: "+base)
    incoming_url = request.url_rule.rule
    if nis_external_client_base in incoming_url:
        # From outside
        if path == "index.html":
            # TODO Possibility of changing both the base and the file name
            # TODO The intention is to NOT show the "Login" possibilities, so
            # TODO users are always anonymous. To be discussed.
            base = get_root_path("clients/web")
            new_name = "index.html"
        else:
            new_name = path
    else:
        # From inside
        new_name = path

    try:
        return send_from_directory(base, new_name)
    except NotFound:
        return send_from_directory(base, "index.html")


# #####################################################################################################################
# >>>> SERVE STATIC FILES <<<<
# #####################################################################################################################


@app.route(nis_api_base + "/static/<path:path>", methods=["GET"])
def send_static_file(path):
    """
    Serve files from the Angular2 client
    To generate these files (ON EACH UPDATE TO THE CLIENT:
    * CD to the Angular2 project directory
    * ng build --prod --aot --base-href /nis_client/
    * CP * <FRONTEND directory>

    :param path:
    :return:
    """
    base = Path(get_root_path("backend.restful_service"))
    base = str(base)+"/static"
    # logger.debug("BASE DIRECTORY: "+base)

    return send_from_directory(base, path)

# #####################################################################################################################
# >>>> RESTFUL INTERFACE <<<<
# #####################################################################################################################

# -- Interactive session --


@app.route(nis_api_base + "/resetdb", methods=["POST"])
def reset_db():
    testing = is_testing_enabled()
    if testing:
        reset_database()
        initialize_database_data()
        interactive_session_close()  # Leave session if already in
        r = build_json_response({}, 204)
    else:
        r = build_json_response({"error": "Illegal operation!!"}, 400)

    return r


@app.route(nis_api_base + "/isession", methods=["POST"])
def interactive_session_open():
    isess = deserialize_isession_and_prepare_db_session(False)
    if isess:
        r = build_json_response({"error": "Close existing interactive session ('DELETE /isession'"}, 400)
    else:
        isess = InteractiveSession(DBSession)
        serialize_isession_and_close_db_session(isess)
        r = build_json_response({}, 204)
    return r


# Set identity at this moment for the interactive session
@app.route(nis_api_base + "/isession/identity", methods=["PUT"])
def interactive_session_set_identity():
    # Recover InteractiveSession
    # if request.method=="OPTIONS":
    #     r = build_json_response({}, 200)
    #     h = r.headers
    #     h['Access-Control-Allow-Origin'] = "http://localhost:4200"
    #     h['Access-Control-Allow-Methods'] = "PUT,POST,DELETE,GET,OPTIONS"
    #     h['Access-Control-Max-Age'] = str(21600)
    #     h['Access-Control-Allow-Credentials'] = "true"
    #     h['Access-Control-Allow-Headers'] = "Content-Type, Authorization, Content-Length, X-Requested-With"
    #     return r

    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # If there is a current identity, issue an error. First "unidentify"
    if isess.get_identity_id():

        testing = is_testing_enabled()
        if testing and request.args.get("user") and isess.get_identity_id() == request.args.get("user"):
            result = True
        else:
            result = False
    else:
        # Two types of identification: external, using OAuth tokens, or application, using user+password
        application_identification = True
        if application_identification:
            if request.args.get("user"):
                testing = is_testing_enabled()
                result = isess.identify({"user": request.args.get("user"),
                                         "password": request.args.get("password", None)
                                         },
                                        testing=testing
                                        )
        else:
            # TODO Check the validity of the token using the right Authentication service
            result = isess.identify({"token": request.headers.get("token"),
                                     "service": request.headers.get("auth_service")
                                     }
                                    )
    serialize_isession_and_close_db_session(isess)

    r = build_json_response({"identity": isess.get_identity_id()} if result else {},
                            200 if result else 401)

    return r


@app.route(nis_api_base + "/isession/identity", methods=["GET"])
def interactive_session_get_identity():
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    return build_json_response({"identity": isess.get_identity_id()})


# Set to anonymous user again (or "logout")
@app.route(nis_api_base + "/isession/identity", methods=["DELETE"])
def interactive_session_remove_identity():
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # Un-identify
    if isess.get_identity_id():

        isess.unidentify()
        serialize_isession_and_close_db_session(isess)

    return build_json_response({"identity": isess.get_identity_id()})


# Close interactive session (has to log out if some identity is active)
@app.route(nis_api_base + "/isession", methods=["DELETE"])
def interactive_session_close():
    isess = deserialize_isession_and_prepare_db_session(False)

    if isess:
        isess.quit()

    flask_session.clear()
    flask_session["__invalidate__"] = True
    return build_json_response({})


@app.route(nis_api_base + '/isession/generator.json', methods=['POST'])
def convert_generator_to_json_generator():
    """
    Send the file to the service
    Convert to native
    Return it in JSON format

    :return:
    """
    # Check Interactive Session is Open. If not, open it
    isess = deserialize_isession_and_prepare_db_session(False)
    if not isess:
        isess = InteractiveSession(DBSession)

    testing = is_testing_enabled()
    if testing:
        result = isess.identify({"user": "test_user", "password": None}, testing=True)

    # Receive file
    generator_type, content_type, buffer, _, _ = receive_file_submission(request)

    output = convert_generator_to_native(generator_type, content_type, buffer)

    # Return the conversion
    r = build_json_response(output, 200)

    serialize_isession_and_close_db_session(isess)

    return r


@app.route(nis_api_base + '/isession/generator.to_dc.xml', methods=['POST'])
def convert_generator_to_dublin_core():
    """
    Send the file to the service
    Convert to native
    Return the Dublin Core XML record

    :return:
    """
    # Check Interactive Session is Open. If not, open it
    isess = deserialize_isession_and_prepare_db_session(False)
    if not isess:
        isess = InteractiveSession(DBSession)

    testing = is_testing_enabled()
    if testing:
        result = isess.identify({"user": "test_user", "password": None}, testing=True)

    # Receive file
    generator_type, content_type, buffer, _, _ = receive_file_submission(request)

    output = convert_generator_to_native(generator_type, content_type, buffer)
    xml = None
    for c in output:
        if "command" in c and c["command"] == "metadata" and "content" in c:
            xml = generate_dublin_core_xml(c["content"])
            break

    # Return the conversion
    if xml:
        r = Response(xml,
                     mimetype="text/xml",
                     status=200)
    else:
        r = build_json_response({"message": "Could not elaborate Dublin Core XML record from the input generator"}, 401)

    serialize_isession_and_close_db_session(isess)

    return r

# -- Reproducible Sessions --


@app.route(nis_api_base + "/isession/rsession", methods=["POST"])
def reproducible_session_open():
    def read_parameters(dd):
        nonlocal uuid2, read_uuid_state, create_new, allow_saving
        # Read query parameters
        uuid2 = dd.get("uuid")
        if "read_version_state" in dd:
            read_uuid_state = dd["read_version_state"]
            read_uuid_state = bool(read_uuid_state)
        if "create_new" in dd:
            create_new = dd["create_new"]
            if create_new.lower() in ["case_study", "casestudy"]:
                create_new = CreateNew.CASE_STUDY
            elif create_new.lower() in ["version", "case_study_version"]:
                create_new = CreateNew.VERSION
            else:
                create_new = CreateNew.NO
        if "allow_saving" in dd:
            allow_saving = dd["allow_saving"]
            allow_saving = allow_saving.lower() == "true"

    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    uuid2 = None
    read_uuid_state = None
    create_new = None
    allow_saving = None

    # First, read uploaded JSON
    if len(request.files) > 0:
        for k in request.files:
            buffer = bytes(request.files[k].stream.getbuffer())
            content_type = request.files[k].content_type
            break
    else:
        buffer = bytes(io.BytesIO(request.get_data()).getbuffer())
        content_type = request.headers["Content-Type"]

    if buffer:
        read_parameters(json.loads(buffer))
    if not uuid2 and not read_uuid_state and not create_new and not allow_saving:
        read_parameters(request.form)
        if not uuid2 and not read_uuid_state and not create_new and not allow_saving:
            read_parameters(request.args)

    if not read_uuid_state:
        read_uuid_state = True
    if not create_new:
        create_new = CreateNew.NO
    if not allow_saving:
        allow_saving = True

    # Persistent object to open: None (new case study), UUID (case study version)
    if isess.reproducible_session_opened():
        r = build_json_response({"error": "There is an open Reproducible Session. Close it first."}, 401)
    else:
        try:
            isess.open_reproducible_session(case_study_version_uuid=uuid2,
                                            recover_previous_state=read_uuid_state,
                                            cr_new=create_new,
                                            allow_saving=allow_saving
                                            )
            r = build_json_response({}, 204)
        except Exception as e:
            s = "Exception trying to open reproducible session: "+str(e)
            logger.error(s)
            r = build_json_response({"error": s}, 401)

    #
    serialize_isession_and_close_db_session(isess)
    return r


@app.route(nis_api_base + "/isession/rsession", methods=["DELETE"])
def reproducible_session_save_close():  # Close the ReproducibleSession, with the option of saving it
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # A reproducible session must be open, signal it if not,
    save = request.form.get("save_before_close", "False")
    if save:
        save = bool(save)

    # Close reproducible session
    if not isess.reproducible_session_opened():
        r = build_json_response({"error": "There is no open Reproducible Session. Cannot close"}, 401)
    else:
        try:
            uuid_, v_uuid, cs_uuid = isess.close_reproducible_session(issues=None, output=None, save=save, from_web_service=True)
            r = build_json_response({"session_uuid": str(uuid_),
                                     "version_uuid": str(v_uuid),
                                     "case_study_uuid": str(cs_uuid)
                                     },
                                    200)
        except Exception as e:
            s = "Exception trying to close reproducible session: " + str(e)
            logger.error(s)
            r = build_json_response({"error": s}, 401)

    serialize_isession_and_close_db_session(isess)
    return r


@app.route(nis_api_base + "/isession/rsession", methods=["GET"])
def reproducible_session_get_status():  # Return current status of ReproducibleSession
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # A reproducible session must be open, signal about it if not
    if isess.reproducible_session_opened():
        r = build_json_response("Opened Reproducible Session", 200)
    else:
        r = build_json_response("No open reproducible Session", 200)

    return r


@app.route(nis_api_base + "/isession/rsession/pyckled_state", methods=["GET"])
def reproducible_session_get_state():  # Return current status of ReproducibleSession
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # A reproducible session must be open, signal about it if not
    if isess.reproducible_session_opened():
        if isess._state:
            r = build_json_response(jsonpickle.encode(isess._state), 200)
        else:
            r = build_json_response({}, 204)
    else:
        r = build_json_response({"error": "Cannot return state, no opened reproducible session"}, 401)

    return r


@app.route(nis_api_base + "/isession/rsession/command", methods=["POST"])
def reproducible_session_append_single_command():  # Receive a JSON or CSV command from some externally executed generator

    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # A reproducible session must be open
    if isess.reproducible_session_opened():
        # Read content type header AND infer "generator_type"
        content_type = request.headers["Content-Type"]
        if content_type.lower() in ["application/json", "text/json", "text/csv"]:
            generator_type = "native"

        # Read binary content
        if len(request.files) > 0:
            for k in request.files:
                buffer = bytes(request.files[k].stream.getbuffer())
                break
        else:
            buffer = bytes(io.BytesIO(request.get_data()).getbuffer())

        # Read Execute and Register parameters
        execute = request.args.get("execute", "True")
        if execute:
            execute = bool(execute)
        register = request.args.get("register", "True")
        if register:
            register = bool(register)

        if isinstance(buffer, bytes):
            d = buffer.decode("utf-8")
        else:
            d = buffer
        d = json.loads(d)
        if isinstance(d, dict) and "command" in d and "content" in d:
            if "label" in d:
                n = d["label"]
            else:
                n = None
            cmd, syntax_issues = create_command(d["command"], n, d["content"])

        if register:
            isess.register_executable_command(cmd)
        if execute:
            issues, output = isess.execute_executable_command(cmd)
            # TODO Process "ret". Add issues to an issues list. Add output to an outputs list.

        r = build_json_response({}, 204)
        serialize_isession_and_close_db_session(isess)
    else:
        r = build_json_response({"error": "A reproducible session must be open in order to submit a command"}, 400)

    return r


def receive_file_submission(req):
    """
    Receive file submitted using multipart/form-data
    Return variables for the processing of the file as a command_executors generator

    :param req: The "request" object
    :return: A tuple (generator_type -str-, content_type -str-, buffer -bytes-, execute -bool-, register -bool-)
    """
    # Read binary content
    if len(req.files) > 0:
        for k in req.files:
            buffer = bytes(req.files[k].stream.getbuffer())
            content_type = req.files[k].content_type
            break
    else:
        buffer = bytes(io.BytesIO(req.get_data()).getbuffer())
        content_type = req.headers["Content-Type"]

    # Infer "generator_type" from content type
    if content_type.lower() in ["application/json", "text/csv"]:
        generator_type = "primitive"
    elif content_type.lower() in ["application/excel",
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        generator_type = "spreadsheet"
    elif content_type.lower() in ["text/x-r-source"]:
        generator_type = "R-script"
    elif content_type.lower() in ["text/x-python", "text/x-python3", "application/x-python3"]:
        generator_type = "python-script"

    # # Write to file
    # with open("/home/rnebot/output_file.xlsx", "wb") as f:
    #     f.write(buffer)

    # Read Register and Execute parameters
    register = req.form.get("register")
    if not register:
        register = req.args.get("register")
        if not register:
            register = False
    execute = req.form.get("execute")
    if not execute:
        execute = req.args.get("execute")
        if not execute:
            execute = False
    execute = bool(execute)
    register = bool(register)

    return generator_type, content_type, buffer, execute, register


@app.route(nis_api_base + "/isession/rsession/generator", methods=["POST"])
def reproducible_session_append_command_generator():  # Receive a command_executors generator, like a Spreadsheet file, an R script, or a full JSON command_executors list (or other)
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # A reproducible session must be open
    if isess.reproducible_session_opened():
        generator_type, content_type, buffer, execute, register = receive_file_submission(request)

        # EXECUTE IT!!!
        ret = isess.register_andor_execute_command_generator(generator_type, content_type, buffer, register, execute)
        # TODO Return the issues if there were any. Return outputs (could be a list of binary files)
        r = build_json_response({}, 204)

        # TODO Important!!! The R script generator can be executed remotely and locally. In the first case, it
        # TODO could be desired to store commands. But the library, when executed at the server, will be passed a flag
        # TODO to perform every call with the registering disabled.
        serialize_isession_and_close_db_session(isess)
    else:
        r = build_json_response({"error": "A reproducible session must be open in order to submit a generator"}, 400)

    return r

# -- Case studies --


@app.route(nis_api_base + '/case_studies/', methods=['POST'])
def new_case_study_from_file():
    """
    Check that the user is authorized to submit a new case study
    Open a reproducible session
    Send the file to the service
    Close the reproducible session

    :return:
    """
    # Check Interactive Session is Open. If not, open it
    isess = deserialize_isession_and_prepare_db_session(False)
    if not isess:
        isess = InteractiveSession(DBSession)

    # TODO Check User Credentials (from Token)
    testing = is_testing_enabled()
    if testing:
        result = isess.identify({"user": "test_user", "password": None}, testing=True)

    # TODO Check User has Create New Case Study permissions

    # Receive file
    generator_type, content_type, buffer, execute, register = receive_file_submission(request)

    # Open Reproducible Session, NEW case study
    try:
        isess.open_reproducible_session(case_study_version_uuid=None,
                                        recover_previous_state=False,
                                        cr_new=CreateNew.CASE_STUDY,
                                        allow_saving=register
                                        )
    except Exception as e:
        s = "Exception trying to open reproducible session: "+str(e)
        logger.error(s)
        return build_json_response({"error": s}, 401)

    # Submit file to the Interactive Session (which has the open reproducible session)
    issues, output = isess.register_andor_execute_command_generator(generator_type, content_type, buffer, register, execute)

    # Close Reproducible Session
    isess.close_reproducible_session(issues=issues, output=output, save=register, from_web_service=False)

    # TODO Return the issues if there were any. Return outputs (could be a list of binary files)
    r = build_json_response({}, 204)

    serialize_isession_and_close_db_session(isess)

    return r


@app.route(nis_api_base + "/case_studies/", methods=["GET"])
def case_studies():  # List case studies
    """
Example:
[
{"resource": "/case_studies/<case study uuid>",
 "uuid": "<uuid>",
 "name": "Food in the EU",
 "oid": "zenodo.org/2098235",
 "internal_code": "CS1_F_E",
 "description": "...",
 "stats":
  {
   "n_versions": "<# of versions>",
   "n_commands": "<# of command_executors latest version>",
   "n_hierarchies": <# of hierarchies latest version>",
  }
 "versions": "/case_studies/<uuid>/short.json"
 "thumbnail": "/case_studies/<uuid>/thumbnail.svg|html|png"
},
...
]

    :return:
    """

    def get_avatar_path(cstudy):
        """
        From the areas of a case study, obtain the file name representing these areas

        :param cstudy: CaseStudy object
        :return: String with the URL subpath to the file name
        """
        areas = cstudy.areas
        if areas:
            name = ""
            if "W" in areas:
                name += "Water"
            if "E" in areas:
                name += "Energy"
            if "F" in areas:
                name += "Food"
            return "/static/images/" + name + "Nexus.png"
        else:
            return "/static/images/NoNexusAreas.png"  # TODO create this image

    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()

    if not user:
        user = "_anonymous"
    # Recover case studies READABLE by current user (or "anonymous")
    if not isess:
        session = DBSession()
    else:
        session = isess.open_db_session()
    # TODO Obtain case studies FILTERED by current user permissions. Show case studies with READ access enabled
    # Access Control
    # CS in acl and user in acl.detail and acl.detail is READ, WRITE,
    # CS in acl and group acl.detail and user in group
    base = app.config["APPLICATION_ROOT"]
    lst = session.query(CaseStudy).all()
    lst2 = []
    for cs in lst:
        uuid2 = str(cs.uuid)
        d = {"resource": nis_api_base + "/case_studies/"+uuid2,
             "uuid": uuid2,
             "name": cs.name if cs.name else "<empty>",
             "oid": cs.oid if cs.oid else "<empty>",  # TODO
             "internal_code": cs.internal_code if cs.internal_code else "",  # TODO
             "description": cs.description if cs.description else "",  # TODO
             "stats": {
                 "n_versions": str(len(cs.versions)),
                 "n_commands": str(len([])),  # TODO
                 "n_hierarchies": str(len([])),  # TODO
             },
             "versions": nis_api_base + "/case_studies/" + uuid2 + "/versions/",
             "thumbnail": nis_api_base + "/case_studies/" + uuid2 + "/default_view.png",
             "thumbnail_png": nis_api_base + "/case_studies/" + uuid2 + "/default_view.png",
             "thumbnail_svg": nis_api_base + "/case_studies/" + uuid2 + "/default_view.svg",
             "avatar": nis_api_base + get_avatar_path(cs),  # Icon representing the type of Nexus study
             "case_study_permissions":
                 {
                     "read": True,
                     "annotate": True,
                     "contribute": True,
                     "share": False,
                     "delete": False
                 }
             }
        lst2.append(d)
        # print(json.dumps(lst2, default=json_serial, sort_keys=True, indent=JSON_INDENT, ensure_ascii=ENSURE_ASCII, separators=(',', ': '))
        #       )
    r = build_json_response(lst2)  # TODO Improve it, it must return the number of versions. See document !!!
    if isess:
        isess.close_db_session()
    else:
        DBSession.remove()

    return r


@app.route(nis_api_base + "/case_studies/<cs_uuid>", methods=["GET"])
@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/", methods=["GET"])
def case_study(cs_uuid):  # Information about case study
    """

{"case_study": "<uuid>",
 "name": "Food in the EU",
 "oid": "zenodo.org/2098235",
 "internal_code": "CS1_F_E",
 "resource": "/case_studies/<case study uuid>",
 "description": "...",
 "versions":
 [
  {"uuid": "<uuid>",
   "resource": "/case_studies/<case study uuid>/<version uuid>",
   "tag": "v0.1",
   "sessions":
   [
    {"uuid": "<uuid>",
     "open_date": "2017-09-20T10:00:00Z",
     "close_date": "2017-09-20T10:00:10Z",
     "client": "spreadsheet",
     "restart": True,
     "author": "<uuid>",
    },
    ...
   ]
   "detail": "/case_studies/<case study uuid>/<version uuid>/long.json"
   "generator": "/case_studies/<case study uuid>/<version uuid>/generator.xlsx",
   "state": "/case_studies/<case study uuid>/<version uuid>/state.xlsx",
   "issues": [{"type": "error", "description": "syntax error in command ..."}, ...],
  },
  ...
 ]
}

    :param cs_uuid:
    :return:
    """
    def get_version_dict(vs):
        # [
        #     {"uuid": "<uuid>",
        #      "resource": "/case_studies/<case study uuid>/<version uuid>",
        #      "tag": "v0.1",
        #      "sessions":
        #          [
        #              {"uuid": "<uuid>",
        #               "open_date": "2017-09-20T10:00:00Z",
        #               "close_date": "2017-09-20T10:00:10Z",
        #               "client": "spreadsheet",
        #               "restart": True,
        #               "author": "<uuid>",
        #               },
        #          ],
        #      "detail": "/case_studies/<case study uuid>/<version uuid>/long.json",
        #      "generator": "/case_studies/<case study uuid>/<version uuid>/generator.xlsx",
        #      },
        # ],
        def get_session_dict(ss):
            uuid4 = str(ss.uuid)
            v_session = {"uuid": uuid4,
                         "open_date": str(ss.open_instant),
                         "close_date": str(ss.close_instant),
                         "client": "spreadsheet",  # TODO Spreadsheet, R script, Python script, <Direct?>
                         "restart": ss.restarts,
                         "author": ss.who.name
                         }
            if mode == "tree":
                v_session = {"data": v_session}
            else:
                pass
            return v_session

        uuid3 = str(vs.uuid)
        version = {"uuid": uuid3,
                   "resource": nis_api_base + "/case_studies/"+uuid2+"/versions/"+uuid3,
                   "tag": "v0.1",
                   "detail": nis_api_base + "/case_studies/"+uuid2+"/versions/"+uuid3,
                   "state": nis_api_base + "/case_studies/" + uuid2 + "/versions/"+uuid3+"/state.xlsx",
                   "issues": None,  # [{"type": "error", "description": "syntax error in command ..."}],
                   "generator": nis_api_base + "/case_studies/"+uuid2+"/versions/"+uuid3+"/generator.xlsx",
                   }
        if mode == "tree":
            version = {"data": version, "children": [get_session_dict(s) for s in vs.sessions]}
        else:
            version["sessions"] = [get_session_dict(s) for s in vs.sessions]
        return version

    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if not user:
        user = "_anonymous"

    mode = "tree"

    # Recover case studies READABLE by current user (or "anonymous")
    session = isess.open_db_session()
    # TODO Obtain case study, filtered by current user permissions
    # Access Control
    # CS in acl and user in acl.detail and acl.detail is READ, WRITE,
    # CS in acl and group acl.detail and user in group
    cs = session.query(CaseStudy).filter(CaseStudy.uuid == cs_uuid).first()
    if cs:
        uuid2 = str(cs.uuid)
        d = {"uuid": uuid2,
             "name": cs.name if cs.name else "<empty>",
             "oid": cs.oid if cs.oid else "<empty>",
             "internal_code": cs.internal_code if cs.internal_code else "",  # TODO
             "description": cs.description if cs.description else "",  # TODO
             "resource": nis_api_base + "/case_studies/"+uuid2,
             "versions": [get_version_dict(v) for v in cs.versions],
             "case_study_permissions":
                 {
                     "read": True,
                     "annotate": True,
                     "contribute": True,
                     "share": False,
                     "delete": False
                 },

             }
        # print(json.dumps(d, default=json_serial, sort_keys=True, indent=JSON_INDENT, ensure_ascii=ENSURE_ASCII, separators=(',', ': ')))
        r = build_json_response(d)
    else:
        r = build_json_response({"error": "The case study '"+cs_uuid+"' does not exist."}, 404)
    isess.close_db_session()

    return r


@app.route(nis_api_base + "/case_studies/<cs_uuid>", methods=["POST"])
@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/", methods=["POST"])
def new_case_study_version_from_file(cs_uuid):
    """
    Check that the user is authorized to submit a new case study version
    Open a reproducible session
    Send the file to the service
    Close the reproducible session

    :param cs_uuid: UUID of case study
    :return:
    """
    # Check Interactive Session is Open. If not, open it
    isess = deserialize_isession_and_prepare_db_session(False)
    if not isess:
        isess = InteractiveSession(DBSession)

    # TODO Check User Credentials (from Token)
    testing = is_testing_enabled()
    if testing:
        result = isess.identify({"user": "test_user", "password": None}, testing=True)

    # TODO Check User has Write Case Study permissions

    # Receive file
    generator_type, content_type, buffer, execute, register = receive_file_submission(request)

    # Open Reproducible Session, NEW case study
    try:
        isess.open_reproducible_session(case_study_version_uuid=cs_uuid,
                                        recover_previous_state=False,
                                        cr_new=CreateNew.VERSION,
                                        allow_saving=register
                                        )
    except Exception as e:
        s = "Exception trying to open reproducible session: "+str(e)
        logger.error(s)
        return build_json_response({"error": s}, 401)


    # Submit file to the Interactive Session (which has the open reproducible session)
    issues, output = isess.register_andor_execute_command_generator(generator_type, content_type, buffer, register, execute)

    # Close Reproducible Session
    isess.close_reproducible_session(issues=issues, output=output, save=register, from_web_service=False)

    # TODO Return the issues if there were any. Return outputs (could be a list of binary files)
    r = build_json_response({}, 204)

    serialize_isession_and_close_db_session(isess)

    return r

# @app.route(nis_api_base + "/case_studies/<cs_uuid>", methods=["DELETE"])
# def case_study_delete(cs_uuid):  # DELETE a case study
#     # Recover InteractiveSession
#     isess = deserialize_isession_and_prepare_db_session()
#     if isess and isinstance(isess, Response):
#         return isess
#
#     # TODO Check permissions
#     # TODO If possible, deleet ALL the case study


@app.route(nis_api_base + "/case_studies/<cs_uuid>/default_view.png", methods=["GET"])
def case_study_default_view_png(cs_uuid):  # Return a view of the case study in PNG format, for preview purposes
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if not user:
        user = "_anonymous"

    return send_static_file("images/case_study_preview_placeholder.png")

    # # Recover case studies READABLE by current user (or "anonymous")
    # session = isess.open_db_session()
    # # TODO Obtain case study, filtered by current user permissions
    # # Access Control
    # # CS in acl and user in acl.detail and acl.detail is READ, WRITE,
    # # CS in acl and group acl.detail and user in group
    # cs = session.query(CaseStudy).filter(CaseStudy.uuid == cs_uuid).first()
    # # TODO Scan variables. Look for the ones most interesting: grammar, data. Maybe cut processors.
    # # TODO Scan also for hints to the elaboration of this thumbnail
    # # TODO Elaborate View in PNG format
    # isess.close_db_session()
    # # TODO Return PNG image


@app.route(nis_api_base + "/case_studies/<cs_uuid>/default_view.svg", methods=["GET"])
def case_study_default_view_svg(cs_uuid):  # Return a view of the case study in SVG format, for preview purposes
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if not user:
        user = "_anonymous"
    # Recover case studies READABLE by current user (or "anonymous")
    session = isess.open_db_session()
    # TODO Obtain case study, filtered by current user permissions
    # Access Control
    # CS in acl and user in acl.detail and acl.detail is READ, WRITE,
    # CS in acl and group acl.detail and user in group
    cs = session.query(CaseStudy).filter(CaseStudy.uuid == cs_uuid).first()
    # TODO Scan variables. Look for the ones most interesting: grammar, data. Maybe cut processors.
    # TODO Scan also for hints to the elaboration of this thumbnail
    # TODO Elaborate View in SVG format
    isess.close_db_session()
    # TODO Return SVG image


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>", methods=["GET"])
def case_study_version(cs_uuid, v_uuid):  # Information about a case study version
    """

{"case_study": "<uuid>",
 "version": "<uuid>",
 "resource": "/case_studies/<case study uuid>/<version uuid>",
 "tag": "v0.1",
 "sessions":
   [
    {"uuid": "<uuid>",
     "open_date": "2017-09-20T10:00:00Z",
     "close_date": "2017-09-20T10:00:10Z",
     "client": "spreadsheet",
     "restart": True,
     "author": "<uuid>",
     "generator": "/case_studies/<case study uuid>/<version uuid>/<session uuid>/generator.xlsx",
     "state": "/case_studies/<case study uuid>/<version uuid>/<session uuid>/state.xlsx",
     "issues": [{"type": "error", "description": "syntax error in command ..."}, ...],
    },
    ...
   ]
 "command_executors":
 [
  {"type": "...",
   "label": "...",
   "definition": "/case_studies/<case study uuid>/<version uuid>/1.json"
  },
  ...
 ],
 "generator": "/case_studies/<case study uuid>/<version uuid>/generator.xlsx",
 "state": "/case_studies/<case study uuid>/<version uuid>/state.xlsx",
 "issues": [{"type": "error", "description": "syntax error in command ..."}, ...],
}

    :param cs_uuid:
    :param v_uuid:
    :return:
    """

    def get_version_dict(vs):
        # [
        #     {"uuid": "<uuid>",
        #      "resource": "/case_studies/<case study uuid>/<version uuid>",
        #      "tag": "v0.1",
        #      "sessions":
        #          [
        #              {"uuid": "<uuid>",
        #               "open_date": "2017-09-20T10:00:00Z",
        #               "close_date": "2017-09-20T10:00:10Z",
        #               "client": "spreadsheet",
        #               "restart": True,
        #               "author": "<uuid>",
        #               },
        #          ],
        #      "detail": "/case_studies/<case study uuid>/<version uuid>/long.json",
        #      "generator": "/case_studies/<case study uuid>/<version uuid>/generator.xlsx",
        #      },
        # ],
        def get_session_dict(ss):
            uuid4 = str(ss.uuid)
            v_session = {"uuid": uuid4,
                         "open_date": str(ss.open_instant),
                         "close_date": str(ss.close_instant),
                         "client": "spreadsheet",  # TODO Spreadsheet, R script, Python script, <Direct?>
                         "restart": ss.restarts,
                         "author": ss.who.name
                         }
            if mode == "tree":
                v_session = {"data": v_session}
            else:
                pass
            return v_session

        # Case Study UUID, Case Study Version UUID
        uuid2 = str(vs.case_study.uuid)
        uuid3 = str(vs.uuid)

        # Get active sessions
        act_sess = []
        for s in vs.sessions:
            if s.restarts:
                act_sess = []
            act_sess.append(s)

        # Load state
        if vs.state:
            # Deserialize
            st = deserialize_to_object(vs.state)
        else:
            st = State()  # Zero State, execute all commands in sequence
            for ws in act_sess:
                for c in ws.commands:
                    execute_command_container(st, c)

        # List command_executors lista -> diccionario ("data": {}, "children": [ ... ])
        lst_cmds = []
        for ws in act_sess:
            for c in ws.commands:
                d = {"type": c.generator_type,
                     "label": c.name if c.name else "<empty>",
                     "definition": nis_api_base + "/case_studies/" + uuid2 + "/versions/"+uuid3+"/sessions/"+str(ws.uuid)+"/command/"+str(c.order)
                     }
                if mode == "tree":
                    d = {"data": d}
                lst_cmds.append(d)

        # List of variables
        lst_vars = []
        for n in st.list_namespaces():
            for t in st.list_namespace_variables(n):
                d = {"name": t[0],
                     "type": str(type(t[1])),
                     "view": nis_api_base + "/case_studies/" + uuid2 + "/versions/"+uuid3+"/variables/"+str(t[0]),
                     "namespace": n
                     }
                if mode == "tree":
                    d = {"data": d}
                lst_vars.append(d)

        version = {"case_study": uuid2,
                   "version": uuid3,
                   "resource": nis_api_base + "/case_studies/"+uuid2+"/versions/"+uuid3,
                   "tag": "v0.1",
                   "generator": nis_api_base + "/case_studies/"+uuid2+"/versions/"+uuid3+"/generator.xlsx",
                   "state": nis_api_base + "/case_studies/" + uuid2 + "/versions/"+uuid3+"/state.xlsx",
                   "issues": [{"type": "error", "description": "syntax error in command ..."}
                              ],
                   "sessions": [get_session_dict(s) for s in vs.sessions],
                   "command_executors": lst_cmds,
                   "variables": lst_vars
                   }
        # if mode == "tree":
        #     version = {"data": version, "children": [get_session_dict(s) for s in vs.sessions]}
        # else:
        #     version["sessions"] = [get_session_dict(s) for s in vs.sessions]
        return version

    mode = "tree"

    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if not user:
        user = "_anonymous"
    # Recover case studies READABLE by current user (or "anonymous")
    session = isess.open_db_session()
    # TODO Obtain case study version, filtered by current user permissions
    # Access Control
    # CS in acl and user in acl.detail and acl.detail is READ, WRITE,
    # CS in acl and group acl.detail and user in group
    vs = session.query(CaseStudyVersion).filter(CaseStudyVersion.uuid == v_uuid).first()
    if not vs:
        r = build_json_response({"error": "The case study version '"+v_uuid+"' does not exist."}, 404)
    else:
        if str(vs.case_study.uuid) != cs_uuid:
            r = build_json_response({"error": "The case study '" + cs_uuid + "' does not exist."}, 404)
        else:
            r = build_json_response(get_version_dict(vs))
    isess.close_db_session()

    return r


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>", methods=["DELETE"])
def case_study_version_delete(cs_uuid, v_uuid):  # DELETE a case study version
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # TODO Check user permissions
    # TODO If authorized, delete a case study version and all its sessions and commands


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>/sessions/<s_uuid>", methods=["GET"])
def case_study_version_session(cs_uuid, v_uuid, s_uuid):  # Information about a session in a case study version
    """

{"case_study": "<uuid>",
 "version": "<uuid>",
 "session": "<uuid>",
 "resource": "/case_studies/<case study uuid>/<version uuid>/<session uuid>",
 "open_date": "2017-09-20T10:00:00Z",
 "close_date": "2017-09-20T10:00:10Z",
 "client": "spreadsheet",
 "restart": True,
 "author": "<uuid>",
 "generator": "/case_studies/<case study uuid>/<version uuid>/<session uuid>/generator.xlsx",
 "state": "/case_studies/<case study uuid>/<version uuid>/<session uuid>/state.xlsx",
 "issues": [{"type": "error", "description": "syntax error in command ..."}, ...],
 "command_executors":
 [
  {"type": "...",
   "label": "...",
   "definition": "/case_studies/<case study uuid>/<version uuid>/1.json"
  },
  ...
 ]
}

    :param cs_uuid:
    :param v_uuid:
    :param s_uuid:
    :return:
    """
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if not user:
        user = "_anonymous"
    # Recover case studies READABLE by current user (or "anonymous")
    session = isess.open_db_session()
    # TODO Obtain case study version session, filtered by current user permissions
    # Access Control
    # CS in acl and user in acl.detail and acl.detail is READ, WRITE,
    # CS in acl and group acl.detail and user in group
    ss = session.query(CaseStudyVersionSession).filter(CaseStudyVersionSession.uuid == s_uuid).first()
    if not ss:
        r = build_json_response({"error": "The case study version session '"+s_uuid+"' does not exist."}, 404)
    else:
        if ss.version.uuid != v_uuid:
            r = build_json_response({"error": "The case study version '" + v_uuid + "' does not exist."}, 404)
        elif ss.version.case_study.uuid != cs_uuid:
            r = build_json_response({"error": "The case study '" + cs_uuid + "' does not exist."}, 404)
        else:
            # TODO Return the command OR generator
            # TODO The generator can be text or BINARY
            r = build_json_response(ss)  # TODO Improve it, it must return the number of versions. See document !!!
    isess.close_db_session()

    return r


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>/sessions/<s_uuid>", methods=["DELETE"])
def case_study_version_session_delete(cs_uuid, v_uuid, s_uuid):  # DELETE a session in a case study version
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # TODO Check user permissions
    # TODO If authorized, delete a case study version SESSION and all its commands


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>/sessions/<s_uuid>/<command_order>", methods=["GET"])
def case_study_version_session_command(cs_uuid, v_uuid, s_uuid, command_order):
    """
        DOWNLOAD a command or generator, using the order, from 0 to number of command_executors - 1
        Commands are enumerated using "case_study_version_session()"
            (URL: "/case_studies/<cs_uuid>/versions/<v_uuid>/sessions/<s_uuid>")

    :param cs_uuid:
    :param v_uuid:
    :param s_uuid:
    :param command_order:
    :return:
    """
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if not user:
        user = "_anonymous"
    # Recover case studies READABLE by current user (or "anonymous")
    session = isess.open_db_session()
    # TODO Obtain case study version session, filtered by current user permissions
    # Access Control
    # CS in acl and user in acl.detail and acl.detail is READ, WRITE,
    # CS in acl and group acl.detail and user in group
    ss = session.query(CaseStudyVersionSession).filter(CaseStudyVersionSession.uuid == s_uuid).first()
    if not ss:
        r = build_json_response({"error": "The case study version session '"+s_uuid+"' does not exist."}, 404)
    else:
        # if ss.version.uuid != v_uuid:
        #     r = build_json_response({"error": "The case study version '" + v_uuid + "' does not exist."}, 404)
        # elif ss.version.case_study.uuid != cs_uuid:
        #     r = build_json_response({"error": "The case study '" + cs_uuid + "' does not exist."}, 404)
        order = int(command_order)
        c = ss.commands[order]
        r = Response(c.content, mimetype=c.content_type)
        #r.headers['Access-Control-Allow-Origin'] = "*"

    isess.close_db_session()

    return r


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>/variables/", methods=["GET"])
def case_study_version_variables(cs_uuid, v_uuid):  # List of variables defined in a case study version
    """
    Return the list of ALL variables defined in the case study version

    :param cs_uuid:
    :param v_uuid:
    :return:
    """
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # TODO Check READ permission of the user to the case study
    # Open temporary reproducible session
    try:
        isess.open_reproducible_session(case_study_version_uuid=v_uuid,
                                        recover_previous_state=True,
                                        cr_new=CreateNew.NO,
                                        allow_saving=False
                                        )

        # A reproducible session must be open, signal about it if not
        if isess._state:
            # List all available variables, from state. A list of dictionaries "name", "type" and "namespace"
            lst = []
            for n in isess._state.list_namespaces():
                # Add ALL variables EXCEPT the internal ones (which start with "_")
                lst.extend([{"name": t[0],
                             "type": str(type(t[1])),
                             "namespace": n} for t in isess._state.list_namespace_variables(n) if not t[0].startswith("_")
                            ]
                           )

            r = build_json_response(lst, 200)
        else:
            r = build_json_response({"error": "No state available for Case Study Version '"+v_uuid+"'"}, 404)

        # Close temporary reproducible session
        isess.close_reproducible_session(issues=None, output=None, save=False, from_web_service=True)
    except Exception as e:
        r = build_json_response({"error": str(e)}, 404)

    return r


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>/variables/<name>", methods=["GET"])
def case_study_version_variable(cs_uuid, v_uuid, name):  # Information about a case study version variable
    """
    Return the value of the requested variable

    :param cs_uuid: Case Study UUID
    :param v_uuid: Version UUID
    :param name: Variable name
    :return:
    """
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # TODO Check READ permission of the user to the case study
    # Open temporary reproducible session
    try:
        isess.open_reproducible_session(case_study_version_uuid=v_uuid,
                                        recover_previous_state=True,
                                        cr_new=CreateNew.NO,
                                        allow_saving=False
                                        )

        # A reproducible session must be open, signal about it if not
        if isess._state:
            # TODO Parse Variable name can be "namespace'::'name"
            # TODO For now, just the variable name
            v = isess._state.get(name)
            if v:
                r = build_json_response({name: v}, 200)
            else:
                r = build_json_response(
                    {"error": "The requested variable name ('"+name+"') has not "
                              "been found in the Case Study Version '" + v_uuid + "'"}, 404)
        else:
            r = build_json_response({"error": "No state available for Case Study Version '" + v_uuid + "'"}, 404)

        # Close temporary reproducible session
        isess.close_reproducible_session(issues=None, output=None, save=False, from_web_service=True)
    except Exception as e:
        r = build_json_response({"error": str(e)}, 404)

    return r


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>/variables/<name>/views/", methods=["GET"])
def case_study_version_variable_views(cs_uuid, v_uuid, name):  # Information about a case study version variable views
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # TODO Check READ permission of the user to the case study
    # TODO Return the different views on a variable


@app.route(nis_api_base + "/case_studies/<cs_uuid>/versions/<v_uuid>/variables/<name>/views/<view_type>", methods=["GET"])
def case_study_version_variable_view(cs_uuid, v_uuid, name, view_type):  # A view of case study version variable
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # TODO Check READ permission of the user to the case study
    # TODO Return a view of the requested variable

# -- Users --


@app.route(nis_api_base + "/users/", methods=["GET"])
def list_users():
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if user and user == "admin":
        session = isess.open_db_session()
        lst = session.query(User).all()
        r = build_json_response(lst)
        isess.close_db_session()
    else:
        r = build_json_response({"error": "Users list can be obtained only by 'admin' user"}, 401)

    return r


@app.route(nis_api_base + "/users/<id>", methods=["GET"])
def get_user(id):
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if user and user == "admin" or user == id:
        session = isess.open_db_session()
        u = session.query(User).filter(User.name == id).first()
        r = build_json_response(u)  # TODO Improve it !!!
        isess.close_db_session()
    else:
        r = build_json_response({"error": "User '"+id+"' can be obtained only by 'admin' or '"+id+"' user"}, 401)

    return r


@app.route(nis_api_base + "/users/<id>", methods=["PUT"])
def put_user(id):  # Used also to deactivate user
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if user and user == "admin" or user == id:
        session = isess.open_db_session()
        u = session.query(User).filter(User.name == id).first()
        if not u:
            r = build_json_response({"error": "User '"+id+"' does not exist"}, 404)
        else:
            # TODO Update "u" fields
            session.commit()
        r = build_json_response(u)  # TODO Improve it !!!
        isess.close_db_session()
    else:
        r = build_json_response({"error": "User '"+id+"' can be modified only by 'admin' or '"+id+"' user"}, 401)

    return r


@app.route(nis_api_base + "/users/", methods=["POST"])
def post_user():
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if user and user == "admin":
        session = isess.open_db_session()
        # Read JSON
        if request.content_type != "application/json":
            raise Exception("Only application/json data is allowed")
        if not request.data:
            raise Exception("No data received")
        j = request.data.decode()
        j = json.loads(j)
        u = session.query(User).filter(User.name == j["name"]).first()
        if not u:
            # Create User
            u = User()
            u.name = j["name"]
            session.add(u)
            session.commit()
            r = build_json_response(u)
        else:
            r = build_json_response({"error": "User '"+j["name"]+"' already exists"}, 422)
        isess.close_db_session()
    else:
        r = build_json_response({"error": "A user can be created only by 'admin'"}, 401)

    return r

# -- Groups --


@app.route(nis_api_base + "/groups/", methods=["GET"])
def list_groups():
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess


@app.route(nis_api_base + "/groups/<id>", methods=["GET"])
def get_group(id):
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if user and user == "admin" or user == id:
        session = isess.open_db_session()
        u = session.query(Group).filter(Group.name == id).first()
        r = build_json_response(u)  # TODO Improve it !!!
        isess.close_db_session()
    else:
        r = build_json_response({"error": "Group '" + id + "' can be obtained only by 'admin' or '" + id + "' user"},
                                401)

    return r


@app.route(nis_api_base + "/groups/<id>", methods=["PUT"])
def put_group(id):
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if user and user == "admin" or user == id:
        session = isess.open_db_session()
        u = session.query(Group).filter(Group.name == id).first()
        if not u:
            r = build_json_response({"error": "Group '" + id + "' does not exist"}, 404)
        else:
            # TODO Update "u" fields
            session.commit()
        r = build_json_response(u)  # TODO Improve it !!!
        isess.close_db_session()
    else:
        r = build_json_response({"error": "Group '" + id + "' can be modified only by 'admin' or '" + id + "' user"},
                                401)

    return r


@app.route(nis_api_base + "/users/", methods=["POST"])
def post_group():
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    user = isess.get_identity_id()
    if user and user == "admin":
        session = isess.open_db_session()
        # Read JSON
        if request.content_type != "application/json":
            raise Exception("Only application/json data is allowed")
        if not request.data:
            raise Exception("No data received")
        j = request.data.decode()
        j = json.loads(j)
        u = session.query(Group).filter(Group.name == j["name"]).first()
        if not u:
            # TODO Create Group
            u = Group()
            u.name = j["name"]
            session.add(u)
            session.commit()
            r = build_json_response(u)
        else:
            r = build_json_response({"error": "Group '" + j["name"] + "' already exists"}, 422)
        isess.close_db_session()
    else:
        r = build_json_response({"error": "A group can be created only by 'admin'"}, 401)

    return r


# -- Permissions --


def acl():
    pass

# -- Reusable objects --


@app.route(nis_api_base + "/sources/", methods=["GET"])
def data_sources():
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # TODO Authentication, authorization

    # Enumerate sources
    dsm = backend.data_source_manager
    lst = dsm.get_supported_sources()

    return build_json_response(dict(sources=lst))


@app.route(nis_api_base + "/sources/<source_id>", methods=["GET"])
@app.route(nis_api_base + "/sources/<source_id>/databases/", methods=["GET"])
def data_source_databases(source_id):
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # Enumerate source databases
    dsm = backend.data_source_manager
    if source_id == "-":
        source_id = None
    lst = dsm.get_databases(source_id)

    ret_lst = []
    for i in lst:
        ret_lst.append(dict(source=i[0], databases=[dict(code=c.code, description=c.description) for c in i[1]]))

    return build_json_response(ret_lst)


@app.route(nis_api_base + "/sources/<source_id>/databases/<database_id>", methods=["GET"])
@app.route(nis_api_base + "/sources/<source_id>/databases/<database_id>/datasets/", methods=["GET"])
def data_source_database_datasets(source_id, database_id):
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    # Enumerate source+database datasets
    dsm = backend.data_source_manager
    if source_id == "-":
        source_id = None
        database_id = None
    if database_id == "-":
        database_id = None
    lst = dsm.get_datasets(source_id, database_id)

    return build_json_response([dict(source=i[0],
                                     datasets=[dict(code=j[0], description=j[1]) for j in i[1]]) for i in lst]
                               )


@app.route(nis_api_base + "/sources/<source_id>/databases/<database_id>/datasets/<dataset_id>", methods=["GET"])
def data_source_database_dataset_detail(source_id, database_id, dataset_id):
    """
    Return a JSON with the method "GET" and the possible values for the dimensions
    Also parameters to return a table of tuples or a precomputed pivot table
    Also return the address of the endpoint to query the dataset using SDMX. This be

    :param id:
    :param database_id:
    :param dataset_id:
    :return:
    """
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess

    dsm = backend.data_source_manager
    if source_id == "-":
        source_id = None
        database_id = None
    if database_id == "-":
        database_id = None
    if not dataset_id:
        raise Exception("It is mandatory to define the dataset name when requesting the dataset parameters")

    if not source_id:
        source_id = obtain_dataset_source(dataset_id)

    ds = dsm.get_dataset_structure(source_id, dataset_id)
    dims = []
    for d in ds.dimensions:
        cl = []
        if d.code_list:
            # CodeList has one or more levels ".levels" property
            # CodeListLevel has zero or more Codes ".codes" property
            for level in d.code_list.levels:
                for code in level.codes:
                    cl.append(dict(code=code.code, description=code.description, level=level.code))

        dims.append(dict(code=d.code, description=d.description, is_time=d.is_time, is_measure=d.is_measure, attributes=d.attributes, code_list=cl))

    d = dict(id=ds.id, code=ds.code, description=ds.description, data_dictionary=ds.data_dictionary, attributes=ds.attributes,
             dimensions=dims)

    return build_json_response(d)


@app.route(nis_api_base + "/sources/<id>/databases/<database_id>/datasets/<dataset_id>", methods=["GET"])
def data_source_database_dataset_query(id, database_id, dataset_id):
    """
    This is the most powerful data method, allowing to

    :param id:
    :param database_id:
    :param dataset_id:
    :return:
    """
    # Recover InteractiveSession
    isess = deserialize_isession_and_prepare_db_session()
    if isess and isinstance(isess, Response):
        return isess


def data_processes():
    pass


def nusap_data_pedigree():
    pass


def grammars():
    pass


def mappings():
    """
    From an external dataset to internal categories
    :return: 
    """
    pass


def heterarchies():
    a= 6
    pass

# -- Test --


@app.route('/test', methods=['GET'])
@app.route(nis_api_base + '/test', methods=['GET'])
def hello():
    return build_json_response({"hello": "world"})


if __name__ == '__main__':
    # cs = CaseStudy()
    # vs1 = CaseStudyVersion()
    # vs1.case_study = cs
    # vs2 = CaseStudyVersion()
    # vs2.case_study = cs
    #
    # lst = [cs, vs1, vs2]
    # d_list = serialize(lst)
    # lst2 = deserialize(d_list)
    # sys.exit(1)
    # >>>>>>>>>> IMPORTANT <<<<<<<<<
    # For debugging in local mode, prepare an environment variable "MAGIC_NIS_SERVICE_CONFIG_FILE", with value "./nis_local.conf"
    # >>>>>>>>>> IMPORTANT <<<<<<<<<

    # >>>>>>>>>> IMPORTANT <<<<<<<<<
    # "cannot connect to X server" error when remote debugging?
    # Execute "Xvfb :99 -ac -noreset" in the remote server and uncomment the following line
    # os.environ["DISPLAY"] = ":99"
    app.run(host='0.0.0.0',
            debug=True,
            use_reloader=False,  # Avoid loading twice the application
            threaded=True)  # Default port, 5000
