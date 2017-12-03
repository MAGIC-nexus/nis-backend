# -*- coding: utf-8 -*-
"""
* Registry of objects. Add, remove, search
* Support for high level operations: directly create and/or modify objects, calling the specification API. Create connections
"""
import logging
import datetime
from enum import Enum
from typing import List
import uuid
import json
import copy

from backend.common.helper import create_dictionary
from backend.domain import IExecutableCommand
from backend.model.rdb_persistence.persistent import (User,
                                                      CaseStudy,
                                                      CaseStudyStatus,
                                                      CaseStudyVersion,
                                                      CaseStudyVersionSession,
                                                      PersistableCommand,
                                                      force_load,
                                                      serialize_from_object,
                                                      deserialize_to_object
                                                      )
from backend.commands.factory import (execute_command,
                                      execute_command_generator,
                                      execute_command_generator_file
                                      )

logger = logging.getLogger(__name__)


class Identity:
    pass

# class IdentityCredentials:
#     pass
#
# class NISystem:
#     def __init__(self):
#         self._authentication_service = None
#         self._authorization_service = None
#         self._base_url = None
#         self._configuration_manager = None
#         self._plugin_manager = None
#
#
#     def gather_credentials(self) -> IdentityCredentials:
#         return IdentityCredentials()
#
#     def login(self, ic: IdentityCredentials):
#         # Check credentials using authentication service
#         return Identity()
#
#     def logout(self, id: Identity):
#         pass


class Scope:
    """ The scope allows to assign names to entities using a registry """
    def __init__(self, name=None):
        self._name = name  # A name for the scope itself
        self._registry = create_dictionary()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __contains__(self, key):  # "in" operator to check if the key is present in the dictionary
        return key in self._registry

    def __getitem__(self, name):
        if name in self._registry:
            return self._registry[name]
        else:
            return None

    def __setitem__(self, name: str, entity):
        if name not in self._registry:
            existing = True
        else:
            existing = False

        self._registry[name] = entity
        return existing

    def __delitem__(self, name):
        del self._registry[name]

    def list(self):
        """ List just the names of variables """
        return self._registry.keys()

    def list_pairs(self):
        """ List tuples of variable name and value object """
        return [(k, v2) for k, v2 in self._registry.items()]


class Namespace:
    def __init__(self):
        self.__scope = []  # List of scopes
        self.__current_scope = None  # type: Scope
        self.__current_scope_idx = -1
        self.new_scope()

    # The registry will have "nested" scopes (names in the current scope take precedence on "higher" scopes)
    # When searching for names, the search will go from the most recent scope to the oldest
    def new_scope(self, name=None):
        """ Create a new scope """
        self.__current_scope = Scope()
        self.__scope.append(self.__current_scope)
        self.__current_scope_idx = len(self.__scope) - 1
        if not name:
            name = "Scope" + str(self.__current_scope_idx)
        self.__current_scope.name = name

    def close_scope(self):
        if self.__current_scope:
            del self.__scope[-1]
        if self.__current_scope_idx >= 0:
            self.__current_scope_idx -= 1
            if self.__current_scope_idx >= 0:
                self.__current_scope = self.__scope[-1]
            else:
                self.__current_scope = None

    def list_names(self, scope=None):
        """ Returns a list of the names of the registered entities of the "scope" or if None, of the CURRENT scope """
        if not scope:
            scope = self.__current_scope

        return scope.list()

    def list(self, scope=None):
        """
            Returns a list of the names and values of the registered entities of
            the "scope" or if None, of the CURRENT scope
        """
        if not scope:
            scope = self.__current_scope

        return scope.list_pairs()

    def list_all_names(self):
        """
            Returns a list of the names of registered entities considering the scopes
            Start from top level, end in bottom level (the current one, which takes precedence)
            :return:
        """
        t = create_dictionary()
        for scope in self.__scope:
            t.update(scope._registry)

        return t.keys()

    def list_all(self):
        """
            Returns a list of the names and variables of registered entities considering the scopes
            Start from top level, end in bottom level (the current one, which takes precedence)

        :return:
        """
        t = create_dictionary()
        for scope in self.__scope:
            t.update(scope._registry)

        return [(k, v2) for k, v2 in t.items()]

    def set(self, name: str, entity):
        """ Set a named entity in the current scope. Previous scopes are not writable. """
        if self.__current_scope:
            var_exists = name in self.__current_scope
            self.__current_scope[name] = entity
            if var_exists:
                logger.warning("'"+name+"' overwritten.")

    def get(self, name: str, scope=None, return_scope=False):
        """ Return the entity named "name". Return also the Scope in which it was found """
        if not scope:
            for scope_idx in range(len(self.__scope)-1, -1, -1):
                if name in self.__scope[scope_idx]:
                    if return_scope:
                        return self.__scope[scope_idx][name], self.__scope[scope_idx]
                    else:
                        return self.__scope[scope_idx][name]
            else:
                logger.warning("The name '"+name+"' was not found in the stack of scopes ("+str(len(self.__scope))+")")
                if return_scope:
                    return None, None
                else:
                    return None
        else:
            # TODO Needs proper implementation !!!! (when scope is a string, not a Scope instance, to be searched in the list of scopes "self.__scope")
            if name in scope:
                if return_scope:
                    return scope[name], scope
                else:
                    return scope[name]
            else:
                logger.error("The name '" + name + "' was not found in scope '"+scope.name+"'")
                if return_scope:
                    return None, None
                else:
                    return None


class State:
    """
    -- "State" in memory --

    Commands may alter State or may just read it
    It uses a dictionary of named Namespaces (and Namespaces can have several scopes)
    Keeps a registry of variable names and the objects behind them.
    
        It is basically a list of Namespaces. One is active by default.
        The others have a name. Variables inside these other Namespaces may be accessed using that 
        name then "::", same as C++
    """
    def __init__(self):
        self._default_namespace = ""
        self._namespaces = create_dictionary()  # type:

    def new_namespace(self, name):
        self._namespaces[name] = Namespace()
        if self._default_namespace is None:
            self._default_namespace = name

    @property
    def default_namespace(self):
        return self._default_namespace

    @default_namespace.setter
    def default_namespace(self, name):
        if name is not None:  # Name has to have some value
            self._default_namespace = name

    def del_namespace(self, name):
        if name in self._namespaces:
            del self._namespaces

    def list_namespaces(self):
        return self._namespaces.keys()

    def list_namespace_variables(self, namespace_name=None):
        if namespace_name is None:
            namespace_name = self._default_namespace

        return self._namespaces[namespace_name].list_all()

    def set(self, name, entity, namespace_name=None):
        if namespace_name is None:
            namespace_name = self._default_namespace

        if namespace_name not in self._namespaces:
            self.new_namespace(namespace_name)

        self._namespaces[namespace_name].set(name, entity)

    def get(self, name, namespace_name=None, scope=None):
        if not namespace_name:
            namespace_name = self._default_namespace

        if namespace_name not in self._namespaces:
            self.new_namespace(namespace_name)

        return self._namespaces[namespace_name].get(name, scope)


class CommandResult:
    pass

# class SessionCreationAction(Enum):  # Used in FlowFund
#     """
#
#         +--------+--------+---------+-------------------------+
#         | branch | clone  | restart |        Behavior         |
#         +--------+--------+---------+-------------------------+
#         | True   | True   | True    | Branch & New WorkSession|
#         | True   | True   | False   | Branch & Clone          |
#         | True   | False  | True    | Branch & New WorkSession|
#         | True   | False  | False   | Branch & New WorkSession|
#         | False  | True   | True    | New CS (not CS version) |
#         | False  | True   | False   | New CS & Clone          |
#         | False  | False  | True    | Restart                 |
#         | False  | False  | False   | Continue                |
#         +--------+--------+---------+-------------------------+
#     """
#     BranchAndNewWS = 7
#     BranchAndCloneWS = 6
#     NewCaseStudy = 3
#     NewCaseStudyCopyFrom = 2
#     Restart = 1
#     Continue = 0


class CreateNew(Enum):
    CASE_STUDY = 1
    VERSION = 2
    NO = 3


class InteractiveSession:
    """ 
    Main class for interaction with NIS
    The first thing would be to identify the user and create a GUID for the session which can be used by the web server
    to store and retrieve the interactive session state.
    
    It receives commands, modifying state accordingly
    If a reproducible session is opened, 
    """
    def __init__(self, session_factory):
        # Session factory with access to business logic database
        self._session_factory = session_factory

        # Interactive session ID
        self._guid = str(uuid.uuid4())

        # User identity, if given (can be an anonymous session)
        self._identity = None  # type: Identity
        self._state = State()  # To keep the state
        self._work_session = None  # type: WorkSession

    def reset_state(self):
        """ Restart state """
        self._state = State()
        # TODO self._recordable_session = None ??

    def get_sf(self):
        return self._session_factory

    def set_sf(self, session_factory):
        self._session_factory = session_factory
        if self._work_session:
            self._work_session.set_sf(session_factory)

    def open_db_session(self):
        return self._session_factory()

    def close_db_session(self):
        self._session_factory.remove()

    def quit(self):
        """
        End interactive session
        :return: 
        """
        pass

    # --------------------------------------------------------------------------------------------

    def identify(self, identity_information, testing=False):
        """
        Given credentials of some type -identity_information-, link an interactive session to an identity.
        The credentials can vary from an OAuth2 Token to user+password.
        Depending on the type of credentials, invoke a type of "identificator" or other
        An interactive session without identification is allowed to perform a subset of available operations
        
        :param identity_information: 
        :return: True if the identification was successful, False if not 
        """
        # TODO Check the credentials
        if isinstance(identity_information, dict):
            if "user" in identity_information and testing:
                # Check if the user is in the User's table
                session = self._session_factory()
                src = session.query(User).filter(User.name == identity_information["user"]).first()
                # Check if the dataset exists. "ETL" it if not
                # ds = session.query(Dataset).\
                #     filter(Dataset.code == dataset).\
                #     join(Dataset.database).join(Database.data_source).\
                #     filter(DataSource.name == src_name).first()
                force_load(src)
                session.close()
                self._session_factory.remove()
                if src:
                    self._identity = src.name
                return src is not None
            elif "token" in identity_information:
                # TODO Validate against some Authentication service
                pass

    def get_identity_id(self):
        return self._identity

    def unidentify(self):
        # TODO The un-identification cannot do in the following circumstances: any?
        self._identity = None

    # --------------------------------------------------------------------------------------------
    # Reproducible sessions and commands INSIDE them
    # --------------------------------------------------------------------------------------------
    def open_reproducible_session(self,
                                  case_study_version_uuid: str,
                                  recover_previous_state=True,
                                  cr_new: CreateNew = CreateNew.NO,
                                  allow_saving=True):
        self._work_session = WorkSession(self)
        self._work_session.open(self._session_factory, case_study_version_uuid, recover_previous_state, cr_new, allow_saving)

    def close_reproducible_session(self, issues=None, output=None, save=False, from_web_service=False):
        if save:
            self._work_session.save(from_web_service)
        uuid_, v_uuid, cs_uuid = self._work_session.close()
        self._work_session = None
        return uuid_, v_uuid, cs_uuid

    def reproducible_session_opened(self):
        return self._work_session is not None

    def execute_executable_command(self, cmd: IExecutableCommand):
        return execute_command(self._state, cmd)

    def register_executable_command(self, cmd: IExecutableCommand):
        self._work_session.register_executable_command(cmd)

    def register_andor_execute_command_generator(self, generator_type, file_type: str, file, register=True, execute=False):
        """
        Creates a generator parser, then it feeds the file type and the file
        The generator parser has to parse the file and to generate commands as a Python generator 

        :param generator_type: 
        :param file_type: 
        :param file: 
        :param register: If True, register the command in the WorkSession
        :param execute: If True, execute the command in the WorkSession
        :return: 
        """
        if not self._work_session:
            raise Exception("In order to execute a command generator, a work session is needed")
        if not register and not execute:
            raise Exception("More than zero of the parameters 'register' and 'execute' must be True")

        # Prepare persistable command
        c = WorkSession.create_persistable_command(generator_type, file_type, file)
        if register:
            self._work_session.register_persistable_command(c)
        if execute:
            return self._work_session.execute_persistable_command(c)
            # Or
            # return execute_command_generator(self._state, c)
        else:
            return None
    # --------------------------------------------------------------------------------------------

    def get_case_studies(self):
        """ Get a list of case studies READABLE by current identity (or public if anonymous) """
        pass

    def get_case_study_versions(self, case_study: str):
        # TODO Check that the current user has READ access to the case study
        pass

    def get_case_study_version(self, case_study_version: str):
        # TODO Check that the current user has READ access to the case study
        pass

    def get_case_study_version_variable(self, case_study_version: str, variable: str):
        pass

    def remove_case_study_version(self, case_study_version: str):
        pass

    def share_case_study(self, case_study: str, identities: List[str], permission: str):
        pass

    def remove_case_study_share(self, case_study: str, identities: List[str]):
        pass

    def get_case_study_shared(self, case_study: str):
        pass

    def export_case_study_version(self, case_study_version: str):
        pass

    def import_case_study(self, file):
        pass

# class Store(ABCMeta):
#     def define(self, definition):
#         pass
#
#     def load(self, uuid):
#         pass
#
#     def save(self, uuid, session: WorkSession):
#         pass


class WorkSession:
    def __init__(self, isess):
        # Containing InteractiveSession. Used to set State when a WorkSession is opened and it overwrites State
        self._isess = isess
        self._identity = isess._identity
        self._sess_factory = None
        self._allow_saving = None
        self._session = None  # type: CaseStudyVersionSession

    def open(self, session_factory, uuid_: str=None, recover_previous_state=True, cr_new:CreateNew=CreateNew.NO, allow_saving=True):
        """
        Open a work session

    +--------+--------+---------+-----------------------------------------------------------------------------------+
    | UUID   | cr_new | recover |        Behavior                                                                   |
    +--------+--------+---------+-----------------------------------------------------------------------------------+ 
    | !=None | True   | True    | Create new CS or version (branch) from "UUID", clone WS, recover State, append WS |
    | !=None | True   | False   | Create new CS or version (branch) from "UUID", Zero State, first WS               |
    | !=None | False  | True    | Recover State, append WS                                                          |
    | !=None | False  | False   | Zero State, append WS (overwrite type)                                            |
    | ==None | -      | -       | New CS and version, Zero State, first WS                                          |
    +--------+--------+---------+-----------------------------------------------------------------------------------+
    Use cases:
    * A new case study, from scratch. uuid_=None
    * A new case study, copied from another case study. uuid_=<something>, cr_new=CreateNew.CASE_STUDY, recover_previous_state=True
    * A new version of a case study
      - Copying previous version
      - Starting from scratch
    * Continue a case study version
      - But restart (from scratch)
    * Can be a Transient session

        :param uuid_: UUID of the case study or case study version. Can be None, for new case studies or for testing purposes.
        :param recover_previous_state: If an existing version is specified, it will recover its state after execution of all commands
        :param cr_new: If != CreateNew.NO, create either a case study or a new version. If == CreateNew.NO, append session to "uuid"
        :param allow_saving: If True, it will allow saving at the end (it will be optional). If False, trying to save will generate an Exception
        :return UUID of the case study version in use. If it is a new case study and it has not been saved, the value will be "None"
        """
        # TODO Just register for now. But in the future it should control that there is no other "allow_saving" WorkSession opened
        # TODO for the same Case Study Version. So it implies modifying some state in CaseStudyVersion to have the UUID
        # TODO of the active WorkSession, even if it is not in the database. Register also the date of "lock", so the
        # TODO lock can be removed in case of "hang" of the locker WorkSession
        self._allow_saving = allow_saving
        self._sess_factory = session_factory
        session = self._sess_factory()
        if uuid_:
            uuid_ = str(uuid_)
            # Find UUID. Is it a Case Study or a Case Study version?
            # If it is the former, look for the active version.
            cs = session.query(CaseStudy).filter(CaseStudy.uuid == uuid_).first()
            if not cs:
                vs = session.query(CaseStudyVersion).filter(CaseStudyVersion.uuid == uuid_).first()
                if not vs:
                    ss = session.query(CaseStudyVersionSession).filter(CaseStudyVersionSession.uuid == uuid_).first()
                    if not ss:
                        raise Exception("Object '"+uuid_+"' not found, when opening a WorkSession")
                    else:
                        vs = ss.version
                        cs = vs.case_study
                else:
                    cs = vs.case_study
            else:  # A case study, find the latest version (the version modified latest -by activity, newest WorkSession-)
                max_date = None
                max_version = None
                for v in cs.versions:
                    for s in v.sessions:
                        if not max_date or s.open_instant > max_date:
                            max_date = s.open_instant
                            max_version = v
                vs = max_version
                cs = vs.case_study

            # List of active sessions
            # NOTE: instead of time ordering, the ID is used, assuming sessions with greater ID were created later
            lst = session.query(CaseStudyVersionSession). \
                filter(CaseStudyVersionSession.version_id == vs.id). \
                order_by(CaseStudyVersionSession.id). \
                all()
            idx = 0
            for i, ws in enumerate(lst):
                if ws.restarts:
                    idx = i
            lst = lst[idx:]  # Cut the list, keep only active sessions

            if cr_new != CreateNew.NO:  # Create either a case study or a case study version
                if cr_new == CreateNew.CASE_STUDY:
                    cs = copy.copy(cs)  # COPY CaseStudy
                else:
                    force_load(cs)
                vs2 = copy.copy(vs)  # COPY CaseStudyVersion
                vs2.case_study = cs  # Assign case study to the new version
                if recover_previous_state:  # If the new version keeps previous state, copy it also
                    vs2.state = vs.state  # Copy state
                    for ws in lst:  # COPY active WorkSessions
                        ws2 = copy.copy(ws)
                        ws2.version = vs2
                        for c in ws.commands:  # COPY commands
                            c2 = copy.copy(c)
                            c2.session = ws2
                vs = vs2
            else:
                # Load into memory
                force_load(vs)
                force_load(cs)

            if recover_previous_state:
                # Load state if it is persisted
                if vs.state:
                    # Deserialize
                    self._isess._state = deserialize_to_object(vs.state)
                else:
                    self._isess._state = State()  # Zero State, execute all commands in sequence
                    for ws in lst:
                        for c in ws.commands:
                            execute_command_generator(self._isess._state, c)
            else:
                self._isess._state = State()

        else:  # New Case Study AND new Case Study Version
            cs = CaseStudy()
            vs = CaseStudyVersion()
            vs.case_study = cs

        # Detach Case Study and Case Study Version
        if cs in session:
            session.expunge(cs)
        if vs in session:
            session.expunge(vs)
        # Create the WorkSession
        usr = session.query(User).filter(User.name == self._identity).first()
        force_load(usr)
        self._session = CaseStudyVersionSession()
        self._session.version = vs
        # If the Version existed, define "restarts" according to parameter "recover_previous_state"
        # ElseIf it is the first Session -> RESTARTS=True
        self._session.restarts = not recover_previous_state if uuid_ else True
        self._session.who = usr

        session.close()
        # session.expunge_all()
        self._sess_factory.remove()

    def save(self, from_web_service=False):
        # if len(self._session.commands) == 0:
        #     raise Exception("The WorkSession will not be saved because it does not contain commands")
        if not self._allow_saving:
            raise Exception("The WorkSession was opened disallowing saving. Please close it and reopen it with the proper value")
        # Serialize state
        st = serialize_from_object(self._isess._state)
        self._session.version.state = st
        self._session.state = st
        # Open DB session
        session = self._sess_factory()
        ws = self._session
        # Append commands, self._session, the version and the case_study
        if not from_web_service:
            for c in self._session.commands:
                session.add(c)
            session.add(ws)
            session.add(ws.version)
            session.add(ws.version.case_study)
        else:
            ws.who = session.merge(ws.who)
            if ws.version.case_study.id:
                ws.version.case_study = session.merge(ws.version.case_study)
            else:
                session.add(ws.version.case_study)
            if ws.version.id:
                ws.version = session.merge(ws.version)
            else:
                session.add(ws.version)
            ws.close_instant = datetime.datetime.utcnow()
            session.add(ws)
            for c in self._session.commands:
                session.add(c)
        # Commit DB session
        session.commit()
        force_load(self._session)
        self._sess_factory.remove()

    @staticmethod
    def create_persistable_command(generator_type, file_type, file):
        return PersistableCommand.create(generator_type, file_type, file)

    def register_persistable_command(self, cmd: PersistableCommand):
        cmd.session = self._session
        return cmd

    def create_and_register_persistable_command(self, generator_type, file_type, file):
        """
        Generates commands from an input stream (string or file)
        There must be a factory to parse stream 
        :param generator_type: 
        :param file_type: 
        :param file: It can be a stream or a URL or a file name
        """
        c = PersistableCommand.create(generator_type, file_type, file)
        c.session = self._session
        return c

    def execute_persistable_command(self, cmd: PersistableCommand):
        execute_command_generator(self._isess._state, cmd)

    def register_executable_command(self, command: IExecutableCommand):
        d = {"command": command._serialization_type,
             "label": command._serialization_label,
             "content": command.json_serialize()}

        self.create_and_register_persistable_command(generator_type="primitive",
                                                     file_type="application/json",
                                                     file=json.dumps(d).encode("utf-8"))

    def set_sf(self, session_factory):
        self._sess_factory = session_factory

    @property
    def case_study(self):
        return self._session.version.case_study

    def close(self) -> tuple:
        if not self._session:
            raise Exception("The WorkSession is not opened")
        id3 = self._session.uuid, self._session.version.uuid, self._session.version.case_study.uuid
        self._session = None
        self._allow_saving = None
        return id3


if __name__ == '__main__':
    # Submit Worksheet as New Case Study
    isess = InteractiveSession()
    isess.quit()

