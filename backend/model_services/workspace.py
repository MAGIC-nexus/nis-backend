# -*- coding: utf-8 -*-
"""
* Registry of objects. Add, remove, search
* Support for high level operations: directly create and/or modify objects, calling the specification API. Create connections
"""
import copy
import datetime
import json
import logging
import uuid
from enum import Enum
from typing import List

from backend.command_generators.parsers_factory import commands_container_parser_factory
from backend.model.persistent_db.persistent import (User,
                                                    CaseStudy,
                                                    CaseStudyVersion,
                                                    CaseStudyVersionSession,
                                                    CommandsContainer,
                                                    force_load
                                                    )
from backend.model_services import IExecutableCommand
from backend.model_services import State
from backend.restful_service.serialization import serialize_state, deserialize_state

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


class CommandResult:
    pass

# class SessionCreationAction(Enum):  # Used in FlowFund
#     """
#
#         +--------+--------+---------+---------------------------------+
#         | branch | clone  | restart |        Behavior                 |
#         +--------+--------+---------+---------------------------------+
#         | True   | True   | True    | Branch & New ReproducibleSession|
#         | True   | True   | False   | Branch & Clone                  |
#         | True   | False  | True    | Branch & New ReproducibleSession|
#         | True   | False  | False   | Branch & New ReproducibleSession|
#         | False  | True   | True    | New CS (not CS version)         |
#         | False  | True   | False   | New CS & Clone                  |
#         | False  | False  | True    | Restart                         |
#         | False  | False  | False   | Continue                        |
#         +--------+--------+---------+---------------------------------+
#     """
#     BranchAndNewWS = 7
#     BranchAndCloneWS = 6
#     NewCaseStudy = 3
#     NewCaseStudyCopyFrom = 2
#     Restart = 1
#     Continue = 0
# ---------------------------------------------------------------------------------------------------------------------

# #####################################################################################################################
# >>>> PRIMITIVE COMMAND & COMMAND GENERATOR PROCESSING FUNCTIONS <<<<
# #####################################################################################################################


def executable_command_to_commands_container(e_cmd: IExecutableCommand):
    """
    IExecutableCommand -> CommandsContainer

    The resulting CommandsContainer will be always in native (JSON) format, because the specification
    to construct an IExecutableCommand has been translated to this native format.

    :param command:
    :return:
    """
    d = {"command": e_cmd._serialization_type,
         "label": e_cmd._serialization_label,
         "content": e_cmd.json_serialize()}

    return CommandsContainer.create("native", "application/json", json.dumps(d).encode("utf-8"))


def persistable_to_executable_command(p_cmd: CommandsContainer, limit=1000):
    """
    A persistable command can be either a single command or a sequence of commands (like a spreadsheet). In the future
    it could even be a full script.

    Because the executable command is DIRECTLY executable, it is not possible to convert from persistable to a single
    executable command. But it is possible to obtain a list of executable commands, and this is the aim of this function

    The size of the list can be limited by the parameter "limit". "0" is for unlimited

    :param p_cmd:
    :return: A list of IExecutableCommand
    """
    # Generator factory (from generator_type and file_type)
    # Generator has to call "yield" whenever an ICommand is generated
    issues_aggreg = []
    outputs = []
    state = State()
    count = 0
    for cmd, issues in commands_container_parser_factory(p_cmd.generator_type, p_cmd.file_type, p_cmd.file, state):
        # If there are syntax ERRORS, STOP!!!
        stop = False
        if issues and len(issues) > 0:
            for t in issues:
                if t[0] == 3:  # Error
                    stop = True
        if stop:
            break

        issues_aggreg.extend(issues)

        count += 1
        if count >= limit:
            break


def execute_command(state, e_cmd: "IExecutableCommand"):
    return e_cmd.execute(state)


def execute_command_container(state, p_cmd: CommandsContainer):
    return execute_command_container_file(state, p_cmd.generator_type, p_cmd.content_type, p_cmd.content)


def execute_command_container_file(state, generator_type, file_type: str, file):
    """
    Creates a generator parser, then it feeds the file type and the file
    The generator parser has to parse the file and to generate command_executors as a Python generator

    :param generator_type:
    :param file_type:
    :param file:
    :return: Issues and outputs (still not defined) -TODO-
    """
    # Generator factory (from generator_type and file_type)
    # Generator has to call "yield" whenever an ICommand is generated
    issues_aggreg = []
    outputs = []
    for cmd, issues in commands_container_parser_factory(generator_type, file_type, file, state):
        # If there are syntax ERRORS, STOP!!!
        stop = False
        if issues and len(issues) > 0:
            for t in issues:
                if isinstance(t, dict):
                    if t["type"] == 3:
                        stop = True
                elif isinstance(t, tuple):
                    if t[0] == 3:  # Error
                        stop = True
        if stop:
            break

        issues_aggreg.extend(issues)
        i, output = execute_command(state, cmd)
        if i:
            issues_aggreg.extend(i)
        if output:
            outputs.append(output)

    return issues_aggreg, outputs


def convert_generator_to_native(generator_type, file_type: str, file):
    """
    Converts a generator
    Creates a generator parser, then it feeds the file type and the file
    The generator parser has to parse the file and to elaborate a native generator (JSON)

    :param generator_type:
    :param file_type:
    :param file:
    :return: Issues and output file
    """
    # Generator factory (from generator_type and file_type)
    # Generator has to call "yield" whenever an ICommand is generated
    output = []
    if generator_type.lower() not in ["json", "native", "primitive"]:
        state = State()
        for cmd, issues in commands_container_parser_factory(generator_type, file_type, file, state):
            # If there are syntax ERRORS, STOP!!!
            stop = False
            if issues and len(issues) > 0:
                for t in issues:
                    if t["type"] == 3:  # Error
                        stop = True
                        break

            output.append({"command": cmd._serialization_type,
                           "label": cmd._serialization_label,
                           "content": cmd.json_serialize(),
                           "issues": issues
                           }
                          )
            if stop:
                break

    return output

# #####################################################################################################################
# >>>> INTERACTIVE SESSION <<<<
# #####################################################################################################################


class CreateNew(Enum):
    CASE_STUDY = 1
    VERSION = 2
    NO = 3


class InteractiveSession:
    """ 
    Main class for interaction with NIS
    The first thing would be to identify the user and create a GUID for the session which can be used by the web server
    to store and retrieve the interactive session state.
    
    It receives command_executors, modifying state accordingly
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
        self._reproducible_session = None  # type: ReproducibleSession

    def reset_state(self):
        """ Restart state """
        self._state = State()
        # TODO self._recordable_session = None ??

    @property
    def state(self):
        return self._state

    def get_sf(self):
        return self._session_factory

    def set_sf(self, session_factory):
        self._session_factory = session_factory
        if self._reproducible_session:
            self._reproducible_session.set_sf(session_factory)

    def open_db_session(self):
        return self._session_factory()

    def close_db_session(self):
        self._session_factory.remove()

    def quit(self):
        """
        End interactive session
        :return: 
        """
        self.close_reproducible_session()
        self.close_db_session()

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
                    if self._state:
                        self._state.set("_identity", self._identity)

                return src is not None
            elif "token" in identity_information:
                # TODO Validate against some Authentication service
                pass

    def get_identity_id(self):
        return self._identity

    def unidentify(self):
        # TODO The un-identification cannot be done in the following circumstances: any?
        self._identity = None
        if self._state:
            self._state.set("_identity", self._identity)

    # --------------------------------------------------------------------------------------------
    # Reproducible sessions and commands INSIDE them
    # --------------------------------------------------------------------------------------------
    def open_reproducible_session(self,
                                  case_study_version_uuid: str,
                                  recover_previous_state=True,
                                  cr_new: CreateNew = CreateNew.NO,
                                  allow_saving=True):
        self._reproducible_session = ReproducibleSession(self)
        self._reproducible_session.open(self._session_factory, case_study_version_uuid, recover_previous_state, cr_new, allow_saving)

    def close_reproducible_session(self, issues=None, output=None, save=False, from_web_service=False):
        if self._reproducible_session:
            if save:
                # TODO Save issues AND (maybe) output
                self._reproducible_session.save(from_web_service)
            uuid_, v_uuid, cs_uuid = self._reproducible_session.close()
            self._reproducible_session = None
            return uuid_, v_uuid, cs_uuid
        else:
            return None, None, None

    def reproducible_session_opened(self):
        return self._reproducible_session is not None

    # --------------------------------------------------------------

    def execute_executable_command(self, cmd: IExecutableCommand):
        return execute_command(self._state, cmd)

    def register_executable_command(self, cmd: IExecutableCommand):
        self._reproducible_session.register_executable_command(cmd)

    def register_andor_execute_command_generator(self, generator_type, file_type: str, file, register=True, execute=False):
        """
        Creates a generator parser, then it feeds the file type and the file
        The generator parser has to parse the file and to generate command_executors as a Python generator

        :param generator_type: 
        :param file_type: 
        :param file: 
        :param register: If True, register the command in the ReproducibleSession
        :param execute: If True, execute the command in the ReproducibleSession
        :return: 
        """
        if not self._reproducible_session:
            raise Exception("In order to execute a command generator, a work session is needed")
        if not register and not execute:
            raise Exception("More than zero of the parameters 'register' and 'execute' must be True")

        # Prepare persistable command
        c = CommandsContainer.create(generator_type, file_type, file)
        if register:
            self._reproducible_session.register_persistable_command(c)
        if execute:
            pass_case_study = self._reproducible_session._session.version.case_study is not None
            return self._reproducible_session.execute_command_generator(c, pass_case_study)
            # Or
            # return execute_command_container(self._state, c)
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

    def get_case_study_version_variables(self, case_study_version: str):
        """ A tree of variables, by type: processors, flows """
        pass

    def get_case_study_version_variable(self, case_study_version: str, variable: str):
        pass

    def remove_case_study_version(self, case_study_version: str):
        pass

    def share_case_study(self, case_study: str, identities: List[str], permission: str):
        pass

    def remove_case_study_share(self, case_study: str, identities: List[str], permission: str):
        pass

    def get_case_study_permissions(self, case_study: str):
        pass

    def export_case_study_version(self, case_study_version: str):
        pass

    def import_case_study(self, file):
        pass

# #####################################################################################################################
# >>>> REPRODUCIBLE SESSION <<<<
# #####################################################################################################################


class ReproducibleSession:
    def __init__(self, isess):
        # Containing InteractiveSession. Used to set State when a ReproducibleSession is opened and it overwrites State
        self._isess = isess  # type: InteractiveSession
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
        :param recover_previous_state: If an existing version is specified, it will recover its state after execution of all command_executors
        :param cr_new: If != CreateNew.NO, create either a case study or a new version. If == CreateNew.NO, append session to "uuid"
        :param allow_saving: If True, it will allow saving at the end (it will be optional). If False, trying to save will generate an Exception
        :return UUID of the case study version in use. If it is a new case study and it has not been saved, the value will be "None"
        """

        # TODO Just register for now. But in the future it should control that there is no other "allow_saving" ReproducibleSession opened
        # TODO for the same Case Study Version. So it implies modifying some state in CaseStudyVersion to have the UUID
        # TODO of the active ReproducibleSession, even if it is not in the database. Register also the date of "lock", so the
        # TODO lock can be removed in case of "hang" of the locker ReproducibleSession
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
                        raise Exception("Object '"+uuid_+"' not found, when opening a ReproducibleSession")
                    else:
                        vs = ss.version
                        cs = vs.case_study
                else:
                    cs = vs.case_study
            else:  # A case study, find the latest version (the version modified latest -by activity, newest ReproducibleSession-)
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
                    cs = copy.copy(cs)  # New Case Study: COPY CaseStudy
                else:
                    force_load(cs)  # New Case Study Version: LOAD CaseStudy (then version it)
                vs2 = copy.copy(vs)  # COPY CaseStudyVersion
                vs2.case_study = cs  # Assign case study to the new version
                if recover_previous_state:  # If the new version keeps previous state, copy it also
                    vs2.state = vs.state  # Copy state
                    for ws in lst:  # COPY active ReproducibleSessions
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
                    self._isess._state = deserialize_state(vs.state)
                else:
                    self._isess._state = State()  # Zero State, execute all commands in sequence
                    for ws in lst:
                        for c in ws.commands:
                            execute_command_container(self._isess._state, c)
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
        # Create the Case Study Version Session
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
        if not self._allow_saving:
            raise Exception("The ReproducibleSession was opened disallowing saving. Please close it and reopen it with the proper value")
        # Serialize state
        st = serialize_state(self._isess._state)
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

    def register_persistable_command(self, cmd: CommandsContainer):
        cmd.session = self._session

    def create_and_register_persistable_command(self, generator_type, file_type, file):
        """
        Generates command_executors from an input stream (string or file)
        There must be a factory to parse stream 
        :param generator_type: 
        :param file_type: 
        :param file: It can be a stream or a URL or a file name
        """
        c = CommandsContainer.create(generator_type, file_type, file)
        self.register_persistable_command(c)
        return c

    def execute_command_generator(self, cmd: CommandsContainer, pass_case_study=False):
        if pass_case_study:  # CaseStudy can be modified by Metadata command, pass a reference to it
            self._isess._state.set("_case_study", self._session.version.case_study)

        ret = execute_command_container(self._isess._state, cmd)

        if pass_case_study:
            self._isess._state.set("_case_study", None)

        return ret

    def register_executable_command(self, command: IExecutableCommand):
        c = executable_command_to_commands_container(command)
        c.session = self._session

    def set_sf(self, session_factory):
        self._sess_factory = session_factory

    @property
    def case_study(self):
        return self._session.version.case_study

    def close(self) -> tuple:
        if not self._session:
            raise Exception("The CaseStudyVersionSession is not opened")
        id3 = self._session.uuid, self._session.version.uuid, self._session.version.case_study.uuid
        self._session = None
        self._allow_saving = None
        return id3


if __name__ == '__main__':
    # Submit Worksheet as New Case Study
    isess = InteractiveSession()
    isess.quit()
