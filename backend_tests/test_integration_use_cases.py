import unittest
import sqlalchemy

# Memory
from nexinfosys.model_services.workspace import InteractiveSession, CreateNew
from nexinfosys.initialization import prepare_and_reset_database_for_tests
from nexinfosys.command_executors import create_command
from nexinfosys.restful_service import tm_default_users, \
    tm_authenticators, \
    tm_object_types, \
    tm_permissions, \
    tm_case_study_version_statuses

# Database (ORM)
from nexinfosys.models.musiasem_methodology_support import *
import nexinfosys


def setUpModule():
    print('In setUpModule()')
    # Setup SQLAlchemy engines as SQLite IN-MEMORY
    nexinfosys.engine = sqlalchemy.create_engine("sqlite://", echo=True)
    nexinfosys.data_engine = sqlalchemy.create_engine("sqlite://", echo=True)

    # global DBSession # global DBSession registry to get the scoped_session
    DBSession.configure(bind=nexinfosys.engine)  # reconfigure the sessionmaker used by this scoped_session
    tables = ORMBase.metadata.tables
    connection = nexinfosys.engine.connect()
    table_existence = [nexinfosys.engine.dialect.has_table(connection, tables[t].name) for t in tables]
    connection.close()
    if False in table_existence:
        ORMBase.metadata.bind = nexinfosys.engine
        ORMBase.metadata.create_all()

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


def tearDownModule():
    print('In tearDownModule()')
    nexinfosys.data_engine.dispose()
    nexinfosys.engine.dispose()
    print(str(len(tm_authenticators)))
    print("a")


# --- Functions containing pieces of code used in several unit tests ----------------

def get_metadata_command():  # UTILITY FUNCTION
    # Fields: ("<field label in Spreadsheet file>", "<field name in Dublin Core>", Mandatory?, Controlled?)
    # [("Case study name", "title", True, False),  # DEPRECATED
    #   ("Case study code", "title", True, False),
    #   ("Title", "title", True, False),
    #   ("Subject, topic and/or keywords", "subject", False, True),
    #   ("Description", "description", False, False),
    #   ("Level", "description", False, True),
    #   ("Dimensions", "subject", True, True),
    #   ("Reference documentation", "source", False, False),
    #   ("Authors", "creator", True, False),
    #   ("Date of elaboration", "date", True, False),
    #   ("Temporal situation", "coverage", True, False),
    #   ("Geographical location", "coverage", True, True),
    #   ("DOI", "identifier", False, False),
    #   ("Language", "language", True, True)
    # ]
    metadata = {"CaseStudyName": "A test",
                "CaseStudyCode": None,
                "Title": "Case study for test only",
                "SubjectTopicKeywords": ["Desalination", "Renewables", "Sustainability"],
                "Description": "A single command case study",
                "Level": "Local",
                "Dimensions": ["Water", "Energy", "Food"],
                "ReferenceDocumentation": ["reference 1", "reference 2"],
                "Authors": ["Author 1", "Author 2"],
                "DateOfElaboration": "2017-11-01",
                "TemporalSituation": "Years 2015 and 2016",
                "GeographicalLocation": "South",
                "DOI": None,
                "Language": "English"
                }

    cmd, issues = create_command("metadata", None, metadata)
    return cmd


def new_case_study(with_metadata_command=0):
    """
    * Prepare an InteractiveSession,
    * Open a ReproducibleSession,
    * (optionally) Add a metadata command,
    * Close the ReproducibleSession

    :param with_metadata_command: 1 -> execute, 2 -> register, 3 -> execute and register
    :return: UUID of the CaseStudyVersionSession, InteractiveSession
    """
    isess = InteractiveSession(DBSession)
    isess.identify({"user": "test_user"}, testing=True)  # Pass just user name.
    isess.open_reproducible_session(case_study_version_uuid=None,
                                    recover_previous_state=None,
                                    cr_new=None,
                                    allow_saving=True)
    issues = None
    output = None
    if with_metadata_command > 0:
        cmd = get_metadata_command()
        if with_metadata_command & 1:
            issues, output = isess.execute_executable_command(cmd)
        if with_metadata_command & 2:
            isess.register_executable_command(cmd)
    uuid, v_uuid, cs_uuid = isess.close_reproducible_session(issues, output, save=True)
    return uuid, isess


class TestHighLevelUseCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass  # Executed BEFORE test methods of the class

    @classmethod
    def tearDownClass(cls):
        pass  # Executed AFTER tests methods of the class

    def setUp(self):
        super().setUp()
        pass  # Repeated BEFORE each test...

    def tearDown(self):
        pass  # Repeated AFTER each test...
        super().tearDown()

    # ###########################################################
    def test_001_isession_open_and_close(self):
        isess = InteractiveSession(DBSession)
        isess.quit()
        self.assertEqual(1, 1)

    def test_002_isession_open_identify_and_close(self):
        isess = InteractiveSession(DBSession)
        isess.identify({"user": "test_user"}, testing=True)  # Pass just user name.
        ide = isess.get_identity_id()
        isess.quit()
        self.assertEqual(ide, "test_user")

    def test_003_new_case_study_with_no_command(self):
        prepare_and_reset_database_for_tests(True)
        uuid2, isess = new_case_study(with_metadata_command=0)
        isess.quit()
        self.assertIsNotNone(uuid2, "UUID should be defined after saving+closing the reproducible session")
        # Check there is a case study, and a case study version and a session
        session = DBSession()
        self.assertEqual(len(session.query(CaseStudy).all()), 1)
        self.assertEqual(len(session.query(CaseStudyVersion).all()), 1)
        self.assertEqual(len(session.query(CaseStudyVersionSession).all()), 1)
        session.close()

    def test_004_new_case_study_with_only_metadata_command(self):
        prepare_and_reset_database_for_tests(True)
        uuid2, isess = new_case_study(with_metadata_command=1)  # 1 means "execute" (do not register)
        # Check State
        md = isess._state.get("_metadata")
        isess.quit()
        self.assertIsNotNone(md)
        self.assertEqual(md["Title"], "Case study for test only")
        # Check Session should not have CommandsContainer
        session = DBSession()
        tmp = len(session.query(CaseStudy).all())
        self.assertEqual(tmp, 1)
        self.assertEqual(len(session.query(CaseStudyVersion).all()), 1)
        self.assertEqual(len(session.query(CaseStudyVersionSession).all()), 1)
        self.assertEqual(len(session.query(CommandsContainer).all()), 0)
        session.close()

    def test_005_new_case_study_with_metadata_plus_dummy_command(self):
        prepare_and_reset_database_for_tests(True)
        # ----------------------------------------------------------------------
        # Create case study, with one CommandsContainer
        uuid_, isess = new_case_study(with_metadata_command=3)
        session = DBSession()
        self.assertEqual(len(session.query(CaseStudy).all()), 1)
        self.assertEqual(len(session.query(CaseStudyVersion).all()), 1)
        self.assertEqual(len(session.query(CaseStudyVersionSession).all()), 1)
        self.assertEqual(len(session.query(CommandsContainer).all()), 1)
        session.close()

        # Reset State
        isess.reset_state()
        self.assertIsNone(isess._state.get("metadata"))  # After reset, no variable should be there

        # ----------------------------------------------------------------------
        # SECOND work session, resume and add DummyCommand
        isess.open_reproducible_session(case_study_version_uuid=uuid_,
                                        recover_previous_state=True,
                                        cr_new=CreateNew.NO,
                                        allow_saving=True)
        self.assertIsNotNone(isess._state.get("_metadata"))
        d_cmd, _ = create_command("dummy", None, {"name": "var_a", "description": "Content"})
        self.assertIsNone(isess._state.get("var_a"))
        issues, output = isess.execute_executable_command(d_cmd)
        self.assertIsNotNone(isess._state.get("var_a"))
        isess.register_executable_command(d_cmd)
        uuid2, _, _ = isess.close_reproducible_session(issues, output, save=True)
        # Check Database objects
        session = DBSession()
        self.assertEqual(len(session.query(CommandsContainer).all()), 2)
        self.assertEqual(len(session.query(CaseStudy).all()), 1)
        self.assertEqual(len(session.query(CaseStudyVersion).all()), 1)
        self.assertEqual(len(session.query(CaseStudyVersionSession).all()), 1)
        session.close()
        isess.reset_state()

        # ----------------------------------------------------------------------
        # Just Check that State was saved. Two variables: metadata and "var_a"
        isess.open_reproducible_session(case_study_version_uuid=uuid2,
                                        recover_previous_state=True,
                                        cr_new=CreateNew.NO,
                                        allow_saving=True)
        self.assertIsNotNone(isess._state.get("_metadata"))
        self.assertIsNotNone(isess._state.get("var_a"))
        isess.close_reproducible_session()  # Dismiss ReproducibleSession

        # ----------------------------------------------------------------------
        # Now create a new version, COPYING the previous, add TWO commands
        # One of them OVERWRITES the state of one variable
        isess.open_reproducible_session(case_study_version_uuid=uuid2,
                                        recover_previous_state=True,
                                        cr_new=CreateNew.VERSION,
                                        allow_saving=True)
        d_cmd, _ = create_command("dummy", None, {"name": "var_b", "description": "To test var storage and retrieval"})
        issues, output = isess.execute_executable_command(d_cmd)
        isess.register_executable_command(d_cmd)
        d_cmd, _ = create_command("dummy", None, {"name": "var_a", "description": "Overwritten"})
        issues, output = isess.execute_executable_command(d_cmd)
        isess.register_executable_command(d_cmd)
        uuid2, _, _ = isess.close_reproducible_session(issues, output, save=True)
        # Database objects
        session = DBSession()
        cs_lst = session.query(CaseStudy).all()
        self.assertEqual(len(cs_lst), 1)
        self.assertEqual(len(session.query(CaseStudyVersion).all()), 2)
        self.assertEqual(len(session.query(CaseStudyVersionSession).all()), 3)
        self.assertEqual(len(session.query(CommandsContainer).all()), 6)
        session.close()

        # ----------------------------------------------------------------------
        # Now create a NEW CASE STUDY, COPYING the second version of the first case study
        # TODO fix it!

        # isess.open_reproducible_session(case_study_version_uuid=uuid2,
        #                                 recover_previous_state=True,
        #                                 cr_new=CreateNew.CASE_STUDY,
        #                                 allow_saving=True)
        # d_cmd, _ = create_command("dummy", None, {"name": "var_b", "description": "Another value"})
        # issues, output = isess.execute_executable_command(d_cmd)
        # isess.register_executable_command(d_cmd)
        # uuid3, _, _ = isess.close_reproducible_session(issues, output, save=True)
        # # Database objects
        # session = DBSession()
        # cs_lst = session.query(CaseStudy).all()
        # self.assertEqual(len(cs_lst), 2)
        # self.assertEqual(len(session.query(CaseStudyVersion).all()), 3)
        # self.assertEqual(len(session.query(CaseStudyVersionSession).all()), 3)
        # self.assertEqual(len(session.query(CommandsContainer).all()), 11)
        # session.close()

        # ----------------------------------------------------------------------
        # Create a NEW VERSION, with restart
        # isess.open_reproducible_session(case_study_version_uuid=uuid3,
        #                                 recover_previous_state=False,
        #                                 cr_new=CreateNew.VERSION,
        #                                 allow_saving=True)
        # d_cmd, _ = create_command("dummy", None, {"name": "var_b", "description": "Another value"})
        # issues, output = isess.execute_executable_command(d_cmd)
        # isess.register_executable_command(d_cmd)
        # uuid4, _, _ = isess.close_reproducible_session(issues, output, save=True)
        #
        # isess.open_reproducible_session(case_study_version_uuid=uuid4,
        #                                 recover_previous_state=True,
        #                                 cr_new=CreateNew.NO,
        #                                 allow_saving=True)
        # self.assertIsNone(isess._state.get("var_a"))
        # self.assertIsNotNone(isess._state.get("var_b"))
        # isess.close_reproducible_session(issues, output, save=False)
        #
        # # Database objects
        # session = DBSession()
        # self.assertEqual(len(session.query(CaseStudy).all()), 2)
        # self.assertEqual(len(session.query(CaseStudyVersion).all()), 4)
        # self.assertEqual(len(session.query(CaseStudyVersionSession).all()), 3)
        # self.assertEqual(len(session.query(CommandsContainer).all()), 12)
        # session.close()

        # ----------------------------------------------------------------------
        # CHECK that we have separate states. Open the first version. "var_b" is not there, "var_a" equals "Content"
        # FIRST VERSION
        # isess.open_reproducible_session(case_study_version_uuid=uuid_,
        #                                 recover_previous_state=True,
        #                                 cr_new=CreateNew.NO,
        #                                 allow_saving=True)
        # self.assertEqual(isess._state.get("var_a"), "Content")
        # self.assertIsNone(isess._state.get("var_b"))
        # isess.close_reproducible_session()  # Dismiss ReproducibleSession
        #
        # # SECOND VERSION same case study
        # isess.open_reproducible_session(case_study_version_uuid=uuid2,
        #                                 recover_previous_state=True,
        #                                 cr_new=CreateNew.NO,
        #                                 allow_saving=True)
        # self.assertEqual(isess._state.get("var_a"), "Overwritten")
        # self.assertIsNotNone(isess._state.get("var_b"))
        # isess.close_reproducible_session()  # Dismiss ReproducibleSession
        #
        # # FIRST VERSION new CASE STUDY, from second VERSION first CASE STUDY
        # isess.open_reproducible_session(case_study_version_uuid=uuid3,
        #                                 recover_previous_state=True,
        #                                 cr_new=CreateNew.NO,
        #                                 allow_saving=True)
        # self.assertEqual(isess._state.get("var_b"), "Another value")
        # self.assertIsNotNone(isess._state.get("var_a"))
        # isess.close_reproducible_session()  # Dismiss ReproducibleSession

        # TODO Test clone BUT without RESTART --> ALL sessions should be dismissed, in this case it makes no sense cloning State, Sessions and Commands

        # TODO What if another session is opened?? It should be a close without saving followed by an open

        isess.quit()

    #
    # def test_submit_worksheet_new_case_study(self):
    #     """
    #     * Submit Worksheet
    #       - Interactive session
    #       - Open work session
    #       - Submit file
    #     	- the file produces a sequence of commands
    #     	- execute
    #     	  - elaborate output file and compile issues
    #       - Close Interactive session (save, new case study version)
    #       - Close user session
    #
    #     """
    #     # Submit Worksheet as New Case Study
    #     ## OTHER Submit with a single Metadata command
    #     ## OTHER Submit but do not save
    #     ## OTHER Submit but only save, do not execute (in that case issues and output cannot be obtained)
    #     ## OTHER Submit but save with no identity
    #     isess = InteractiveSession()
    #     isess.identify({"user": "rnebot"}) # Pass just user name.
    #     isess.open_reproducible_session(None, SessionCreationAction.NewCaseStudy)
    #     issues, output = isess.execute_command_container("worksheet", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "<whatever>")
    #     isess.close_reproducible_session(issues, output, save=True)  # It has to create a case study, a case study version, and a work session
    #     isess.quit()
    #
    # def test_submit_worksheet_new_version(self):
    #     pass
    #
    # def test_submit_worksheet_evolve_version(self):
    #     pass
    #
    # def test_submit_r_script_new_case_study(self):
    #     pass
    #
    # def test_delete_case_study_version(self):
    #     pass
    #
    # def test_delete_case_study(self):
    #     pass
    #
    # def test_list_case_studies_1(self):
    #     # Submit Worksheet as New Case Study
    #     isess = InteractiveSession()
    #     isess.identify({"user": "test_user"}, testing=True) # Pass just user name.
    #     lst = isess.get_case_studies()
    #     cs_guid = None
    #     vs_lst = isess.get_case_study_versions(cs_guid)
    #     vs_guid = None
    #     vs = isess.get_case_study_version(vs_guid)  # Includes a list of variables with their types
    #     obj = isess.get_case_study_version_variable(vs, "var_name")  # Single object
    #     isess.quit()
    #
    # def test_list_case_studies_2(self):
    #     # Submit Worksheet as New Case Study
    #     isess = InteractiveSession()
    #     isess.identify({"user": "rnebot"}) # Pass just user name.
    #     lst = isess.get_case_studies()
    #     cs_guid = None
    #     vs_lst = isess.get_case_study_versions(cs_guid)
    #     vs_guid = None
    #     vs = isess.get_case_study_version(vs_guid)  # Includes a list of variables with their types
    #     obj = isess.get_case_study_version_variable(vs, "var_name")  # Single object
    #     isess.quit()
    #
    # def test_list_case_studies_3(self):
    #     # Submit Worksheet as New Case Study
    #     isess = InteractiveSession()
    #     isess.identify({"user": "rnebot"}) # Pass just user name.
    #     lst = isess.get_case_studies()
    #     cs_guid = None
    #     vs_lst = isess.get_case_study_versions(cs_guid)
    #     vs_guid = None
    #     vs = isess.get_case_study_version(vs_guid)  # Includes a list of variables with their types
    #     obj = isess.get_case_study_version_variable(vs, "var_name")  # Single object
    #     isess.quit()


if __name__ == '__main__':
    setUpModule()
    i = TestHighLevelUseCases()
    i.test_003_new_case_study_with_no_command()
    #i.test_005_new_case_study_with_only_metadata_command()

    print("Just a placeholder (useful when running to check if the file is syntactically valid")
