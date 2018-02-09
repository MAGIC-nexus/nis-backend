import unittest
import os

from backend.model_services import get_case_study_registry_objects
from backend_tests.test_integration_use_cases import setUpModule, tearDownModule, new_case_study, reset_database
from backend.model_services.workspace import InteractiveSession, CreateNew
from backend.command_generators.json import create_command

# Database (ORM)
from backend.model.persistent_db.persistent import *
import backend


def execute_file(file_name, generator_type):
    if generator_type == "spreadsheet":
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        read_type = "rb"
    elif generator_type == "native":
        content_type = "application/json"
        read_type = "r"

    reset_database()
    isess = InteractiveSession(DBSession)
    isess.identify({"user": "test_user"}, testing=True)  # Pass just user name.
    isess.open_reproducible_session(case_study_version_uuid=None,
                                    recover_previous_state=None,
                                    cr_new=CreateNew.CASE_STUDY,
                                    allow_saving=False)
    with open(file_name, read_type) as f:
        buffer = f.read()
    ret = isess.register_andor_execute_command_generator(generator_type, content_type, buffer, False, True)
    isess.close_reproducible_session()
    isess.close_db_session()
    return isess


class TestCommandFiles(unittest.TestCase):
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

    def test_001_execute_file_one(self):
        """
        A file containing QQs for three different sets of processor: Crop, Farm, AgrarianRegion
        (extracted from Almeria case study)
        Test number of processors read for each category, using processor sets and PartialRetrievalDictionary
        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_1.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # Three processor sets
        self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_002_execute_file_two(self):
        """
        A file containing QQs for three different sets of processor: Crop, Farm, AgrarianRegion
        AND Upscaling
        (extracted from Almeria case study)
        Test number of processors read for each category, using processor sets and PartialRetrievalDictionary
        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_2.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # Three processor sets
        self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_003_execute_file_three(self):
        """
        * Declares TWO mappings
        * Reads TWO Eurostat datasets
        * Uses one of the datasets to feed QQs

        Test number of processors read for each category, using processor sets and PartialRetrievalDictionary
        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_3_dataset.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # Three processor sets
        self.assertEqual(len(p_sets), 1)
        # Close interactive session
        isess.close_db_session()

    def test_004_execute_file_four(self):
        """

        Parameters
        Simple Expression evaluation in QQs

        :return:
        """

    def test_005_execute_file_five(self):
        """
        Just Structure. From Soslaires.

        :return:
        """
