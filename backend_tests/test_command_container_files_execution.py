import unittest

import os

from backend.model_services import get_case_study_registry_objects
from backend_tests.test_integration_use_cases import setUpModule, tearDownModule, new_case_study, reset_database
from backend.model_services.workspace import InteractiveSession, CreateNew
from backend.model.memory.musiasem_concepts import Observer, \
    Processor, FactorType, Factor, \
    Hierarchy, \
    FactorQuantitativeObservation, RelationObservation, ProcessorsRelationPartOfObservation, \
    ProcessorsRelationUndirectedFlowObservation, ProcessorsRelationUpscaleObservation, \
    FactorsRelationDirectedFlowObservation

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
        # file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_2.xlsx"
        # isess = execute_file(file_path, generator_type="spreadsheet")
        # # Check State of things
        # glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # # Three processor sets
        # self.assertEqual(len(p_sets), 3)
        # # Close interactive session
        # isess.close_db_session()

    def test_003_execute_file_three(self):
        """
        * Declares TWO mappings
        * Reads TWO Eurostat datasets
        * Uses one of the datasets to feed QQs

        Test number of processors read for each category, using processor sets and PartialRetrievalDictionary
        :return:
        """
        # file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_3_dataset.xlsx"
        # isess = execute_file(file_path, generator_type="spreadsheet")
        # # Check State of things
        # glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # # Three processor sets
        # self.assertEqual(len(p_sets), 1)
        # # Close interactive session
        # isess.close_db_session()

    def test_004_execute_file_four(self):
        """

        Parameters
        Simple Expression evaluation in QQs

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_4.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # Three processor sets
        self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_005_execute_file_five(self):
        """
        Just Structure. From Soslaires.

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/Soslaires.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # Four processor sets
        self.assertEqual(len(p_sets), 4)
        # Obtain all Observers
        print("---- Observer ----")
        oers = glb_idx.get(Observer.partial_key())
        for i in oers:
            print(i.name)
        # Obtain all processors
        print("---- Processor ----")
        procs = glb_idx.get(Processor.partial_key())
        for i in procs:
            print(i.name)
        # Obtain all FactorTypes
        print("---- FactorType ----")
        fts = glb_idx.get(FactorType.partial_key())
        for i in fts:
            print(i.name)
        # Obtain all Factors
        print("---- Factor ----")
        fs = glb_idx.get(Factor.partial_key())
        for i in fs:
            print(i.processor.name + ":" + i.taxon.name)
        # Obtain all Quantitative Observations
        print("---- Quantities ----")
        qqs = glb_idx.get(FactorQuantitativeObservation.partial_key())
        for i in qqs:
            print(i.factor.processor.name + ":" + i.factor.taxon.name + "= " + str(i.value.expression if i.value else ""))
        # Obtain all part-of Relation Observations
        print("---- Part-of relations (P-P) ----")
        po_rels = glb_idx.get(ProcessorsRelationPartOfObservation.partial_key())
        for i in po_rels:
            print(i.parent_processor.name + " \/ " + i.child_processor.name)
        # Obtain all undirected flow Relation Observations
        print("---- Undirected flow relations (P-P) ----")
        uf_rels = glb_idx.get(ProcessorsRelationUndirectedFlowObservation.partial_key())
        for i in uf_rels:
            print(i.source_processor.name + " <> " + i.target_processor.name)
        # Obtain all upscale Relation Observations
        print("---- Upscale relations (P-P) ----")
        up_rels = glb_idx.get(ProcessorsRelationUpscaleObservation.partial_key())
        for i in up_rels:
            print(i.parent_processor.name + " \/ " + i.child_processor.name + "(" + i.factor_name+ ": " + str(i.quantity) + ")")
        # Obtain all directed flow Relation Observations
        print("---- Directed flow relations (F-F) ----")
        df_rels = glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key())
        for i in df_rels:
            print(i.source_factor.processor.name + ":" + i.source_factor.taxon.name + " -> " +
                  i.target_factor.processor.name + ":" + i.target_factor.taxon.name + (" (" + str(i.weight) + ")" if i.weight else ""))
        # Obtain all hierarchies
        print("---- FactorType Hierarchies ----")
        hies = glb_idx.get(Hierarchy.partial_key())
        for i in hies:
            print(i.name)
        # Close interactive session
        isess.close_db_session()
