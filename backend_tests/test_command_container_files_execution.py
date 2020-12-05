import os
import unittest

import nexinfosys
from nexinfosys.ie_exports.json_export import export_model_to_json
from nexinfosys.model_services import get_case_study_registry_objects
from nexinfosys.model_services.workspace import execute_file, prepare_and_solve_model, execute_file_return_issues
from nexinfosys.initialization import prepare_and_reset_database_for_tests
from nexinfosys.models.musiasem_concepts import Observer, \
    Processor, FactorType, Factor, \
    Hierarchy, \
    FactorQuantitativeObservation, ProcessorsRelationPartOfObservation, \
    ProcessorsRelationUndirectedFlowObservation, ProcessorsRelationUpscaleObservation, \
    FactorsRelationDirectedFlowObservation
# Database (ORM)
from nexinfosys import register_external_datasources
from nexinfosys.serialization import serialize_state, deserialize_state
from nexinfosys.solving import get_processor_names_to_processors_dictionary
from nexinfosys.common.helper import add_label_columns_to_dataframe


class TestFAOCommandFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Executed BEFORE test methods of the class
        prepare_and_reset_database_for_tests(
            prepare=True,
            metadata_string="sqlite:////home/rnebot/GoogleDrive/AA_MAGIC/nis_metadata.db",
            data_string="sqlite:////home/rnebot/GoogleDrive/AA_MAGIC/nis_cached_datasets.db")
        nexinfosys.data_source_manager = register_external_datasources(
            {"FAO_DATASETS_DIR": "/home/marco/temp/Data/FAOSTAT/"})

    @classmethod
    def tearDownClass(cls):
        pass  # Executed AFTER tests methods of the class

    def setUp(self):
        super().setUp()
        pass  # Repeated BEFORE each test...

    def tearDown(self):
        pass  # Repeated AFTER each test...
        super().tearDown()

    def test_001_fao(self):
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/16_fao_fbs_test.xlsx"
        isess, issues = execute_file_return_issues(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        df = datasets["fbs2"].data
        df = add_label_columns_to_dataframe("fbs2", df, glb_idx)
        tmp = df.to_csv(date_format="%Y-%m-%d %H:%M:%S", index=False, na_rep="")
        self.assertTrue(tmp)
        # Close interactive session
        isess.close_db_session()


class TestCommandFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Executed BEFORE test methods of the class
        prepare_and_reset_database_for_tests(prepare=True)
        nexinfosys.data_source_manager = register_external_datasources(
            {"FAO_DATASETS_DIR": "/home/marco/temp/Data/FAOSTAT/"})

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

    # TEST USING DEPRECATED COMMANDS
    #
    # def test_002_execute_file_two(self):
    #     """
    #     A file containing QQs for three different sets of processor: Crop, Farm, AgrarianRegion
    #     AND UPSCALING
    #     (extracted from Almeria case study)
    #     Test number of processors read for each category, using processor sets and PartialRetrievalDictionary
    #     :return:
    #     """
    #     file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_upscale_reduced.xlsx"
    #     isess = execute_file(file_path, generator_type="spreadsheet")
    #     # # Save state
    #     s = serialize_state(isess.state)
    #     # Changed "wt" to "wb": the output of "serialize_state" is a byte array (it is compressed now)
    #     with open("/home/rnebot/GoogleDrive/AA_MAGIC/MiniAlmeria.serialized", "wb") as f:
    #         f.write(s)
    #     local_state = deserialize_state(s)
    #     # Check State of things
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(local_state)
    #     # Three processor sets
    #     self.assertEqual(len(p_sets), 3)
    #     # Close interactive session
    #     isess.close_db_session()

    def test_003_execute_file_three(self):
        """
        * Declares TWO mappings
        * Reads TWO Eurostat datasets
        * Uses one of the datasets to feed QQs

        Test number of processors read for each category, using processor sets and PartialRetrievalDictionary
        :return:
        """
        # ####
        # COMMENTED OUT BECAUSE IT IS VERY SLOW
        # ####

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
        # # Save state
        s = serialize_state(isess.state)
        # Changed "wt" to "wb": the output of "serialize_state" is a byte array (it is compressed now)
        with open("/home/rnebot/GoogleDrive/AA_MAGIC/Soslaires.serialized", "wb") as f:
            f.write(s)
        local_state = deserialize_state(s)
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(local_state)
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

    def test_006_execute_file_five(self):
        """

        Parameters
        Simple Expression evaluation in QQs

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/mapping_example_maddalena.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # Three processor sets
        self.assertEqual(len(p_sets), 1)
        # Close interactive session
        isess.close_db_session()

    def test_007_execute_file_v2_one(self):
        """
        Two connected Processors
        Test parsing and execution of a file with basic commands, and only literals (very basic syntax)

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/01_declare_two_connected_processors.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    # ------------------------------------------------------------------------------------------------------------------
    #                                -------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def test_008_execute_file_v2_two(self):
        """
        Processors from Soslaires
        Test parsing and execution of a file with basic commands, and only literals (very basic syntax)

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/02_declare_hierarchies_and_cloning_and_scaling.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        processor_dict = get_processor_names_to_processors_dictionary(glb_idx)
        for p in processor_dict:
            print(p)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_009_execute_file_v2_three(self):
        """
        Soslaires, without parameters
        With regard to the two previous, introduces the syntax of a Selector of many Processors

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/03_Soslaires_no_parameters.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    # File "brazilian oil" deleted
    # (the following test does not work)

    # def test_010_execute_file_v2_four(self):
    #     """
    #     Brazilian oil case study
    #
    #     :return:
    #     """
    #     file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/04_brazilian_oil.xlsx"
    #     isess = execute_file(file_path, generator_type="spreadsheet")
    #     # Check State of things
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     # TODO Check things!!!
    #     # self.assertEqual(len(p_sets), 3)
    #     # Close interactive session
    #     isess.close_db_session()

    def test_011_execute_file_v2_five(self):
        """
        Dataset processing using old commands

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/05_caso_energia_eu_old_commands.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_012_execute_file_v2_six(self):
        """
        Almeria upscaling with new syntax
        * References
        * InterfaceTypes
        * BareProcessors
          * Dynamic attribute columns
        * Interfaces
        * Old Upscale (really efficient)

        :return:
        """
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/06_upscale_almeria.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_013_execute_file_v2_seven(self):
        """
        Parsing of Custom datasets

        :return:
        """
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/07_custom_datasets.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_014_execute_file_v2_eight(self):
        """
        Dataset queries using Mappings

        :return:
        """
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/08_caso_energia_eu_new_commands_CASE_SENSITIVE.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        name = "ds2"
        ds = datasets[name]  # type: Dataset
        ds2 = ds.data
        print("Preparing Dataset labels")
        ds2 = add_label_columns_to_dataframe(name, ds2, glb_idx)
        print(ds2.head())
        isess.close_db_session()

    # def test_015_execute_file_v2_nine(self):
    #     """
    #     Dataset queries using Mappings, then use of Datasets to create Processors and Interfaces
    #
    #     :return:
    #     """
    #     file_path = os.path.dirname(
    #         os.path.abspath(__file__)) + "/z_input_files/v2/09_mapping_dataset_qry_and_dataset_expansion.xlsx"
    #     isess = execute_file(file_path, generator_type="spreadsheet")
    #     # Check State of things
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     # TODO Check things!!!
    #     # self.assertEqual(len(p_sets), 3)
    #     # Close interactive session
    #     isess.close_db_session()

    # def test_016_execute_file_v2_ten(self):
    #     """
    #     Upscaling using Instantiations. Translation of Louisa's file "Electricity state of the play 16.03.xlsm"
    #
    #     :return:
    #     """
    #     file_path = os.path.dirname(
    #         os.path.abspath(__file__)) + "/z_input_files/v2/10_electricity_state_of_the_play.xlsx"
    #     isess = execute_file(file_path, generator_type="spreadsheet")
    #     # Check State of things
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     # TODO Check things!!!
    #     # self.assertEqual(len(p_sets), 3)
    #     # Close interactive session
    #     isess.close_db_session()

    # def test_017_execute_file_v2_eleven(self):
    #     """
    #     Dataset queries using Mappings, then use of resulting Datasets to create Processors and Interfaces
    #
    #     :return:
    #     """
    #     file_path = os.path.dirname(
    #         os.path.abspath(__file__)) + "/z_input_files/v2/11_dataset_to_musiasem_maddalena.xlsx"
    #     isess = execute_file(file_path, generator_type="spreadsheet")
    #     # Check State of things
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     # TODO Check things!!!
    #     # self.assertEqual(len(p_sets), 3)
    #     # Close interactive session
    #     isess.close_db_session()

    def test_018_many_to_many_mappings(self):
        """
        Testing many to many mappings

        :return:
        """
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/09_many_to_many_mapping.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        datasets["ds1"].data.to_csv("/tmp/09_many_to_many_mapping_ds1_results.csv", index=False)
        isess.close_db_session()

    # Cannot be tested properly (must place file in "/tmp" directory first)
    #
    # def test_019_import_commands(self):
    #     """
    #     Testing import commands
    #
    #     :return:
    #     """
    #     file_path = os.path.dirname(
    #         os.path.abspath(__file__)) + "/z_input_files/v2/12_import_commands_example.xlsx"
    #     isess = execute_file(file_path, generator_type="spreadsheet")
    #     # Check State of things
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     # TODO Check things!!!
    #     # self.assertEqual(len(p_sets), 3)
    #     # Close interactive session
    #     isess.close_db_session()

    def test_020_list_of_commands(self):
        """
        Testing list of commands

        :return:
        """
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/13_list_of_commands_example_using_soslaires.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_021_export_to_json(self):
        """
        Testing model export

        :return:
        """
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/03_Soslaires_no_parameters.xlsx"
        #file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/02_declare_hierarchies_and_cloning_and_scaling.xlsx"
        #file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/06_upscale_almeria.xlsx"
        #file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/test_spreadsheet_4.xlsx"
        #file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/08_caso_energia_eu_new_commands.xlsx"
        #file_path = os.path.dirname(os.path.abspath(__file__)) + "/z_input_files/v2/09_many_to_many_mapping.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        json_string = export_model_to_json(isess.state)
        print(json_string)
        isess.close_db_session()

    def test_022_processor_scalings(self):
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/14_processor_scalings_example.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        json_string = export_model_to_json(isess.state)
        print(json_string)
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    def test_023_solving(self):
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/15_graph_solver_example.xlsx"

        isess = execute_file(file_path, generator_type="spreadsheet")

        issues = prepare_and_solve_model(isess.state)
        for idx, issue in enumerate(issues):
            print(f"Issue {idx + 1}/{len(issues)} = {issue}")

        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # Close interactive session
        isess.close_db_session()

    def test_024_maddalena_dataset(self):
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/MAGIC_n_1_CC_Spain.xlsx"
        isess = execute_file(file_path, generator_type="spreadsheet")
        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        # TODO Check things!!!
        # self.assertEqual(len(p_sets), 3)
        # Close interactive session
        isess.close_db_session()

    # def test_025_biofuel(self):
    #     file_path = os.path.dirname(
    #         os.path.abspath(__file__)) + "/z_input_files/v2/Biofuel_NIS.xlsx"
    #     isess, issues = execute_file_return_issues(file_path, generator_type="spreadsheet")
    #     issues2 = prepare_and_solve_model(isess.state)
    #     serialize_state(isess.state)
    #     # Check State of things
    #     self.assertEqual(len(issues), 6)  # One issue
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     p = glb_idx.get(Processor.partial_key("Society"))
    #     self.assertEqual(len(p), 1)
    #     p = p[0]
    #     self.assertEqual(len(p.factors), 2)
    #     f = glb_idx.get(Factor.partial_key(p, None, name="Bioethanol"))
    #     self.assertEqual(len(f), 1)
    #     self.assertEqual(len(f[0].observations), 2)
    #     # TODO These Observations are not registered, uncomment in case they are
    #     # obs = glb_idx.get(FactorQuantitativeObservation.partial_key(f[0]))
    #     #self.assertEqual(len(obs), 2)
    #     f = glb_idx.get(Factor.partial_key(p, None, name="Biodiesel"))
    #     self.assertEqual(len(f), 1)
    #     self.assertEqual(len(f[0].observations), 2)
    #     # Close interactive session
    #     isess.close_db_session()

    # def test_026_NL_ES(self):
    #     file_path = os.path.dirname(
    #         os.path.abspath(__file__)) + "/../../nis-internal-tests/NL_ES.xlsx"
    #     isess, issues = execute_file_return_issues(file_path, generator_type="spreadsheet")
    #     # Check State of things
    #     self.assertEqual(len(issues), 1)  # Just one issue
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     # Close interactive session
    #     isess.close_db_session()

    def test_027_solving_it2it(self):
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/15_graph_solver_euro_example.xlsx"

        isess = execute_file(file_path, generator_type="spreadsheet")

        issues = prepare_and_solve_model(isess.state)
        for idx, issue in enumerate(issues):
            print(f"Issue {idx + 1}/{len(issues)} = {issue}")

        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)

        # Close interactive session
        isess.close_db_session()

    # def test_028_dataset_expansion_and_integration(self):
    #     file_path = os.path.dirname(
    #         os.path.abspath(__file__)) + "/../../nis-internal-tests/test_dataset_expansion.xlsx"
    #
    #     isess = execute_file(file_path, generator_type="spreadsheet")
    #
    #     # issues = prepare_and_solve_model(isess.state)
    #     # for idx, issue in enumerate(issues):
    #     #     print(f"Issue {idx + 1}/{len(issues)} = {issue}")
    #
    #     # Check State of things
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
    #     p = glb_idx.get(Processor.partial_key())
    #     self.assertGreater(len(p), 50)
    #     p = p[0]
    #
    #     # Close interactive session
    #     isess.close_db_session()

    def test_029_dataset_expansion2(self):
        """
        Test dataset expansion using advanced expansion expression
        (function calls returning either InterfaceTypes or Processors)

        :return:
        """
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/18_dataset_expansion_2.xlsx"

        isess = execute_file(file_path, generator_type="spreadsheet")

        # issues = prepare_and_solve_model(isess.state)
        # for idx, issue in enumerate(issues):
        #     print(f"Issue {idx + 1}/{len(issues)} = {issue}")

        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        p = glb_idx.get(Processor.partial_key())
        self.assertEqual(len(p), 2)
        p = p[0]

        # Close interactive session
        isess.close_db_session()

    def test_030_solving_flow_graph_matrix(self):
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/15_graph_solver_example.xlsx"
        output_path =  os.path.dirname(
            os.path.abspath(__file__)) + "/tmp/flow_graph_matrix.csv"

        isess = execute_file(file_path, generator_type="spreadsheet")

        issues = prepare_and_solve_model(isess.state)
        for idx, issue in enumerate(issues):
            print(f"Issue {idx + 1}/{len(issues)} = {issue}")

        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        datasets.get("flow_graph_matrix").data.to_csv(output_path, index = False)
        df = datasets.get("flow_graph_matrix").data

        sankey = {}
        for p in list(set(df['Period'])):
            df_period = df[df['Period'] == p]
            tmp = {}
            for s in list(set(df_period['Scenario'])):
                ds_scenario = df_period[df_period['Scenario'] == s]
                processors = list(set(ds_scenario['source_processor'].append(ds_scenario['target_processor'])))
                source = [processors.index(i) for i in list(ds_scenario['source_processor'])]
                target = [processors.index(i) for i in list(ds_scenario['target_processor'])]
                label = list(ds_scenario['source'] + ' to ' + ds_scenario['target'])
                data = dict(
                    type='sankey',
                    node=dict(
                        pad=50,
                        thickness=100,
                        line=dict(
                            color="black",
                            width=0.5
                        ),
                        label=processors,

                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=list(ds_scenario['Value']),
                        label=label
                    ))

                tmp[s] = data
            sankey[p] = tmp
        print(sankey['2011']['Scenario1'])
        # Close interactive session
        isess.close_db_session()

    def test_031_export_JupyterNotebook(self):
        from nbformat import write, v4
        text = []
        code = []
        nb = v4.new_notebook()
        text.append( """\
        # My first automatic Jupyter Notebook
        This is an auto-generated notebook.""")

        code.append( """\
        fname = "Biofuel_NIS.xlsx";""")

        code.append( """\
        from nexinfosys import NISClient, display_visjs_jupyterlab
        import io
        import pandas as pd
        import networkx as nx
        
        c = NISClient("https://one.nis.magic-nexus.eu/nis_api")
        #c = NISClient("http://localhost:5000/nis_api")
        # Login, open session, load a workbook (which is in Nextcloud), submit (execute!)
        c.login("test_user")
        #print("Logged in")
        c.open_session()
        #print("Session opened")
        n = c.load_workbook(fname)
        #print("N worksheets: "+str(n))
        r = c.submit()
        #print("Returned from submit")
        # Check if submission was successful (it should be with the provided workbook), then query 
        # available datasets, and get one of them, converting it into a pd.DataFrame
        any_error = False
        if len(r) > 0:
            for i in r:
                if i["type"] == 3:
                    any_error = True
                    print(str(i))
        
        if not any_error:
            # Obtain available datasets
            r = c.query_available_datasets()
            if len(r) > 0:
                results = {}
                for ds in r:
                    results[ds["name"]] = {d["format"].lower(): d["url"] for d in ds["formats"]}
                    #print(str(ds))
                #r = c.download_results([(results["FG"]["visjs"])])
                #visjs_data = r[0].decode("utf-8")
                #unique_name = None
                r = c.download_results([(results["PG"]["visjs"])])
                visjs_data2 = r[0].decode("utf-8")
                un2 = None
        
                #unique_name = display_visjs_jupyterlab(visjs_data, 800, unique_name)
                un2 = display_visjs_jupyterlab(visjs_data2, 1200, un2)
              
        c.close_session()
        c.logout()""")


        nb['cells'] = [v4.new_markdown_cell(text),
                       v4.new_code_cell(code[0]),v4.new_code_cell(code[1])]
        fname = 'test.ipynb'

        with open(fname, 'w') as f:
            write(nb, f)

    def test_032_new_processor_declaration_convention(self):
        """
        Test new convention for creation of Processors, using BareProcessors command
        See the descriptions in the file

        :return:
        """
        file_path = os.path.dirname(
            os.path.abspath(__file__)) + "/z_input_files/v2/19_naming_processors_new_convention.xlsx"

        isess = execute_file(file_path, generator_type="spreadsheet")

        # issues = prepare_and_solve_model(isess.state)
        # for idx, issue in enumerate(issues):
        #     print(f"Issue {idx + 1}/{len(issues)} = {issue}")

        # Check State of things
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(isess.state)
        ps = glb_idx.get(Processor.partial_key())
        lst1 = []
        for p in ps:
            lst = p.full_hierarchy_names(glb_idx)
            lst1.extend(lst)
        p1 = glb_idx.get(Processor.partial_key("P2.C"))
        p2 = glb_idx.get(Processor.partial_key("P3.C"))
        self.assertEqual(p1[0], p2[0])
        p = glb_idx.get(Processor.partial_key("P1.C.P2"))
        self.assertEqual(len(p), 1)

        # Close interactive session
        isess.close_db_session()


if __name__ == '__main__':
    # import pandas as pd
    # import openpyxl
    # from openpyxl.utils.dataframe import dataframe_to_rows
    #
    # sname = "Hoja1"
    # sidx = 0
    # fname = "/home/rnebot/Downloads/dynamic_table.xlsx"
    # fname = "/home/rnebot/Downloads/dynamic_table - Editable.xlsx"
    # xl = openpyxl.load_workbook(fname, keep_vba=True)
    # ws = xl["Hoja2"]
    # pivot = ws._pivots[0]
    # # pivot.cache.refreshOnload = True
    # wsd = xl["Hoja1"]
    # print(f"{wsd.max_row}:{wsd.max_column} ---------------")
    # for r_idx in range(wsd.max_row - 1):
    #     print(f"{r_idx}")
    #     for c_idx in range(wsd.max_column):
    #         wsd.cell(row=r_idx + 2, column=c_idx + 1).value = None
    #
    # df = pd.read_csv("/home/rnebot/Downloads/flow_graph_solution.csv")
    # df = df.loc[1500:]
    #
    # # xl.remove(xl.get_sheet_by_name(sname))
    # # ws = xl.create_sheet(sname, sidx)
    # rows = dataframe_to_rows(df)
    # for r_idx, row in enumerate(rows):
    #     if r_idx < 2:
    #         continue
    #     for c_idx, value in enumerate(row[1:], 1):
    #         wsd.cell(row=r_idx, column=c_idx).value = value
    # xl.save("/home/rnebot/Downloads/dt_new.xlsx")
    #
    # a = 1 / 0

    is_fao_test = False
    fao_dir = "/home/marco/temp/Data/FAOSTAT/"
    if is_fao_test:
        i = TestFAOCommandFiles()
        prepare_and_reset_database_for_tests(
            prepare=True,
            metadata_string="sqlite:////home/rnebot/GoogleDrive/AA_MAGIC/nis_metadata.db",
            data_string="sqlite:////home/rnebot/GoogleDrive/AA_MAGIC/nis_cached_datasets.db")
        nexinfosys.data_source_manager = register_external_datasources(
            {"FAO_DATASETS_DIR": fao_dir})
        i.test_001_fao()
    else:
        i = TestCommandFiles()
        prepare_and_reset_database_for_tests(prepare=True)
        nexinfosys.data_source_manager = register_external_datasources(
            {"FAO_DATASETS_DIR": fao_dir})

        #i.test_002_execute_file_two()
        # i.test_006_execute_file_five()  # TODO: This test from v1 has problems with the case sensitiveness!
        #i.test_008_execute_file_v2_two()
        #i.test_009_execute_file_v2_three()  # Soslaires. v2 syntax
        #i.test_011_execute_file_v2_five()  # Dataset
        #i.test_012_execute_file_v2_six()  # Almeria using v2 commands and v1 upscale
        #i.test_013_execute_file_v2_seven()  # Custom datasets
        #i.test_014_execute_file_v2_eight()
        #i.test_018_many_to_many_mappings()
        #i.test_019_import_commands()
        #i.test_020_list_of_commands()
        #i.test_021_export_to_json()
        #i.test_022_processor_scalings()
        #i.test_023_solving()
        #i.test_024_maddalena_dataset()
        #i.test_025_biofuel()
        #i.test_026_NL_ES()
        #i.test_027_solving_it2it()
        #i.test_028_dataset_expansion_and_integration()
        i.test_029_dataset_expansion2()
        #i.test_030_solving_flow_graph_matrix()
        #i.test_031_export_JupyterNotebook()
        #i.test_032_new_processor_declaration_convention()
