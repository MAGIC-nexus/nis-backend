import unittest
import pandas as pd
import pyximport

pyximport.install(reload_support=True, language_level=3)

from nexinfosys.restful_service.helper_accel import augment_dataframe_with_mapped_columns2
import nexinfosys.common.helper
from nexinfosys.common.helper import PartialRetrievalDictionary, augment_dataframe_with_mapped_columns, create_dictionary
from nexinfosys.models.musiasem_concepts import Processor, ProcessorsRelationPartOfObservation, Observer
from nexinfosys.models.musiasem_methodology_support import (
                                                      serialize_from_object,
                                                      deserialize_to_object
                                                      )


def setUpModule():
    print('In setUpModule()')
    nexinfosys.common.helper.case_sensitive = True


def tearDownModule():
    print('In tearDownModule()')


# --- Functions containing pieces of code used in several unit tests ----------------

def prepare_partial_key_dictionary():
    prd = PartialRetrievalDictionary()
    oer = Observer("tester")
    p0 = Processor("A1")
    p1 = Processor("A2")
    p2 = Processor("B")
    p3 = Processor("C")
    prd.put({"_type": "Processor", "_name": "A1"}, p0)
    prd.put({"_type": "Processor", "_name": "A2"}, p1)
    prd.put({"_type": "Processor", "_name": "B"}, p2)
    prd.put({"_type": "Processor", "_name": "C"}, p3)
    prd.put({"_type": "PartOf", "_parent": "A1", "_child": "B"}, ProcessorsRelationPartOfObservation(p0, p2, oer))
    prd.put({"_type": "PartOf", "_parent": "A2", "_child": "B"}, ProcessorsRelationPartOfObservation(p1, p2, oer))
    prd.put({"_type": "PartOf", "_parent": "B", "_child": "C"}, ProcessorsRelationPartOfObservation(p2, p3, oer))
    return prd


class TestMapFunction(unittest.TestCase):
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
    def test_001_many_to_one_1(self):
        # Prepare a many to one map from category set to category set
        m = create_dictionary()
        m["cat_o_1"] = ("cat_d_1",
              {
                  "c11": [{"d": "c21", "w": 1.0}],
                  "c12": [{"d": "c23", "w": 1.0}],
                  "c13": [{"d": "c23", "w": 1.0}]
              }
        )
        # Prepare a simple DataFrame
        df = pd.DataFrame(data=[["c11", 4], ["c12", 3], ["c13", 1.5]], columns=["cat_o_1", "value"])
        # Call
        df2 = augment_dataframe_with_mapped_columns(df, m, ["value"])
        # Check result
        self.assertEqual(list(df2.columns), ["cat_o_1", "cat_d_1", "value"])
        self.assertEqual(df2.shape, (3, 3))

    def test_002_many_to_many_1(self):
        # Prepare a many to many map from category set to category set
        # Prepare a simple DataFrame containing
        m = create_dictionary()
        m["cat_o_1"] = ("cat_d_1",
              {
                  "c11": [{"d": "c21", "w": 0.6},
                          {"d": "c22", "w": 0.4}],
                  "c12": [{"d": "c23", "w": 1.0}],
                  "c13": [{"d": "c23", "w": 1.0}]
              }
        )
        # Prepare a simple DataFrame
        df = pd.DataFrame(data=[["c11", 4], ["c12", 3], ["c13", 1.5]], columns=["cat_o_1", "value"])
        # Call
        df2 = augment_dataframe_with_mapped_columns(df, m, ["value"])
        # Check result
        self.assertEqual(list(df2.columns), ["cat_o_1", "cat_d_1", "value"])
        self.assertEqual(df2.shape, (4, 3))

    def test_003_many_to_many_2(self):
        # Prepare a many to many map from category set to category set
        # Prepare a simple DataFrame containing
        m = create_dictionary()
        m["cat_o_1"] = ("cat_d_1",
                {
                  "c11": [{"d": "c21", "w": 0.6},
                          {"d": "c22", "w": 0.4}],
                  "c12": [{"d": "c23", "w": 1.0}],
                  "c13": [{"d": "c23", "w": 1.0}]
                }
        )
        m["cat_o_2"] = ("cat_d_2",
              {
                  "c31": [{"d": "c41", "w": 0.3},
                          {"d": "c42", "w": 0.7}],
                  "c32": [{"d": "c43", "w": 1.0}],
                  "c33": [{"d": "c43", "w": 1.0}]
              }
        )
        # Prepare a simple DataFrame
        df = pd.DataFrame(data=[["c11", "c31", 4], ["c12", "c32", 3], ["c13", "c31", 1.5]], columns=["cat_o_1", "cat_o_2", "value"])
        # >>>>> Call Cython ACCELERATED Function <<<<<
        df2 = augment_dataframe_with_mapped_columns2(df, m, ["value"])
        # Check result
        self.assertEqual(list(df2.columns), ["cat_o_1", "cat_o_2", "cat_d_1", "cat_d_2", "value"])
        self.assertEqual(df2.shape, (7, 5))


class TestPartialKeyDictionary(unittest.TestCase):
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
    def test_001_put_and_get(self):
        prd = prepare_partial_key_dictionary()
        # Tests
        res = prd.get({"_type": "PartOf"})
        self.assertEqual(len(res), 3)
        res = prd.get({"_type": "PartOf", "_child": "B"})
        self.assertEqual(len(res), 2)
        res = prd.get({"_type": "PartOf", "_parent": "A1"})
        self.assertEqual(len(res), 1)
        res = prd.get({"_type": "PartOf", "_parent": "C"})
        self.assertEqual(len(res), 0)
        res = prd.get({"_typer": "PartOf", "_parent": "C"})
        self.assertEqual(len(res), 0)
        res = prd.get({"_type": "Processor", "_name": "C"})
        self.assertEqual(len(res), 1)

    def test_002_serialization_deserialization(self):
        prd = prepare_partial_key_dictionary()
        prd2 = prd.to_pickable()
        s = serialize_from_object(prd2)
        prd2 = deserialize_to_object(s)
        prd = PartialRetrievalDictionary().from_pickable(prd2)
        # Tests
        res = prd.get({"_type": "PartOf"})
        self.assertEqual(len(res), 3)
        res = prd.get({"_type": "PartOf", "_child": "B"})
        self.assertEqual(len(res), 2)
        res = prd.get({"_type": "PartOf", "_parent": "A1"})
        self.assertEqual(len(res), 1)
        res = prd.get({"_type": "PartOf", "_parent": "C"})
        self.assertEqual(len(res), 0)
        res = prd.get({"_typer": "PartOf", "_parent": "C"})
        self.assertEqual(len(res), 0)

    def test_003_delete(self):
        prd = prepare_partial_key_dictionary()
        # Before deletion
        res = prd.get({"_type": "PartOf"})
        self.assertEqual(len(res), 3)
        # Delete
        res = prd.delete({"_type": "PartOf"})
        self.assertEqual(res, 3)
        # Confirm deletion
        res = prd.get({"_type": "PartOf"})
        self.assertEqual(len(res), 0)

        prd = prepare_partial_key_dictionary()
        # Before deletion
        res = prd.get({"_type": "PartOf", "_child": "B"})
        self.assertEqual(len(res), 2)
        # Delete
        res = prd.delete({"_type": "PartOf", "_child": "B"})
        self.assertEqual(res, 2)
        # Confirm deletion
        res = prd.get({"_type": "PartOf", "_child": "B"})
        self.assertEqual(len(res), 0)

        prd = prepare_partial_key_dictionary()
        # Before deletion
        res = prd.get({"_type": "PartOf", "_parent": "A1"})
        self.assertEqual(len(res), 1)
        # Delete
        res = prd.delete({"_type": "PartOf", "_parent": "A1"})
        self.assertEqual(res, 1)
        # Confirm deletion
        res = prd.get({"_type": "PartOf", "_parent": "A1"})
        self.assertEqual(len(res), 0)

        prd = prepare_partial_key_dictionary()
        # Before deletion
        res = prd.get({"_type": "PartOf", "_parent": "C"})
        self.assertEqual(len(res), 0)
        # Delete
        res = prd.delete({"_type": "PartOf", "_parent": "C"})
        self.assertEqual(res, 0)
        # Confirm deletion
        res = prd.get({"_type": "PartOf", "_parent": "C"})
        self.assertEqual(len(res), 0)

        prd = prepare_partial_key_dictionary()
        # Before deletion
        res = prd.get({"_type": "Processor", "_name": "C"})
        self.assertEqual(len(res), 1)
        # Delete
        res = prd.delete({"_type": "Processor", "_name": "C"})
        self.assertEqual(res, 1)
        # Confirm deletion
        res = prd.get({"_type": "Processor", "_name": "C"})
        self.assertEqual(len(res), 0)

        prd = prepare_partial_key_dictionary()
        # Before deletion
        res = prd.get({"_type": "Processor"})
        self.assertEqual(len(res), 4)
        res = prd.get({"_type": "PartOf", "_child": "B"})
        self.assertEqual(len(res), 2)
        res = prd.delete([{"_type": "PartOf", "_child": "B"},
                          {"_type": "Processor"}
                          ]
                         )
        self.assertEqual(res, 6)
        # Confirm deletion
        res = prd.get({"_type": "Processor"})
        self.assertEqual(len(res), 0)
        res = prd.get({"_type": "PartOf", "_child": "B"})
        self.assertEqual(len(res), 0)

    def test_004_case_insensitive(self):
        nexinfosys.common.helper.case_sensitive = False
        prd = prepare_partial_key_dictionary()
        # Tests
        res = prd.get({"_type": "partOf"})
        self.assertEqual(len(res), 3)
        res = prd.get({"_type": "pArtOf", "_child": "b"})
        self.assertEqual(len(res), 2)
        res = prd.get({"_type": "PaRtOf", "_parent": "a1"})
        self.assertEqual(len(res), 1)
        res = prd.get({"_type": "Partof", "_parent": "C"})
        self.assertEqual(len(res), 0)


if __name__ == '__main__':
    unittest.main()
