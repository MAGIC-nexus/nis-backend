import unittest

import backend.common.helper
from backend.common.helper import PartialRetrievalDictionary
from backend.model.memory.musiasem_concepts import Processor, ProcessorsRelationPartOfObservation, Observer
from backend.model.persistent_db.persistent import (
                                                      serialize_from_object,
                                                      deserialize_to_object
                                                      )


def setUpModule():
    print('In setUpModule()')
    backend.common.helper.case_sensitive = True


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
        backend.common.helper.case_sensitive = False
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


