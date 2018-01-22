import unittest

from backend.model.memory.expressions import ExpressionsEngine
from backend.common.helper_2 import build_hierarchy
from backend.model.memory.musiasem_concepts import *

""" Integration tests for in memory model structures """


def setUpModule():
    print('In setUpModule()')


def tearDownModule():
    print('In tearDownModule()')


class ModelBuildingHierarchies(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('In setUpClass()')
        cls.good_range = range(1, 10)

    @classmethod
    def tearDownClass(cls):
        print('In tearDownClass()')
        del cls.good_range

    def setUp(self):
        super().setUp()
        print('\nIn setUp()')

    def tearDown(self):
        print('In tearDown()')
        super().tearDown()

    # ###########################################################

    def test_hierarchy(self):
        h = Heterarchy("Test")
        t1 = Taxon("T1", None, h)
        t2 = Taxon("T2", t1, h)
        t3 = Taxon("T3", None, h)
        roots = [t1, t3]
        h.roots_append(roots)
        # Same roots
        self.assertEqual(len(set(roots).intersection(h.roots)), len(roots))
        # Relations
        self.assertEqual(t1.get_children(h)[0], t2)
        self.assertEqual(t2.parent, t1)  # Special test, to test parent of node in a single hierarchy
        self.assertEqual(t3.get_parent(h), None)

    def test_hierarchy_2(self):
        h = build_hierarchy("Test_auto", "Taxon", None, {"T1": {"T2": None}, "T3": None})
        self.assertEqual(len(h.roots), 2)

    def test_hierarchy_of_factors(self):
        h = Heterarchy("Test2")
        f1 = FactorTaxon("F1", None, h)
        f2 = FactorTaxon("F2", f1, h)
        t1 = Taxon("T1")
        with self.assertRaises(Exception):
            FactorTaxon("F3", t1)
        f3 = FactorTaxon("F3", None, h)
        roots = [f1, f3]
        h.roots_append(roots)
        # Same roots
        self.assertEqual(len(set(roots).intersection(h.roots)), len(roots))
        # Relations
        self.assertEqual(f1.get_children(h)[0], f2)
        self.assertEqual(f2.get_parent(h), f1)
        self.assertEqual(f3.get_parent(h), None)

    def test_hierarchy_of_processors(self):
        h = Heterarchy("Test3")
        p1 = Processor("P1", None, h)
        p2 = Processor("P2", p1, h)
        p3 = Processor("P3", None, h)
        roots = [p1, p3]
        h.roots_append(roots)
        # Same roots
        self.assertEqual(len(set(roots).intersection(h.roots)), len(roots))
        # Relations
        self.assertEqual(p1.get_children(h)[0], p2)
        self.assertEqual(p2.get_parent(h), p1)
        self.assertEqual(p3.get_parent(h), None)


class ModelBuildingProcessors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        pass
        super().tearDown()

    # ###########################################################

    def test_tagged_processors(self):
        # Create a taxonomy
        h = build_hierarchy("Test_auto", "Taxon", None, {"T1": {"T2": None}, "T3": None})
        # Create a processor and tag it
        p1 = Processor("P1")
        t1 = h.get_node("T1")
        t2 = h.get_node("T2")
        p1.tags_append(t1)
        self.assertTrue(t1 in p1.tags)
        self.assertFalse(t2 in p1.tags)
        p1.tags_append(t2)
        # Check if the processor meets the tags
        self.assertTrue(t1 in p1.tags)
        self.assertTrue(t2 in p1.tags)
        self.assertFalse(h.get_node("T3") in p1.tags)

    def test_processor_with_attributes(self):
        # Create a Processor
        p1 = Processor("P1")
        # Create a Location
        geo = Geolocation("Spain")
        # Assign it as "location" attribute
        p1.attributes_append("location", geo)
        # Check
        self.assertEqual(p1.attributes["location"], geo)

    def test_sequentially_connected_processors(self):
        # Create two processors. No hierarchy
        ps = build_hierarchy("Procs", "Processor", None, {"P1": None, "P2": None})
        # Connect from one to the other
        p1 = ps.get_node("P1")
        p2 = ps.get_node("P2")
        c = p1.connect_to(p2, ps)
        # Check that the connection exists, and that it is sequential
        self.assertTrue(c in p1.connections)
        self.assertTrue(c in p2.connections)
        self.assertFalse(c.hierarchical)
        # Create a FactorTaxon Hierarchy
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Connect from one processor to one FactorTaxon. An exception should be issued
        with self.assertRaises(Exception):
            p2.connect_to(ft.get_node("F1"))

    def test_hierarchically_connected_processors(self):
        # Create two processors in a hierarchy
        ps = build_hierarchy("Procs", "Processor", None, {"P1": {"P2": None}})
        # Connect from one to the other
        p1 = ps.get_node("P1")
        p2 = ps.get_node("P2")
        c = p1.connect_to(p2, ps)
        # Check that the connection exists in the two processors, and that it is hierarchical
        self.assertTrue(c in p1.connections)
        self.assertTrue(c in p2.connections)
        self.assertTrue(c.hierarchical)

    def test_hybrid_connected_processors(self):
        # Create three processors in a hierarchy
        ps = build_hierarchy("Procs", "Processor", None, {"P1": {"P2": None}, "P3": None})
        # Connect two of them hierarchically
        p1 = ps.get_node("P1")
        p2 = ps.get_node("P2")
        ch = p1.connect_to(p2, ps)
        # Connect two sequentially
        p3 = ps.get_node("P3")
        cs = p1.connect_to(p3, ps)
        # Check both connections. One should be sequential, the other should be hierarchical
        self.assertTrue(ch in p1.connections)
        self.assertTrue(ch in p2.connections)
        self.assertTrue(cs in p1.connections)
        self.assertTrue(cs in p3.connections)
        self.assertTrue(ch.hierarchical)
        self.assertFalse(cs.hierarchical)

    def test_processors_with_factors(self):
        """ Processors adorned with factors. No need for connections. 
            Need to specify if input and other. Only Funds do not need this specification
        """
        # Create a Hierarchy of FactorTaxon
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Create a processor
        p1 = Processor("P1")
        # Create a Factor and append it to the Processor (¿register it into the FactorType also?)
        f = Factor("", p1, FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        p1.factors_append(f)
        # Check that the processor contains the Factor
        self.assertTrue(f in p1.factors)


class ModelBuildingFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    # ###########################################################

    def test_sequentially_connected_factors(self):
        """ Two or more processors, with factors. Connect some of them """
        # Create a Hierarchy of FactorTaxon
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Create two Processors, same level
        ps = build_hierarchy("Procs", "Processor", None, {"P1": None, "P2": None})
        # Create a Factor for each processor
        f1 = Factor("", ps.get_node("P1"), FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f2 = Factor("", ps.get_node("P2"), FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        # Connect from one to the other
        c = f1.connect_to(f2)
        # Check that the connection exists in both sides, and that it is sequential
        self.assertTrue(c in f1.connections)
        self.assertTrue(c in f2.connections)
        self.assertFalse(c.hierarchical)

    def test_hierarchically_connected_factors(self):
        """ Two or more processors, with factors. Connect some of them """
        # Create a Hierarchy of FactorTaxon
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Create two Processors, parent and child
        ps = build_hierarchy("Procs", "Processor", None, {"P1": {"P2": None}})
        # Create a Factor for each processor
        f1 = Factor.create("", ps.get_node("P1"), FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f2 = Factor.create("", ps.get_node("P2"), FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        # Connect from one to the other
        c = f1.connect_to(f2, ps)
        # Check that the connection exists in both sides, and that it is Hierarchical
        self.assertTrue(c in f1.connections)
        self.assertTrue(c in f2.connections)
        self.assertTrue(c.hierarchical)

    def test_hybrid_connected_factors(self):
        # Create a Hierarchy of FactorTaxon
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Create Three Processors, parent and child, and siblings
        ps = build_hierarchy("Procs", "Processor", None, {"P1": {"P2": None}, "P3": None})
        # Create a Factor for each processor, and an additional factor for the processor which is parent and sibling
        f11 = Factor.create("", ps.get_node("P1"), FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f2 = Factor.create("", ps.get_node("P2"), FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        f12 = Factor.create("", ps.get_node("P1"), FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f3 = Factor.create("", ps.get_node("P3"), FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        # Do the connections
        ch = f11.connect_to(f2, ps)
        cs = f12.connect_to(f3, ps)
        # Check each connection
        self.assertTrue(ch in f11.connections)
        self.assertTrue(ch in f2.connections)
        self.assertTrue(cs in f12.connections)
        self.assertTrue(cs in f3.connections)
        self.assertTrue(ch.hierarchical)
        self.assertFalse(cs.hierarchical)

    def test_create_qq(self):
        # Create a value with incorrect unit
        with self.assertRaises(Exception) as ctx:
            QualifiedQuantityExpression.nu(5, "non existent unit")

        q2 = QualifiedQuantityExpression.nu(5, "m²")

    def test_processors_with_factors_with_one_observation(self):
        # Create a Hierarchy of FactorTaxon
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Create a processor
        p1 = Processor("P1")
        # Create a Factor and assign it to the Processor
        f1 = Factor.create("", p1, FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        # Observer of the Value
        oer = Observer("oer1")
        # Create an Observation with its value
        fo = FactorQuantitativeObservation(QualifiedQuantityExpression.nu(5, "m²"), oer, f1)
        # Assign to the factor
        f1.observations_append(fo)
        # Check
        self.assertTrue(fo in f1.observations)

    def test_processors_with_factors_with_more_than_one_observation(self):
        # Create a Hierarchy of FactorTaxon
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Create a processor
        p1 = Processor("P1")
        # Create a Factor and assign it to the Processor
        f1 = Factor.create("", p1, FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        # Observer of the Value
        oer1 = Observer("oer1")
        oer2 = Observer("oer2")
        # Create a Value
        fo1 = FactorQuantitativeObservation.create_and_append(QualifiedQuantityExpression.nu(5, "m²"), f1, oer1)
        fo2 = FactorQuantitativeObservation.create_and_append(QualifiedQuantityExpression.nu(5, "m²"), f1, oer2)
        f1.observations_append(fo1)
        f1.observations_append(fo2)
        # Check
        self.assertTrue(fo1 in f1.observations)
        self.assertTrue(fo2 in f1.observations)


class ModelBuildingExpressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        pass
        super().tearDown()

    # ###########################################################

    def test_processors_with_expression_in_taxonomy(self):
        # A hierarchy of Taxon
        h = build_hierarchy("H1", "Taxon", None, {"T1": {"T2": None}, "T3": None})
        # A taxon is a function of others
        t3 = h.get_node("T3")
        t3.expression = {"op": "*", "oper": [{"n": 0.5}, {"v": "T1"}]}
        t2 = h.get_node("T2")
        t2.expression = {"n": 3, "u": "kg"}
        # INJECT EXPRESSIONS into the ExpEvaluator:
        expev = ExpressionsEngine()
        expev.append_expressions({"lhs": {"v": "H1.T1"}, "rhs": {"v": "H1.T2"}})  # t1 = t2
        expev.append_expressions({"lhs": {"v": "H1.T3"}, "rhs": {"op": "*", "oper": [{"n": 0.5}, {"v": "H1.T1"}]}})  # t3 = 0.5*t1
        expev.append_expressions({"lhs": {"v": "H1.T2"}, "rhs": {"n": 3, "u": "kg"}})  # t2 = 3 kg
        expev.cascade_solver()
        self.assertEqual(expev.variables["H1.T1"].values[0][0], ureg("3 kg"))
        self.assertEqual(expev.variables["H1.T2"].values[0][0], ureg("3 kg"))
        self.assertEqual(expev.variables["H1.T3"].values[0][0], ureg("1.5 kg"))
        # TODO Check: cascade up to T1
        expev.reset()
        #expev.r
        # TODO Check: cascade side to T3

    def test_processors_with_factors_with_expression_observation(self):
        # A hierarchy of FactorTaxon
        ft = build_hierarchy("Taxs", "FactorTaxon", None, {"F1": None, "F2": None})
        # Hierarchy of Processors
        ps = build_hierarchy("Procs", "Processor", None, {"P1": {"P2": None, "P4": None}, "P3": None})
        # Attach Factor
        #connect_processors(source_p: Processor, dest_p: Processor, h: "Hierarchy", weight: float, taxon: FactorTaxon, source_name: str = None, dest_name: str = None)
        # TODO A taxon is a function of others

        pass

    def test_processors_with_factors_with_more_than_expression_observation(self):
        pass


class ModelBuildingWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        pass
        super().tearDown()

    # ###########################################################

    def test_register_entities(self):
        """ Create context.
            Register hierarchies, processors. Check no double registration is done
        """

    def test_connect_processors_using_registry(self):
        pass

    def test_build_hierarchy_using_registry(self):
        pass

    def test_import(self):
        # TODO Create a Space context
        # TODO Create a FactorTaxon Hierarchy
        # TODO Close Space context
        # TODO Create another space context
        # TODO
        pass


class ModelSolvingExpressionsEvaluationSimple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        pass
        super().tearDown()

    """ Build simple models including all types of expression """


class ModelBuildingIndicatorsSimple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        pass
        super().tearDown()

    # ###########################################################

    def test_intensive_processor_to_extensive(self):
        pass


if __name__ == '__main__':
    unittest.main()
