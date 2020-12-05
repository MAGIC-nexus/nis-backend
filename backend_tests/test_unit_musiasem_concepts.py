import unittest


import nexinfosys.common.helper
from nexinfosys.models.experiments.expressions import ExpressionsEngine
from nexinfosys.models.musiasem_concepts import *
from nexinfosys.models.musiasem_concepts_helper import *
from nexinfosys.models.musiasem_concepts_helper import _get_observer, _find_or_create_relation
from nexinfosys.serialization import serialize_state, deserialize_state

""" Integration tests for in memory model structures """


def setUpModule():
    print('In setUpModule()')


def tearDownModule():
    print('In tearDownModule()')


def prepare_partial_key_dictionary():
    glb_idx = PartialRetrievalDictionary()

    oer = Observer("tester")
    p0 = Processor("A1")
    p1 = Processor("A2")
    p2 = Processor("B")
    p3 = Processor("C")
    glb_idx.put(p0.key(), p0)
    glb_idx.put(p1.key(), p1)
    glb_idx.put(p2.key(), p2)
    glb_idx.put(p3.key(), p3)
    obs = ProcessorsRelationPartOfObservation(p0, p2, oer)
    glb_idx.put(obs.key(), obs)
    obs = ProcessorsRelationPartOfObservation(p1, p2, oer)
    glb_idx.put(obs.key(), obs)
    obs = ProcessorsRelationPartOfObservation(p2, p3, oer)
    glb_idx.put(obs.key(), obs)
    return glb_idx


def prepare_simple_processors_hierarchy():
    state = State()
    p1,_,_ = find_or_create_observable(state, "P1", "test_observer",
                                   aliases=None,
                                   proc_attributes=None, proc_location=None,
                                   fact_roegen_type=None, fact_attributes=None,
                                   fact_incoming=None, fact_external=None, fact_location=None
                                   )
    p2,_,_ = find_or_create_observable(state, "P1.P2", "test_observer",
                                   aliases=None,
                                   proc_attributes=None, proc_location=None,
                                   fact_roegen_type=None, fact_attributes=None,
                                   fact_incoming=None, fact_external=None, fact_location=None
                                   )
    p3,_,_ = find_or_create_observable(state, "P3", "test_observer",
                                   aliases=None,
                                   proc_attributes=None, proc_location=None,
                                   fact_roegen_type=None, fact_attributes=None,
                                   fact_incoming=None, fact_external=None, fact_location=None
                                   )
    p4,_,_ = find_or_create_observable(state, "P1.P2.P3", "test_observer",
                                   aliases=None,
                                   proc_attributes=None, proc_location=None,
                                   fact_roegen_type=None, fact_attributes=None,
                                   fact_incoming=None, fact_external=None, fact_location=None
                                   )
    p5,_,_ = find_or_create_observable(state, "P1.P2b", "test_observer",
                                   aliases=None,
                                   proc_attributes=None, proc_location=None,
                                   fact_roegen_type=None, fact_attributes=None,
                                   fact_incoming=None, fact_external=None, fact_location=None
                                   )
    return state


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

    def test_hierarchy_construction(self):
        prd = PartialRetrievalDictionary()

        for node_type in (Taxon, FactorType):
            h = build_hierarchy("hierarchy_" + node_type.__name__, node_type.__name__, prd,
                            [dict(code="NODE1", children=[dict(code="NODE2")]),
                            dict(code="NODE3")])
            node1 = prd.get(node_type.partial_key("NODE1"))[0]
            node2 = prd.get(node_type.partial_key("NODE1.NODE2"))[0]
            node3 = prd.get(node_type.partial_key("NODE3"))[0]
            roots = [node1, node3]

            # Roots
            self.assertEqual(len(h.roots), 2, "Size of hierarchy is not correct")
            self.assertEqual(len(set(roots).intersection(h.roots)), len(roots),
                             "Number of roots elements in hierarchy is not correct")
            # Relations
            self.assertIn(node2, node1.get_children(), "Children of a node do not match")
            self.assertEqual(node2.parent, node1, "Parent of a node do not match")
            self.assertEqual(node3.parent, None, "The parent of a root node should not exist")

    def test_hierarchy_nodes_linking(self):
        interface_type = FactorType("Node1")
        interface = Taxon("Node2")

        # An interface type cannot be linked to a interface
        with self.assertRaises(Exception):
            FactorType("Node3", interface)

        # An interface cannot be linked to a interface type
        with self.assertRaises(Exception):
            Taxon("Node3", interface_type)

    def test_hierarchy_of_processors(self):
        state = prepare_simple_processors_hierarchy()
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        p2 = glb_idx.get(Processor.partial_key("P1.P2"))[0]
        p4 = glb_idx.get(Processor.partial_key("P1.P2.P3"))[0]
        p5 = glb_idx.get(Processor.partial_key("P1.P2b"))[0]
        names = p2.full_hierarchy_names(glb_idx)
        self.assertEqual(names[0], "P1.P2")
        # Make "p1.p2.p3" processor descend from "p1.p2b" so it will be also "p1.p2b.p3"
        r = _find_or_create_relation(p5, p4, RelationClassType.pp_part_of, "test_observer", None, state)
        names = p4.full_hierarchy_names(glb_idx)
        self.assertIn("P1.P2.P3", names)
        # self.assertEqual(names[0], "P1.P2.P3")

        # TODO Register Aliases for the Processor (in "obtain_relation")

    def test_hierarchy_of_processors_after_serialization_deserialization(self):
        state = prepare_simple_processors_hierarchy()
        # Serialize, deserialize
        s = serialize_state(state)
        state = deserialize_state(s)
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        p2 = glb_idx.get(Processor.partial_key("P1.P2"))[0]
        p4 = glb_idx.get(Processor.partial_key("P1.P2.P3"))[0]
        p5 = glb_idx.get(Processor.partial_key("P1.P2b"))[0]

        names = p2.full_hierarchy_names(glb_idx)
        self.assertEqual(names[0], "P1.P2")
        # Make "p1.p2.p3" processor descend from "p1.p2b" so it will be also "p1.p2b.p3"
        r = _find_or_create_relation(p5, p4, RelationClassType.pp_part_of, "test_observer", None, state)
        names = p4.full_hierarchy_names(glb_idx)
        self.assertIn("P1.P2.P3", names)
        # self.assertEqual(names[0], "P1.P2.P3")

        # TODO Register Aliases for the Processor (in "obtain_relation")


class ModelBuildingQuantativeObservations(unittest.TestCase):
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

    def test_001_soslaires_windfarm_observations(self):
        state = State()
        k = {"spread": None, "assessment": None, "pedigree": None, "pedigree_template": None,
             "relative_to": None,
             "time": None,
             "geolocation": None,
             "comments": None,
             "tags": None,
             "other_attributes": None,
             "proc_aliases": None,
             "proc_external": False,
             "proc_location": None,
             "proc_attributes": None,
             "ftype_roegen_type": FlowFundRoegenType.fund,
             "ftype_attributes": None,
             "fact_incoming": True,
             "fact_external": None,
             "fact_location": None
             }
        create_or_append_quantitative_observation(state, "WindFarm:LU.cost", "17160", "€", **(k.copy()))
        create_or_append_quantitative_observation(state, "WindFarm:HA.cost", "1800", "€", **(k.copy()))
        create_or_append_quantitative_observation(state, "WindFarm:PC.cost", "85600", "€", **(k.copy()))
        create_or_append_quantitative_observation(state, "WindFarm:LU", "8800", "m2", **(k.copy()))
        create_or_append_quantitative_observation(state, "WindFarm:HA", "660", "hours", **(k.copy()))
        create_or_append_quantitative_observation(state, "WindFarm:PC", "2.64", "MW", **(k.copy()))
        k["ftype_roegen_type"] = FlowFundRoegenType.flow
        create_or_append_quantitative_observation(state, "WindFarm:WindElectricity", "9.28", "GWh", **(k.copy()))
        create_or_append_quantitative_observation(state, "WindFarm:WindElectricity.max_production", "23.2", "GWh", **(k.copy()))
        k["proc_external"] = True
        create_or_append_quantitative_observation(state, "ElectricGrid:GridElectricity", "6.6", "GWh", **(k.copy()))
        # ============================= READS AND ASSERTIONS =============================
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        # Check function "get_factor_or_processor_or_factor_type"
        wf = find_observable_by_name("WindFarm", glb_idx) # Get a Processor
        self.assertIsInstance(wf, Processor)
        wf = find_observable_by_name("WindFarm:", glb_idx) # Get a Processor
        self.assertIsInstance(wf, Processor)
        lu = find_observable_by_name(":LU", glb_idx) # Get a FactorType
        self.assertIsInstance(lu, FactorType)
        wf_lu = find_observable_by_name("WindFarm:LU", glb_idx) # Get a Factor, using full name
        self.assertIsInstance(wf_lu, Factor)
        wf_lu = find_observable_by_name(":LU", glb_idx, processor=wf) # Get a Factor, using already known Processor
        self.assertIsInstance(wf_lu, Factor)
        wf_lu = find_observable_by_name("WindFarm:", glb_idx, factor_type=lu) # Get a Factor, using already known FactorType
        self.assertIsInstance(wf_lu, Factor)
        # Check things about the Factor
        self.assertEqual(wf_lu.processor.name, "WindFarm")
        self.assertEqual(wf_lu.taxon.name, "LU")
        self.assertEqual(wf_lu.name, "LU")
        # Get observations from the Factor
        obs = glb_idx.get(FactorQuantitativeObservation.partial_key(wf_lu))
        self.assertEqual(len(obs), 0)
        obs = [o for o in find_quantitative_observations(glb_idx) if o.factor.ident == wf_lu.ident]
        self.assertEqual(len(obs), 1)
        # Get observations from the Observer
        oer = _get_observer(Observer.no_observer_specified, glb_idx)
        obs = glb_idx.get(FactorQuantitativeObservation.partial_key(observer=oer))
        self.assertEqual(len(obs), 0)  # NINE !!!!
        obs = [o for o in find_quantitative_observations(glb_idx) if o.observer.ident == oer.ident]
        self.assertEqual(len(obs), 9)  # NINE !!!!
        # Get observations from both Factor and Observer
        obs = glb_idx.get(FactorQuantitativeObservation.partial_key(factor=wf_lu, observer=oer))
        self.assertEqual(len(obs), 0)
        obs = [o for o in find_quantitative_observations(glb_idx) if o.factor.ident == wf_lu.ident and o.observer.ident == oer.ident]
        self.assertEqual(len(obs), 1)


class ModelBuildingRelationObservations(unittest.TestCase):
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

    def test_001_build_soslaires_relations(self):
        nexinfosys.common.helper.case_sensitive = False
        state = State()
        create_relation_observations(state, "WindFarm:WindElectricity", ["DesalinationPlant", ("ElectricGrid")])
        create_relation_observations(state, "ElectricGrid", "DesalinationPlant:GridElectricity")
        create_relation_observations(state, "DesalinationPlant:DesalinatedWater", "Farm:BlueWater")
        crop_processors = ["Cantaloupe", "Watermelon", "Tomato", "Zucchini", "Beans", "Pumpkin", "Banana", "Moringa"]
        create_relation_observations(state, "Farm", crop_processors, RelationClassType.pp_part_of)
        crop_processors = ["Farm."+p for p in crop_processors]
        create_relation_observations(state, "Farm:LU", crop_processors)
        create_relation_observations(state, "Farm:HA", crop_processors)
        create_relation_observations(state, "Farm:IrrigationCapacity", crop_processors)
        create_relation_observations(state, "Farm:BlueWater", crop_processors)
        create_relation_observations(state, "Farm:Agrochemicals", crop_processors)
        create_relation_observations(state, "Farm:Fuel", crop_processors)
        create_relation_observations(state, "Farm:GreenWater", crop_processors, RelationClassType.ff_reverse_directed_flow)
        create_relation_observations(state, "Farm:MaterialWaste", crop_processors, RelationClassType.ff_reverse_directed_flow)
        create_relation_observations(state, "Farm:DiffusivePollution", crop_processors, RelationClassType.ff_reverse_directed_flow)
        create_relation_observations(state, "Farm:CO2", crop_processors, RelationClassType.ff_reverse_directed_flow)
        create_relation_observations(state, "Farm:Vegetables", [p + ":Vegetables." + p for p in crop_processors], RelationClassType.ff_reverse_directed_flow)
        # ============================= READS AND ASSERTIONS =============================
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        # Check Observables and FlowTypes existence
        processors = glb_idx.get(Processor.partial_key(None))
        processors = set(processors)
        self.assertEqual(len(processors), 12)
        dplant = glb_idx.get(Processor.partial_key("desalinationplant"))
        farm = glb_idx.get(Processor.partial_key("farm"))
        banana = glb_idx.get(Processor.partial_key("farm.banana"))
        lu = glb_idx.get(FactorType.partial_key("lU"))
        farm_lu = glb_idx.get(Factor.partial_key(processor=farm[0], factor_type=lu[0]))

        # Check Relations between observables
        rels = glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(source=farm_lu[0]))
        self.assertEqual(len(rels), 8)


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
    def test_001_processor_hierarchical_names_from_part_of_relations(self):
        prd = prepare_partial_key_dictionary()

        p = prd.get(Processor.partial_key("C"))
        self.assertEqual(len(p), 1)
        p = p[0]  # Get the processor
        n = p.full_hierarchy_names(prd)
        self.assertEqual(len(n), 2)

    def test_tagged_processors(self):
        # Create a taxonomy
        prd = PartialRetrievalDictionary()
        h = build_hierarchy("Test_auto", "Taxon", prd, [dict(code="T1", children=[dict(code="T2")]),
                                                        dict(code="T3")])
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
        # Create a Location
        geo = Geolocation("NUTS", "ES")
        # Create a Processor
        p1 = Processor("P1", geolocation=geo)
        # Check
        self.assertEqual(p1.geolocation, geo)

    def test_processors_with_factors(self):
        """ Processors adorned with factors. No need for connections.
            Need to specify if input and other. Only Funds do not need this specification
        """
        # Create a Hierarchy of FactorType
        prd = PartialRetrievalDictionary()
        ft = build_hierarchy("Taxs", "FactorType", prd, [dict(code="P1"),
                                                         dict(code="P2")])
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
        # Create a Hierarchy of FactorType
        prd = PartialRetrievalDictionary()
        ft = build_hierarchy("Taxs", "FactorType", prd, [dict(code="F1"),
                                                         dict(code="F2")])
        # Create two Processors, same level
        ps = build_hierarchy("Procs", "Processor", prd, [dict(code="P1"),
                                                         dict(code="P2")])
        # Create a Factor for each processor
        f1 = Factor("", prd.get(Processor.partial_key("P1")), FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f2 = Factor("", prd.get(Processor.partial_key("P2")), FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        # Connect from one to the other
        c = _find_or_create_relation(f1, f2, RelationClassType.ff_directed_flow, Observer.no_observer_specified, None, prd)  # c = f1.connect_to(f2)
        # Check that the connection exists in both sides, and that it is sequential
        self.assertTrue(c in f1.observations)
        self.assertTrue(c in f2.observations)

    def test_hierarchically_connected_factors(self):
        """ Two or more processors, with factors. Connect some of them """
        # Create a Hierarchy of FactorType
        prd = PartialRetrievalDictionary()
        ft = build_hierarchy("Taxs", "FactorType", prd, [dict(code="F1"),
                                                         dict(code="F2")])
        # Create two Processors, parent and child
        ps = build_hierarchy("Procs", "Processor", prd, [dict(code="P1", children=[dict(code="P2")]),
                                                         ])
        # Create a Factor for each processor
        p1 = prd.get(Processor.partial_key("P1"))[0]
        p2 = prd.get(Processor.partial_key("P1.P2"))[0]
        f1 = Factor.create_and_append("", p1, FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f2 = Factor.create_and_append("", p2, FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        # Connect from one to the other
        c = _find_or_create_relation(f1, f2, RelationClassType.ff_directed_flow, Observer.no_observer_specified, None, prd)
        # Check that the connection exists in both sides
        self.assertTrue(c in f1.observations)
        self.assertTrue(c in f2.observations)

    def test_hybrid_connected_factors(self):
        # Create a Hierarchy of FactorType
        prd = PartialRetrievalDictionary()
        ft = build_hierarchy("Taxs", "FactorType", prd, [dict(code="F1"),
                                                         dict(code="F2")])
        # Create Three Processors, parent and child, and siblings
        ps = build_hierarchy("Procs", "Processor", prd, [dict(code="P1", children=[dict(code="P2")]),
                                                         dict(code="P3")])
        # Create a Factor for each processor, and an additional factor for the processor which is parent and sibling
        f11 = Factor.create_and_append("", prd.get(Processor.partial_key("P1"))[0], FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f2 = Factor.create_and_append("", prd.get(Processor.partial_key("P1.P2"))[0], FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        f12 = Factor.create_and_append("", prd.get(Processor.partial_key("P1"))[0], FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        f3 = Factor.create_and_append("", prd.get(Processor.partial_key("P3"))[0], FactorInProcessorType(external=False, incoming=True), ft.get_node("F1"))
        # Do the connections
        ch = _find_or_create_relation(f11, f2, RelationClassType.ff_directed_flow, Observer.no_observer_specified, None, prd)  # ch = f11.connect_to(f2, ps)
        cs = _find_or_create_relation(f12, f3, RelationClassType.ff_directed_flow, Observer.no_observer_specified, None, prd)  # cs = f12.connect_to(f3, ps)
        # Check each connection
        self.assertTrue(ch in f11.observations)
        self.assertTrue(ch in f2.observations)
        self.assertTrue(cs in f12.observations)
        self.assertTrue(cs in f3.observations)

    def test_processors_with_factors_with_one_observation(self):
        # Create a Hierarchy of FactorType
        prd = PartialRetrievalDictionary()
        ft = build_hierarchy("Taxs", "FactorType", prd, [dict(code="F1"),
                                                         dict(code="F2")])
        # Create a processor
        p1 = Processor("P1")
        # Create a Factor and assign it to the Processor
        f1 = Factor.create_and_append("", p1, FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        # Observer of the Value
        oer = Observer("oer1")
        # Create an Observation with its value
        fo = FactorQuantitativeObservation(5, oer, f1, attributes={"unit": "m²"})
        # Assign to the factor
        f1.observations_append(fo)
        # Check
        self.assertTrue(fo in f1.observations)

    def test_processors_with_factors_with_more_than_one_observation(self):
        # Create a Hierarchy of FactorType
        prd = PartialRetrievalDictionary()
        ft = build_hierarchy("Taxs", "FactorType", prd, [dict(code="F1"),
                                                         dict(code="F2")])
        # Create a processor
        p1 = Processor("P1")
        # Create a Factor and assign it to the Processor
        f1 = Factor.create_and_append("", p1, FactorInProcessorType(external=False, incoming=False), ft.get_node("F1"))
        # Observer of the Value
        oer1 = Observer("oer1")
        oer2 = Observer("oer2")
        # Create a Value
        fo1 = FactorQuantitativeObservation.create_and_append(5, f1, oer1, attributes={"unit": "m²"})
        fo2 = FactorQuantitativeObservation.create_and_append(5, f1, oer2, attributes={"unit": "m²"})
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
        prd = PartialRetrievalDictionary()
        h = build_hierarchy("H1", "Taxon", prd, [dict(code="T1", children=[dict(code="T2")]),
                                                 dict(code="T3")])
        # A taxon is a function of others
        t3 = h.get_node("T3")
        t3.expression = {"op": "*", "oper": [{"n": 0.5}, {"v": "T1"}]}
        t2 = h.get_node("T1.T2")
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
        # A hierarchy of FactorType
        prd = PartialRetrievalDictionary()
        ft = build_hierarchy("Taxs", "FactorType", prd, [dict(code="F1"),
                                                         dict(code="F2")])
        # Hierarchy of Processors
        ps = build_hierarchy("Procs", "Processor", prd, [dict(code="P1", children=[dict(code="P2"), dict(code="P4")]),
                                                         dict(code="P3")])
        # Attach Factor
        #connect_processors(source_p: Processor, dest_p: Processor, h: "Hierarchy", weight: float, taxon: FactorType, source_name: str = None, dest_name: str = None)
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
        # TODO Create a FactorType Hierarchy
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
