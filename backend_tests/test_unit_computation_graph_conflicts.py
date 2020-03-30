import unittest
from typing import List, Set, Dict, Tuple, Optional, Callable, NamedTuple

from nexinfosys.solving.graph.computation_graph import ComputationGraph
from nexinfosys.solving.graph.flow_graph import FlowGraph
from nexinfosys.solving.graph import EdgeType, Weight, Node, Value


class TestDataTuple(NamedTuple):
    new_computed_nodes: Set[Node]
    prev_computed_nodes: Set[Node]
    conflicts: Dict[Node, Set[Node]]


def create_test_data1() -> Tuple[ComputationGraph, List[TestDataTuple]]:
    comp_graph = ComputationGraph()

    comp_graph.add_edge('D', 'A', Weight(1.0), None)
    comp_graph.add_edge('E', 'A', Weight(1.0), None)
    comp_graph.add_edge('A', 'B', Weight(2.0), None)
    comp_graph.add_edge('B', 'C', Weight(0.7), None)
    comp_graph.add_edge('F', 'C', Weight(1.0), None)
    comp_graph.add_edge('C', 'G', Weight(1.0), None)
    comp_graph.add_edge('H', 'F', Weight(0.5), None)
    comp_graph.add_edge('I', 'F', Weight(0.4), None)

    return comp_graph, [
        TestDataTuple(
            new_computed_nodes={'D'},
            prev_computed_nodes=set(),
            conflicts={}
        ),
        TestDataTuple(
            new_computed_nodes={'D'},
            prev_computed_nodes={'H', 'I'},
            conflicts={}
        ),
        TestDataTuple(
            new_computed_nodes={'D', 'E'},
            prev_computed_nodes=set(),
            conflicts={}
        ),
        TestDataTuple(
            new_computed_nodes={'F'},
            prev_computed_nodes={'D', 'E'},
            conflicts={}
        ),
        TestDataTuple(
            new_computed_nodes={'H'},
            prev_computed_nodes={'D', 'E', 'F'},
            conflicts={'H': {'F'}}
        ),
        TestDataTuple(
            new_computed_nodes={'F'},
            prev_computed_nodes={'D', 'E', 'H'},
            conflicts={'H': {'F'}}
        ),
        TestDataTuple(
            new_computed_nodes={'D', 'E'},
            prev_computed_nodes={'B'},
            conflicts={'D': {'B'}, 'E': {'B'}, }
        ),
        TestDataTuple(
            new_computed_nodes={'D'},
            prev_computed_nodes={'A', 'B', 'H', 'G', 'E'},
            conflicts={'D': {'A', 'B', 'G'}}
        )
    ]


def create_test_data2() -> Tuple[ComputationGraph, List[TestDataTuple]]:
    comp_graph = ComputationGraph()

    comp_graph.add_edge('D', 'A', Weight(1.0), None)
    comp_graph.add_edge('E', 'A', Weight(1.0), None)
    comp_graph.add_edge('A', 'B', Weight(2.0), Weight(1.0))
    comp_graph.add_edge('B', 'C', Weight(0.7), Weight(1.0))
    comp_graph.add_edge('F', 'C', Weight(1.0), Weight(1.0))
    comp_graph.add_edge('C', 'G', Weight(1.0), Weight(1.0))
    comp_graph.add_edge('H', 'F', Weight(0.5), Weight(1.0))
    comp_graph.add_edge('I', 'F', Weight(0.4), Weight(1.0))

    return comp_graph, [
        TestDataTuple(
            new_computed_nodes={'D'},
            prev_computed_nodes=set(),
            conflicts={}
        ),
        TestDataTuple(
            new_computed_nodes={'D'},
            prev_computed_nodes={'H', 'I'},
            conflicts={'D': {'H', 'I'}}
        ),
        TestDataTuple(
            new_computed_nodes={'D', 'E'},
            prev_computed_nodes=set(),
            conflicts={}
        ),
        TestDataTuple(
            new_computed_nodes={'F'},
            prev_computed_nodes={'D', 'E'},
            conflicts={'D': {'F'}, 'E': {'F'}}
        ),
        TestDataTuple(
            new_computed_nodes={'H'},
            prev_computed_nodes={'D', 'E', 'F'},
            conflicts={'H': {'F'}, 'F': {'H'}, 'D': {'H'}, 'E': {'H'}}
        ),
        TestDataTuple(
            new_computed_nodes={'F'},
            prev_computed_nodes={'D', 'E', 'H'},
            conflicts={'H': {'F'}, 'F': {'H'}, 'D': {'F'}, 'E': {'F'}}
        ),
        TestDataTuple(
            new_computed_nodes={'D', 'E'},
            prev_computed_nodes={'B'},
            conflicts={'D': {'B'}, 'E': {'B'}, }
        ),
        TestDataTuple(
            new_computed_nodes={'D'},
            prev_computed_nodes={'A', 'B', 'H', 'G', 'E'},
            conflicts={'D': {'A', 'B', 'H', 'G'}}
        )
    ]


class TestComputationGraphConflicts(unittest.TestCase):
    test_data: List[Tuple[ComputationGraph, List[TestDataTuple]]] = []

    @classmethod
    def setUpClass(cls):
        """ Executed BEFORE test methods of the class """

        cls.test_data.append(create_test_data1())
        cls.test_data.append(create_test_data2())

    @classmethod
    def tearDownClass(cls):
        """ Executed AFTER tests methods of the class """
        pass

    def setUp(self):
        """ Repeated BEFORE each test """
        super().setUp()

    def tearDown(self):
        """ Repeated AFTER each test """
        super().tearDown()

    def test_conflicts(self):
        for case, (computation_graph, input_output_data) in enumerate(self.test_data):
            with self.subTest(case=case):

                computation_graph.compute_descendants()

                for new_computed_nodes, prev_computed_nodes, expected_conflicts in input_output_data:

                    conflicts = computation_graph.compute_conflicts(new_computed_nodes, prev_computed_nodes)

                    self.assertDictEqual(conflicts, expected_conflicts)
