import unittest
from typing import List, Set, Dict, Tuple, Optional, Callable

from nexinfosys.solving.graph.computation_graph import ComputationGraph
from nexinfosys.solving.graph.flow_graph import FlowGraph
from nexinfosys.solving.graph import EdgeType, Weight, Node, Value


class SubTestCase:
    def __init__(self, params: Dict[str, Value], conflicts: Dict[str, Set[str]],
                 combinations: Set[frozenset], results: Dict[Tuple[str, frozenset], Optional[Value]]):
        self.params = params
        self.conflicts = conflicts
        self.combinations = combinations
        self.results = results


def create_test_data_flow_cycle() -> Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]:
    flow_graph = FlowGraph()

    # flow_graph.add_edge('A', 'B', 2.0, 0.5)
    # flow_graph.add_edge('B', 'C', 0.7, 0.1)

    comp_graph = ComputationGraph()

    comp_graph.add_edge('A', 'B', Weight(2.0), Weight(0.5))
    comp_graph.add_edge('B', 'A', Weight(0.7), Weight(0.1))
    # NOTE: the reverse weight is ignored for this computation

    # Set split info
    #

    subtest_cases: List[SubTestCase] = []

    # Case 0
    params = {'B': Weight(5)}
    conflicts = {
        'B': set()
    }
    combinations = {
        frozenset({'B'})
    }
    results = {
        ('A', frozenset({'B'})): Value(3.5)
    }
    subtest_cases.append(SubTestCase(params, conflicts, combinations, results))

    return flow_graph, comp_graph, subtest_cases


def create_test_data_flow() -> Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]:
    flow_graph = FlowGraph()

    flow_graph.add_edge('D', 'A', Weight(1.0), None)
    flow_graph.add_edge('E', 'A', Weight(1.0), None)
    flow_graph.add_edge('A', 'B', Weight(2.0), None)
    flow_graph.add_edge('B', 'C', Weight(0.7), Weight(0.1))
    flow_graph.add_edge('F', 'C', Weight(1.0), None)
    flow_graph.add_edge('C', 'G', None, None)
    flow_graph.add_edge('H', 'G', Weight(0.5), Weight(0.9))
    flow_graph.add_edge('I', 'G', Weight(0.4), Weight(0.1))

    comp_graph = ComputationGraph()

    comp_graph.add_edge('D', 'A', Weight(1.0), None)
    comp_graph.add_edge('E', 'A', Weight(1.0), None)
    comp_graph.add_edge('A', 'B', Weight(2.0), Weight(0.5))
    comp_graph.add_edge('B', 'C', Weight(0.7), Weight(0.1))
    comp_graph.add_edge('F', 'C', Weight(1.0), Weight(0.9))
    comp_graph.add_edge('C', 'G', Weight(1.0), Weight(0.0))
    comp_graph.add_edge('H', 'G', Weight(0.5), Weight(0.9))
    comp_graph.add_edge('I', 'G', Weight(0.4), Weight(0.1))

    # Set split info
    comp_graph.mark_node_split('C', EdgeType.REVERSE)
    comp_graph.mark_node_split('G', EdgeType.REVERSE)

    return flow_graph, comp_graph, []


def create_test_data_reverse() -> Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]:
    flow_graph = FlowGraph()

    flow_graph.add_edge('A', 'B', Weight(2.0), Weight(0.5))
    flow_graph.add_edge('B', 'C', Weight(0.7), Weight(0.1))
    flow_graph.add_edge('C', 'D', Weight(0.7), Weight(0.1))
    flow_graph.add_edge('C', 'E', Weight(0.3), Weight(0.1))
    flow_graph.add_edge('F', 'G', Weight(0.7), Weight(0.1))
    flow_graph.add_edge('G', 'E', Weight(0.7), Weight(0.1))
    flow_graph.add_edge('B', 'G', Weight(0.7), Weight(0.1))

    comp_graph = ComputationGraph()

    comp_graph.add_edge('A', 'B', Weight(2.0), Weight(0.5))
    comp_graph.add_edge('B', 'C', Weight(0.7), Weight(0.1))
    comp_graph.add_edge('C', 'D', Weight(0.7), Weight(0.1))
    comp_graph.add_edge('C', 'E', Weight(0.3), Weight(0.1))
    comp_graph.add_edge('F', 'G', Weight(0.7), Weight(0.1))
    comp_graph.add_edge('G', 'E', Weight(0.7), Weight(0.1))
    comp_graph.add_edge('B', 'G', Weight(0.7), Weight(0.1))

    # Set split info
    comp_graph.mark_node_split('C', EdgeType.DIRECT)

    subtest_cases: List[SubTestCase] = []

    # Case 0
    params = {'B': Value(5), 'C': Value(10)}
    conflicts = {
        'B': {'C'},
        'C': {'B'}
    }
    combinations = {
        frozenset({'B'}),
        frozenset({'C'})
    }
    results = {
        ('A', frozenset({'B'})): Value(2.5),
        ('A', frozenset({'C'})): None
    }
    subtest_cases.append(SubTestCase(params, conflicts, combinations, results))

    return flow_graph, comp_graph, subtest_cases


def create_test_data_conflict_cycle() -> Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]:
    flow_graph = FlowGraph()

    flow_graph.add_edge('A', 'B', Weight(2.0), Weight(0.5))
    flow_graph.add_edge('B', 'C', Weight(0.7), Weight(0.1))

    comp_graph = ComputationGraph()

    comp_graph.add_edge('A', 'B', Weight(2.0), Weight(0.5))
    comp_graph.add_edge('B', 'C', Weight(0.7), Weight(0.1))

    # Set split info
    #

    subtest_cases: List[SubTestCase] = []

    # Case 0
    params = {'B': Value(5), 'C': Value(10)}
    conflicts = {
        'B': {'C'},
        'C': {'B'}
    }
    combinations = {
        frozenset({'B'}),
        frozenset({'C'})
    }
    results = {
        ('A', frozenset({'B'})): Value(2.5),
        ('A', frozenset({'C'})): Value(0.5)
    }
    subtest_cases.append(SubTestCase(params, conflicts, combinations, results))

    return flow_graph, comp_graph, subtest_cases


def create_test_data_simple() -> Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]:
    flow_graph = FlowGraph()

    flow_graph.add_edge('A', 'B', Weight(2.0), None)
    flow_graph.add_edge('B', 'C', Weight(0.7), None)
    flow_graph.add_edge('C', 'D', Weight(0.1), None)
    flow_graph.add_edge('E', 'D', Weight(0.1), None)

    comp_graph = ComputationGraph()

    comp_graph.add_edge('A', 'B', Weight(2.0), Weight(0.5))
    comp_graph.add_edge('B', 'C', Weight(0.7), Weight(1.4285714))
    comp_graph.add_edge('C', 'D', Weight(0.1), None)
    comp_graph.add_edge('E', 'D', Weight(0.1), None)

    # Set split info
    #

    subtest_cases: List[SubTestCase] = []

    # Case 0
    params = {'A': Value(5), 'B': Value(10), 'C': Value(4), 'E': Value(8)}
    conflicts = {
        'A': {'B', 'C'},
        'B': {'C', 'A'},
        'C': {'B', 'A'},
        'E': set()
    }
    combinations = {
        frozenset({'A', 'E'}),
        frozenset({'B', 'E'}),
        frozenset({'C', 'E'})
    }
    results = {
        ('D', frozenset({'A', 'E'})): Value(1.5),
        ('D', frozenset({'B', 'E'})): Value(1.5),
        ('D', frozenset({'C', 'E'})): Value(1.2)
    }
    subtest_cases.append(SubTestCase(params, conflicts, combinations, results))

    return flow_graph, comp_graph, subtest_cases


def create_test_data_star() -> Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]:
    flow_graph = FlowGraph()

    flow_graph.add_edge('A', 'B', Weight(0.05), None)
    flow_graph.add_edge('A', 'C', Weight(0.7), None)
    flow_graph.add_edge('A', 'D', Weight(0.1), None)
    flow_graph.add_edge('A', 'E', Weight(0.15), None)

    comp_graph = ComputationGraph()

    comp_graph.add_edge('A', 'B', Weight(0.05), Weight(20.0))
    comp_graph.add_edge('A', 'C', Weight(0.7), Weight(1.4285714))
    comp_graph.add_edge('A', 'D', Weight(0.1), Weight(10.0))
    comp_graph.add_edge('A', 'E', Weight(0.15), Weight(6.66666667))

    # Set split info
    comp_graph.mark_node_split('A', EdgeType.DIRECT)

    subtest_cases: List[SubTestCase] = []

    # Case 0
    params = {'B': Value(5), 'C': Value(10), 'D': Value(8), 'E': Value(12)}
    conflicts = {
        'B': {'C', 'D', 'E'},
        'C': {'B', 'D', 'E'},
        'D': {'B', 'C', 'E'},
        'E': {'B', 'C', 'D'}
    }
    combinations = {
        frozenset({'B'}),
        frozenset({'C'}),
        frozenset({'D'}),
        frozenset({'E'})
    }
    results = {
        ('A', frozenset({'B'})): Value(100),
        ('A', frozenset({'C'})): Value(14.285714),
        ('A', frozenset({'D'})): Value(80),
        ('A', frozenset({'E'})): Value(80)
    }
    subtest_cases.append(SubTestCase(params, conflicts, combinations, results))

    # Case 1
    params = {'A': Value(5), 'B': Value(10)}
    conflicts = {
        'A': {'B'},
        'B': {'A'}
    }
    combinations = {
        frozenset({'A'}),
        frozenset({'B'})
    }
    results = {
        ('A', frozenset({'B'})): Value(200),
        ('A', frozenset({'A'})): Value(5),
        ('B', frozenset({'B'})): Value(10),
        ('B', frozenset({'A'})): Value(0.25),
        ('C', frozenset({'B'})): Value(140),
        ('C', frozenset({'A'})): Value(3.5)
    }
    subtest_cases.append(SubTestCase(params, conflicts, combinations, results))

    return flow_graph, comp_graph, subtest_cases


def create_test_data_star2() -> Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]:
    flow_graph = FlowGraph()

    flow_graph.add_edge('A', 'B', None, None)
    flow_graph.add_edge('A', 'C', None, None)
    flow_graph.add_edge('A', 'D', None, None)
    flow_graph.add_edge('A', 'E', None, None)

    comp_graph = ComputationGraph()

    comp_graph.add_edge('A', 'B', None, Weight(1.0))
    comp_graph.add_edge('A', 'C', None, Weight(1.0))
    comp_graph.add_edge('A', 'D', None, Weight(1.0))
    comp_graph.add_edge('A', 'E', None, Weight(1.0))

    # Set split info
    #

    subtest_cases: List[SubTestCase] = []

    # Case 0
    params = {'B': Value(5), 'C': Value(10), 'D': Value(8), 'E': Value(12)}
    conflicts = {
        'B': set(),
        'C': set(),
        'D': set(),
        'E': set()
    }
    combinations = {
        frozenset({'B', 'C', 'D', 'E'})
    }
    results = {
        ('A', frozenset({'B', 'C', 'D', 'E'})): Value(35)
    }
    subtest_cases.append(SubTestCase(params, conflicts, combinations, results))

    return flow_graph, comp_graph, subtest_cases


class TestComputationGraph(unittest.TestCase):
    computation_graphs: Dict[FlowGraph, ComputationGraph] = {}
    subtest_cases: Dict[ComputationGraph, List[SubTestCase]] = {}

    @classmethod
    def setUpClass(cls):
        """ Executed BEFORE test methods of the class """

        def add_test_case(
                create_test_data: Callable[[], Tuple[FlowGraph, ComputationGraph, List[SubTestCase]]]) -> None:
            flow_graph, comp_graph, subtest_cases = create_test_data()
            cls.computation_graphs[flow_graph] = comp_graph
            cls.subtest_cases[comp_graph] = subtest_cases

        add_test_case(create_test_data_flow_cycle)
        add_test_case(create_test_data_flow)
        add_test_case(create_test_data_reverse)
        add_test_case(create_test_data_conflict_cycle)
        add_test_case(create_test_data_simple)
        add_test_case(create_test_data_star)
        add_test_case(create_test_data_star2)

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

    def test_flow_computation_graph_conversion(self):
        for g, (flow_graph, comp_graph) in enumerate(self.computation_graphs.items()):
            with self.subTest(graph=g):

                computed_comp_graph, _ = flow_graph.get_computation_graph()  # type: ComputationGraph

                for u, v, weight in computed_comp_graph.graph.edges.data("weight"):  # type: Node, Node, Weight
                    self.assertAlmostEqual(comp_graph.graph[u][v]["weight"], weight)

                for n, split in computed_comp_graph.graph.nodes.data("split"):  # type: Node, List[bool]
                    self.assertListEqual(comp_graph.graph.nodes[n]["split"], split)

    def test_parameters_conflicts(self):
        for g, graph in enumerate(self.subtest_cases):
            for c, case in enumerate(self.subtest_cases[graph]):
                with self.subTest(graph=g, case=c):
                    conflicts = graph.compute_param_conflicts(set(case.params.keys()))
                    self.assertDictEqual(case.conflicts, conflicts)

    def test_parameters_combinations(self):
        for g, graph in enumerate(self.subtest_cases):
            for c, case in enumerate(self.subtest_cases[graph]):
                with self.subTest(graph=g, case=c):
                    combinations = ComputationGraph.compute_param_combinations(case.conflicts)
                    self.assertSetEqual(case.combinations, combinations)

    def test_compute_values(self):
        for g, graph in enumerate(self.subtest_cases):
            for c, case in enumerate(self.subtest_cases[graph]):
                with self.subTest(graph=g, case=c):
                    nodes = list({k[0] for k in case.results})
                    for combination in case.combinations:
                        filtered_params = {k: case.params[k] for k in combination}
                        results, _ = graph.compute_values(nodes, filtered_params)

                        for param, value in results.items():
                            self.assertAlmostEqual(value, case.results[(param, combination)])
