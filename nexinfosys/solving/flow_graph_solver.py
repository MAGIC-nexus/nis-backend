"""
A solver based on the elaboration of a flow graph to quantify interfaces, connected by flow or scale relationships.
It is assumed that other kinds of relationship (part-of, upscale, ...) are translated into these two basic ones.
Another type of relationship considered is linear transform from InterfaceType to InterfaceType, which is cascaded into
appearances of its instances.

Before the elaboration of flow graphs, several preparatory steps:
* Find the separate contexts. Each context is formed by the "local", "environment" and "external" sets of processors,
  and is -totally- isolated from other contexts
  - Context defining attributes are defined in the Problem Statement command. If not, defined, a "context" attribute in
    Processors would be assumed
    If nothing is found, all Processors are assumed to be under the same context (what will happen??!!)
  - Elaborate (add necessary entities) the "environment", "external" top level processors if none have been specified.
    - Opposite processor can be specified when defining Interface
      - This attribute is taken into account if NO relationship originates or ends in this Interface. Then, a default
        relationship would be created
      - If Processors are defined for environment or
* Unexecuted model parts
  - Connection of Interfaces
  - Dataset expansion
* [Datasets]
* Scenarios
  - Parameters
* Time. Classify QQs by time, on storage
* Observers (different versions). Take average always

"""
from _operator import add
from collections import defaultdict
from enum import Enum
from functools import reduce
from itertools import chain

import lxml
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Any, Tuple, Union, Optional, NamedTuple, Generator, Type

from nexinfosys import case_sensitive
from nexinfosys.command_field_definitions import orientations
from nexinfosys.command_generators.parser_ast_evaluators import ast_evaluator
from nexinfosys.command_generators.parser_field_parsers import string_to_ast, expression_with_parameters, is_year, \
    is_month, indicator_expression
from nexinfosys.common.helper import create_dictionary, PartialRetrievalDictionary, ifnull, Memoize, istr, strcmp
from nexinfosys.ie_exports.xml import export_model_to_xml
from nexinfosys.models import CodeImmutable
from nexinfosys.models.musiasem_concepts import ProblemStatement, Parameter, FactorsRelationDirectedFlowObservation, \
    FactorsRelationScaleObservation, Processor, FactorQuantitativeObservation, Factor, \
    ProcessorsRelationPartOfObservation, FactorType, Indicator, MatrixIndicator, IndicatorCategories
from nexinfosys.model_services import get_case_study_registry_objects, State
from nexinfosys.models.musiasem_concepts_helper import find_quantitative_observations
from nexinfosys.solving.graph.computation_graph import ComputationGraph
from nexinfosys.solving.graph.flow_graph import FlowGraph, IType
from nexinfosys.models.statistical_datasets import Dataset, Dimension, CodeList
from nexinfosys.command_generators import Issue
from lxml import etree


class SolvingException(Exception):
    pass


class InterfaceNode:
    registry: PartialRetrievalDictionary = None

    def __init__(self, interface_or_type: Union[Factor, FactorType], processor: Processor = None,
                 orientation: str = None, processor_name: str = None):
        if isinstance(interface_or_type, Factor):
            self.interface: Optional[Factor] = interface_or_type
            self.interface_type = self.interface.taxon
            self.orientation: Optional[str] = orientation if orientation else self.interface.orientation
            self.processor = processor if processor else self.interface.processor
        elif isinstance(interface_or_type, FactorType):
            self.interface: Optional[Factor] = None
            self.interface_type = interface_or_type
            self.orientation = orientation
            self.processor = processor
        else:
            raise Exception(f"Invalid object type '{type(interface_or_type)}' for the first parameter. "
                            f"Valid object types are [Factor, FactorType].")

        self.interface_name: str = interface_or_type.name
        self.processor_name: str = get_processor_name(self.processor, self.registry) if self.processor else processor_name
        self.name: str = self.processor_name + ":" + self.interface_name + ":" + self.orientation

    @property
    def key(self) -> Tuple:
        return self.processor_name, self.interface_name, self.orientation

    @staticmethod
    def key_labels() -> List[str]:
        return ["Processor", "Interface", "Orientation"]

    @property
    def unit(self):
        return self.interface_type.attributes.get('unit')

    @property
    def roegen_type(self):
        if self.interface and self.interface.roegen_type:
            if isinstance(self.interface.roegen_type, str):
                return self.interface.roegen_type
            else:
                return self.interface.roegen_type.name
        elif self.interface_type and self.interface_type.roegen_type:
            if isinstance(self.interface_type.roegen_type, str):
                return self.interface_type.roegen_type
            else:
                return self.interface_type.roegen_type.name
        else:
            return ""

    @property
    def sphere(self):
        if self.interface and self.interface.sphere:
            return self.interface.sphere
        else:
            return self.interface_type.sphere

    @property
    def system(self):
        return self.processor.processor_system if self.processor else None

    @property
    def subsystem(self):
        return self.processor.subsystem_type if self.processor else None

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return istr(str(self))

    def __hash__(self):
        return hash(repr(self))


class Scope(Enum):
    Total = 1
    Internal = 2
    External = 3


class Computed(Enum):
    No = 1
    Yes = 2


NodeFloatDict = Dict[InterfaceNode, float]
# Key = Tuple[scenario_name, time_period, Scope, Computed]
ResultDict = Dict[Tuple[str, str, str], NodeFloatDict]


@Memoize
def get_processor_name(processor: Processor, registry: PartialRetrievalDictionary) -> str:
    """ Get the processor hierarchical name with caching enabled """
    return processor.full_hierarchy_names(registry)[0]


def get_circular_dependencies(parameters: Dict[str, Tuple[Any, list]]) -> list:
    # Graph, for evaluation of circular dependencies
    G = nx.DiGraph()
    for param, (_, dependencies) in parameters.items():
        for param2 in dependencies:
            G.add_edge(param2, param)  # We need "param2" to obtain "param"
    return list(nx.simple_cycles(G))


def evaluate_parameters_for_scenario(base_params: List[Parameter], scenario_params: Dict[str, str]):
    """
    Obtain a dictionary (parameter -> value), where parameter is a string and value is a literal: number, boolean,
    category or string.

    Start from the base parameters then overwrite with the values in the current scenario.

    Parameters may depend on other parameters, so this has to be considered before evaluation.
    No cycles are allowed in the dependencies, i.e., if P2 depends on P1, P1 cannot depend on P2.
    To analyze this, first expressions are evaluated, extracting which parameters appear in each of them. Then a graph
    is elaborated based on this information. Finally, an algorithm to find cycles is executed.

    :param base_params:
    :param scenario_params:
    :return:
    """
    # Create dictionary without evaluation
    result_params = create_dictionary()
    result_params.update({p.name: p.default_value for p in base_params if p.default_value is not None})
    param_types = create_dictionary()
    param_types.update({p.name: p.type for p in base_params})

    # Overwrite with scenario expressions or constants
    result_params.update(scenario_params)

    state = State()
    known_params = create_dictionary()
    unknown_params = create_dictionary()

    # Now, evaluate ALL expressions
    for param, expression in result_params.items():
        ptype = param_types[param]
        if strcmp(ptype, "Number") or strcmp(ptype, "Boolean"):
            value, ast, params, issues = evaluate_numeric_expression_with_parameters(expression, state)
            if value is None:  # It is not a constant, store the parameters on which this depends
                unknown_params[param] = (ast, set([istr(p) for p in params]))
            else:  # It is a constant, store it
                result_params[param] = value  # Overwrite
                known_params[param] = value
        elif strcmp(ptype, "Code") or strcmp(ptype, "String"):
            result_params[param] = expression
            known_params[param] = expression

    cycles = get_circular_dependencies(unknown_params)
    if len(cycles) > 0:
        raise SolvingException(
            IType.ERROR, f"Parameters cannot have circular dependencies. {len(cycles)} cycles were detected: "
                         f"{':: '.join(cycles)}")

    # Initialize state with known parameters
    state.update(known_params)

    known_params_set = set([istr(p) for p in known_params.keys()])
    # Loop until no new parameters can be evaluated
    previous_len_unknown_params = len(unknown_params) + 1
    while len(unknown_params) < previous_len_unknown_params:
        previous_len_unknown_params = len(unknown_params)

        for param in list(unknown_params):  # A list(...) is used because the dictionary can be modified inside
            ast, params = unknown_params[param]

            if params.issubset(known_params_set):
                value, _, _, issues = evaluate_numeric_expression_with_parameters(ast, state)
                if value is None:
                    raise SolvingException(
                        IType.ERROR, f"It should be possible to evaluate the parameter '{param}'. "
                                     f"Issues: {', '.join(issues)}")
                else:
                    del unknown_params[param]
                    result_params[param] = value
                    # known_params[param] = value  # Not necessary
                    known_params_set.add(istr(param))
                    state.set(param, value)

    if len(unknown_params) > 0:
        raise SolvingException(IType.ERROR, f"Could not evaluate the following parameters: {', '.join(unknown_params)}")

    return result_params


TimeObservationsType = Dict[str, List[Tuple[float, FactorQuantitativeObservation]]]


def get_evaluated_observations_by_time(prd: PartialRetrievalDictionary) -> TimeObservationsType:
    """
        Get all interface observations (intensive or extensive) by time.
        Also resolve expressions without parameters. Cannot resolve expressions depending only on global parameters
        because some of them can be overridden by scenario parameters.

        Each evaluated observation is stored as a tuple:
        * First: the evaluated result as a float or the prepared AST
        * Second: the observation

    :param prd: the global objects dictionary
    :return: a time dictionary with a list of observation on each time
    """
    observations: TimeObservationsType = defaultdict(list)
    state = State()

    # Get all observations by time
    for observation in find_quantitative_observations(prd, processor_instances_only=True):

        # Try to evaluate the observation value
        value, ast, _, issues = evaluate_numeric_expression_with_parameters(observation.value, state)

        # Store: (Value, FactorQuantitativeObservation)
        time = observation.attributes["time"].lower()
        observations[time].append((ifnull(value, ast), observation))

    if len(observations) == 0:
        return {}

    # Check all time periods are consistent. All should be Year or Month, but not both.
    time_period_type = check_type_consistency_from_all_time_periods(list(observations.keys()))
    assert(time_period_type in ["year", "month"])

    # Remove generic period type and insert it into all specific periods. E.g. "Year" into "2010", "2011" and "2012"
    if time_period_type in observations:
        # Generic monthly ("Month") or annual ("Year") data
        periodic_observations = observations.pop(time_period_type)

        for time in observations:
            observations[time] += periodic_observations

    return observations


def evaluate_numeric_expression_with_parameters(expression: Union[float, str, dict], state: State) \
        -> Tuple[Optional[float], Optional[Dict], Set, List[str]]:

    issues: List[Tuple[int, str]] = []
    ast = None
    value = None
    params = set()

    if expression is None:
        value = None

    elif isinstance(expression, float):
        value = expression

    elif isinstance(expression, dict):
        ast = expression
        value, params = ast_evaluator(ast, state, None, issues)
        if value is not None:
            ast = None

    elif isinstance(expression, str):
        try:
            value = float(expression)
        except ValueError:
            print(f"{expression} before")
            ast = string_to_ast(expression_with_parameters, expression)
            print(f"{expression} after")
            value, params = ast_evaluator(ast, state, None, issues)
            if value is not None:
                ast = None

    else:
        issues.append((3, f"Invalid type '{type(expression)}' for expression '{expression}'"))

    return value, ast, params, [i[1] for i in issues]


def check_type_consistency_from_all_time_periods(time_periods: List[str]) -> str:
    """ Check if all time periods are of the same period type, either Year or Month:
         - general "Year" & specific year (YYYY)
         - general "Month" & specific month (mm-YYYY or YYYY-mm, separator can be any of "-/")

    :param time_periods:
    :return:
    """
    # Based on the first element we will check the rest of elements
    period = next(iter(time_periods))

    if period == "year" or is_year(period):
        period_type = "year"
        period_check = is_year
    elif period == "month" or is_month(period):
        period_type = "month"
        period_check = is_month
    else:
        raise SolvingException(IType.ERROR, f"Found invalid period type '{period}'")

    for time_period in time_periods:
        if time_period != period_type and not period_check(time_period):
            raise SolvingException(IType.ERROR,
                f"Found period type inconsistency: accepting '{period_type}' but found '{time_period}'")

    return period_type


def split_observations_by_relativeness(observations_by_time: TimeObservationsType) -> Tuple[TimeObservationsType, TimeObservationsType]:
    observations_by_time_norelative = defaultdict(list)
    observations_by_time_relative = defaultdict(list)
    for time, observations in observations_by_time.items():
        for value, obs in observations:
            if obs.is_relative:
                observations_by_time_relative[time].append((value, obs))
            else:
                observations_by_time_norelative[time].append((value, obs))

    return observations_by_time_norelative, observations_by_time_relative


def compute_graph_values(comp_graph: ComputationGraph, params: NodeFloatDict, other_values: NodeFloatDict) \
        -> Tuple[NodeFloatDict, List[InterfaceNode]]:
    print(f"****** NODES: {comp_graph.nodes}")

    # Filter params in graph
    graph_params: NodeFloatDict = {k: v for k, v in params.items() if k in comp_graph.nodes}
    print("Missing values: ", [k for k, v in graph_params.items() if v is None])

    conflicts: Dict[InterfaceNode, Set[InterfaceNode]] = comp_graph.compute_param_conflicts(set(graph_params.keys()))

    raise_error_if_conflicts(conflicts, graph_params)

    graph_params = {**graph_params, **other_values}

    # Obtain nodes without a value
    compute_nodes = [n for n in comp_graph.nodes if graph_params.get(n) is None]

    # Compute the missing information with the computation graph
    if len(compute_nodes) == 0:
        print("All nodes have a value. Nothing to solve.")
        return {}, []

    print(f"****** UNKNOWN NODES: {compute_nodes}")
    print(f"****** PARAMS: {graph_params}")

    results, _ = comp_graph.compute_values(compute_nodes, graph_params)

    results_with_values: NodeFloatDict = {}
    unknown_nodes: List[InterfaceNode] = []
    for k, v in results.items():
        if v is not None:
            results_with_values[k] = v
        else:
            unknown_nodes.append(k)

    return results_with_values, unknown_nodes


def raise_error_if_conflicts(conflicts: Dict[InterfaceNode, Set[InterfaceNode]], graph_params: NodeFloatDict):
    conflict_strings: List[str] = []
    for param, conf_params in conflicts.items():
        if len(conf_params) > 0:
            conf_params_string = "{" + ', '.join([f"{p} ({round(graph_params[p], 3)})" for p in conf_params]) + "}"
            conflict_strings.append(f"{param} ({round(graph_params[param], 3)}) -> {conf_params_string}")

    if len(conflict_strings) > 0:
        raise SolvingException(IType.ERROR, f"There are conflicts: {', '.join(conflict_strings)}")


def split_name(processor_interface: str) -> Tuple[str, Optional[str]]:
    l = processor_interface.split(":")
    if len(l) > 1:
        return l[0], l[1]
    else:
        return l[0], None


class Edge(NamedTuple):
    src: Factor
    dst: Factor
    weight: Optional[str]


InterfacesRelationClassType = Type[Union[FactorsRelationDirectedFlowObservation, FactorsRelationScaleObservation]]


def create_interface_edges(edges: List[Tuple[Factor, Factor, Optional[str]]]) \
        -> Generator[Tuple[InterfaceNode, InterfaceNode, Dict], None, None]:
    for src, dst, weight in edges:
        src_node = InterfaceNode(src)
        dst_node = InterfaceNode(dst)
        if "Archetype" in [src.processor.instance_or_archetype, dst.processor.instance_or_archetype]:
            print(f"WARNING: excluding relation from '{src_node}' to '{dst_node}' because of Archetype processor")
        else:
            yield src_node, dst_node, dict(weight=weight)


def resolve_weight_expressions(graph_list: List[nx.DiGraph], state: State, raise_error=False) -> None:
    for graph in graph_list:
        for u, v, data in graph.edges(data=True):
            expression = data["weight"]
            if expression is not None:
                value, ast, params, issues = evaluate_numeric_expression_with_parameters(expression, state)
                if raise_error and value is None:
                    raise SolvingException(
                        IType.ERROR,
                        f"Cannot evaluate expression "
                        f"'{expression}' for weight from interface '{u}' to interface '{v}'. Params: {params}. "
                        f"Issues: {', '.join(issues)}"
                    )

                data["weight"] = ifnull(value, ast)


def iterative_solving(comp_graph_list: List[ComputationGraph], observations: NodeFloatDict) -> Tuple[NodeFloatDict, List[InterfaceNode]]:
    num_unknown_nodes: int = reduce(add, [len(g.nodes) for g in comp_graph_list])
    prev_num_unknown_nodes: int = num_unknown_nodes + 1
    params: List[Optional[NodeFloatDict]] = [None] * len(comp_graph_list)
    other_values: List[NodeFloatDict] = [{}] * len(comp_graph_list)
    unknown_nodes: List[List[InterfaceNode]] = [[]] * len(comp_graph_list)
    data: NodeFloatDict = {}
    all_data: NodeFloatDict = {}

    while prev_num_unknown_nodes > num_unknown_nodes > 0:
        prev_num_unknown_nodes = num_unknown_nodes

        for i, comp_graph in enumerate(comp_graph_list):
            if params[i] is None:
                params[i] = {**data, **observations}
            else:
                params[i] = {**data}
            data, unknown_nodes[i] = compute_graph_values(comp_graph, params[i], other_values[i])
            all_data.update(data)
            other_values[i].update({**data, **params[i]})

        num_unknown_nodes = reduce(add, [len(l) for l in unknown_nodes])

    all_unknown_nodes = [node for node_list in unknown_nodes for node in node_list]

    return all_data, all_unknown_nodes


def create_scale_change_relations_and_update_flow_relations(relations_flow: nx.DiGraph, registry) -> nx.DiGraph:
    relations_scale_change = nx.DiGraph()

    edges = [(r.source_factor, r.target_factor, r.back_factor, r.weight, r.scale_change_weight)
             for r in registry.get(FactorsRelationDirectedFlowObservation.partial_key())
             if r.scale_change_weight is not None or r.back_factor is not None]

    for src, dst, bck, weight, scale_change_weight in edges:

        source_node = InterfaceNode(src)
        dest_node = InterfaceNode(dst)
        back_node = InterfaceNode(bck) if bck else None

        if "Archetype" in [src.processor.instance_or_archetype,
                           dst.processor.instance_or_archetype,
                           bck.processor.instance_or_archetype if bck else None]:
            print(f"WARNING: excluding relation from '{source_node}' to '{dest_node}' "
                  f"and back to '{back_node}' because of Archetype processor")
            continue

        hidden_node = InterfaceNode(src.taxon,
                                    processor_name=f"{get_processor_name(src.processor, registry)}-"
                                                   f"{get_processor_name(dst.processor, registry)}",
                                    orientation="Input/Output")

        relations_flow.add_edge(source_node, hidden_node, weight=weight)
        relations_scale_change.add_edge(hidden_node, dest_node, weight=scale_change_weight, add_reverse_weight="yes")
        if back_node:
            relations_scale_change.add_edge(hidden_node, back_node, weight=scale_change_weight, add_reverse_weight="yes")

        relations_scale_change.nodes[hidden_node]["add_split"] = "yes"

        real_dest_node = InterfaceNode(source_node.interface_type, dest_node.processor,
                                       orientation="Input" if source_node.orientation.lower() == "output" else "Output")

        if relations_flow.has_edge(source_node, real_dest_node):
            # weight = relations_flow[source_node][real_dest_node]['weight']
            relations_flow.remove_edge(source_node, real_dest_node)
            # relations_flow.add_edge(source_node, hidden_node, weight=weight)  # This "weight" should be the same
            relations_flow.add_edge(hidden_node, real_dest_node, weight=1.0)

    return relations_scale_change


def get_scenario_evaluated_observation_results(scenario_states: Dict[str, State],
                                               time_observations: TimeObservationsType) -> ResultDict:
    results: ResultDict = {}

    for scenario_name, scenario_state in scenario_states.items():  # type: str, State
        for time_period, observations in time_observations.items():
            resolved_observations: NodeFloatDict = {}

            # Second and last pass to resolve observation expressions with parameters
            for expression, obs in observations:
                value, _, params, issues = evaluate_numeric_expression_with_parameters(expression, scenario_state)
                if value is None:
                    raise SolvingException(IType.ERROR,
                        f"Scenario '{scenario_name}' - period '{time_period}'. Cannot evaluate expression "
                        f"'{expression}' for observation at interface '{obs.factor.name}'. Params: {params}. "
                        f"Issues: {', '.join(issues)}"
                    )

                resolved_observations[InterfaceNode(obs.factor)] = value

            # results[(scenario_name, time_period, Scope.Total.name, Computed.No.name)] = resolved_observations
            results[(scenario_name, time_period, Scope.Total.name)] = resolved_observations

            empty_graph: ComputationGraph = ComputationGraph()
            internal_data, external_data = compute_separated_results(resolved_observations, empty_graph)
            if len(external_data) > 0:
                # results[(scenario_name, time_period, Scope.External.name, Computed.No.name)] = external_data
                results[(scenario_name, time_period, Scope.External.name)] = external_data

            if len(internal_data) > 0:
                # results[(scenario_name, time_period, Scope.Internal.name, Computed.No.name)] = internal_data
                results[(scenario_name, time_period, Scope.Internal.name)] = internal_data

    return results


def compute_flow_and_scale_results(state: State, glb_idx, scenario_states: Dict[str, State],
                                   observation_results: ResultDict,
                                   time_observations_relative: TimeObservationsType) -> ResultDict:
    # Add Interfaces -Flow- relations (time independent)
    relations_flow = nx.DiGraph(
        incoming_graph_data=create_interface_edges(
            [(r.source_factor, r.target_factor, r.weight)
             for r in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key())
             if r.scale_change_weight is None and r.back_factor is None]
        )
    )

    # Add Processors -Scale- relations (time independent)
    relations_scale = nx.DiGraph(
        incoming_graph_data=create_interface_edges(
            [(r.origin, r.destination, r.quantity)
             for r in glb_idx.get(FactorsRelationScaleObservation.partial_key())]
        )
    )

    # Add Interfaces -Scale Change- relations (time independent). Also update Flow relations.
    relations_scale_change = create_scale_change_relations_and_update_flow_relations(relations_flow, glb_idx)

    # First pass to resolve weight expressions: only expressions without parameters can be solved
    resolve_weight_expressions([relations_flow, relations_scale, relations_scale_change], state)

    results: ResultDict = {}

    for scenario_name, scenario_state in scenario_states.items():  # type: str, State
        print(f"********************* SCENARIO: {scenario_name}")

        for time_period, observations in [(k[1], v) for k, v in observation_results.items() if k[0] == scenario_name]:
            print(f"********************* TIME PERIOD: {time_period}")

            # Create a copy of the main relations structures that are modified with time-dependent values
            time_relations_flow = relations_flow.copy()
            time_relations_scale = relations_scale.copy()
            time_relations_scale_change = relations_scale_change.copy()

            # RNEBOT - This "for" iterator was AFTER the elaboration of "known_observations". Changing it to BEFORE
            #          enables specifying unitary processors which are DIRECTLY SCALED by specifying the value of the
            #          dimensioning Interface.
            
            # Add Processors internal -RelativeTo- relations (time dependent)
            # Transform relative observations into graph edges
            for expression, obs in time_observations_relative[time_period]:
                time_relations_scale.add_edge(InterfaceNode(obs.relative_factor, obs.factor.processor),
                                              InterfaceNode(obs.factor),
                                              weight=expression)

            # Are there observations we can use to resolve the graphs? If not continue
            if observations.keys().isdisjoint(set().union(time_relations_flow.nodes, time_relations_scale.nodes,
                                                          time_relations_scale_change.nodes)):
                # raise SolvingException(IType.ERROR, f"Scenario '{scenario_name}' - period '{time_period}'. No 'flowable' or 'scalable' observations available.")
                print(f"WARNING: Scenario '{scenario_name}' - period '{time_period}'. No 'flowable' or 'scalable' observations available.")
                continue

            # Second and last pass to resolve weight expressions: expressions with parameters can be solved
            resolve_weight_expressions([time_relations_flow, time_relations_scale, time_relations_scale_change],
                                       scenario_state, raise_error=True)

            # Show graphs with Matplotlib. Use only for debugging!
            # if scenario_name == 'Scenario1' and time_period == '2011':
            #     for graph in [time_relations_scale]:  # time_relations_flow
            #         for component in nx.weakly_connected_components(graph):
            #             # plt.figure(1, figsize=(8, 8))
            #             nx.draw_spring(graph.subgraph(component), with_labels=True, font_size=8, node_size=60)
            #             # nx.draw_kamada_kawai(graph.subgraph(component), with_labels=True)
            #             plt.show()

            # Create computation graphs
            comp_graph_flow = create_computation_graph_from_flows(time_relations_flow, time_relations_scale)
            comp_graph_scale = ComputationGraph(time_relations_scale)
            comp_graph_scale_change = ComputationGraph(relations_scale_change)

            # Solve computation graphs
            data, unknown_data = iterative_solving([comp_graph_scale, comp_graph_scale_change, comp_graph_flow], observations)

            print(f"Unknown data: {unknown_data}")

            # Filter out results without a real processor
            data: NodeFloatDict = {k: v for k, v in data.items() if k.processor}

            # results[(scenario_name, time_period, Scope.Total.name, Computed.Yes.name)] = data
            results[(scenario_name, time_period, Scope.Total.name)] = data

            # TODO INDICATORS

            internal_data, external_data = compute_separated_results(data, comp_graph_flow)
            if len(external_data) > 0:
                # results[(scenario_name, time_period, Scope.External.name, Computed.Yes.name)] = external_data
                results[(scenario_name, time_period, Scope.External.name)] = external_data

            if len(internal_data) > 0:
                # results[(scenario_name, time_period, Scope.Internal.name, Computed.Yes.name)] = internal_data
                results[(scenario_name, time_period, Scope.Internal.name)] = internal_data

    return results


def create_computation_graph_from_flows(relations_flow: nx.DiGraph, relations_scale: Optional[nx.DiGraph] = None) -> ComputationGraph:
    flow_graph = FlowGraph(relations_flow)
    comp_graph_flow, issues = flow_graph.get_computation_graph(relations_scale)

    for issue in issues:
        print(issue)

    error_issues = [e.description for e in issues if e.itype == IType.ERROR]
    if len(error_issues) > 0:
        raise SolvingException(
            IType.ERROR, f"The computation graph cannot be generated. Issues: {', '.join(error_issues)}"
        )

    return comp_graph_flow


def compute_separated_results(values: NodeFloatDict, comp_graph: ComputationGraph) -> Tuple[NodeFloatDict, NodeFloatDict]:
    internal_results: NodeFloatDict = {}
    external_results: NodeFloatDict = {}
    for node, value in values.items():

        if node in comp_graph.graph:
            if node.orientation.lower() == 'input':
                # res = compute resulting vector based on INCOMING flows processor.subsystem_type
                edges: Set[Tuple[InterfaceNode, InterfaceNode, Dict]] = comp_graph.graph.in_edges(node, data=True)
            else:
                # res = compute resulting vector based on OUTCOMING flows processor.subsystem_type
                edges: Set[Tuple[InterfaceNode, InterfaceNode, Dict]] = comp_graph.graph.out_edges(node, data=True)

            external_value: Optional[float] = None
            internal_value: Optional[float] = None
            for opposite_node, _, data in edges:
                if data['weight'] and opposite_node in values:
                    edge_value = values[opposite_node] * data['weight']

                    if opposite_node.subsystem.lower() in ["external", "externalenvironment"]:
                        external_value = (0.0 if external_value is None else external_value) + edge_value
                    else:
                        internal_value = (0.0 if internal_value is None else internal_value) + edge_value

            if external_value is not None:
                external_results[node] = external_value

            if internal_value is not None:
                internal_results[node] = internal_value

        if node not in chain(external_results, internal_results):
            # res = compute resulting vector based on containing PROCESSOR
            if node.subsystem.lower() in ["external", "externalenvironment"]:
                external_results[node] = value
            else:
                internal_results[node] = value

    return internal_results, external_results


def compute_interfacetype_aggregates(glb_idx, results):

    def get_sum(processor: Processor, children: Set[FactorType]) -> float:
        sum_children = None
        for child in children:
            child_value = values.get(InterfaceNode(child, processor, orientation))
            if child_value is None:
                if child in parent_interfaces:
                    child_value = get_sum(processor, parent_interfaces[child])

            if child_value is not None:
                if sum_children is None:
                    sum_children = child_value
                else:
                    sum_children += child_value

        return sum_children

    agg_results: ResultDict = {}
    processors: Set[Processor] = {node.processor for values in results.values() for node in values}

    # Get all different existing interfaces types that can be computed based on children interface types
    parent_interfaces: Dict[FactorType, Set[FactorType]] = \
        {ft: ft.get_children() for ft in glb_idx.get(FactorType.partial_key()) if len(ft.get_children()) > 0}

    for key, values in results.items():
        for parent_interface, children_interfaces in parent_interfaces.items():
            for proc in processors:
                for orientation in ["Input", "Output"]:
                    interface_node = InterfaceNode(parent_interface, proc, orientation)

                    if values.get(interface_node) is None:
                        sum_result = get_sum(proc, children_interfaces)
                        if sum_result is not None:
                            agg_results.setdefault(key, {}).update({
                                interface_node: sum_result
                            })

    return agg_results


def get_processor_partof_hierarchies(glb_idx, system):
    # Handle Processors -PartOf- relations
    proc_hierarchies = nx.DiGraph()

    # Just get the -PartOf- relations of the current system
    part_of_relations = [(r.child_processor, r.parent_processor)
                         for r in glb_idx.get(ProcessorsRelationPartOfObservation.partial_key())
                         if system in [r.parent_processor.processor_system, r.child_processor.processor_system]]

    for child_processor, parent_processor in part_of_relations:  # type: Processor, Processor
        # Parent and child should be of the same system
        assert (child_processor.processor_system == parent_processor.processor_system)

        if "Archetype" in [parent_processor.instance_or_archetype, child_processor.instance_or_archetype]:
            print(f"WARNING: excluding relation from '{get_processor_name(child_processor, glb_idx)}' to "
                  f"'{get_processor_name(parent_processor, glb_idx)}' because of Archetype processor")
        else:
            proc_hierarchies.add_edge(child_processor, parent_processor, weight=1.0)

    return proc_hierarchies


def compute_partof_aggregates(glb_idx: PartialRetrievalDictionary,
                              systems: Dict[str, Set[Processor]],
                              results: ResultDict):
    agg_results: ResultDict = {}

    # Get all different existing interfaces and their units
    # TODO: interfaces could need a unit transformation according to interface type
    interfaces: Set[FactorType] = glb_idx.get(FactorType.partial_key())

    for system in systems:
        print(f"********************* SYSTEM: {system}")

        proc_hierarchies = get_processor_partof_hierarchies(glb_idx, system)

        # Iterate over all independent trees created by the PartOf relations
        for component in nx.weakly_connected_components(proc_hierarchies):
            proc_hierarchy: nx.DiGraph = proc_hierarchies.subgraph(component)
            print(f"********************* HIERARCHY: {[p.name for p in proc_hierarchy.nodes]}")

            # plt.figure(1, figsize=(8, 8))
            # nx.draw_spring(proc_hierarchy, with_labels=True, font_size=8, node_size=60)
            # plt.show()

            for interface in interfaces:
                print(f"********************* INTERFACE: {interface.name}")

                for orientation in orientations:
                    print(f"********************* ORIENTATION: {orientation}")

                    interfaced_proc_hierarchy = nx.DiGraph(
                        incoming_graph_data=[(InterfaceNode(interface, u, orientation),
                                              InterfaceNode(interface, v, orientation))
                                             for u, v in proc_hierarchy.edges]
                    )

                    for key, values in results.items():

                        filtered_values: NodeFloatDict = {k: values[k]
                                                          for k in interfaced_proc_hierarchy.nodes
                                                          if values.get(k) is not None}

                        if len(filtered_values) > 0:
                            print(f"*** (Results key = {key})")

                            # Aggregate top-down, starting from root
                            agg_res, error_msg = compute_aggregate_results(interfaced_proc_hierarchy, filtered_values)

                            if error_msg is not None:
                                raise SolvingException(
                                    IType.ERROR, f"System: '{system}'. Interface: '{interface.name}'. "
                                                 f"Orientation: '{orientation}'. Error: {error_msg}")

                            if len(agg_res) > 0:
                                agg_results.setdefault(key, {}).update(agg_res)

    return agg_results


def flow_graph_solver(global_parameters: List[Parameter], problem_statement: ProblemStatement,
                      input_systems: Dict[str, Set[Processor]], state: State, dynamic_scenario: bool) -> List[Issue]:
    """
    A solver using the graph composed by the interfaces and the relationships (flows, part-of, scale, change-of-scale and relative-to)

    :param global_parameters: Parameters including the default value (if defined)
    :param problem_statement: ProblemStatement object, with scenarios (parameters changing the default)
                              and parameters for the solver
    :param state:             All variables available: object model, registry, datasets (inputs and outputs), ...
    :param input_systems:     A dictionary of the different systems to be solved
    :param dynamic_scenario:  If "True" store results in datasets separated from "fixed" scenarios.
                              Also "problem_statement" MUST have only one scenario with the parameters.
    :return: List of Issues
    """
    glb_idx, _, _, datasets, _ = get_case_study_registry_objects(state)
    InterfaceNode.registry = glb_idx

    absolute_observations, relative_observations = \
        split_observations_by_relativeness(get_evaluated_observations_by_time(glb_idx))

    if len(absolute_observations) == 0:
        raise SolvingException(
            IType.WARNING, f"No absolute observations have been found. The solver has nothing to solve."
        )

    scenario_states: Dict[str, State] = \
        {scenario_name: State(evaluate_parameters_for_scenario(global_parameters, scenario_params))
         for scenario_name, scenario_params in problem_statement.scenarios.items()}

    observation_results = get_scenario_evaluated_observation_results(scenario_states, absolute_observations)

    try:
        results = compute_flow_and_scale_results(state, glb_idx, scenario_states, observation_results, relative_observations)

        # Add "observation_results" to "results"
        for key, value in observation_results.items():
            results[key].update(value)

        agg_results = compute_partof_aggregates(glb_idx, input_systems, results)

    except SolvingException as e:
        return [Issue(e.args[0], str(e.args[1]))]

    # Add "agg_results" to "results"
    for key, value in agg_results.items():
        results[key].update(value)

    agg_results = compute_interfacetype_aggregates(glb_idx, results)

    # Add "agg_results" to "results"
    for key, value in agg_results.items():
        results[key].update(value)

    data = {k + node.key: {"RoegenType": node.roegen_type if node else "-",
                           "Value": value,
                           "Unit": node.unit if node else "-",
                           "Level": node.processor.attributes.get('level', '') if node else "-",
                           "System": node.system if node else "-",
                           "Subsystem": node.subsystem if node else "-",
                           "Sphere": node.sphere if node else "-"
                           }
            for k, v in results.items()
            for node, value in v.items()}

    df = pd.DataFrame.from_dict(data, orient='index')

    # Round all values to 3 decimals
    df = df.round(3)

    # Give a name to the dataframe indexes
    index_names = ["Scenario", "Period", "Scope"] + InterfaceNode.key_labels()  # "Processor", "Interface", "Orientation"
    df.index.names = index_names

    # Sort the dataframe based on indexes. Not necessary, only done for debugging purposes.
    df = df.sort_index(level=index_names)

    print(df)

    # Convert model to XML and to DOM tree
    xml, p_map = export_model_to_xml(glb_idx)
    dom_tree = etree.fromstring(xml).getroottree()

    # Calculate ScalarIndicators
    indicators = glb_idx.get(Indicator.partial_key())
    df_local_indicators = calculate_local_scalar_indicators(indicators, dom_tree, p_map, df, global_parameters, problem_statement)
    df_global_indicators = calculate_global_scalar_indicators(indicators, dom_tree, p_map, df, df_local_indicators, global_parameters, problem_statement)

    # Create datasets and store in State
    if not dynamic_scenario:
        ds_name = "flow_graph_solution"
        ds_indicators_name = "flow_graph_solution_indicators"
        df_global_indicators_name = "flow_graph_global_indicators"
    else:
        ds_name = "dyn_flow_graph_solution"
        ds_indicators_name = "dyn_flow_graph_solution_indicators"
        df_global_indicators_name = "dyn_flow_graph_global_indicators"

    datasets[ds_name] = get_dataset(df, ds_name, "Flow Graph Solver - Interfaces")

    if not df_local_indicators.empty:
        # Register dataset
        datasets[ds_indicators_name] = get_dataset(df_local_indicators, ds_indicators_name, "Flow Graph Solver - Local Indicators")

    if not df_global_indicators.empty:
        # Register dataset
        datasets[df_global_indicators_name] = get_dataset(df_global_indicators, df_global_indicators_name, "Flow Graph Solver - Global Indicators")

    #Create Matrix to Sanskey graph

    FactorsRelationDirectedFlowObservation_list = glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key())

    ds_flows = pd.DataFrame({'source': [i._source.full_name for i in FactorsRelationDirectedFlowObservation_list],
                              'source_processor': [i._source._processor._name for i in FactorsRelationDirectedFlowObservation_list],
                          'source_level':[ i._source._processor._attributes['level']  if ('level' in i._source._processor._attributes) else None  for i in FactorsRelationDirectedFlowObservation_list],
                              'target': [i._target.full_name for i in FactorsRelationDirectedFlowObservation_list],
                          'target_processor': [i._target._processor._name for i in FactorsRelationDirectedFlowObservation_list],
                              'target_level': [i._target._processor._attributes['level'] if 'level' in i._target._processor._attributes else None for i in FactorsRelationDirectedFlowObservation_list ],
                             # 'RoegenType_target': [i.target_factor._attributes['roegen_type']for i in FactorsRelationDirectedFlowObservation_list],
                             'Sphere_target': [i.target_factor._attributes['sphere'] for i in
                                                   FactorsRelationDirectedFlowObservation_list],
                             'Subsystem_target': [i._target._processor._attributes['subsystem_type'] for i in
                                               FactorsRelationDirectedFlowObservation_list],
                             'System_target': [i._target._processor._attributes['processor_system'] for i in
                                                  FactorsRelationDirectedFlowObservation_list]
                              }
                             )


    # I suppose that relations between processors (source-target) doesn't change between different scenarios.
    df2 = df.reset_index()
    processor = df2["Processor"].apply(lambda x: x.split("."))
    df2["lastprocessor"] = [i[-1] for i in processor]
    df2["source"] = df2["lastprocessor"] + ":" + df2["Interface"]
   # df2 = df2[df2["Orientation"]=="Output"] It is not necessary?
    ds_flow_values = pd.merge(df2,ds_flows, on="source")
    ds_flow_values = ds_flow_values.drop(columns = ["Orientation","lastprocessor","Processor", "Interface", 'RoegenType'], axis = 1)
    ds_flow_values = ds_flow_values.rename(columns = {'Sphere':'Sphere_source', 'System' : 'System_source', 'Subsystem': 'Subsystem_source' })
    # ds_flow_values.reset_index()
    ds_flows_name = "flow_graph_matrix"
    # if not ds_flows.empty:
    # Register flow dataset
    datasets[ds_flows_name] = get_dataset(ds_flow_values, ds_flows_name, "Flow Graph Matrix - Interfaces")

    # Calculate and publish MatrixIndicators
    indicators = glb_idx.get(MatrixIndicator.partial_key())
    matrices = prepare_matrix_indicators(indicators, glb_idx, dom_tree, p_map, df, df_local_indicators, dynamic_scenario)
    for n, ds in matrices.items():
        datasets[n] = ds

    # Create dataset and store in State (specific of "Biofuel case study")
    # datasets["end_use_matrix"] = get_ eum_dataset(df)

    return []


def obtain_subset_processors(processors_selector: str, serialized_model: lxml.etree._ElementTree,
                             registry: PartialRetrievalDictionary,
                             p_map: Dict[str, Processor], df: pd.DataFrame) -> pd.DataFrame:
    # TODO Processor names can be accompanied by: Level, System, Subsystem, Sphere
    processors = set()
    if processors_selector:
        try:
            r = serialized_model.xpath(processors_selector if case_sensitive else processors_selector.lower())
            for e in r:
                fname = e.get("fullname")
                if fname:
                    processors.add(p_map[fname])
                else:
                    pass  # Interfaces...
        except lxml.etree.XPathEvalError:
            # TODO Try CSSSelector syntax
            # TODO Generate Issue
            pass
    else:
        # If empty, ALL processors (do not filter)
        pass

    # Filter Processors
    if len(processors) > 0:
        # Obtain names of processor to KEEP
        processor_names = set([_.full_hierarchy_names(registry)[0] for _ in processors])
        if not case_sensitive:
            processor_names = set([_.lower() for _ in processor_names])

        p_names = df.index.unique(level="Processor").values
        p_names_case = [_ if case_sensitive else _.lower() for _ in p_names]
        p_names_corr = dict(zip(p_names_case, p_names))
        # https://stackoverflow.com/questions/18453566/python-dictionary-get-list-of-values-for-list-of-keys
        p_names = [p_names_corr[_] for _ in processor_names.intersection(p_names_case)]
        # Filter dataframe to only the desired Processors
        df2 = df.query('Processor in [' + ', '.join(['"' + p + '"' for p in p_names]) + ']')
    else:
        df2 = df

    return df2  # , processors


def prepare_matrix_indicators(indicators: List[MatrixIndicator],
                              registry: PartialRetrievalDictionary,
                              serialized_model: lxml.etree._ElementTree, p_map: Dict,
                              results: pd.DataFrame, indicator_results: pd.DataFrame,
                              dynamic_scenario: bool) -> Dict[str, Dataset]:
    """
    Compute Matrix Indicators

    :param indicators:
    :param registry:
    :param serialized_model:
    :param p_map:
    :param results: The matrix with all the input results
    :param indicator_results: Matrix with local scalar indicators
    :param dynamic_scenario: True if the matrices have to be prepared for a dynamic scenario
    :return: A dictionary <dataset_name> -> <dataset>
    """
    def prepare_matrix_indicator(indicator: MatrixIndicator) -> pd.DataFrame:
        """
        Compute a Matrix Indicator

        :param indicator: the MatrixIndicator to consider
        :param serialized_model: model as result of parsing XML serialization of model (Processors and Interfaces; no relationships nor observations)
        :param results: result of graph solver
        :return: a pd.DataFrame containing the desired matrix indicator
        """
        # Filter "Scope", if defined
        if indicator.scope:
            # TODO Consider case sensitiveness of "indicator.scope" (it is entered by the user)
            df = results.query('Scope in ("'+indicator.scope+'")')
        else:
            df = results

        # Apply XPath to obtain the dataframe filtered by the desired set of processors
        df = obtain_subset_processors(indicator.processors_selector, serialized_model, registry, p_map, df)

        # Filter Interfaces
        if indicator.interfaces_selector:
            ifaces = set([_.strip() for _ in indicator.interfaces_selector.split(",")])
            if not case_sensitive:
                ifaces = set([_.lower() for _ in ifaces])

            i_names = results.index.unique(level="Interface").values
            i_names_case = [_ if case_sensitive else _.lower() for _ in i_names]
            i_names_corr = dict(zip(i_names_case, i_names))
            i_names = [i_names_corr[_] for _ in ifaces]
            # Filter dataframe to only the desired Interfaces.
            df = df.query('Interface in [' + ', '.join(['"' + _ + '"' for _ in i_names]) + ']')

        # TODO Filter ScalarIndicators
        # TODO Indicator (scalar) names are accompanied by: Unit

        # Pivot Table: Dimensions (rows) are (Scenario, Period, Processor[, Scope])
        #              Dimensions (columns) are (Interface, Orientation -of Interface-)
        #              Measures (cells) are (Value)
        idx_columns = ["Scenario", "Period", "Processor"]
        if indicator.scope:
            idx_columns.append("Scope")
        df = df.pivot_table(values="Value", index=idx_columns, columns=["Interface", "Orientation"])
        # Flatten columns, concatenating levels
        df.columns = [f"{x} {y}" for x, y in zip(df.columns.get_level_values(0), df.columns.get_level_values(1))]

        # TODO Interface names are accompanied by: Orientation, RoegenType, Unit
        # TODO Output columns (MultiIndex?): external/internal/total,
        #  <scenarios>, <times>, <interfaces>, <scalar_indicators>

        return df

    # For each MatrixIndicator...
    result = {}
    for mi in indicators:
        df = prepare_matrix_indicator(mi)
        ds_name = mi.name
        if dynamic_scenario:
            ds_name = "dyn_"+ds_name
        ds = get_dataset(df, ds_name, mi.description)
        result[ds_name] = ds

    return result


def calculate_local_scalar_indicators(indicators: List[Indicator],
                                      serialized_model: lxml.etree._ElementTree, p_map: Dict[str, Processor],
                                      results: pd.DataFrame,
                                      global_parameters: List[Parameter], problem_statement: ProblemStatement) -> pd.DataFrame:
    """
    Compute local scalar indicators using data from "results", and return a pd.DataFrame

    :param indicators: List of indicators to compute
    :param serialized_model:
    :param p_map:
    :param results: Result of the graph solving process
    :param global_parameters: List of parameter definitions
    :param problem_statement: Object with a list of scenarios (defining Parameter sets)
    :return: pd.DataFrame with all the local indicators
    """

    def prepare_scalar_indicator(indicator: Indicator) -> pd.DataFrame:
        """

        :param indicator:
        :return:
        """

        # TODO Ignore "processors_selector". Considering it complicates the algorithm to
        #  append a new column (a Join is required)
        #  df = obtain_subset_processors(indicator.processors_selector, serialized_model, registry, p_map, results)
        df = results

        # Parse the expression
        ast = string_to_ast(indicator_expression, indicator.formula if case_sensitive else indicator.formula.lower())

        # Scenario parameters
        scenario_params = create_dictionary()
        for scenario_name, scenario_exp_params in problem_statement.scenarios.items():  # type: str, dict
            scenario_params[scenario_name] = evaluate_parameters_for_scenario(global_parameters, scenario_exp_params)

        issues = []
        new_df_rows_idx = []
        new_df_rows_data = []
        for t, g in df.groupby(idx_names):
            params = scenario_params[t[0]]
            # Elaborate a dictionary with dictionary values
            d = {}
            for row, sdf in g.iterrows():
                iface = sdf["Interface"]
                d[iface+"_"+sdf["Orientation"]] = sdf["Value"]
                if iface in d:
                    del d[iface]  # If exists, delete (only one appearance permitted, to avoid ambiguity)
                else:
                    d[iface] = sdf["Value"]  # First appearance allowed, insert
            d.update(params)
            # Variables for aggregator functions
            d["_processors_map"] = p_map
            d["_processors_dom"] = serialized_model
            if not case_sensitive:
                d = {k.lower(): v for k, v in d.items()}

            state = State(d)
            val, variables = ast_evaluator(ast, state, None, issues)
            if val:
                new_df_rows_idx.append(t)
                new_df_rows_data.append((indicator.name, val, None))
        df2 = pd.DataFrame(data=new_df_rows_data,
                           index=pd.MultiIndex.from_tuples(new_df_rows_idx, names=idx_names),
                           columns=["Indicator", "Value", "Unit"])
        return df2

    idx_names = ["Scenario", "Period", "Scope", "Processor"]
    idx_to_change = ["Interface", "Orientation"]
    results.reset_index(idx_to_change, inplace=True)
    # For each ScalarIndicator...
    dfs = []
    for si in indicators:
        if si._indicator_category == IndicatorCategories.factors_expression:
            dfi = prepare_scalar_indicator(si)
            if not dfi.empty:
                dfs.append(dfi)
    # Restore index
    results.set_index(idx_to_change, append=True, inplace=True)

    if dfs:
        return pd.concat(dfs)
    else:
        return pd.DataFrame()


def calculate_global_scalar_indicators(indicators: List[Indicator],
                                      serialized_model: lxml.etree._ElementTree, p_map: Dict[str, Processor],
                                      results: pd.DataFrame, local_indicators: pd.DataFrame,
                                      global_parameters: List[Parameter], problem_statement: ProblemStatement) -> pd.DataFrame:
    """
    Compute global scalar indicators using data from "results", and return a pd.DataFrame

    :param indicators: List of indicators to compute
    :param serialized_model:
    :param p_map:
    :param results: Result of the graph solving process
    :param global_parameters: List of parameter definitions
    :param problem_statement: Object with a list of scenarios (defining Parameter sets)
    :return: pd.DataFrame with all the local indicators
    """

    def prepare_global_scalar_indicator(indicator: Indicator) -> pd.DataFrame:
        """

        :param indicator:
        :return:
        """
        df = results

        # Parse the expression
        ast = string_to_ast(indicator_expression, indicator.formula if case_sensitive else indicator.formula.lower())

        # Scenario parameters
        scenario_params = create_dictionary()
        for scenario_name, scenario_exp_params in problem_statement.scenarios.items():  # type: str, dict
            scenario_params[scenario_name] = evaluate_parameters_for_scenario(global_parameters, scenario_exp_params)

        issues = []
        new_df_rows_idx = []
        new_df_rows_data = []
        for t, g in df.groupby(idx_names):  # GROUP BY Scenario, Period
            params = scenario_params[t[0]]  # Obtain parameter values from scenario, in t[0]
            # TODO ALL interfaces and local indicators from the selected processors, for the selected Scenario, Period, Scope should go to a DataFrame: rows->Processors, columns->(interfaces and indicators)
            extract = pd.DataFrame()
            # TODO If a specific indicator or interface from a processor is mentioned, put it as a variable
            # Elaborate a dictionary with dictionary values
            d = {}
            for row, sdf in g.iterrows():
                iface = sdf["Interface"]
                d[iface+"_"+sdf["Orientation"]] = sdf["Value"]
                if iface in d:
                    del d[iface]  # If exists, delete (only one appearance permitted, to avoid ambiguity)
                else:
                    d[iface] = sdf["Value"]  # First appearance allowed, insert
            d.update(params)
            # Variables for aggregator functions
            d["_processors_map"] = p_map
            d["_processors_dom"] = serialized_model
            d["_results"] = extract
            # d["_indicators"] = local_indicators
            if not case_sensitive:
                d = {k.lower(): v for k, v in d.items()}

            state = State(d)
            val, variables = ast_evaluator(ast, state, None, issues)

            if val:
                new_df_rows_idx.append(t)
                new_df_rows_data.append((indicator.name, val, None))
        df2 = pd.DataFrame(data=new_df_rows_data,
                           index=pd.MultiIndex.from_tuples(new_df_rows_idx, names=idx_names),
                           columns=["Indicator", "Value", "Unit"])
        return df2

    idx_names = ["Scenario", "Period", "Scope"]
    idx_to_change = []
    results.reset_index(idx_to_change, inplace=True)
    # For each ScalarIndicator...
    dfs = []
    if False:  # DISABLED, "prepare_global_scalar_indicator" STILL NOT FINISHED
        for si in indicators:
            if si._indicator_category == IndicatorCategories.case_study:
                dfi = prepare_global_scalar_indicator(si)
                if not dfi.empty:
                    dfs.append(dfi)
    # Restore index
    results.set_index(idx_to_change, append=True, inplace=True)

    if dfs:
        return pd.concat(dfs)
    else:
        return pd.DataFrame()


def compute_aggregate_results(tree: nx.DiGraph, params: NodeFloatDict) -> Tuple[NodeFloatDict, Optional[str]]:
    def compute_node(node: str) -> float:
        if params.get(node) is not None:
            return params[node]

        sum_children = None

        for pred in tree.predecessors(node):
            pred_value = compute_node(pred)
            if pred_value is not None:
                sum_children = (0 if sum_children is None else sum_children) + pred_value

        if sum_children is not None:
            values[node] = sum_children

        return sum_children

    root_nodes = [node for node, degree in tree.out_degree if degree == 0]
    if len(root_nodes) != 1 or root_nodes[0] is None:
        return {}, f"Root node cannot be taken from list '{root_nodes}'"

    values: NodeFloatDict = {}
    compute_node(root_nodes[0])
    return values, None


def get_eum_dataset(dataframe: pd.DataFrame) -> "Dataset":
    # EUM columns
    df = dataframe.query('Orientation == "Input" and '
                         'Interface in ["Biofuel", "CropProduction", "Fertilizer", "HA", "LU"]')

    # EUM rows
    df = df.query(
        'Processor in ['
        '"Society", "Society.Biodiesel", "Society.Bioethanol", '
        '"Society.CommerceImports", "Society.CommerceExports", '
        '"Society.Bioethanol.Cereals", '
        '"Society.Bioethanol.Cereals.Wheat", "Society.Bioethanol.Cereals.Maize", '
        '"Society.Bioethanol.Cereals.ExternalWheat", "Society.Bioethanol.Cereals.ExternalMaize", '
        '"Society.Bioethanol.SugarCrops", '
        '"Society.Bioethanol.SugarCrops.SugarBeet", "Society.Bioethanol.SugarCrops.SugarCane", '
        '"Society.Bioethanol.SugarCrops.ExternalSugarBeet", "Society.Bioethanol.SugarCrops.ExternalSugarCane", '
        '"Society.Biodiesel.OilCrops", '
        '"Society.Biodiesel.OilCrops.PalmOil", "Society.Biodiesel.OilCrops.RapeSeed", '
        '"Society.Biodiesel.OilCrops.SoyBean", '
        '"Society.Biodiesel.OilCrops.ExternalPalmOil", "Society.Biodiesel.OilCrops.ExternalRapeSeed", '
        '"Society.Biodiesel.OilCrops.ExternalSoyBean"'
        ']'
    )

    df = df.pivot_table(values="Value", index=["Scenario", "Period", "Processor", "Level"], columns="Interface")

    # Adding units to column name
    # TODO: remove hardcoded
    df = df.rename(columns={"Biofuel": "Biofuel (tonnes)",
                            "CropProduction": "CropProduction (tonnes)",
                            "Fertilizer": "Fertilizer (kg)",
                            "HA": "HA (h)",
                            "LU": "LU (ha)"})

    print(df)

    return get_dataset(df, "end_use_matrix", "End use matrix")


def get_dataset(dataframe: pd.DataFrame, code: str, description: str) -> "Dataset":
    ds = Dataset()
    ds.data = dataframe.reset_index()
    ds.code = code
    ds.description = description
    ds.attributes = {}
    ds.metadata = None
    ds.database = None

    if dataframe.index.names[0] != None:
        for dimension in dataframe.index.names:  # type: str
            d = Dimension()
            d.code = dimension
            d.description = None
            d.attributes = None
            d.is_time = (dimension.lower() == "period")
            d.is_measure = False
            cl = dataframe.index.unique(level=dimension).values
            d.code_list = CodeList.construct(
                dimension, dimension, [""],
                codes=[CodeImmutable(c, c, "", []) for c in cl]
            )
            d.dataset = ds

    for measure in dataframe.columns.values:  # type: str
        d = Dimension()
        d.code = measure
        d.description = None
        d.attributes = None
        d.is_time = False
        d.is_measure = True
        d.dataset = ds

    return ds
